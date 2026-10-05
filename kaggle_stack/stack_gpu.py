"""S6E10 stack members on a Kaggle GPU: XGBoost, CatBoost and RealMLP on the fe.py feature set.

Same folds as pipeline.py / fe.py (StratifiedKFold(5, shuffle=True, random_state=42) on the train order),
so the out-of-fold predictions line up row for row with the LightGBM members trained locally.
Writes oof_<name>.npy / pred_<name>.npy to /kaggle/working after each member; the stack is fitted locally.

Feature blocks (see fe.py): ote = per-value satisfaction rate in the original data; og = model fitted on
the original rows only; route = per-Flight-Distance mean of every other feature; cnt = value counts;
te = fold-safe target encoding (inner 5-fold on the training part of each fold).
RealMLP picks its epoch on an internal 5% split of the training fold, never on the scored fold.
"""
import glob
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

SMOKE = False                      # True: 30k-row dry run to check the code end to end
warnings.filterwarnings("ignore")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytabkit"], check=False)
import torch  # noqa: E402

GPU = torch.cuda.is_available()
DEVICE = "cuda" if GPU else "cpu"
OUT = "/kaggle/working"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def find(name, hint):
    hits = sorted(p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if hint in p)
    assert hits, f"{name} ({hint}) not found"
    return hits[0]


TARGET, FD = "satisfaction", "Flight Distance"
RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
           "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
           "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMS = ["Age", FD, "Departure Delay in Minutes", "Arrival Delay in Minutes"]
TE_KEYS = [[FD], ["Age"], [FD, "Age"], [FD, "Class"], [FD, "Type of Travel"],
           ["Age", "Class", "Type of Travel", "Customer Type"]]

tr = pd.read_csv(find("train.csv", "playground-series-s6e10"))
te = pd.read_csv(find("test.csv", "playground-series-s6e10"))
og = pd.concat([pd.read_csv(find(f, "airline-passenger-satisfaction")) for f in ("train.csv", "test.csv")],
               ignore_index=True)
FEATS = [c for c in tr.columns if c not in ("id", TARGET)]
og[TARGET] = (og[TARGET] == "satisfied").astype(int)
og = og[FEATS + [TARGET]].reset_index(drop=True)
if SMOKE:
    tr, te = tr.sample(30000, random_state=0).reset_index(drop=True), te.head(5000).copy()
y = tr[TARGET].astype(int).values
NTR = len(tr)
A = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(NTR), y))
log(f"train {tr.shape} test {te.shape} original {og.shape} | GPU {GPU}")


# ----------------------------------------------------------------------------- static features
def orig_te(cols_list, prior_w=20):
    prior = og[TARGET].mean()
    out = pd.DataFrame(index=A.index)
    for cols in cols_list:
        g = og.groupby(cols, dropna=False)[TARGET].agg(["sum", "count"])
        g["te"] = (g["sum"] + prior * prior_w) / (g["count"] + prior_w)
        m = A[cols].merge(g[["te", "count"]], left_on=cols, right_index=True, how="left")
        name = "ote_" + "|".join(c[:10] for c in cols)
        out[name] = m["te"].fillna(prior).values
        out[name + "_n"] = m["count"].fillna(0).values
    return out


def og_model():
    import lightgbm as lgb
    B = pd.concat([A, og[FEATS]], ignore_index=True)
    for c in CATS:
        B[c] = B[c].astype("category")
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, reg_lambda=5.0, random_state=0, verbose=-1)
    m.fit(B.iloc[len(A):], og[TARGET].values)
    p = np.clip(m.predict_proba(B.iloc[:len(A)])[:, 1], 1e-6, 1 - 1e-6)
    return pd.DataFrame({"og_p": p, "og_logit": np.log(p / (1 - p))})


def route_profile():
    coded = A.copy()
    for c in CATS:
        coded[c] = coded[c].astype("category").cat.codes
    g = coded.groupby(FD)
    out = pd.DataFrame({f"fdm_{c}": g[c].transform("mean").astype(np.float32) for c in FEATS if c != FD})
    out["fd_cnt"] = g["Age"].transform("size").astype(np.float32)
    return out


def counts():
    return pd.DataFrame({f"cnt_{c}": A[c].map(A[c].value_counts()).fillna(0).astype(np.float32) for c in NUMS})


def te_keys():
    keys = {}
    for cols in TE_KEYS:
        code = np.zeros(len(A), dtype=np.int64)
        for c in cols:
            k = pd.factorize(A[c], use_na_sentinel=False)[0]
            code = code * (k.max() + 1) + k
        keys["te_" + "|".join(c[:6] for c in cols)] = pd.factorize(code)[0]
    return keys


def target_encode(k_fit, y_fit, k_apply, smooth=20.0):
    prior = y_fit.mean()
    s = pd.DataFrame({"k": k_fit, "y": y_fit}).groupby("k")["y"].agg(["sum", "count"])
    enc = (s["sum"] + prior * smooth) / (s["count"] + smooth)
    return pd.Series(k_apply).map(enc).fillna(prior).values.astype(np.float32)


def add_te(trn, y_trn, others):
    tr_cols, oth = {}, [dict() for _ in others]
    for name, codes in KEYS.items():
        ka, col = codes[trn], np.zeros(len(trn), dtype=np.float32)
        for i, j in KFold(5, shuffle=True, random_state=0).split(ka):
            col[j] = target_encode(ka[i], y_trn[i], ka[j])
        tr_cols[name] = col
        for d, idx in zip(oth, others):
            d[name] = target_encode(ka, y_trn, codes[idx])
    return pd.DataFrame(tr_cols), [pd.DataFrame(d) for d in oth]


OTE = orig_te([[c] for c in FEATS if c != "Arrival Delay in Minutes"])
OG = og_model()
ROUTE, CNT, KEYS = route_profile(), counts(), te_keys()
log(f"static features ready | og AUC vs competition labels {roc_auc_score(y, OG['og_p'].values[:NTR]):.5f}")

TREE = pd.concat([A, OTE, OG, ROUTE, CNT], axis=1)            # same blocks as fe.py ote+og+route+cnt (+te per fold)
for c in CATS:
    TREE[c] = TREE[c].astype("category")


def nn_frame(X):
    """RealMLP input: every raw numeric column also gets a categorical twin (one embedding per value)."""
    X = X.copy()
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(0.0)
    X = X.drop(columns=[c for c in ("og_p",) if c in X.columns])
    cat_cols = list(CATS)
    for c in NUMS + RATINGS:
        X[c + "_cat_"] = X[c].astype(int).astype(str)
        cat_cols.append(c + "_cat_")
    for c in cat_cols:
        X[c] = X[c].astype(str).astype("category")
    return X, cat_cols


NN_RAW, NN_CATS = nn_frame(A)
NN_V4, _ = nn_frame(pd.concat([A, OTE, OG, ROUTE, CNT], axis=1))


# ----------------------------------------------------------------------------- members
N_EST, ES = (300, 50) if SMOKE else (20000, 400)


def fit_xgb(Xa, ya, Xb, yb):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=N_EST, learning_rate=0.02, max_depth=8, subsample=0.8, colsample_bytree=0.5,
                          min_child_weight=5, reg_lambda=1.0, max_bin=256, tree_method="hist", device=DEVICE,
                          eval_metric="auc", early_stopping_rounds=ES, enable_categorical=True, max_cat_to_onehot=1,
                          random_state=42)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return lambda X: m.predict_proba(X)[:, 1]


def fit_cat(Xa, ya, Xb, yb):
    from catboost import CatBoostClassifier
    to = lambda X: X.astype({c: str for c in CATS})
    m = CatBoostClassifier(iterations=N_EST, learning_rate=0.05, depth=8, l2_leaf_reg=3.0, border_count=254,
                           task_type="GPU" if GPU else "CPU", eval_metric="Logloss", early_stopping_rounds=ES,
                           random_seed=42, cat_features=CATS, verbose=0, allow_writing_files=False)
    m.fit(to(Xa), ya, eval_set=(to(Xb), yb))
    return lambda X: m.predict_proba(to(X))[:, 1]


REALMLP = dict(n_ens=8, n_epochs=3, batch_size=256, use_early_stopping=False,
               lr=0.053, wd=0.0236, sq_mom=0.988, lr_sched="lin_cos_log_15", wd_sched="cos_log_15",
               first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=18,
               hidden_sizes=[512, 256, 128], act="silu", p_drop=0.05, p_drop_sched="expm4t",
               plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=2.33,
               ls_eps=0.01, ls_eps_sched="sqrt_cos", add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
               tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"])


def fit_realmlp(seed=42, **over):
    def fit(Xa, ya, Xb, yb):
        from pytabkit import RealMLP_TD_Classifier
        p = {**REALMLP, **over}
        if SMOKE:
            p.update(n_epochs=1, n_ens=2)
        m = RealMLP_TD_Classifier(**p, device=DEVICE, random_state=seed, verbosity=0,
                                  val_fraction=0.05, val_metric_name="1-auc_ovr")
        m.fit(Xa, ya, cat_col_names=NN_CATS)          # no X_val: the scored fold never picks the epoch
        return lambda X: m.predict_proba(X)[:, 1]
    return fit


def run(name, fit, X, use_te):
    Xtr, Xte = X.iloc[:NTR].reset_index(drop=True), X.iloc[NTR:].reset_index(drop=True)
    oof, tst, aucs, t0 = np.zeros(NTR), np.zeros(len(Xte)), [], time.time()
    test_idx = np.arange(NTR, len(X))
    for a, b in FOLDS:
        Xa, Xb, Xt = Xtr.iloc[a].reset_index(drop=True), Xtr.iloc[b].reset_index(drop=True), Xte
        if use_te:
            ta, (tb, tt) = add_te(a, y[a], [b, test_idx])
            Xa, Xb, Xt = pd.concat([Xa, ta], axis=1), pd.concat([Xb, tb], axis=1), pd.concat([Xte, tt], axis=1)
        predict = fit(Xa, y[a], Xb, y[b])
        oof[b] = predict(Xb)
        tst += predict(Xt) / len(FOLDS)
        aucs.append(roc_auc_score(y[b], oof[b]))
        if GPU:
            torch.cuda.empty_cache()
    np.save(f"{OUT}/oof_{name}.npy", oof)
    np.save(f"{OUT}/pred_{name}.npy", tst)
    log(f"{name:14s} OOF AUC {roc_auc_score(y, oof):.5f} folds {np.round(aucs, 5).tolist()} ({time.time() - t0:.0f}s)")


MEMBERS = [
    ("xgb_route", fit_xgb, TREE, True),
    ("cat_route", fit_cat, TREE, True),
    ("realmlp_v4", fit_realmlp(42, n_epochs=6), NN_V4, True),
    ("realmlp_pub", fit_realmlp(42), NN_RAW, False),
    ("realmlp_v4_s1", fit_realmlp(1, n_epochs=6), NN_V4, True),
]
for name, fit, X, use_te in MEMBERS:
    try:
        run(name, fit, X, use_te)
    except Exception as e:                            # one failing member must not lose the others
        log(f"{name} FAILED: {type(e).__name__}: {e}")
np.save(f"{OUT}/y.npy", y)
log("done")
