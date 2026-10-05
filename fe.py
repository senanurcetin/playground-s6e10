"""Feature-engineering screen for S6E10 (airline satisfaction, AUC).

Every variant is scored with the same LightGBM, the same 5 folds and the same screening budget
(lr 0.1), and compared with the base features fold by fold: a gain that does not hold on most
folds is noise at this sample size.

  python fe.py screen [--variants base,agg,catr,ote,ote2,all]   # the first variant is the reference
    blocks, joined with +: agg catr ote ote2 | og (original-only model) route (per-distance profile)
    cnt (value counts) te (fold-safe target encoding of distance/age keys) mb (max_bin 4095)
    aux (expected rating given the other columns; work/aux_features.parquet from kaggle_stack, or $AUX_PATH)
  python fe.py full --variant all --tag _fe      # lr 0.03, saves oof/pred for pipeline.py blend
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

import pipeline as P

ORIG = os.environ.get("ORIGINAL", "C:/Users/MONSTER/kaggle-data/airline-original/original.csv")
RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
           "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
           "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
THREADS = int(os.environ.get("THREADS", "8"))


def load_all():
    tr, te, ss, id_col, targets = P.load()
    target = targets[0]
    y = tr[target].astype(int).values
    X, Xt = tr.drop(columns=[id_col, target]), te.drop(columns=[id_col])
    orig = pd.read_csv(ORIG)
    orig[target] = orig[target].astype(int)
    return X, y, Xt, orig, target, ss, id_col


def agg(df):
    r = df[RATINGS]
    out = pd.DataFrame(index=df.index)
    out["r_sum"] = r.sum(1)
    out["r_std"] = r.std(1)
    out["r_zero"] = (r == 0).sum(1)
    out["r_five"] = (r == 5).sum(1)
    out["r_low"] = ((r >= 1) & (r <= 2)).sum(1)
    out["delay_sum"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
    out["delay_diff"] = df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]
    out["inflight"] = df[["Seat comfort", "Inflight entertainment", "Cleanliness", "Food and drink"]].mean(1)
    out["digital"] = df[["Inflight wifi service", "Ease of Online booking", "Online boarding"]].mean(1)
    return out


def orig_te(df, orig, target, cols_list, prior_w=20):
    """Smoothed target mean and count of each value (or value combination) in the ORIGINAL data.
    The original set is disjoint from train/test, so this needs no out-of-fold scheme."""
    prior = orig[target].mean()
    out = pd.DataFrame(index=df.index)
    for cols in cols_list:
        cols = list(cols)
        name = "ote_" + "|".join(c[:10] for c in cols)
        g = orig.groupby(cols, dropna=False)[target].agg(["sum", "count"])
        g["te"] = (g["sum"] + prior * prior_w) / (g["count"] + prior_w)
        m = df[cols].merge(g[["te", "count"]], left_on=cols, right_index=True, how="left")
        out[name] = m["te"].fillna(prior).values
        out[name + "_n"] = m["count"].fillna(0).values
    return out


FD = "Flight Distance"
NUMS = ["Age", FD, "Departure Delay in Minutes", "Arrival Delay in Minutes"]
TE_KEYS = [[FD], ["Age"], [FD, "Age"], [FD, "Class"], [FD, "Type of Travel"],
           ["Age", "Class", "Type of Travel", "Customer Type"]]


def og_model(A, orig, target):
    """LightGBM fitted on the original rows only, scored on train+test: no competition label involved."""
    import lightgbm as lgb
    cols = list(A.columns)
    B = pd.concat([A[cols], orig[cols]], ignore_index=True)
    for c in CATS:
        B[c] = B[c].astype("category")
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, reg_lambda=5.0, random_state=0, n_jobs=THREADS, verbose=-1)
    m.fit(B.iloc[len(A):], orig[target].values)
    p = np.clip(m.predict_proba(B.iloc[:len(A)])[:, 1], 1e-6, 1 - 1e-6)
    return pd.DataFrame({"og_p": p, "og_logit": np.log(p / (1 - p))}, index=A.index)


def route_profile(A):
    """Flight Distance behaves like a route ID: per-value mean of every other feature (no target)."""
    coded = A.copy()
    for c in CATS:
        coded[c] = coded[c].astype("category").cat.codes
    g = coded.groupby(FD)
    out = pd.DataFrame({f"fdm_{c}": g[c].transform("mean").astype(np.float32) for c in A.columns if c != FD})
    out["fd_cnt"] = g["Age"].transform("size").astype(np.float32)
    return out


def counts(A):
    return pd.DataFrame({f"cnt_{c}": A[c].map(A[c].value_counts()).fillna(0).astype(np.float32) for c in NUMS})


def te_keys(A):
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


def add_te(keys, trn, y_trn, others):
    """Fold-safe: training rows get inner 5-fold encodings; the other row sets use the whole training fold.
    Indices are positions in train+test."""
    tr_cols, oth_cols = {}, [dict() for _ in others]
    for name, codes in keys.items():
        ka, col = codes[trn], np.zeros(len(trn), dtype=np.float32)
        for i, j in KFold(5, shuffle=True, random_state=0).split(ka):
            col[j] = target_encode(ka[i], y_trn[i], ka[j])
        tr_cols[name] = col
        for d, idx in zip(oth_cols, others):
            d[name] = target_encode(ka, y_trn, codes[idx])
    return pd.DataFrame(tr_cols), [pd.DataFrame(d) for d in oth_cols]


def build(X, Xt, orig, target, variant):
    parts = set(variant.split("+")) if variant != "all" else {"agg", "catr", "ote", "ote2"}
    A = pd.concat([X, Xt], ignore_index=True)
    feats = [A.copy()]
    if "og" in parts:
        feats.append(og_model(A, orig, target))
    if "route" in parts:
        feats.append(route_profile(A))
    if "cnt" in parts:
        feats.append(counts(A))
    if "aux" in parts:                 # expected value of each rating given the other 20 columns (kaggle_stack)
        aux = pd.read_parquet(os.environ.get("AUX_PATH", str(P.OUT / "aux_features.parquet")))
        assert len(aux) == len(A)
        feats.append(aux[[c for c in aux.columns if c.startswith("aux_ev_")]].reset_index(drop=True))
    if "agg" in parts:
        feats.append(agg(A))
    if "ote" in parts:
        feats.append(orig_te(A, orig, target, [[c] for c in X.columns if c not in ("Arrival Delay in Minutes",)]))
    if "ote2" in parts:
        key = ["Online boarding", "Inflight wifi service", "Type of Travel", "Class", "Customer Type",
               "Inflight entertainment", "Seat comfort", "Ease of Online booking", "Leg room service", "Age"]
        feats.append(orig_te(A, orig, target, itertools.combinations(key, 2)))
    F = pd.concat(feats, axis=1)
    for c in CATS:
        F[c] = F[c].astype("category")
    if "catr" in parts:
        for c in RATINGS:
            F[c] = F[c].astype("category")
    keys = te_keys(A) if "te" in parts else None
    return F.iloc[:len(X)].reset_index(drop=True), F.iloc[len(X):].reset_index(drop=True), keys


def cv(F, y, Ft, lr, folds=5, return_pred=False, keys=None, max_bin=255):
    import lightgbm as lgb
    oof, pred, fold_auc = np.zeros(len(F)), np.zeros(len(Ft)), []
    test_idx = np.arange(len(F), len(F) + len(Ft))
    for trn, val in StratifiedKFold(folds, shuffle=True, random_state=P.SEED).split(F, y):
        Ftr, Fva, Fte = F.iloc[trn].reset_index(drop=True), F.iloc[val].reset_index(drop=True), Ft
        if keys is not None:
            a, (b, c) = add_te(keys, trn, y[trn], [val, test_idx if return_pred else val[:1]])
            Ftr, Fva = pd.concat([Ftr, a], axis=1), pd.concat([Fva, b], axis=1)
            if return_pred:
                Fte = pd.concat([Ft.reset_index(drop=True), c], axis=1)
        m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=lr, num_leaves=63, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=1.0, min_child_samples=40, random_state=P.SEED,
                               max_bin=max_bin, n_jobs=THREADS, verbose=-1)
        m.fit(Ftr, y[trn], eval_set=[(Fva, y[val])], eval_metric="auc",
              callbacks=[lgb.early_stopping(200 if lr < 0.05 else 100, verbose=False)])
        oof[val] = m.predict_proba(Fva)[:, 1]
        fold_auc.append(roc_auc_score(y[val], oof[val]))
        if return_pred:
            pred += m.predict_proba(Fte)[:, 1] / folds
    return oof, pred, np.array(fold_auc)


def max_bin_of(variant):
    return 4095 if "mb" in variant.split("+") else 255


def screen(variants):
    X, y, Xt, orig, target, *_ = load_all()
    res, ref = {}, variants[0]
    for v in variants:
        t0 = time.time()
        F, Ft, keys = build(X, Xt, orig, target, v)
        oof, _, fa = cv(F, y, Ft, lr=0.1, keys=keys, max_bin=max_bin_of(v))
        res[v] = fa
        line = f"{v:24s} CV {roc_auc_score(y, oof):.5f} | {F.shape[1] + len(keys or {})} feats | {time.time() - t0:.0f}s"
        if v != ref:
            d = fa - res[ref]
            line += f" | vs {ref} {d.mean():+.5f}, folds better {int((d > 0).sum())}/5 {np.round(d, 5).tolist()}"
        print(line, flush=True)


def full(variant, tag):
    X, y, Xt, orig, target, *_ = load_all()
    F, Ft, keys = build(X, Xt, orig, target, variant)
    t0 = time.time()
    oof, pred, fa = cv(F, y, Ft, lr=0.03, return_pred=True, keys=keys, max_bin=max_bin_of(variant))
    P.OUT.joinpath("oof").mkdir(parents=True, exist_ok=True); P.OUT.joinpath("pred").mkdir(exist_ok=True)
    np.save(P.OUT / "oof" / f"lgb{tag}.npy", oof); np.save(P.OUT / "pred" / f"lgb{tag}.npy", pred)
    print(f"lgb{tag} ({variant}): CV {roc_auc_score(y, oof):.5f} folds {np.round(fa, 5).tolist()} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "full"])
    ap.add_argument("--variants", default="base,agg,catr,ote,ote2,all")
    ap.add_argument("--variant", default="all")
    ap.add_argument("--tag", default="_fe")
    a = ap.parse_args()
    screen(a.variants.split(",")) if a.cmd == "screen" else full(a.variant, a.tag)
