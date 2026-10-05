"""Playground tabular pipeline: honest CV, three GBDTs, OOF ensemble, submission.

  python pipeline.py check            # schema, target/task/metric guess, adversarial validation
  python pipeline.py train [--models lgb,xgb,cat] [--folds 5] [--original PATH]
  python pipeline.py blend            # non-negative OOF weights -> submission.csv

Data (outside OneDrive): C:/Users/MONSTER/kaggle-data/<COMP>/{train,test,sample_submission}.csv
Every model writes oof/<name>.npy and pred/<name>.npy; the blend never sees test labels, so the
reported CV is the number to trust — and adversarial validation says how far to trust it.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

COMP = os.environ.get("COMP", "playground-series-s6e10")
DATA = Path(os.environ.get("DATA", f"C:/Users/MONSTER/kaggle-data/{COMP}"))
OUT = Path(__file__).resolve().parent / "work"
SEED = 42


# ----------------------------------------------------------------------------- data
def load():
    tr, te = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    ss = pd.read_csv(DATA / "sample_submission.csv")
    id_col, target = ss.columns[0], ss.columns[1] if ss.shape[1] == 2 else None
    targets = list(ss.columns[1:])
    return tr, te, ss, id_col, targets


def task_of(y: pd.Series):
    if y.dtype.kind in "fc" and y.nunique() > 20:
        return "regression"
    return "binary" if y.nunique() == 2 else "multiclass"


def metric_fn(task, name=None):
    name = name or os.environ.get("METRIC") or {"binary": "auc", "multiclass": "logloss", "regression": "rmse"}[task]
    fns = {"auc": (lambda y, p: roc_auc_score(y, p), True), "logloss": (lambda y, p: log_loss(y, p), False),
           "rmse": (lambda y, p: mean_squared_error(y, p) ** 0.5, False),
           "rmsle": (lambda y, p: mean_squared_error(np.log1p(y), np.log1p(np.clip(p, 0, None))) ** 0.5, False),
           "accuracy": (lambda y, p: accuracy_score(y, p.argmax(1) if p.ndim > 1 else (p > 0.5)), True)}
    f, higher = fns[name]
    return name, f, higher


def features(tr, te, id_col, target, original=None):
    X = tr.drop(columns=[id_col, target])
    Xt = te.drop(columns=[id_col])
    y = tr[target]
    if original is not None:                       # Playground data is synthesised from an original set
        o = original[[c for c in X.columns if c in original.columns] + [target]].dropna(subset=[target])
        X = pd.concat([X, o.drop(columns=[target])], ignore_index=True)
        y = pd.concat([y, o[target]], ignore_index=True)
    cats = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "category"]
    for c in cats:
        vals = pd.Categorical(pd.concat([X[c], Xt[c]]).astype(str))
        X[c] = pd.Categorical(X[c].astype(str), categories=vals.categories)
        Xt[c] = pd.Categorical(Xt[c].astype(str), categories=vals.categories)
    return X, y, Xt, cats


# ----------------------------------------------------------------------------- check
def check():
    tr, te, ss, id_col, targets = load()
    target = targets[0]
    task = task_of(tr[target])
    print(f"train {tr.shape}, test {te.shape}, id {id_col}, target {targets}, task {task}, metric {metric_fn(task)[0]}")
    print(tr.dtypes.value_counts().to_dict(), "| missing:", int(tr.isna().sum().sum()))
    print(tr[target].describe() if task == "regression" else tr[target].value_counts(normalize=True).round(4))
    # adversarial validation: can a model tell train from test? AUC ~0.5 means CV transfers
    import lightgbm as lgb
    X, _, Xt, _ = features(tr, te, id_col, target)
    A = pd.concat([X, Xt], ignore_index=True)
    a = np.r_[np.zeros(len(X)), np.ones(len(Xt))]
    oof = np.zeros(len(A))
    for trn, val in StratifiedKFold(5, shuffle=True, random_state=SEED).split(A, a):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
        m.fit(A.iloc[trn], a[trn])
        oof[val] = m.predict_proba(A.iloc[val])[:, 1]
    auc = roc_auc_score(a, oof)
    imp = pd.Series(m.feature_importances_, A.columns).sort_values(ascending=False).head(5)
    print(f"adversarial AUC {auc:.3f} ({'CV should transfer' if auc < 0.55 else 'train/test differ: check'}) | top: {imp.to_dict()}")


# ----------------------------------------------------------------------------- models
def fit_predict(name, Xtr, ytr, Xva, yva, Xte, task, n_class, cats):
    if name == "lgb":
        import lightgbm as lgb
        obj = {"binary": "binary", "multiclass": "multiclass", "regression": "regression"}[task]
        m = lgb.LGBMClassifier if task != "regression" else lgb.LGBMRegressor
        m = m(objective=obj, n_estimators=5000, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=1.0, min_child_samples=40, random_state=SEED, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(200, verbose=False)])
    elif name == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier if task != "regression" else xgb.XGBRegressor
        m = m(n_estimators=5000, learning_rate=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.7,
              enable_categorical=True, tree_method="hist", device=os.environ.get("XGB_DEVICE", "cuda"), early_stopping_rounds=200, random_state=SEED)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif name == "cat":
        import catboost as cb
        m = cb.CatBoostClassifier if task != "regression" else cb.CatBoostRegressor
        m = m(iterations=5000, learning_rate=0.05, depth=7, od_type="Iter", od_wait=200, random_seed=SEED,
              task_type="GPU", cat_features=cats, verbose=0)
        to = lambda X: X.astype({c: str for c in cats})
        Xtr, Xva, Xte = to(Xtr), to(Xva), to(Xte)
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
    else:
        raise ValueError(name)
    if task == "regression":
        return m.predict(Xva), m.predict(Xte)
    pv, pt = m.predict_proba(Xva), m.predict_proba(Xte)
    return (pv[:, 1], pt[:, 1]) if task == "binary" else (pv, pt)


def train(models, folds, original_path=None, tag=""):
    tr, te, ss, id_col, targets = load()
    target = targets[0]
    task = task_of(tr[target])
    original = pd.read_csv(original_path) if original_path else None
    X, y, Xt, cats = features(tr, te, id_col, target, original)
    n_train = len(tr)                                   # score only on competition rows
    classes = None
    if task == "multiclass":
        classes = sorted(y.unique())
        y = y.map({c: i for i, c in enumerate(classes)})
    mname, mfn, _ = metric_fn(task)
    split = (StratifiedKFold if task != "regression" else KFold)(folds, shuffle=True, random_state=SEED)
    OUT.mkdir(exist_ok=True); (OUT / "oof").mkdir(exist_ok=True); (OUT / "pred").mkdir(exist_ok=True)
    for name in models:
        t0 = time.time()
        shape = (len(X), len(classes)) if task == "multiclass" else (len(X),)
        oof, pred = np.zeros(shape), np.zeros((len(Xt),) + shape[1:])
        for k, (trn, val) in enumerate(split.split(X.iloc[:n_train], y.iloc[:n_train])):
            trn = np.r_[trn, np.arange(n_train, len(X))]   # original rows only ever train
            pv, pt = fit_predict(name, X.iloc[trn], y.iloc[trn], X.iloc[val], y.iloc[val], Xt, task,
                                 len(classes or []), cats)
            oof[val], pred = pv, pred + pt / folds
        score = mfn(y.iloc[:n_train], oof[:n_train])
        np.save(OUT / "oof" / f"{name}{tag}.npy", oof[:n_train]); np.save(OUT / "pred" / f"{name}{tag}.npy", pred)
        print(f"{name}{tag}: CV {mname} {score:.5f} ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "meta.json").write_text(json.dumps({"task": task, "target": target, "id": id_col, "classes": classes}))


def blend():
    tr, te, ss, id_col, targets = load()
    meta = json.loads((OUT / "meta.json").read_text())
    task, target = meta["task"], meta["target"]
    y = tr[target] if task != "multiclass" else tr[target].map({c: i for i, c in enumerate(meta["classes"])})
    names = sorted(p.stem for p in (OUT / "oof").glob("*.npy"))
    oofs = [np.load(OUT / "oof" / f"{n}.npy") for n in names]
    preds = [np.load(OUT / "pred" / f"{n}.npy") for n in names]
    mname, mfn, higher = metric_fn(task)
    # greedy hill climbing with replacement on OOF (Caruana et al.), weights stay non-negative
    best, w = None, np.zeros(len(names))
    for _ in range(50):
        cand = []
        for i in range(len(names)):
            w2 = w.copy(); w2[i] += 1
            p = sum(wi * o for wi, o in zip(w2, oofs)) / w2.sum()
            cand.append((mfn(y, p), i))
        s, i = (max if higher else min)(cand)
        if best is not None and (s <= best if higher else s >= best):
            break
        best, w[i] = s, w[i] + 1
    w = w / w.sum()
    print("single:", {n: round(mfn(y, o), 5) for n, o in zip(names, oofs)}, "| blend", round(best, 5), dict(zip(names, w.round(3))))
    p = sum(wi * pr for wi, pr in zip(w, preds))
    sub = ss.copy()
    if task == "multiclass" and len(targets) == 1:        # label submission
        sub[target] = np.array(meta["classes"])[p.argmax(1)]
    elif task == "multiclass":
        sub[targets] = p
    else:
        sub[target] = p
    sub.to_csv(OUT / "submission.csv", index=False)
    print(f"wrote {OUT / 'submission.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "train", "blend"])
    ap.add_argument("--models", default="lgb,xgb,cat")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--original", default=None)
    ap.add_argument("--tag", default="", help="suffix for saved oof/pred files, e.g. _orig")
    a = ap.parse_args()
    {"check": check, "train": lambda: train(a.models.split(","), a.folds, a.original, a.tag), "blend": blend}[a.cmd]()
