"""Feature-engineering screen for S6E10 (airline satisfaction, AUC).

Every variant is scored with the same LightGBM, the same 5 folds and the same screening budget
(lr 0.1), and compared with the base features fold by fold: a gain that does not hold on most
folds is noise at this sample size.

  python fe.py screen [--variants base,agg,catr,ote,ote2,all]
  python fe.py full --variant all --tag _fe      # lr 0.03, saves oof/pred for pipeline.py blend
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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


def build(X, Xt, orig, target, variant):
    parts = set(variant.split("+")) if variant != "all" else {"agg", "catr", "ote", "ote2"}
    A = pd.concat([X, Xt], ignore_index=True)
    feats = [A.copy()]
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
    return F.iloc[:len(X)].reset_index(drop=True), F.iloc[len(X):].reset_index(drop=True)


def cv(F, y, Ft, lr, folds=5, return_pred=False):
    import lightgbm as lgb
    oof, pred, fold_auc = np.zeros(len(F)), np.zeros(len(Ft)), []
    for trn, val in StratifiedKFold(folds, shuffle=True, random_state=P.SEED).split(F, y):
        m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=lr, num_leaves=63, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=1.0, min_child_samples=40, random_state=P.SEED,
                               n_jobs=THREADS, verbose=-1)
        m.fit(F.iloc[trn], y[trn], eval_set=[(F.iloc[val], y[val])], eval_metric="auc",
              callbacks=[lgb.early_stopping(200 if lr < 0.05 else 100, verbose=False)])
        oof[val] = m.predict_proba(F.iloc[val])[:, 1]
        fold_auc.append(roc_auc_score(y[val], oof[val]))
        if return_pred:
            pred += m.predict_proba(Ft)[:, 1] / folds
    return oof, pred, np.array(fold_auc)


def screen(variants):
    X, y, Xt, orig, target, *_ = load_all()
    res = {}
    for v in variants:
        t0 = time.time()
        F, Ft = build(X, Xt, orig, target, v)
        oof, _, fa = cv(F, y, Ft, lr=0.1)
        res[v] = fa
        line = f"{v:18s} CV {roc_auc_score(y, oof):.5f} | {F.shape[1]} feats | {time.time() - t0:.0f}s"
        if v != "base" and "base" in res:
            d = fa - res["base"]
            line += f" | vs base {d.mean():+.5f}, folds better {int((d > 0).sum())}/5 {np.round(d, 5).tolist()}"
        print(line, flush=True)


def full(variant, tag):
    X, y, Xt, orig, target, *_ = load_all()
    F, Ft = build(X, Xt, orig, target, variant)
    t0 = time.time()
    oof, pred, fa = cv(F, y, Ft, lr=0.03, return_pred=True)
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
