"""Why do blends that win on OOF not win on the leaderboard?

Hypothesis: an OOF prediction comes from ONE fold model, a test prediction is the AVERAGE of five fold
models. Part of a blend's OOF gain is variance reduction, which the fold-averaged test prediction already
has, so the gain shrinks on test.

Test: hold out 20% of train as a fake test set H. On the other 80% run 5-fold CV for three models:
  A  LightGBM
  B  LightGBM, other seed and column sampling     (A+B: pure variance diversity)
  X  XGBoost                                       (A+X: a different model family)
and compare each 50/50 blend with its better member on (a) OOF, (b) H with a single fold model's
predictions, (c) H with fold-averaged predictions. Repeated over several H splits.

  python blend_gap.py [--splits 3]
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import fe

LR = 0.1


def make(k):
    import lightgbm as lgb
    import xgboost as xgb
    common = dict(n_estimators=20000, learning_rate=LR, subsample=0.8, n_jobs=fe.THREADS)
    if k == "A":
        return lgb.LGBMClassifier(num_leaves=63, subsample_freq=1, reg_lambda=1.0, min_child_samples=40,
                                  colsample_bytree=0.7, random_state=42, verbose=-1, **common)
    if k == "B":
        return lgb.LGBMClassifier(num_leaves=63, subsample_freq=1, reg_lambda=1.0, min_child_samples=40,
                                  colsample_bytree=0.5, random_state=7, verbose=-1, **common)
    return xgb.XGBClassifier(max_depth=8, colsample_bytree=0.5, min_child_weight=5, tree_method="hist",
                             enable_categorical=True, eval_metric="auc", early_stopping_rounds=100,
                             random_state=42, **common)


def fit(k, m, Xa, ya, Xb, yb):
    if k == "X":
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    else:
        import lightgbm as lgb
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def one_split(F, y, seed):
    d_idx, h_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    D, yD, H, yH = F.iloc[d_idx].reset_index(drop=True), y[d_idx], F.iloc[h_idx].reset_index(drop=True), y[h_idx]
    keys = ["A", "B", "X"]
    oof = {k: np.zeros(len(yD)) for k in keys}
    h_avg = {k: np.zeros(len(yH)) for k in keys}
    h_one = {}
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(D, yD)):
        for k in keys:
            t0 = time.time()
            m = fit(k, make(k), D.iloc[a], yD[a], D.iloc[b], yD[b])
            oof[k][b] = m.predict_proba(D.iloc[b])[:, 1]
            ph = m.predict_proba(H)[:, 1]
            h_avg[k] += ph / 5
            if f == 0:
                h_one[k] = ph
            print(f"split {seed} fold {f} model {k} ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for view, P_, yy in [("OOF", oof, yD), ("test, single fold model", h_one, yH), ("test, 5-fold average", h_avg, yH)]:
        for pair in [("A", "B"), ("A", "X")]:
            s = {k: roc_auc_score(yy, P_[k]) for k in pair}
            blend = roc_auc_score(yy, (P_[pair[0]] + P_[pair[1]]) / 2)
            rows.append({"split": seed, "view": view, "pair": "+".join(pair), "best member": max(s.values()),
                         "blend": blend, "gain over best member": blend - max(s.values())})
    return rows


def run(splits):
    X, y, Xt, orig, target, *_ = fe.load_all()
    F, _, _ = fe.build(X, Xt.head(10), orig, target, "ote")
    rows = []
    for s in range(1, splits + 1):
        rows += one_split(F, y, s)
        r = pd.DataFrame(rows)
        r.to_csv(fe.P.OUT / "blend_gap.csv", index=False)
        print(r.pivot_table(index=["pair", "view"], columns="split", values="gain over best member").round(5)
              .to_string(), flush=True)
    r = pd.DataFrame(rows)
    print("\nmean gain of the 50/50 blend over its better member\n",
          r.groupby(["pair", "view"])["gain over best member"].agg(["mean", "std"]).round(5).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=3)
    run(ap.parse_args().splits)
