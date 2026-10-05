"""Why do blends that win on OOF not win on the leaderboard?

Hypothesis: an OOF prediction comes from ONE fold model, a test prediction is the AVERAGE of five fold
models. Part of a blend's OOF gain is variance reduction, which the fold-averaged test prediction already
has, so the gain shrinks on test.

Test: hold out 20% of train as a fake test set H. On the other 80% run 5-fold CV for two LightGBMs that
differ only in seed and column sampling (pure variance diversity). Compare the 50/50 blend's gain over a
single model on (a) OOF, (b) H with fold-averaged predictions, (c) H with a single fold model's predictions.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import fe

LR = 0.1


def run():
    X, y, Xt, orig, target, *_ = fe.load_all()
    F, _, _ = fe.build(X, Xt.head(10), orig, target, "ote")
    d_idx, h_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=1)
    D, yD, H, yH = F.iloc[d_idx].reset_index(drop=True), y[d_idx], F.iloc[h_idx].reset_index(drop=True), y[h_idx]
    import lightgbm as lgb
    cfg = {"A": dict(random_state=42, colsample_bytree=0.7), "B": dict(random_state=7, colsample_bytree=0.5)}
    oof = {k: np.zeros(len(yD)) for k in cfg}
    h_avg = {k: np.zeros(len(yH)) for k in cfg}
    h_one = {}
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(D, yD)):
        for k, c in cfg.items():
            t0 = time.time()
            m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=LR, num_leaves=63, subsample=0.8, subsample_freq=1,
                                   reg_lambda=1.0, min_child_samples=40, n_jobs=fe.THREADS, verbose=-1, **c)
            m.fit(D.iloc[a], yD[a], eval_set=[(D.iloc[b], yD[b])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[k][b] = m.predict_proba(D.iloc[b])[:, 1]
            ph = m.predict_proba(H)[:, 1]
            h_avg[k] += ph / 5
            if f == 0:
                h_one[k] = ph
            print(f"fold {f} model {k} ({time.time() - t0:.0f}s)", flush=True)
    auc = roc_auc_score
    rows = []
    for name, pa, pb, yy in [("OOF (one fold model per row)", oof["A"], oof["B"], yD),
                             ("fake test, single fold model", h_one["A"], h_one["B"], yH),
                             ("fake test, average of 5 fold models", h_avg["A"], h_avg["B"], yH)]:
        rows.append({"scored on": name, "A": auc(yy, pa), "B": auc(yy, pb), "blend 50/50": auc(yy, (pa + pb) / 2),
                     "blend gain over A": auc(yy, (pa + pb) / 2) - auc(yy, pa)})
    r = pd.DataFrame(rows).set_index("scored on")
    print(r.round(5).to_string())
    r.to_csv(fe.P.OUT / "blend_gap.csv")


if __name__ == "__main__":
    run()
