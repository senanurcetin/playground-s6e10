"""View-B LightGBM, trained locally on the view-B frame that kaggle_stack/stack_gpu.py saves (view_b.parquet).

  python lgb_vb.py <path to view_b.parquet>

View B (idea: goodpjw2008, S6E10 Part 2): digits of the numbers, GPT-2 tokens of the numbers, the original data's
satisfaction rate per value, the original-data model's logit, value counts, expected ratings; plus fold-safe target
encodings of its own keys (columns teb_*, encoded inside each fold with an inner 5-fold). Same folds as everything
else, so the OOF lines up with the other stack members. Writes work/oof/lgb_vb_aux.npy and work/pred/lgb_vb_aux.npy.
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import fe
import pipeline as P


def main(path):
    import lightgbm as lgb
    tr, te, ss, id_col, targets = P.load()
    y = tr[targets[0]].astype(int).values
    V = pd.read_parquet(path)
    assert len(V) == len(tr) + len(te)
    kcols = [c for c in V.columns if c.startswith("teb_")]
    keys = {c: V[c].values for c in kcols}
    X = V.drop(columns=kcols)
    X.columns = [str(c).replace("/", "_") for c in X.columns]
    Xtr, Xte = X.iloc[:len(tr)].reset_index(drop=True), X.iloc[len(tr):].reset_index(drop=True)
    test_idx = np.arange(len(tr), len(V))
    oof, pred, aucs, t0 = np.zeros(len(tr)), np.zeros(len(te)), [], time.time()
    for a, b in StratifiedKFold(5, shuffle=True, random_state=P.SEED).split(Xtr, y):
        ta, (tb, tt) = fe.add_te(keys, a, y[a], [b, test_idx])
        Xa = pd.concat([Xtr.iloc[a].reset_index(drop=True), ta], axis=1)
        Xb = pd.concat([Xtr.iloc[b].reset_index(drop=True), tb], axis=1)
        Xt = pd.concat([Xte, tt], axis=1)
        m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.02, num_leaves=127, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                               n_jobs=fe.THREADS, verbose=-1, random_state=42)
        m.fit(Xa, y[a], eval_set=[(Xb, y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[b] = m.predict_proba(Xb)[:, 1]
        pred += m.predict_proba(Xt)[:, 1] / 5
        aucs.append(roc_auc_score(y[b], oof[b]))
        print(f"fold {len(aucs)} AUC {aucs[-1]:.5f} ({time.time() - t0:.0f}s)", flush=True)
    np.save(P.OUT / "oof" / "lgb_vb_aux.npy", oof)
    np.save(P.OUT / "pred" / "lgb_vb_aux.npy", pred)
    print(f"lgb_vb_aux: CV {roc_auc_score(y, oof):.5f} folds {np.round(aucs, 5).tolist()} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
