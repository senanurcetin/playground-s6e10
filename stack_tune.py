"""Tune the 162-member stacker by nested CV only (never by the public LB).

  python stack_tune.py [--write NAME]

  1. L2 strength C in {0.01, 0.1, 1, 10}
  2. members whose stack weight is tiny and that are near-duplicates (logit corr > 0.999 with a stronger member)
     are dropped, then the stack is re-scored
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import pipeline as P
from stack_all import load, nested, to_logit


def main(write):
    y, oof, pred, group, ss, target = load()
    names = list(group)
    Z = np.column_stack([to_logit(oof[n]) for n in names]).astype(np.float32)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(y)), y))
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        s, fa = nested(Z, y, folds, C)
        print(f"C={C:<5} nested CV {s:.6f} folds {np.round(fa, 5).tolist()}", flush=True)
        if best is None or s > best[0]:
            best = (s, C, fa)
    s_ref, C, fa_ref = best
    # near-duplicate pruning
    auc = np.array([roc_auc_score(y, Z[:, j]) for j in range(Z.shape[1])])
    corr = np.corrcoef(Z[::7].T)
    order = np.argsort(-auc)
    keep = []
    for j in order:
        if all(corr[j, k] < 0.999 for k in keep):
            keep.append(j)
    Zk = Z[:, sorted(keep)]
    s, fa = nested(Zk, y, folds, C)
    d = fa - fa_ref
    print(f"pruned to {len(keep)} members (corr < 0.999): nested CV {s:.6f} | vs best C {d.mean():+.6f}, better {(d > 0).sum()}/5",
          flush=True)
    use = sorted(keep) if s > s_ref else list(range(Z.shape[1]))
    print(f"chosen: C={C}, {len(use)} members")
    if write:
        ZT = np.column_stack([to_logit(pred[names[j]]) for j in use]).astype(np.float32)
        st = LogisticRegression(C=C, max_iter=5000).fit(Z[:, use], y)
        sub = ss.copy()
        sub[target] = 1 / (1 + np.exp(-st.decision_function(ZT)))
        sub.to_csv(P.OUT / write, index=False)
        print("wrote", P.OUT / write)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default="")
    main(ap.parse_args().write)
