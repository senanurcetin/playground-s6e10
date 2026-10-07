"""Non-negative hill-climbing stack over all 169 members (the robust combiner in S6E9's private LB), nested CV.

  python stack_hill.py [--write NAME]

Caruana hill climbing with replacement on member logits, AUC objective on a 250k-row sample of the training part,
at most 80 steps; scored on the held-out fold of the seed-7 split. The same 169 members as stack_new.py.
"""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import pipeline as P
from stack_all import load
from stack_new import extra, matrix


def hill(Z, y, steps=80, sample=250_000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.choice(len(y), min(sample, len(y)), replace=False)
    Zs, ys = Z[i], y[i]
    w = np.zeros(Z.shape[1])
    cur = np.zeros(len(ys))
    best = -1.0
    for _ in range(steps):
        scores = [roc_auc_score(ys, cur + Zs[:, j]) for j in range(Z.shape[1])]
        j = int(np.argmax(scores))
        if scores[j] <= best:
            break
        best, w[j], cur = scores[j], w[j] + 1, cur + Zs[:, j]
    return w / w.sum()


def main(write):
    y, oof, pred, group, ss, target = load()
    for n, (o, t) in extra().items():
        oof[n], pred[n], group[n] = o, t, "new"
    names = list(group)
    for n in names:
        oof[n], pred[n] = oof[n].astype(np.float32), pred[n].astype(np.float32)
    Z = matrix(oof, names)
    out, fa = np.zeros(len(y)), []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(Z, y):
        w = hill(Z[a], y[a])
        out[b] = Z[b] @ w
        fa.append(roc_auc_score(y[b], out[b]))
        print(f"fold {len(fa)} AUC {fa[-1]:.6f} | {int((w > 0).sum())} members", flush=True)
    print(f"hill nested CV {roc_auc_score(y, out):.6f} folds {np.round(fa, 6).tolist()}", flush=True)
    w = hill(Z, y)
    print("weights:", {names[j]: round(float(w[j]), 3) for j in np.flatnonzero(w)})
    if write:
        ZT = matrix(pred, names)
        sub = ss.copy()
        sub[target] = 1 / (1 + np.exp(-(ZT @ w)))
        sub.to_csv(P.OUT / write, index=False)
        print("wrote", P.OUT / write)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default="")
    main(ap.parse_args().write)
