"""Compare stack combiners on the same members, every one scored nested (weights fitted on 4 folds, scored on the 5th).

  python stack_alt.py <kaggle dirs...> --csv-dir DIR --members a,b,...

  lr        logistic regression on logits (the current stacker; weights may be negative)
  lr_c01    the same with C=0.01
  hill      Caruana hill climbing with replacement on logits, non-negative weights, AUC objective
  mean      plain mean of logits
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import pipeline as P
from stack import LOCAL, logit


def load(kdirs, csv_dir, members):
    tr, te, ss, id_col, targets = P.load()
    y = tr[targets[0]].astype(int).values
    oof, pred = {}, {}
    for n in LOCAL + ["lgb_vb_aux"]:
        f = P.OUT / "oof" / f"{n}.npy"
        if f.exists():
            oof[n], pred[n] = np.load(f), np.load(P.OUT / "pred" / f"{n}.npy")
    for kdir in kdirs:
        for f in sorted(Path(kdir).glob("oof_*.npy")):
            n = f.stem[4:]
            oof[n], pred[n] = np.load(f), np.load(Path(kdir) / f"pred_{n}.npy")
    if csv_dir:
        for f in sorted(Path(csv_dir).glob("oof_*.csv")):
            n = f.stem[4:]
            oof[n] = pd.read_csv(f).set_index("id")["pred"].loc[tr[id_col].values].values
            pred[n] = pd.read_csv(Path(csv_dir) / f"test_{n}.csv").set_index("id")["pred"].loc[te[id_col].values].values
    names = members or list(oof)
    Z = np.column_stack([logit(oof[n]) for n in names])
    ZT = np.column_stack([logit(pred[n]) for n in names])
    return y, names, Z, ZT, ss, targets[0]


def hill(Z, y, steps=60, sample=250_000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.choice(len(y), min(sample, len(y)), replace=False)
    Zs, ys = Z[i], y[i]
    w = np.zeros(Z.shape[1])
    best = -1
    for _ in range(steps):
        cand = []
        for j in range(Z.shape[1]):
            w2 = w.copy(); w2[j] += 1
            cand.append((roc_auc_score(ys, Zs @ w2), j))
        s, j = max(cand)
        if s <= best:
            break
        best, w[j] = s, w[j] + 1
    return w / w.sum()


def main(kdirs, csv_dir, members, write):
    y, names, Z, ZT, ss, target = load(kdirs, csv_dir, members)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(Z, y))
    methods = {
        "lr": lambda Za, ya: (lambda Zb: LogisticRegression(C=1.0, max_iter=3000).fit(Za, ya).decision_function(Zb)),
        "lr_c01": lambda Za, ya: (lambda Zb: LogisticRegression(C=0.01, max_iter=3000).fit(Za, ya).decision_function(Zb)),
        "hill": lambda Za, ya: (lambda Zb, w=hill(Za, ya): Zb @ w),
        "mean": lambda Za, ya: (lambda Zb: Zb.mean(1)),
    }
    res = {}
    for name, make in methods.items():
        out, fa = np.zeros(len(y)), []
        for a, b in folds:
            out[b] = make(Z[a], y[a])(Z[b])
            fa.append(roc_auc_score(y[b], out[b]))
        res[name] = np.array(fa)
        line = f"{name:8s} nested CV {roc_auc_score(y, out):.5f} folds {np.round(fa, 5).tolist()}"
        if name != "lr":
            d = res[name] - res["lr"]
            line += f" | vs lr {d.mean():+.6f}, better {(d > 0).sum()}/5"
        print(line, flush=True)
    w = hill(Z, y)
    print("hill weights:", {n: round(x, 3) for n, x in zip(names, w) if x > 0})
    if write:
        sub = ss.copy()
        sub[target] = 1 / (1 + np.exp(-(ZT @ w)))
        sub.to_csv(P.OUT / write, index=False)
        print("wrote", P.OUT / write)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kdirs", nargs="+")
    ap.add_argument("--csv-dir", default="")
    ap.add_argument("--members", default="")
    ap.add_argument("--write-hill", default="")
    a = ap.parse_args()
    main(a.kdirs, a.csv_dir, [m for m in a.members.split(",") if m], a.write_hill)
