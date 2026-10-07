"""Add members published after the cdeotte collection and re-score the stack (nested CV, C=0.01).

  python stack_new.py [--write NAME]
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import pipeline as P
from stack_all import load, nested, to_logit

NEW = Path("C:/Users/MONSTER/kaggle-data")


def extra():
    out = {}
    z = np.load(NEW / "new-s6e10-stack-oof-predictions" / "xgb_A_auxall.npz")
    out["sa2:xgb_A_auxall"] = (z["oof"], z["test"])
    fg = NEW / "new-s6e10-forge-own-pairs"
    for f in sorted(fg.glob("oof_*.npy")):
        out[f"forge:{f.stem[4:]}"] = (np.load(f), np.load(fg / f"test_{f.stem[4:]}.npy"))
    return out


def matrix(src, names):
    Z = np.empty((len(src[names[0]]), len(names)), dtype=np.float32)
    for j, n in enumerate(names):
        Z[:, j] = to_logit(src[n])
    return Z


def main(write):
    y, oof, pred, group, ss, target = load()
    for n, (o, t) in extra().items():
        assert len(o) == len(y) and len(t) == len(ss), n
        oof[n], pred[n], group[n] = o, t, "new"
    folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(y)), y))
    base = [n for n, g in group.items() if g != "new"]
    allm = list(group)
    for n in allm:                                     # keep memory low: float32 everywhere
        oof[n], pred[n] = oof[n].astype(np.float32), pred[n].astype(np.float32)
    Za = matrix(oof, allm)
    Zb = Za[:, :len(base)]                             # base members come first in `allm`
    s0, f0 = nested(Zb, y, folds, 0.01)
    s1, f1 = nested(Za, y, folds, 0.01)
    d = f1 - f0
    print(f"base {len(base)} members {s0:.6f} | + {len(allm) - len(base)} new {s1:.6f} | diff {d.mean():+.6f}, better {(d > 0).sum()}/5 "
          f"{np.round(d, 6).tolist()}", flush=True)
    if write and s1 > s0 and (d > 0).sum() >= 4:
        ZT = matrix(pred, allm)
        st = LogisticRegression(C=0.01, max_iter=5000).fit(Za, y)
        sub = ss.copy()
        sub[target] = 1 / (1 + np.exp(-st.decision_function(ZT)))
        sub.to_csv(P.OUT / write, index=False)
        print("wrote", P.OUT / write)
    elif write:
        print("no consistent gain: nothing written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default="")
    main(ap.parse_args().write)
