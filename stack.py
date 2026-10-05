"""Stack local LightGBM members with the Kaggle-GPU members (kaggle_stack/stack_gpu.py outputs).

  python stack.py <dir with oof_*.npy, pred_*.npy, y.npy from Kaggle> [--members lgb_route,xgb_route,...]

Logistic regression on the members' logits (AUC depends only on ranks). The stack's own score is
cross-validated on a different split (seed 7), so the reported number is not fitted on its own rows.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import pipeline as P

LOCAL = ["lgb_route", "lgb_ote", "lgb_aux"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main(kdir, members):
    tr, te, ss, id_col, targets = P.load()
    y = tr[targets[0]].astype(int).values
    ky = np.load(Path(kdir) / "y.npy")
    assert len(ky) == len(y) and (ky == y).all(), "Kaggle rows are not in the local train order"
    oof, pred = {}, {}
    for n in LOCAL:
        oof[n], pred[n] = np.load(P.OUT / "oof" / f"{n}.npy"), np.load(P.OUT / "pred" / f"{n}.npy")
    for f in sorted(Path(kdir).glob("oof_*.npy")):
        n = f.stem[4:]
        oof[n], pred[n] = np.load(f), np.load(Path(kdir) / f"pred_{n}.npy")
    names = [n for n in (members or oof) if n in oof]
    Z = np.column_stack([logit(oof[n]) for n in names])
    ZT = np.column_stack([logit(pred[n]) for n in names])
    cv = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(Z, y):
        cv[b] = LogisticRegression(C=1.0, max_iter=2000).fit(Z[a], y[a]).decision_function(Z[b])
    st = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)
    corr = pd.DataFrame(Z, columns=names).corr().round(4)
    print(pd.DataFrame({"OOF AUC": [roc_auc_score(y, oof[n]) for n in names], "weight": st.coef_[0]},
                       index=names).round(5).to_string())
    print("\nlogit correlation\n", corr.to_string())
    best = max(names, key=lambda n: roc_auc_score(y, oof[n]))
    s = roc_auc_score(y, cv)
    print(f"\nbest single {best} {roc_auc_score(y, oof[best]):.5f} | mean of logits {roc_auc_score(y, Z.mean(1)):.5f}"
          f" | stack (cross-validated) {s:.5f}")
    sub = ss.copy()
    sub[targets[0]] = 1 / (1 + np.exp(-st.decision_function(ZT)))
    sub.to_csv(P.OUT / "submission_stack.csv", index=False)
    print("wrote", P.OUT / "submission_stack.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kdir")
    ap.add_argument("--members", default="")
    a = ap.parse_args()
    main(a.kdir, [m for m in a.members.split(",") if m])
