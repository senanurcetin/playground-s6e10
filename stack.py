"""Stack local LightGBM members with the Kaggle-GPU members (kaggle_stack/stack_gpu.py outputs).

  python stack.py <dir with oof_*.npy, pred_*.npy, y.npy> [<more such dirs>] [--csv-dir DIR] [--members a,b,...]
                  [--out submission_stack.csv]

--csv-dir reads id-keyed members (oof_<name>.csv / test_<name>.csv with columns id, pred), e.g. the public
TabPFN-3.5 / TabICL predictions of goodpjw2008 (Kaggle dataset s6e10-tabpfn-member, same 5 folds).
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

LOCAL = ["lgb_route", "lgb_ote", "lgb_aux", "lgb_vb_aux"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main(kdirs, csv_dir, members, out_name):
    tr, te, ss, id_col, targets = P.load()
    y = tr[targets[0]].astype(int).values
    oof, pred = {}, {}
    for n in LOCAL:
        oof[n], pred[n] = np.load(P.OUT / "oof" / f"{n}.npy"), np.load(P.OUT / "pred" / f"{n}.npy")
    for kdir in kdirs:
        ky = np.load(Path(kdir) / "y.npy")
        assert len(ky) == len(y) and (ky == y).all(), f"{kdir}: rows are not in the local train order"
        for f in sorted(Path(kdir).glob("oof_*.npy")):
            n = f.stem[4:]
            oof[n], pred[n] = np.load(f), np.load(Path(kdir) / f"pred_{n}.npy")
    if csv_dir:
        for f in sorted(Path(csv_dir).glob("oof_*.csv")):
            n = f.stem[4:]
            oof[n] = pd.read_csv(f).set_index("id")["pred"].loc[tr[id_col].values].values
            pred[n] = pd.read_csv(Path(csv_dir) / f"test_{n}.csv").set_index("id")["pred"].loc[te[id_col].values].values
    names = [n for n in (members or oof) if n in oof]
    missing = [m for m in members if m not in oof]
    assert not missing, f"unknown members: {missing}"
    Z = np.column_stack([logit(oof[n]) for n in names])
    ZT = np.column_stack([logit(pred[n]) for n in names])
    cv, fold_auc = np.zeros(len(y)), []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(Z, y):
        cv[b] = LogisticRegression(C=1.0, max_iter=3000).fit(Z[a], y[a]).decision_function(Z[b])
        fold_auc.append(roc_auc_score(y[b], cv[b]))
    st = LogisticRegression(C=1.0, max_iter=3000).fit(Z, y)
    print(pd.DataFrame({"OOF AUC": [roc_auc_score(y, oof[n]) for n in names], "weight": st.coef_[0]},
                       index=names).round(5).to_string())
    best = max(names, key=lambda n: roc_auc_score(y, oof[n]))
    print(f"\n{len(names)} members | best single {best} {roc_auc_score(y, oof[best]):.5f} | mean of logits "
          f"{roc_auc_score(y, Z.mean(1)):.5f} | stack (cross-validated) {roc_auc_score(y, cv):.5f} "
          f"folds {np.round(fold_auc, 5).tolist()}")
    sub = ss.copy()
    sub[targets[0]] = 1 / (1 + np.exp(-st.decision_function(ZT)))
    sub.to_csv(P.OUT / out_name, index=False)
    print("wrote", P.OUT / out_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kdirs", nargs="+")
    ap.add_argument("--csv-dir", default="")
    ap.add_argument("--members", default="")
    ap.add_argument("--out", default="submission_stack.csv")
    a = ap.parse_args()
    main(a.kdirs, a.csv_dir, [m for m in a.members.split(",") if m], a.out)
