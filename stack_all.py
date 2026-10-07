"""Stack our members with public prediction libraries and compare (nested CV, seed-7 folds).

  python stack_all.py [--write NAME]

Sources (all on StratifiedKFold(5, shuffle=True, random_state=42) unless their author used 10 folds):
  ours   : local LightGBM + Kaggle-GPU members (kaggle_stack/stack_gpu.py) + public TabPFN-3.5 (goodpjw2008)
  tfm    : nine tabular foundation models (cdeotte/s6e10-tfm-oof-and-test-predictions)
  pub124 : 124 public OOF/test pairs collected by Chris Deotte (cdeotte/s6e10-124-public-oof-and-test-predictions)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import pipeline as P

S = Path(r"C:/Users/MONSTER/AppData/Local/Temp/claude/C--Users-MONSTER-OneDrive-Belgeler-Kaggle/b14d68ed-4101-4188-83d1-e68b77d55492/scratchpad")
TFM = Path("C:/Users/MONSTER/kaggle-data/deotte-tfm")
PUB = Path("C:/Users/MONSTER/kaggle-data/deotte-124")
TABPFN = Path("C:/Users/MONSTER/kaggle-data/tabpfn-member")
OURS = ("lgb_route,lgb_ote,lgb_aux,cat_route,realmlp_pub,realmlp_v4,realmlp_v4_s1,xgb_route,xgb_aux,realmlp_aux,cat_aux,"
        "tabpfn_full_raw,tabpfn_full_raw_s1,tabpfn_te,tabpfn_te4,tabpfn_catfd,tabpfn_catfd_s1,tabpfn_catfd10,"
        "realmlp_pub_aux,cat_ctr,xgb_aux_d10,glm_xgb,glm6_xgb,realmlp_vb_aux,xgb_vb_aux,cat_ctrx,xgb_aux_10f,"
        "realmlp_aux_10f,cat_ctrx_10f").split(",")


def to_logit(v):
    v = v.astype(np.float64)
    if v.min() < 0 or v.max() > 1:                      # already a logit
        return np.clip(v, -30, 30)
    v = np.clip(v, 1e-7, 1 - 1e-7)
    return np.clip(np.log(v / (1 - v)), -30, 30)


def load():
    tr, te, ss, id_col, targets = P.load()
    y = tr[targets[0]].astype(int).values
    oof, pred, group = {}, {}, {}
    for n in ["lgb_route", "lgb_ote", "lgb_aux"]:
        oof[n], pred[n] = np.load(P.OUT / "oof" / f"{n}.npy"), np.load(P.OUT / "pred" / f"{n}.npy")
    for d in ["stack_full", "more_full", "glm_full", "ten_full"]:
        for f in (S / d).glob("oof_*.npy"):
            n = f.stem[4:]
            oof[n], pred[n] = np.load(f), np.load(S / d / f"pred_{n}.npy")
    for f in TABPFN.glob("oof_*.csv"):
        n = f.stem[4:]
        oof[n] = pd.read_csv(f).set_index("id")["pred"].loc[tr[id_col].values].values
        pred[n] = pd.read_csv(TABPFN / f"test_{n}.csv").set_index("id")["pred"].loc[te[id_col].values].values
    for n in OURS:
        group[n] = "ours"
    for src, tag in [(TFM, "tfm"), (PUB, "pub124")]:
        for f in src.glob("oof_*.npy"):
            n = f"{tag}:{f.stem[4:]}"
            if tag == "tfm" and f.stem[4:] in ("xgboost", "realmlp"):
                continue
            oof[n], pred[n] = np.load(f), np.load(src / f"test_{f.stem[4:]}.npy")
            group[n] = tag
    for n in group:
        assert len(oof[n]) == len(y) and len(pred[n]) == len(te), n
    return y, oof, pred, group, ss, targets[0]


def nested(Z, y, folds, C=1.0):
    out, fa = np.zeros(len(y)), []
    for a, b in folds:
        out[b] = LogisticRegression(C=C, max_iter=5000).fit(Z[a], y[a]).decision_function(Z[b])
        fa.append(roc_auc_score(y[b], out[b]))
    return roc_auc_score(y, out), np.array(fa)


def main(write):
    y, oof, pred, group, ss, target = load()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(y)), y))
    sets = {
        "ours+tfm": [n for n, g in group.items() if g in ("ours", "tfm")],
        "pub124+tfm": [n for n, g in group.items() if g in ("pub124", "tfm")],
        "ours+tfm+pub124": list(group),
    }
    res = {}
    for k, names in sets.items():
        Z = np.column_stack([to_logit(oof[n]) for n in names]).astype(np.float32)
        s, fa = nested(Z, y, folds)
        res[k] = fa
        line = f"{k:18s} {len(names):3d} members  nested CV {s:.5f} folds {np.round(fa, 5).tolist()}"
        if k != "ours+tfm":
            d = fa - res["ours+tfm"]
            line += f" | vs ours+tfm {d.mean():+.5f}, better {(d > 0).sum()}/5"
        print(line, flush=True)
    if write:
        names = sets["ours+tfm+pub124"]
        Z = np.column_stack([to_logit(oof[n]) for n in names]).astype(np.float32)
        ZT = np.column_stack([to_logit(pred[n]) for n in names]).astype(np.float32)
        st = LogisticRegression(C=1.0, max_iter=5000).fit(Z, y)
        sub = ss.copy()
        sub[target] = 1 / (1 + np.exp(-st.decision_function(ZT)))
        sub.to_csv(P.OUT / write, index=False)
        print("wrote", P.OUT / write)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default="")
    main(ap.parse_args().write)
