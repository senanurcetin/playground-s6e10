"""Generates the public analysis notebook: why OOF blend gains shrink on test.

  python build_blend_gap_notebook.py [out.ipynb]
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell as C, new_markdown_cell as M, new_notebook

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "nb_blend" / "s6e10-oof-blend-gain.ipynb"
cells = []
A = cells.append

A(M(r'''# S6E10: Why Your OOF Blend Gain Vanishes on Test

Twice this month I blended models whose out-of-fold (OOF) AUC beat my best single model, and twice the
blend scored *lower* on the public leaderboard:

| submission | OOF gain over best single | public LB gain |
|---|---|---|
| LightGBM + XGBoost + LightGBM (original data appended) | +0.00014 | −0.00007 |
| three LightGBMs on different feature sets | +0.00018 | −0.00005 |

The usual explanation is "the public LB is noisy". This notebook checks that explanation, finds it too
small, and tests a different one:

> **An OOF prediction comes from one fold model. A test prediction is the average of five.**
> Part of what a blend buys on OOF is variance reduction, and the fold-averaged test prediction already
> has it.

The experiment holds out a fake test set, so the effect is measured, not argued.

**Result (three splits):** a blend of two reseeded LightGBMs gains about +0.00023 on OOF but only about
+0.00005 on test predictions built the way submissions are built (fold-averaged). A LightGBM + XGBoost
blend keeps more: about +0.00030 on OOF, +0.00011 on test. Exact figures vary a little by run; the gap
between the views is the finding.'''))

A(C(r'''import glob, time
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

T0 = time.time()
path = sorted(glob.glob("/kaggle/input/**/playground-series-s6e10/train.csv", recursive=True)
              or glob.glob("/kaggle/input/**/train.csv", recursive=True))[0]
train = pd.read_csv(path)
y = train["satisfaction"].astype(int).values
X = train.drop(columns=["id", "satisfaction"])
for c in ["Gender", "Customer Type", "Type of Travel", "Class"]:
    X[c] = X[c].astype("category")
try:
    import torch; DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
print(X.shape, "| positive rate", y.mean().round(4), "| XGBoost on", DEVICE)'''))

A(M(r'''## 1. The setup

For each of three random splits, 20% of train becomes a **fake test set** (140k rows, more than twice
the public LB's ~60k). On the other 80% I run ordinary 5-fold CV for three models:

| model | what it is | blended with A, it tests |
|---|---|---|
| **A** | LightGBM | |
| **B** | LightGBM, other seed and column sampling | pure variance diversity |
| **X** | XGBoost | a different model family |

Each 50/50 blend (no fitted weights, so no weight-fitting optimism) is compared with its *better* member
in three views:

1. **OOF**: what we usually look at; every row is scored by one fold model;
2. **fake test, single fold model**: like OOF, but on unseen rows;
3. **fake test, average of the 5 fold models**: exactly how a submission is built.

Raw features only and learning rate 0.1: the point is the gap between the views, not the score.'''))

A(C(r'''def make(k):
    common = dict(n_estimators=20000, learning_rate=0.1, subsample=0.8, n_jobs=-1)
    if k == "A":
        return lgb.LGBMClassifier(num_leaves=63, subsample_freq=1, min_child_samples=40, colsample_bytree=0.7,
                                  random_state=42, verbose=-1, **common)
    if k == "B":
        return lgb.LGBMClassifier(num_leaves=63, subsample_freq=1, min_child_samples=40, colsample_bytree=0.5,
                                  random_state=7, verbose=-1, **common)
    return xgb.XGBClassifier(max_depth=8, colsample_bytree=0.5, min_child_weight=5, tree_method="hist", device=DEVICE,
                             enable_categorical=True, eval_metric="auc", early_stopping_rounds=100, random_state=42, **common)

def fit(k, Xa, ya, Xb, yb):
    m = make(k)
    if k == "X":
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    else:
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    return m

KEYS, rows, keep = ["A", "B", "X"], [], {}
for split in (1, 2, 3):
    d, h = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=split)
    D, yD, H, yH = X.iloc[d], y[d], X.iloc[h], y[h]
    oof = {k: np.zeros(len(yD)) for k in KEYS}
    avg = {k: np.zeros(len(yH)) for k in KEYS}
    one = {}
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(D, yD)):
        for k in KEYS:
            m = fit(k, D.iloc[a], yD[a], D.iloc[b], yD[b])
            oof[k][b] = m.predict_proba(D.iloc[b])[:, 1]
            p = m.predict_proba(H)[:, 1]
            avg[k] += p / 5
            if f == 0:
                one[k] = p
    for view, P, yy in [("OOF", oof, yD), ("fake test, single fold model", one, yH),
                        ("fake test, 5-fold average", avg, yH)]:
        for pair in [("A", "B"), ("A", "X")]:
            best = max(roc_auc_score(yy, P[k]) for k in pair)
            blend = roc_auc_score(yy, (P[pair[0]] + P[pair[1]]) / 2)
            rows.append({"split": split, "view": view, "pair": "+".join(pair), "gain": blend - best})
    if split == 1:
        keep = dict(oof=oof, yD=yD)
    print(f"split {split} done ({time.time() - T0:.0f}s)")
res = pd.DataFrame(rows)'''))

A(M("## 2. Result: gain of the 50/50 blend over its better member"))
A(C(r'''order = ["OOF", "fake test, single fold model", "fake test, 5-fold average"]
table = (res.pivot_table(index=["pair", "view"], columns="split", values="gain")
            .assign(mean=lambda t: t.mean(1), sd=lambda t: t.iloc[:, :3].std(1)))
table = table.reindex([(p, v) for p in ["A+B", "A+X"] for v in order])
print((table * 1e5).round(1).to_string(), "\n(units: 1e-5 AUC)")'''))
A(C(r'''fig, ax = plt.subplots(figsize=(9, 3.6))
w = 0.38
for i, (pair, col) in enumerate([("A+B", "#c4553a"), ("A+X", "#3a7dc4")]):
    m = table.loc[pair, "mean"].reindex(order) * 1e5
    s = table.loc[pair, "sd"].reindex(order) * 1e5
    ax.bar(np.arange(3) + (i - 0.5) * w, m, w, yerr=s, capsize=3, color=col,
           label={"A+B": "A+B: same model, other seed", "A+X": "A+X: LightGBM + XGBoost"}[pair])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels(["OOF", "fake test,\nsingle fold model", "fake test,\n5-fold average"])
ax.set_ylabel("blend gain over better member\n(1e-5 AUC)"); ax.legend(fontsize=8); ax.set_title("mean ± sd over 3 splits")
plt.tight_layout(); plt.show()'''))

A(M(r'''## 3. Is it just public-LB noise?

thisray measured this carefully in
[S6E10: is your 0.0001 gain real? Noise measured](https://www.kaggle.com/code/thisray/s6e10-is-your-0-0001-gain-real-noise-measured):
on simulated public splits, two reseeds of the same model come out in the wrong order 44% of the time and a
0.00011 tweak 16% of the time, while a 0.0005 feature change is always ordered correctly. (The leaderboard
page does state the share: "approximately 20% of the test data", so ~60k rows.)

The same check for the A+B blend here, resampling its OOF predictions at public-LB size:'''))
A(C(r'''rng = np.random.default_rng(0)
oof, yD = keep["oof"], keep["yD"]
blend = (oof["A"] + oof["B"]) / 2
full = roc_auc_score(yD, blend) - max(roc_auc_score(yD, oof[k]) for k in "AB")
best_k = max("AB", key=lambda k: roc_auc_score(yD, oof[k]))
diffs = []
for _ in range(300):
    i = rng.choice(len(yD), 60_000, replace=False)
    diffs.append(roc_auc_score(yD[i], blend[i]) - roc_auc_score(yD[i], oof[best_k][i]))
diffs = np.array(diffs)
print(f"A+B blend vs better member on all OOF rows: {full:+.5f}")
print(f"on 60k-row samples: sd {diffs.std():.5f}, share with the opposite sign {np.mean(np.sign(diffs) != np.sign(full)):.2f}")'''))
A(M(r'''At ~0.0002 the OOF blend gain is two to three standard deviations of the paired public-LB difference, so
sampling noise alone would rarely reverse it. My two blends lost by about three. They more likely never had
the gain on test, which is what section 2 shows.

## Takeaways

- **Do not trust an OOF blend gain that comes from seed or sampling diversity.** The 5-fold-averaged test
  prediction is already an ensemble of that kind, and the gain largely disappears.
- **Gains from a different model family survive in part.** That is where a stack earns its keep: in my own
  stack the weight went to RealMLP, XGBoost and CatBoost, and a second seed of the same RealMLP got none.
  That stack's gain did reach the leaderboard.
- **Score candidate blends the way you submit.** Hold out a slice, average your fold models on it, and
  compare there. A 50/50 blend is a cheaper first check than fitted weights.
- **Read a leaderboard loss with the paired noise in mind.** Below ~0.0001 the public LB cannot order two
  submissions (thisray's numbers); a 0.0002 OOF gain that turns into a loss is more likely not there.

Earlier in this series: [S6E10: Will Your CV Transfer? Check This First](https://www.kaggle.com/code/senanuretin/s6e10-will-your-cv-transfer-check-this-first).'''))
A(C(r'''print(f"total runtime {time.time() - T0:.0f}s")'''))

nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                         "language_info": {"name": "python"}})
OUT.parent.mkdir(exist_ok=True)
nbformat.write(nb, OUT)
print(f"wrote {OUT}")
