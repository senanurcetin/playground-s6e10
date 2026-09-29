"""Generates the early public notebook (English): adversarial validation + an honest CV baseline.

  python build_notebook.py [out.ipynb]
The notebook finds the competition files under /kaggle/input (or $DATA locally), so it needs no
edits on launch day beyond the title/intro once the dataset is known.
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell as C, new_markdown_cell as M, new_notebook

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("s6e10-will-your-cv-transfer.ipynb")
cells = []
A = cells.append

A(M(r'''# S6E10: Will Your CV Transfer? Adversarial Validation First

Playground datasets are generated from an original dataset, and the generator does not always
reproduce the same distribution in train and test. Before tuning anything, it is worth asking one
question: **can a model tell a train row from a test row?**

- If it cannot (AUC ≈ 0.5), your cross-validation score is a fair estimate of the leaderboard.
- If it can, the features it uses are where your CV and the leaderboard will disagree.

This notebook answers that question on this month's data, then builds a baseline whose CV you can trust.'''))
A(C(r'''import glob, os, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss
from sklearn.model_selection import StratifiedKFold, KFold

hits = sorted(glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True))
DATA = Path(hits[0]).parent if hits else Path(os.environ["DATA"])
train, test, sub = (pd.read_csv(DATA / f) for f in ("train.csv", "test.csv", "sample_submission.csv"))
ID, TARGET = sub.columns[0], sub.columns[1]
y = train[TARGET]
TASK = "regression" if (y.dtype.kind in "fc" and y.nunique() > 20) else ("binary" if y.nunique() == 2 else "multiclass")
print(f"train {train.shape} | test {test.shape} | target {TARGET} | task {TASK}")
train.head()'''))
A(M("## 1. The data at a glance"))
A(C(r'''X, Xt = train.drop(columns=[ID, TARGET]), test.drop(columns=[ID])
cats = [c for c in X.columns if X[c].dtype == object]
for c in cats:
    cat_type = pd.CategoricalDtype(sorted(pd.concat([X[c], Xt[c]]).astype(str).unique()))
    X[c], Xt[c] = X[c].astype(str).astype(cat_type), Xt[c].astype(str).astype(cat_type)
summary = pd.DataFrame({"dtype": X.dtypes.astype(str), "n_unique": X.nunique(), "missing_train": X.isna().mean().round(4),
                        "missing_test": Xt.isna().mean().round(4)})
print(summary.to_string())
print("\ntarget:", y.describe().round(4).to_dict() if TASK == "regression" else y.value_counts(normalize=True).round(4).to_dict())'''))
A(M(r'''## 2. Adversarial validation

Label train rows 0 and test rows 1, then train a classifier to separate them with 5-fold CV.
The out-of-fold AUC measures how different the two sets are.'''))
A(C(r'''A = pd.concat([X, Xt], ignore_index=True)
a = np.r_[np.zeros(len(X)), np.ones(len(Xt))]
oof, imp = np.zeros(len(A)), pd.Series(0.0, index=A.columns)
for trn, val in StratifiedKFold(5, shuffle=True, random_state=42).split(A, a):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
    m.fit(A.iloc[trn], a[trn])
    oof[val] = m.predict_proba(A.iloc[val])[:, 1]
    imp += pd.Series(m.booster_.feature_importance("gain"), index=A.columns) / 5
auc = roc_auc_score(a, oof)
print(f"adversarial AUC: {auc:.4f}")
print("features that separate train from test most:")
print((imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_string())'''))
A(C(r'''verdict = ("train and test look alike: CV should transfer to the leaderboard" if auc < 0.55 else
           "train and test differ: validate on the train rows that look most like test")
print(verdict)
# the train rows the adversarial model finds most test-like make a stricter hold-out if needed
test_likeness = oof[:len(X)]
print("share of train rows scored as test-like (p > 0.5):", round(float((test_likeness > 0.5).mean()), 4))'''))
A(M(r'''## 3. A baseline whose CV you can trust

5-fold CV, early stopping on the validation fold only, and the out-of-fold score reported on
exactly the rows the model never trained on.'''))
A(C(r'''metric = {"binary": ("AUC", lambda t, p: roc_auc_score(t, p)),
          "regression": ("RMSE", lambda t, p: mean_squared_error(t, p) ** 0.5),
          "multiclass": ("log loss", lambda t, p: log_loss(t, p))}[TASK]
yy = y if TASK != "multiclass" else y.astype("category").cat.codes
split = (KFold if TASK == "regression" else StratifiedKFold)(5, shuffle=True, random_state=42)
Model = lgb.LGBMRegressor if TASK == "regression" else lgb.LGBMClassifier
n_class = yy.nunique() if TASK == "multiclass" else 1
oof = np.zeros((len(X), n_class)) if TASK == "multiclass" else np.zeros(len(X))
pred = np.zeros((len(Xt), n_class)) if TASK == "multiclass" else np.zeros(len(Xt))
scores = []
for k, (trn, val) in enumerate(split.split(X, yy)):
    m = Model(n_estimators=5000, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, min_child_samples=40, random_state=42, verbose=-1)
    m.fit(X.iloc[trn], yy.iloc[trn], eval_set=[(X.iloc[val], yy.iloc[val])], callbacks=[lgb.early_stopping(200, verbose=False)])
    p_val = m.predict(X.iloc[val]) if TASK == "regression" else m.predict_proba(X.iloc[val])
    p_te = m.predict(Xt) if TASK == "regression" else m.predict_proba(Xt)
    if TASK == "binary":
        p_val, p_te = p_val[:, 1], p_te[:, 1]
    oof[val] = p_val
    pred += p_te / 5
    scores.append(metric[1](yy.iloc[val], p_val))
    print(f"fold {k}: {metric[0]} {scores[-1]:.5f} ({m.best_iteration_} trees)")
print(f"\nCV {metric[0]}: {metric[1](yy, oof):.5f} (fold std {np.std(scores):.5f})")'''))
A(C(r'''if TASK == "multiclass":
    sub[TARGET] = np.array(y.astype("category").cat.categories)[pred.argmax(1)]
else:
    sub[TARGET] = pred
sub.to_csv("submission.csv", index=False)
sub.head()'''))
A(M(r'''## Takeaways
- Run adversarial validation **before** tuning. It takes a minute and tells you whether the CV you are
  about to optimise will mean anything on the leaderboard.
- If the AUC is high, the listed features are where train and test differ. Validate on the test-like
  train rows, or drop or reshape the features that give the difference away.
- Report CV as out-of-fold on rows the model never saw, with the fold spread. A gain smaller than the
  fold standard deviation is not a gain yet.'''))

nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                         "language_info": {"name": "python"}})
nbformat.write(nb, OUT)
print(f"wrote {OUT}")
