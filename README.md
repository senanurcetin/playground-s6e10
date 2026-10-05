# Playground S6E10 — launch-day kit

Generic tabular pipeline (`pipeline.py`), tested end to end on synthetic data.

## Launch day (competition opens 2026-10-01)
1. Join the competition on Kaggle (rules must be accepted by you).
2. `kaggle competitions download -c playground-series-s6e10 -p C:/Users/MONSTER/kaggle-data/playground-series-s6e10` and unzip.
3. `python pipeline.py check` — schema, task, default metric, **adversarial validation** (AUC ≈ 0.5 ⇒ CV transfers).
   Set `METRIC=<name>` if the competition metric differs (auc, logloss, rmse, rmsle, accuracy).
4. `python pipeline.py train --models lgb,xgb,cat` (add `--original PATH` once the source dataset is identified —
   original rows only ever train, never score).
5. `python pipeline.py blend` → `work/submission.csv`; submit.
   (Steps 2–5 in one go for a fast LightGBM baseline: `bash launch.sh`.)
6. Early public notebook: `python build_notebook.py` → "S6E10: Will Your CV Transfer? Adversarial Validation First" (tested; week 1 is when votes happen)
   — fits the honest-validation niche.

Data stays outside OneDrive: `C:/Users/MONSTER/kaggle-data/<competition>/`.

## Results log (Predicting Airline Satisfaction, AUC, ends 2026-10-31)

| submission | CV AUC | public LB |
|---|---|---|
| LightGBM baseline | 0.95904 | 0.95862 |
| blend: LGB 0.5 + LGB with original data 0.25 + XGB 0.25 | 0.95918 | 0.95855 |
| LGB + original-data target encoding per feature (`fe.py`, variant `ote`) | 0.96008 | 0.95965 |
| LGB, variant `ote+og+route+cnt+te` | **0.96092** | **0.96070** |
| blend: that model 0.714 + `ote` 0.143 + LGB with original rows 0.143 | 0.96110 | 0.96065 |

Adversarial AUC 0.500, so CV is the guide; the LB is a small sample. The blend's +0.00014 CV did not show on
the public LB (−0.00007): both gaps are inside noise. The original dataset (teejmahal20, 129,880 rows) adds
nothing measurable. CatBoost skipped (GPU memory shared with other jobs). Next: different model families or
target-encoded interactions rather than more of the same trees.

### Feature screen (`python fe.py screen`; same LightGBM, same 5 folds, lr 0.1)

| variant | CV AUC | vs base | folds better |
|---|---|---|---|
| base (raw features) | 0.95874 | | |
| agg: rating sum/std/zero/five counts, delay sum/diff | 0.95848 | −0.00025 | 0/5 |
| catr: ratings as categorical | 0.95867 | −0.00007 | 2/5 |
| **ote: per-feature target mean + count from the original dataset** | **0.95978** | **+0.00104** | **5/5** |
| ote2: the same for 45 feature pairs (no single-feature encodings) | 0.95859 | −0.00014 | 2/5 |

Appending the original rows as training data added nothing; using them as a lookup table of
smoothed satisfaction rates per value adds +0.001 on every fold. The original set is disjoint from
train and test, so the encoding needs no out-of-fold scheme and cannot leak competition labels.

### Round 2 (reference `ote`; same protocol)

| variant | CV AUC | vs ote | folds better |
|---|---|---|---|
| ote | 0.95978 | | |
| + mb: `max_bin` 4095 | 0.95986 | +0.00008 | 3/5 |
| + og: LightGBM fitted on the original rows only, its probability and logit as features | 0.96004 | +0.00026 | 5/5 |
| + og + route: per-`Flight Distance` mean of every other feature, and the value's count | 0.96050 | +0.00071 | 5/5 |
| **+ og + route + cnt + te**: value counts, fold-safe target encoding of distance/age keys | **0.96062** | **+0.00084** | **5/5** |
| + og + route + cnt + te + mb | 0.96063 | +0.00085 | 5/5 |

`Flight Distance` has ~3.5k distinct values, each repeated hundreds of times: it acts as a route ID, and
the route profile (no target involved) carries most of the gain. Idea credit: the public notebooks by
goodpjw2008 ("Route-ID FE + OG-Model Stack") and Дворкин Евгений ("LightGBM CV 5 folds"). Blends whose
OOF gain is under ~0.0002 have not shown on the public LB twice now: submit the single model unless the
blend adds a genuinely different model family.
