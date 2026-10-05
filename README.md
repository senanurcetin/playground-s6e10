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
| LightGBM baseline | 0.95904 | **0.95862** |
| blend: LGB 0.5 + LGB with original data 0.25 + XGB 0.25 | 0.95918 | 0.95855 |

Adversarial AUC 0.500, so CV is the guide; the LB is a small sample. The blend's +0.00014 CV did not show on
the public LB (−0.00007): both gaps are inside noise. The original dataset (teejmahal20, 129,880 rows) adds
nothing measurable. CatBoost skipped (GPU memory shared with other jobs). Next: different model families or
target-encoded interactions rather than more of the same trees.
