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
