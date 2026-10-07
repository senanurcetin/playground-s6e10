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
| LGB, variant `ote+og+route+cnt+te` | 0.96092 | 0.96070 |
| blend: that model 0.714 + `ote` 0.143 + LGB with original rows 0.143 | 0.96110 | 0.96065 |
| **LR stack on logits**: 2 LGB (local) + XGB + CatBoost + 3 RealMLP (Kaggle GPU, `kaggle_stack/`) | **0.96139** | **0.96097** |

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

### Why OOF blend gains did not reach the LB (`python blend_gap.py`)

Public-LB sampling noise does not explain it: on random public-size subsets of the OOF the blend-vs-single
difference has sd 0.00007, and both blends lost by ~3 sd. The likelier cause is that an OOF prediction comes
from one fold model while a test prediction already averages five. Check with a 20% fake test set and two
LightGBMs that differ only in seed and column sampling (one split, so indicative):

| scored on | A | B | blend 50/50 | blend gain over A |
|---|---|---|---|---|
| OOF (one fold model per row) | 0.95963 | 0.95969 | 0.95994 | +0.00031 |
| fake test, single fold model | 0.95892 | 0.95924 | 0.95933 | +0.00041 |
| fake test, average of 5 fold models | 0.95947 | 0.95965 | 0.95964 | +0.00017 |

With fold-averaged predictions the blend no longer beats the better member (0.95964 vs 0.95965), although
its OOF says +0.00025. Seed/sampling diversity is already absorbed by fold averaging; a stack has to earn its
gain from genuinely different model families.

### Stack (`kaggle_stack/stack_gpu.py` on a Kaggle GPU, 38 min; `python stack.py <outputs>`)

| member | OOF AUC | stack weight |
|---|---|---|
| RealMLP, engineered features, 6 epochs | 0.96108 | 0.280 |
| XGBoost, engineered features (GPU) | 0.96105 | 0.269 |
| RealMLP, engineered features, seed 1 | 0.96105 | −0.029 |
| LightGBM, engineered features (local) | 0.96092 | 0.004 |
| CatBoost, engineered features (GPU) | 0.96090 | 0.123 |
| RealMLP, public recipe (raw columns + categorical twins) | 0.96070 | 0.285 |
| LightGBM, `ote` only (local) | 0.96008 | 0.069 |

Stack cross-validated 0.96139 (+0.00031 over the best member); public LB 0.96097 (+0.00027 over the best
single submission). Unlike the seed blends, this gain reached the LB: the weight sits on different model
families (RealMLP on raw columns, XGBoost, CatBoost), and the second seed of the same RealMLP gets none.
RealMLP picks its epoch on an internal 5% split, never on the scored fold.

Public write-up of the blend-gap experiment (3 splits, raw features): [S6E10: Why Your OOF Blend Gain Vanishes on Test](https://www.kaggle.com/code/senanuretin/s6e10-why-your-oof-blend-gain-vanishes-on-test).

### Auxiliary-task features (idea: goodpjw2008, S6E10 Part 4)

Predict each rating and three categoricals from the other 20 columns (train+test, no label, 5-fold
cross-fitted XGBoost on a Kaggle GPU) and add the expected rating sum_k k*P(k) as 13 features.

| member | before | with expected ratings | folds better |
|---|---|---|---|
| XGBoost | 0.96105 | 0.96116 | 4/5 |
| RealMLP | 0.96108 | 0.96118 | 4/5 |
| CatBoost | 0.96090 | 0.96101 | 4/5 |
| LightGBM (local, `fe.py` variant `+aux`) | 0.96092 | 0.96108 | 5/5 |

Stack: 0.96139 → **0.96150** (+0.00011, all five folds up, +0.00009 to +0.00012); public LB 0.96097 → **0.96120**. Submitted 03:05 on
2026-10-06 (`work/submit_at_0305.sh`; daily submissions reset at 00:00 UTC).

### Foundation-model members (TabPFN-3.5), from a public dataset

TabPFN-3.5 with the full 560k-row context needs ~17.5 GB for its KV cache and hours per fold on a Kaggle T4,
so these members come from goodpjw2008's public Kaggle dataset
[s6e10-tabpfn-member](https://www.kaggle.com/datasets/goodpjw2008/s6e10-tabpfn-member) (same
`StratifiedKFold(5, shuffle=True, random_state=42)`; its per-fold AUCs reproduce exactly on our folds).
`stack.py --csv-dir` reads them by id. Thank you to goodpjw2008 and to Prior Labs (TabPFN-3.5 licence:
non-commercial; Kaggle competitions allowed).

| stack | members | CV | public LB |
|---|---|---|---|
| ours, with auxiliary-task members | 11 | 0.96150 | 0.96120 |
| **+ 7 TabPFN-3.5 members** | 18 | **0.96179** (5/5 folds up) | **0.96140** (rank 139/847) |
| + TabICL v2 as well | 19 | 0.96179 | not submitted |
| + realmlp_pub_aux, cat_ctr, xgb_aux_d10 (Kaggle GPU) | 21 | **0.96183** | **0.96150** (rank 135/889) |
| + GLM-margin XGBoost x2, view-B RealMLP and XGBoost (`MEMBER_SET="glm"`) | 25 | **0.96185** | **0.96154** |
| + RealMLP flat-anneal (all 30 aux), XGBoost (all 30 aux), CatBoost CTR depth 6 (`glm2`) | 28 | 0.96185 (no gain, 3/5 folds) | not submitted |
| + view-B LightGBM (local `lgb_vb.py`, 0.96100 alone) | 26 | 0.96185 (no gain) | not submitted |

**Saturation (2026-10-06).** After the 25-member stack, six more members (two rounds) moved the nested CV by
less than 0.00001. Other combiners did not help either (`stack_alt.py`, nested): logistic regression 0.96185,
the same with C=0.01 0.96185, non-negative hill climbing 0.96182, mean of logits 0.96175. The public reference
stack of 48 members reaches 0.96192, and our 14 own members add nothing to it, so this feature set and these
families are near their ceiling. The remaining gap to the public top-100 cut (0.00006) is below what the public
LB can resolve. Final selection: 25-member stack (LB 0.96154) + 11-member stack without external members
(LB 0.96120).

### 10-fold members and rating x context crosses (2026-10-07, `MEMBER_SET="ten"`)

Ideas from the S6E10 discussion: 10-fold members add to a saturated stack (Kratosyan, #746148), and CatBoost
gains from each rating crossed with Class / Type of Travel / Customer Type (Shelton Wang, #745892).

| member | OOF AUC | 5-fold counterpart |
|---|---|---|
| CatBoost CTR + 39 rating x context crosses, 10 folds | **0.96137** | 0.96122 (5 folds), 0.96109 (no crosses) |
| XGBoost with expected ratings, 10 folds | 0.96133 | 0.96116 |
| RealMLP with expected ratings, 10 folds | 0.96121 | 0.96118 |

| stack | members | CV | public LB |
|---|---|---|---|
| **+ the four members above** | 29 | **0.96192** (5/5 folds up) | **0.96158** |
| + 10-fold view-B RealMLP/XGBoost, RealMLP public+aux, XGBoost all-30 (`ten2`) | 33 | 0.96192 (no gain) | not submitted |
