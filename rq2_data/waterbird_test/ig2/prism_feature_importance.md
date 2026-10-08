# Feature importance: waterbird_test

Only labels/tests that cleared FDR-corrected significance (p < 0.1) appear below, each table sorted by FDR-corrected p-value ascending -- lowest (strongest) p-value first, the row closest to the 0.1 cutoff last. See **`feature_importance_full.md`** (same folder) for every label regardless of significance, full methodology, Key Findings, and the per-image summary.

## Presence -- significant only (1 of 12 label(s))

| label | pass | fail | diff | odds ratio | 95% CI | FDR p |
|---|---|---|---|---|---|---|
| house | 0/87 (0%) | 2/13 (15%) | -15.4% | 0.026 | [0.01, 0.15] | 0.081 |

## Mass Share -- significant only (3 of 12 label(s))

| label | pass | fail | diff | mean mass_share (pass) | mean mass_share (fail) | AUC | 95% CI | FDR p |
|---|---|---|---|---|---|---|---|---|
| house | 0/87 (0%) | 2/13 (15%) | -15.4% |  | 0.116 | 0.423 | [31%, 50%] | 0.005 |
| boat | 1/87 (1%) | 2/13 (15%) | -14.2% | 0.005 | 0.04 | 0.429 | [31%, 51%] | 0.05 |
| people | 0/87 (0%) | 1/13 (8%) | -7.7% |  | 0.003 | 0.462 | [38%, 50%] | 0.063 |

## Peak Ratio -- significant only (3 of 12 label(s))

| label | pass | fail | diff | mean peak_ratio (pass) | mean peak_ratio (fail) | AUC | 95% CI | FDR p |
|---|---|---|---|---|---|---|---|---|
| house | 0/87 (0%) | 2/13 (15%) | -15.4% |  | 0.59 | 0.423 | [31%, 50%] | 0.005 |
| boat | 1/87 (1%) | 2/13 (15%) | -14.2% | 0.079 | 0.227 | 0.428 | [31%, 51%] | 0.05 |
| people | 0/87 (0%) | 1/13 (8%) | -7.7% |  | 0.12 | 0.462 | [38%, 50%] | 0.063 |

## Feature Combinations (Random Forest + SHAP)

**How to read this section, in plain terms**

Every table above tests ONE label at a time. This section instead asks: *if a model uses ALL labels together, can it separate pass from fail -- and which labels is it actually relying on?*

**Cross-validated ROC-AUC** -- plain meaning: train a Random Forest on most of the images (every label's mass_share as input), test it on held-out images it never saw, repeat across folds. Same 0.5=chance / 1.0=perfect scale as AUC above.
- Formula: standard ROC-AUC on out-of-fold predictions, averaged across up to 5 StratifiedKFold folds.
- What the value means: **read this FIRST -- it's a gate.** Near 0.5 means the model cannot actually separate pass/fail, so the SHAP numbers below are explaining noise, not a finding, no matter how large they look. Only trust the rest below if this is comfortably above 0.5.

**Feature importance (mean |SHAP|)** -- plain meaning: on average, how much does this label's value push the prediction away from baseline (in either direction)?
- Formula: SHAP assigns each feature a signed contribution per image; this is the average of the *absolute value* of that contribution across all images. Large = the model leans on it a lot; small = barely used.

**direction** -- plain meaning: does this label push predictions toward PASS or FAIL overall?
- Formula: Pearson correlation between the feature's own value and its own per-image SHAP value, across all images. Positive -> PASSING, negative -> FAILING. (Not just the average SHAP sign -- that can mislead when a feature's push is much bigger on one side, e.g. when absent, than the other.)

**Cross-validated ROC-AUC**: 0.455 (+/- 0.120 across folds), predicting pass/fail from all 12 feature(s) together over 100 images.

**Feature importance** (all 12 feature(s), sorted highest to lowest by mean |SHAP|; direction is which class this feature's presence/intensity typically pushes the prediction toward, jointly with every other feature):

| label | mean abs SHAP | direction |
|---|---|---|
| bird | 0.0803 | PASSING |
| boat | 0.0305 | FAILING |
| Unknown Background | 0.0299 | FAILING |
| tree | 0.0228 | FAILING |
| house | 0.0195 | FAILING |
| person | 0.0055 | PASSING |
| duck | 0.0051 | PASSING |
| water | 0.0041 | PASSING |
| rock | 0.0038 | FAILING |
| chair | 0.0009 | PASSING |
| statue | 0.0007 | PASSING |
| people | 0.0 | neutral |