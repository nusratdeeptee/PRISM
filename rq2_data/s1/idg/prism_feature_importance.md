# Feature importance: S1

Only labels/tests that cleared FDR-corrected significance (p < 0.1) appear below, each table sorted by FDR-corrected p-value ascending -- lowest (strongest) p-value first, the row closest to the 0.1 cutoff last. See **`feature_importance_full.md`** (same folder) for every label regardless of significance, full methodology, Key Findings, and the per-image summary.

## Presence -- significant only (4 of 26 label(s))

| label | pass | fail | diff | odds ratio | 95% CI | FDR p |
|---|---|---|---|---|---|---|
| bush | 0/62 (0%) | 5/38 (13%) | -13.2% | 0.049 | [0.03, 0.20] | 0.058 |
| building | 0/62 (0%) | 5/38 (13%) | -13.2% | 0.049 | [0.03, 0.20] | 0.058 |
| wheel | 3/62 (5%) | 9/38 (24%) | -18.8% | 0.183 | [0.03, 0.58] | 0.058 |
| tree | 6/62 (10%) | 12/38 (32%) | -21.9% | 0.244 | [0.07, 0.66] | 0.058 |

## Mass Share -- significant only (4 of 26 label(s))

| label | pass | fail | diff | mean mass_share (pass) | mean mass_share (fail) | AUC | 95% CI | FDR p |
|---|---|---|---|---|---|---|---|---|
| bush | 0/62 (0%) | 5/38 (13%) | -13.2% |  | 0.032 | 0.434 | [38%, 49%] | 0.048 |
| building | 0/62 (0%) | 5/38 (13%) | -13.2% |  | 0.033 | 0.434 | [38%, 49%] | 0.048 |
| tree | 6/62 (10%) | 12/38 (32%) | -21.9% | 0.005 | 0.02 | 0.378 | [29%, 46%] | 0.048 |
| wheel | 3/62 (5%) | 9/38 (24%) | -18.8% | 0.022 | 0.008 | 0.411 | [34%, 48%] | 0.058 |

## Peak Ratio -- significant only (4 of 26 label(s))

| label | pass | fail | diff | mean peak_ratio (pass) | mean peak_ratio (fail) | AUC | 95% CI | FDR p |
|---|---|---|---|---|---|---|---|---|
| bush | 0/62 (0%) | 5/38 (13%) | -13.2% |  | 0.237 | 0.434 | [38%, 49%] | 0.048 |
| building | 0/62 (0%) | 5/38 (13%) | -13.2% |  | 0.228 | 0.434 | [38%, 49%] | 0.048 |
| tree | 6/62 (10%) | 12/38 (32%) | -21.9% | 0.095 | 0.164 | 0.377 | [29%, 46%] | 0.048 |
| wheel | 3/62 (5%) | 9/38 (24%) | -18.8% | 0.78 | 0.43 | 0.412 | [34%, 48%] | 0.058 |

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

**Cross-validated ROC-AUC**: 0.943 (+/- 0.029 across folds), predicting pass/fail from all 26 feature(s) together over 100 images.

**Feature importance** (all 26 feature(s), sorted highest to lowest by mean |SHAP|; direction is which class this feature's presence/intensity typically pushes the prediction toward, jointly with every other feature):

| label | mean abs SHAP | direction |
|---|---|---|
| coca-cola van | 0.1425 | PASSING |
| car | 0.1399 | PASSING |
| road | 0.0493 | PASSING |
| tree | 0.0371 | FAILING |
| wheel | 0.0256 | FAILING |
| coca-cola truck | 0.0206 | PASSING |
| building | 0.01 | FAILING |
| bush | 0.0092 | FAILING |
| truck | 0.008 | FAILING |
| license plate | 0.0065 | FAILING |
| traffic light | 0.0059 | FAILING |
| tire | 0.0045 | FAILING |
| Unknown Background | 0.0042 | FAILING |
| sidewalk | 0.0041 | FAILING |
| street road | 0.0034 | PASSING |
| bench | 0.0033 | FAILING |
| street | 0.0026 | FAILING |
| car wheel | 0.0016 | PASSING |
| car taillight | 0.0012 | PASSING |
| car license plate | 0.0011 | FAILING |
| car bumper | 0.0011 | PASSING |
| car windshield | 0.0008 | FAILING |
| window | 0.0007 | FAILING |
| taillight | 0.0004 | PASSING |
| fence | 0.0002 | PASSING |
| shadow | 0.0001 | PASSING |