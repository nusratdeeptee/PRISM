# Feature importance: S2

Only labels/tests that cleared FDR-corrected significance (p < 0.1) appear below, each table sorted by FDR-corrected p-value ascending -- lowest (strongest) p-value first, the row closest to the 0.1 cutoff last. See **`feature_importance_full.md`** (same folder) for every label regardless of significance, full methodology, Key Findings, and the per-image summary.

## Presence -- significant only (0 of 34 label(s))

(no label was significant on this test)

## Mass Share -- significant only (1 of 34 label(s))

| label | pass | fail | diff | mean mass_share (pass) | mean mass_share (fail) | AUC | 95% CI | FDR p |
|---|---|---|---|---|---|---|---|---|
| hill | 3/21 (14%) | 0/79 (0%) | +14.3% | 0.067 |  | 0.571 | [50%, 64%] | 0.037 |

## Peak Ratio -- significant only (1 of 34 label(s))

| label | pass | fail | diff | mean peak_ratio (pass) | mean peak_ratio (fail) | AUC | 95% CI | FDR p |
|---|---|---|---|---|---|---|---|---|
| hill | 3/21 (14%) | 0/79 (0%) | +14.3% | 0.617 |  | 0.571 | [50%, 64%] | 0.037 |

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

**Cross-validated ROC-AUC**: 0.493 (+/- 0.190 across folds), predicting pass/fail from all 34 feature(s) together over 100 images.

**Feature importance** (all 34 feature(s), sorted highest to lowest by mean |SHAP|; direction is which class this feature's presence/intensity typically pushes the prediction toward, jointly with every other feature):

| label | mean abs SHAP | direction |
|---|---|---|
| traffic light | 0.0713 | PASSING |
| car | 0.0419 | PASSING |
| building | 0.0381 | PASSING |
| street road | 0.032 | FAILING |
| tree | 0.0315 | PASSING |
| streetlight | 0.0222 | PASSING |
| road pavement | 0.0185 | PASSING |
| traffic light traffic signal | 0.0185 | PASSING |
| road | 0.0178 | FAILING |
| sky | 0.0171 | FAILING |
| hill | 0.0169 | PASSING |
| window | 0.0093 | FAILING |
| fence | 0.0088 | PASSING |
| Unknown Background | 0.0073 | PASSING |
| bush | 0.0066 | FAILING |
| pole | 0.0061 | PASSING |
| lamp post streetlight | 0.006 | FAILING |
| street lamp streetlight | 0.0051 | PASSING |
| bridge | 0.0048 | FAILING |
| lamp post | 0.0035 | FAILING |
| street road pavement | 0.0028 | FAILING |
| mountain | 0.0017 | FAILING |
| tire | 0.0015 | FAILING |
| street pavement | 0.0015 | FAILING |
| road pavement asphalt | 0.0009 | FAILING |
| vehicle | 0.0008 | FAILING |
| car vehicle | 0.0008 | FAILING |
| street | 0.0008 | FAILING |
| pedestrian person | 0.0005 | FAILING |
| sidewalk | 0.0005 | PASSING |
| pedestrian | 0.0004 | FAILING |
| wheel | 0.0004 | FAILING |
| car tire | 0.0004 | FAILING |
| traffic light traffic light fixture | 0.0 | neutral |