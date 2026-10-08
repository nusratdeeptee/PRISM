# Feature importance: S2

Only labels/tests that cleared FDR-corrected significance (p < 0.1) appear below, each table sorted by FDR-corrected p-value ascending -- lowest (strongest) p-value first, the row closest to the 0.1 cutoff last. See **`feature_importance_full.md`** (same folder) for every label regardless of significance, full methodology, Key Findings, and the per-image summary.

## Presence -- significant only (0 of 27 label(s))

(no label was significant on this test)

## Mass Share -- significant only (0 of 27 label(s))

(no label was significant on this test)

## Peak Ratio -- significant only (0 of 27 label(s))

(no label was significant on this test)

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

**Cross-validated ROC-AUC**: 0.529 (+/- 0.195 across folds), predicting pass/fail from all 27 feature(s) together over 100 images.

**Feature importance** (all 27 feature(s), sorted highest to lowest by mean |SHAP|; direction is which class this feature's presence/intensity typically pushes the prediction toward, jointly with every other feature):

| label | mean abs SHAP | direction |
|---|---|---|
| sky | 0.0773 | FAILING |
| car | 0.0508 | FAILING |
| tree | 0.0414 | FAILING |
| building | 0.031 | PASSING |
| Unknown Background | 0.0227 | PASSING |
| traffic light | 0.0169 | PASSING |
| lamp post | 0.0165 | FAILING |
| hill | 0.0118 | PASSING |
| pole | 0.0117 | FAILING |
| fence | 0.0104 | PASSING |
| window | 0.009 | FAILING |
| lamp post streetlight | 0.007 | FAILING |
| streetlight | 0.0069 | PASSING |
| bush | 0.0068 | FAILING |
| traffic light traffic signal | 0.0066 | PASSING |
| street lamp streetlight | 0.0054 | PASSING |
| road pavement | 0.0054 | PASSING |
| vehicle | 0.0047 | FAILING |
| utility pole | 0.003 | FAILING |
| pedestrian | 0.0029 | FAILING |
| car vehicle | 0.0019 | FAILING |
| mountain | 0.0015 | FAILING |
| tire | 0.0012 | FAILING |
| wheel | 0.0007 | FAILING |
| road | 0.0007 | FAILING |
| bridge | 0.0004 | PASSING |
| traffic light traffic light fixture | 0.0 | neutral |