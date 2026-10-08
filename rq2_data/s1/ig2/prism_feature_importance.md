# Feature importance: S1

Only labels/tests that cleared FDR-corrected significance (p < 0.1) appear below, each table sorted by FDR-corrected p-value ascending -- lowest (strongest) p-value first, the row closest to the 0.1 cutoff last. See **`feature_importance_full.md`** (same folder) for every label regardless of significance, full methodology, Key Findings, and the per-image summary.

## Presence -- significant only (0 of 25 label(s))

(no label was significant on this test)

## Mass Share -- significant only (0 of 25 label(s))

(no label was significant on this test)

## Peak Ratio -- significant only (0 of 25 label(s))

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

**Cross-validated ROC-AUC**: 0.831 (+/- 0.072 across folds), predicting pass/fail from all 25 feature(s) together over 100 images.

**Feature importance** (all 25 feature(s), sorted highest to lowest by mean |SHAP|; direction is which class this feature's presence/intensity typically pushes the prediction toward, jointly with every other feature):

| label | mean abs SHAP | direction |
|---|---|---|
| car | 0.0844 | PASSING |
| coca-cola van | 0.0747 | PASSING |
| tree | 0.0705 | FAILING |
| road | 0.06 | PASSING |
| wheel | 0.0327 | FAILING |
| traffic light | 0.0278 | FAILING |
| fence | 0.0161 | FAILING |
| tire | 0.0151 | FAILING |
| truck | 0.0115 | FAILING |
| license plate | 0.0093 | FAILING |
| coca-cola truck | 0.0076 | PASSING |
| street | 0.0071 | FAILING |
| building | 0.0037 | FAILING |
| street road | 0.0029 | FAILING |
| car wheel | 0.0014 | PASSING |
| car windshield | 0.0011 | FAILING |
| car bumper | 0.001 | PASSING |
| streetlight | 0.001 | FAILING |
| shadow | 0.0009 | FAILING |
| taillight | 0.0009 | FAILING |
| car license plate | 0.0008 | PASSING |
| Unknown Background | 0.0007 | FAILING |
| car taillight | 0.0005 | FAILING |
| car tire car wheel | 0.0 | FAILING |
| car car car door mirror | 0.0 | neutral |