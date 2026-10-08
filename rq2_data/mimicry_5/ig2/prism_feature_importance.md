# Feature importance: mimicry_5

Only labels/tests that cleared FDR-corrected significance (p < 0.1) appear below, each table sorted by FDR-corrected p-value ascending -- lowest (strongest) p-value first, the row closest to the 0.1 cutoff last. See **`feature_importance_full.md`** (same folder) for every label regardless of significance, full methodology, Key Findings, and the per-image summary.

## Presence -- significant only (0 of 17 label(s))

(no label was significant on this test)

## Mass Share -- significant only (0 of 17 label(s))

(no label was significant on this test)

## Peak Ratio -- significant only (0 of 17 label(s))

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

**Cross-validated ROC-AUC**: 0.389 (+/- 0.116 across folds), predicting pass/fail from all 17 feature(s) together over 100 images.

**Feature importance** (all 17 feature(s), sorted highest to lowest by mean |SHAP|; direction is which class this feature's presence/intensity typically pushes the prediction toward, jointly with every other feature):

| label | mean abs SHAP | direction |
|---|---|---|
| hat | 0.0329 | FAILING |
| rock | 0.0324 | PASSING |
| Unknown Background | 0.0221 | PASSING |
| dog | 0.0195 | PASSING |
| shoe | 0.0141 | FAILING |
| fish | 0.0101 | PASSING |
| shell | 0.0083 | FAILING |
| bird | 0.0083 | PASSING |
| ray | 0.0065 | FAILING |
| rat | 0.0061 | FAILING |
| ground | 0.0044 | PASSING |
| turtle | 0.0044 | FAILING |
| cat | 0.0044 | PASSING |
| seal | 0.0017 | FAILING |
| shark | 0.0011 | FAILING |
| dirt | 0.0009 | FAILING |
| ear | 0.0 | neutral |