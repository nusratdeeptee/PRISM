# B2T+PRISM concept groups for Waterbirds ERM seed 0 (2026-09-27):
# candidate vocabulary = PRISM's SHAP concept bank (report_idg/feature_importance.md, mean|SHAP| >
# 0.001) instead of BLIP-caption keywords; scoring + selection = B2T's own unmodified
# mechanism (function/calculate_similarity.py, unmodified) on the VALIDATION split, magnitude
# threshold ONLY (score > 1.0, arXiv 2301.11104v4 Section 4.2 -- NOT the S/M sign
# label prism_shap_bias_labels.py used, see
# output/model_repair/B2T_archive/waterbird_celeba_clip_threshold_fix_20260927/README.md).
# Full Keyword/Score/Acc./Bias tables (Bias column kept for visibility only, unused for selection):
# ../../../../output/b2t_discovery_waterbird/seed0/diff_prism_clip1/waterbird_prism_clip1_seed0_{landbird,waterbird}.csv
# Order matters: index 0 = land, index 1 = water (same as waterbirds_templates.py).
from waterbirds_templates import templates, class_templates

class_keywords_all = [
    ['bamboo', 'forest', 'tree', 'branch'],   # land side (4)
    ['ocean', 'seal', 'wave', 'seagull', 'boat', 'ship', 'water', 'fish', 'duck'],   # water side (9)
]
