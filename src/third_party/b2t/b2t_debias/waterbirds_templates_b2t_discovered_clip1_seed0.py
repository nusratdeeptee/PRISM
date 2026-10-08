# B2T-discovered concept groups for Waterbirds ERM seed 0, re-filtered
# 2026-09-27 at the paper's real selection rule (CLIP score > 1.0,
# arXiv 2301.11104v4 Section 4.2), replacing the old hand-picked selection -- see
# output/model_repair/B2T_archive/waterbird_celeba_clip_threshold_fix_20260927/README.md.
# Candidates and Score/Acc. values are UNCHANGED from B2T's own Phase A discovery
# (output/b2t_discovery_waterbird/seed0/diff/*.csv, produced by b2t.py, unmodified) --
# only the selection rule differs (magnitude threshold, not the S/M sign label).
# Order matters: index 0 = land, index 1 = water (same as waterbirds_templates.py).
from waterbirds_templates import templates, class_templates

class_keywords_all = [
    ['bamboo forest', 'bamboo', 'forest', 'woods', 'trees', 'garden', 'tree'],   # land side (7)
    ['seagull', 'beach', 'boat', 'water', 'lake', 'sunset', 'rocks'],   # water side (7)
]
