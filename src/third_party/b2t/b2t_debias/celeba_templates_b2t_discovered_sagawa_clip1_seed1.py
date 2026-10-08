# B2T-discovered concept group for CelebA Sagawa ERM seed 1, re-filtered
# 2026-09-27 at the paper's real selection rule (CLIP score > 1.0,
# arXiv 2301.11104v4 Section 4.2), replacing the old hand-picked selection -- see
# output/model_repair/B2T_archive/waterbird_celeba_clip_threshold_fix_20260927/README.md.
# Candidates and Score/Acc. values are UNCHANGED from B2T's own Phase A discovery
# (output/b2t_discovery_celeba/seed1/diff/*_blond.csv, produced by b2t.py, unmodified) --
# only the selection rule differs (magnitude threshold, not the S/M sign label).
# Bucket 0 stays [""] (B2T's own convention, unchanged); bucket 1 = blond-class S-scored keywords.
from celeba_templates import templates, class_templates

class_keywords_all = [
    [""],
    ['man'],
]
