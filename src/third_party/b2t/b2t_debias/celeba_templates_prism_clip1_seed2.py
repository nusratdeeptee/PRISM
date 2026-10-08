# B2T+PRISM concept group for CelebA Sagawa ERM seed 2 (2026-09-27):
# candidate vocabulary = PRISM's own per-seed analysis of this Sagawa ERM
# (output/celeba_val_sagawa_s2/finetuned_classifier/report_idg/) instead of BLIP-caption
# keywords; scoring + selection = B2T's own unmodified mechanism
# (function/calculate_similarity.py, unmodified) on the VALIDATION split, magnitude threshold
# ONLY (score > 1.0, arXiv 2301.11104v4 Section 4.2). Full Keyword/Score/Acc.
# table (Bias column present but not meaningful for CelebA's single-bucket structure, unused for
# selection): ../../../../output/b2t_discovery_celeba/seed2/diff_prism_clip1/celeba_prism_clip1_seed2_blond.csv
# Bucket 0 stays [""] (B2T's own convention, unchanged); bucket 1 = blond-class score-selected concepts.
from celeba_templates import templates, class_templates

class_keywords_all = [
    [""],
    ['man', 'man person'],
]
