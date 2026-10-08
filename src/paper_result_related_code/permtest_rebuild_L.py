"""Part 2 of the PermTest sensitivity check: rebuild Alg. 2's C/L/V at a different
Alg. 1 per-image alpha (0.05 vs. the calibrated p* from permtest_pvalue_calibration.py),
and compare.

Scope (by explicit user request, to keep runtime tractable -- see the module-level
DATASETS below): S1, S2, mimicry_5, waterbird_test (the same 4 conditions used for the
human open-coding / kappa / attention-rate study earlier), each at N=1000 (the full
cached pool) and N=100 using ONLY the "first" draw (analysis_<attn>_sample_100, no
_repeatK suffix -- the same sample already used for that human-eval study), not the 4
extra repeat draws rq1.py's stability sweep also covers.

Does not touch generate_analysis.py or its cached _stats.json files -- those are the
inputs, read-only. This script only re-derives C (Select, Alg. 2 Eq. 8) at a chosen
alpha from each image's ALREADY-COMPUTED permutation_pvalue (Alg. 1 Line 11's real
output, cached per object+UNCLAIMED row in every <id>_stats.json by generate_analysis.py
itself) -- the permutation test is never rerun/reimplemented; only the significance gate
sig_i = [p_i < alpha] (Alg. 1 Line 12) is re-applied at a different alpha. Everything
downstream of C is the real Alg. 2 code, called directly (not through the markdown-report
wrappers, which also run the unrelated, expensive Random Forest+SHAP stage this script
has no need for):
  - balanced branch (ContrastTest):  common.stats_tables.label_stats_table()
  - imbalanced branch (EnrichmentTest): common.low_failure_stats.compute_label_df()
  - canonicalization: common.canonicalize.build_canonical_map / normalize_label
  - branch selection: common.low_failure_stats.dataset_fail_rate/pass_rate +
    LOW_FAILURE_THRESHOLD (depends only on the sample's own pass/fail labels, not alpha)

Reproduction check (--verify-rq1): before trusting any p*-rebuilt number, every
(dataset, attention, N) combination is first rebuilt at alpha=0.05 and compared against
the already-published results/rq1_feature_counts.json. |L| is compared directly;
|V| (balanced branch only) is compared after rounding the three FDR columns to 3
decimals, matching the rounding build_report() itself applies before the markdown table
rq1.py originally parsed its counts from -- found and confirmed on S1/idg/N=1000 during
development: raw (unrounded) FDR comparison gives |V|=21, the archived count is 20,
and the one label responsible ("street sign", mass_share_pvalue_fdr=0.099775) rounds to
exactly 0.100 in the real pipeline's own markdown output, which rq1.py's parser then
reads back as NOT "< 0.1". The imbalanced branch's V (enrichment_diff_mean > 0.1) needs
no such correction -- feature_importance_csv_table() never rounds that column.
Any other mismatch is reported and this script stops rather than silently trusting a
disagreement with the published numbers.

Caching/resumability: each (dataset, attention, n, alpha_label) rebuild is cached to
results/permtest_rebuild_cache/<alpha_label>/<dataset>__<attention>__N<n>.json; a rerun
skips whatever already has a cache file (these are the slow ~60-90s calls, dominated by
canonicalization + label_stats_table's bootstrap CIs).

Usage:
    python permtest_rebuild_L.py --p-star 0.000999                 # pooled p* from Part 1
    python permtest_rebuild_L.py --p-star 0.000999 --verify-rq1    # also run the alpha=0.05 check
    python permtest_rebuild_L.py --p-star-by-family SGSM=0.0014,Mimicry=0.000999,...
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths
from common.canonicalize import build_canonical_map, normalize_label
from common.stats_tables import label_stats_table, SIGNIFICANCE_ALPHA as FDR_ALPHA
from common.low_failure_stats import (compute_label_df, dataset_fail_rate, dataset_pass_rate,
                                       LOW_FAILURE_THRESHOLD)

DATASETS = ["S1", "S2", "mimicry_5", "waterbird_test"]
ATTENTIONS = ["idg", "ig2"]
N_LEVELS = [100, 1000]   # N=100 uses ONLY the first draw (no _repeatK), per explicit scope above

IMPORTANCE_THRESHOLD = 0.1   # Alg. 2's epsilon (imbalanced-branch EnrichmentTest cutoff), unchanged by alpha

REBUILD_CACHE_DIR = paths.RQ_RESULTS_DIR / "permtest_rebuild_cache"
RQ1_COUNTS_PATH = paths.RQ_RESULTS_DIR / "rq1_feature_counts.json"


def family_of(dataset_id: str) -> str:
    if dataset_id.startswith("S"):
        return "SGSM"
    if dataset_id.startswith("mimicry"):
        return "Mimicry"
    if dataset_id.startswith("biggan_ai_tigs"):
        return "GenAI-TIG"
    if dataset_id.startswith("ep_dnn"):
        return "EP-DNN"
    return "Waterbirds"


def analysis_dir_for(cfg: dict, attention: str, n: int):
    stage = f"analysis_{attention}" if n == 1000 else f"analysis_{attention}_sample_{n}"
    return paths.dataset_results_dir(cfg, stage)


def load_rows(analysis_dir: Path, alpha: float):
    """Reads every <id>_stats.json in analysis_dir once and builds (all_images,
    chosen_rows, all_occurrence_rows) -- mirrors generate_analysis.py's
    load_dataset_for_report(), except is_significant is RE-DERIVED from the cached
    permutation_pvalue at the given alpha instead of trusting the stored boolean
    (which was baked in at alpha=0.05)."""
    files = sorted(glob.glob(str(analysis_dir / "*_stats.json")), key=lambda f: int(Path(f).stem.split("_")[0]))
    chosen_rows, all_rows, image_ids = [], [], []
    for f in files:
        d = json.loads(Path(f).read_text())
        iid = int(d["image_id"])
        image_ids.append(iid)
        for o in d["objects"]:
            if o["label"] != "UNCLAIMED":
                all_rows.append({"image_id": iid, "label": o["label"]})
            if o["permutation_pvalue"] < alpha:
                chosen_rows.append({"image_id": iid, "label": o["label"], "mass_share": o["mass_share"],
                                     "peak_ratio": o["peak_ratio"], "area_share": o["area_share"]})
    return image_ids, pd.DataFrame(chosen_rows), pd.DataFrame(all_rows)


def canonicalize(chosen_df: pd.DataFrame, all_occ_df: pd.DataFrame):
    raw_labels = list(pd.concat([chosen_df.get("label", pd.Series(dtype=str)),
                                  all_occ_df.get("label", pd.Series(dtype=str))], ignore_index=True))
    canonical_map = build_canonical_map(raw_labels) if raw_labels else {}
    if len(chosen_df):
        chosen_df = chosen_df.copy()
        chosen_df["canonical_label"] = chosen_df["label"].apply(
            lambda l: canonical_map.get(normalize_label(l), normalize_label(l)))
    if len(all_occ_df):
        all_occ_df = all_occ_df.copy()
        all_occ_df["canonical_label"] = all_occ_df["label"].apply(
            lambda l: canonical_map.get(normalize_label(l), normalize_label(l)))
    return chosen_df, all_occ_df


def rebuild_one(dataset_id: str, attention: str, n: int, alpha: float) -> dict:
    cfg = paths.load_dataset_config(dataset_id)
    analysis_dir = analysis_dir_for(cfg, attention, n)
    image_ids, chosen_df, all_occ_df = load_rows(analysis_dir, alpha)

    meta_df = pd.read_csv(cfg["metadata_csv"])
    pf_map = meta_df.drop_duplicates("image_id").set_index("image_id")["label"].to_dict()
    all_images = {i: pf_map[i] for i in image_ids if i in pf_map}

    chosen_df, all_occ_df = canonicalize(chosen_df, all_occ_df)

    fail_rate, pass_rate = dataset_fail_rate(all_images), dataset_pass_rate(all_images)
    balanced = fail_rate >= LOW_FAILURE_THRESHOLD and pass_rate >= LOW_FAILURE_THRESHOLD

    if balanced:
        label_df = label_stats_table(f"{dataset_id}_{attention}_N{n}_a{alpha}", all_images, chosen_df)
        total = len(label_df)
        if total:
            rounded = label_df[["presence_pvalue_fdr", "mass_share_pvalue_fdr", "peak_ratio_pvalue_fdr"]].round(3)
            sig_mask = (rounded["presence_pvalue_fdr"] < FDR_ALPHA) | \
                       (rounded["mass_share_pvalue_fdr"] < FDR_ALPHA) | \
                       (rounded["peak_ratio_pvalue_fdr"] < FDR_ALPHA)
            validated = int(sig_mask.sum())
            labels_in_L = sorted(label_df["label"])
            labels_in_V = sorted(label_df.loc[sig_mask, "label"])
        else:
            validated, labels_in_L, labels_in_V = 0, [], []
        branch = "balanced"
    else:
        label_df = compute_label_df(chosen_df, all_occ_df)
        total = len(label_df)
        if total:
            sig_mask = label_df["enrichment_diff_mean"] > IMPORTANCE_THRESHOLD
            validated = int(sig_mask.sum())
            labels_in_L = sorted(label_df["label"])
            labels_in_V = sorted(label_df.loc[sig_mask, "label"])
        else:
            validated, labels_in_L, labels_in_V = 0, [], []
        branch = "imbalanced"

    return {
        "dataset": dataset_id, "attention": attention, "n": n, "alpha": alpha, "branch": branch,
        "fail_rate": fail_rate, "pass_rate": pass_rate,
        "n_images": len(image_ids), "total": total, "validated": validated,
        "labels_in_L": labels_in_L, "labels_in_V": labels_in_V,
    }


def cache_path(alpha_label: str, dataset_id: str, attention: str, n: int) -> Path:
    return REBUILD_CACHE_DIR / alpha_label / f"{dataset_id}__{attention}__N{n}.json"


def load_or_rebuild(dataset_id: str, attention: str, n: int, alpha: float, alpha_label: str) -> dict:
    cp = cache_path(alpha_label, dataset_id, attention, n)
    if cp.exists():
        return json.loads(cp.read_text())
    result = rebuild_one(dataset_id, attention, n, alpha)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(result, indent=2))
    return result


def verify_against_rq1(result: dict) -> list:
    """Compares one rebuild (alpha=0.05 only) against rq1_feature_counts.json. Returns a
    list of human-readable mismatch strings, empty if everything matches."""
    counts = json.loads(RQ1_COUNTS_PATH.read_text())
    ds, att, n = result["dataset"], result["attention"], result["n"]
    rec = counts[ds][att][str(n)]
    if n == 1000:
        expected_total, expected_validated, expected_branch = rec["total"], rec["validated"], rec["branch"]
    else:
        expected_total, expected_validated, expected_branch = rec["total"][0], rec["validated"][0], rec["branch"][0]

    mismatches = []
    if result["total"] != expected_total:
        mismatches.append(f"{ds}/{att}/N={n}: |L| mismatch: rebuilt={result['total']} vs rq1={expected_total}")
    if result["validated"] != expected_validated:
        mismatches.append(f"{ds}/{att}/N={n}: |V| mismatch: rebuilt={result['validated']} vs rq1={expected_validated}")
    if result["branch"] != expected_branch:
        mismatches.append(f"{ds}/{att}/N={n}: branch mismatch: rebuilt={result['branch']} vs rq1={expected_branch}")
    return mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--verify-rq1", action="store_true",
                     help="also rebuild at alpha=0.05 and compare against rq1_feature_counts.json; "
                          "STOPS (no p*-rebuild output written) if any mismatch is found")
    ap.add_argument("--p-star", type=float, required=True, help="pooled p* (main setting)")
    ap.add_argument("--p-star-by-family", default=None,
                     help="comma-separated Family=value pairs (secondary setting), e.g. "
                          "SGSM=0.0014,Mimicry=0.000999,Waterbirds=0.000999")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dataset_ids = args.datasets.split(",")
    family_p_star = {}
    if args.p_star_by_family:
        for pair in args.p_star_by_family.split(","):
            fam, val = pair.split("=")
            family_p_star[fam] = float(val)

    if args.verify_rq1:
        print("===== Verifying alpha=0.05 reproduction against rq1_feature_counts.json =====")
        all_mismatches = []
        for dataset_id in dataset_ids:
            for attention in ATTENTIONS:
                for n in N_LEVELS:
                    r = load_or_rebuild(dataset_id, attention, n, 0.05, "alpha_0.05")
                    mism = verify_against_rq1(r)
                    status = "OK" if not mism else "MISMATCH"
                    print(f"[{dataset_id}/{attention}/N={n}] branch={r['branch']} |L|={r['total']} "
                          f"|V|={r['validated']} -- {status}", flush=True)
                    all_mismatches.extend(mism)
        if all_mismatches:
            print(f"\n{len(all_mismatches)} MISMATCH(ES) -- stopping before the p*-rebuild:")
            for m in all_mismatches:
                print(f"  - {m}")
            sys.exit(1)
        print("All reproductions matched rq1_feature_counts.json exactly.\n")

    print(f"===== Rebuilding at p*_pooled={args.p_star} (+ per-family p* if given) =====")
    # The alpha value itself is embedded in the cache label (not just "p_star_pooled") so a
    # rerun with a DIFFERENT --p-star can never silently reuse a stale cache computed at the
    # old value -- the cache path is a pure function of (dataset, attention, n, alpha).
    settings = [("alpha_0.05", 0.05, "alpha=0.05 (original)"),
                (f"p_star_pooled_{args.p_star}", args.p_star, f"p*_pooled={args.p_star}")]

    pooled_label = settings[1][0]  # the dynamic f"p_star_pooled_{args.p_star}" key, used by the summary below

    results = {}
    for label, alpha, desc in settings:
        results[label] = {}
        for dataset_id in dataset_ids:
            fam = family_of(dataset_id)
            results[label][dataset_id] = {}
            for attention in ATTENTIONS:
                results[label][dataset_id][attention] = {}
                for n in N_LEVELS:
                    r = load_or_rebuild(dataset_id, attention, n, alpha, label)
                    results[label][dataset_id][attention][str(n)] = r
                    print(f"[{desc}] [{dataset_id}/{attention}/N={n}] branch={r['branch']} "
                          f"|L|={r['total']} |V|={r['validated']}", flush=True)

    if family_p_star:
        results["p_star_family"] = {}
        for dataset_id in dataset_ids:
            fam = family_of(dataset_id)
            if fam not in family_p_star:
                continue
            alpha = family_p_star[fam]
            results["p_star_family"][dataset_id] = {}
            for attention in ATTENTIONS:
                results["p_star_family"][dataset_id][attention] = {}
                for n in N_LEVELS:
                    r = load_or_rebuild(dataset_id, attention, n, alpha, f"p_star_family_{fam}_{alpha}")
                    results["p_star_family"][dataset_id][attention][str(n)] = r
                    print(f"[p*_family({fam})={alpha}] [{dataset_id}/{attention}/N={n}] branch={r['branch']} "
                          f"|L|={r['total']} |V|={r['validated']}", flush=True)

    # Summary: reduction and validated share, per setting, pooled + per dataset.
    print("\n===== Summary: alpha=0.05 vs p*_pooled =====")
    print(f"{'dataset':14s}{'att':5s}{'N':>6s}  {'alpha=0.05 |L|/|V|':>20s}  {'p*_pooled |L|/|V|':>20s}  "
          f"{'labels dropped from L':>24s}  {'labels dropped from V':>24s}")
    for dataset_id in dataset_ids:
        for attention in ATTENTIONS:
            for n in N_LEVELS:
                a = results["alpha_0.05"][dataset_id][attention][str(n)]
                b = results[pooled_label][dataset_id][attention][str(n)]
                dropped_L = sorted(set(a["labels_in_L"]) - set(b["labels_in_L"]))
                dropped_V = sorted(set(a["labels_in_V"]) - set(b["labels_in_V"]))
                print(f"{dataset_id:14s}{attention:5s}{n:6d}  {a['total']:>8d}/{a['validated']:<8d}     "
                      f"{b['total']:>8d}/{b['validated']:<8d}     "
                      f"{len(dropped_L):>6d} of {len(a['labels_in_L']):<6d}        "
                      f"{len(dropped_V):>6d} of {len(a['labels_in_V']):<6d}")

    out_path = Path(args.out) if args.out else paths.RQ_RESULTS_DIR / "permtest_rebuild_L.json"
    if not out_path.is_absolute():
        out_path = paths.PRISM_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
