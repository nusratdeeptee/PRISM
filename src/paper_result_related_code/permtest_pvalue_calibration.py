"""Part 1 of the PermTest sensitivity check: the null p-value distribution and the
calibrated cutoff p* (paper Sec. 3.2 Eq. 7 / Alg. 1 Line 11/12).

This is a SEPARATE script from permtest_calibration.py -- it does not modify that
script or its output files. It imports and reuses its exact helpers (derangement,
mask loading, saliency loading/resizing, random_translate_mask, the same fixed seeds)
rather than duplicating them, and it calls the exact same real per-image significance
function (common.permutation_test.add_permutation_pvalues, Alg. 1 Line 11's own
implementation -- never reimplemented). The only difference from permtest_calibration.py
is WHAT is kept from each trial: that script only kept the boolean is_significant flag
and a running count; this one keeps every trial's raw permutation_pvalue, split by
whether the row is an object mask or the synthetic "unclaimed" row.

Why: Alg. 2's Select (Eq. 8) builds C from sig_i = [p_i < alpha]. permtest_calibration.py
established that alpha=0.05 is themselves miscalibrated (~33% null rejection rate,
untranslated; ~37%, translated) against this test's own pixel-shuffle null. p* here is
the empirical cutoff that WOULD give a 5% null rejection rate under that same null -- the
threshold Part 2 (permtest_rebuild_L.py) substitutes for 0.05 when rebuilding C/L/V.

p-value resolution: permutation_test.py's PermTest (Eq. 7) draws P=1000 random same-size
windows per object (common.permutation_test.N_PERMUTATIONS) and returns
(1 + #{random >= real}) / (1 + P), so every p-value is one of exactly P+1=1001 discrete
levels, k/1001 for k=1..1001. p* can therefore never be refined finer than this
resolution floor (1/1001 ~ 0.000999) -- if p* lands AT that floor (the 5th percentile of
null p-values happens to be the smallest level the test can ever emit), this is flagged
explicitly in the output, since no finer cutoff is achievable with this P.

Storage: rather than persist the ~750k raw p-values from each mode, every trial is
folded into a histogram over its 1001 possible discrete levels (exact, not an
approximation -- a quantile computed from the histogram is bit-identical to one computed
from the raw array, since every trial's p-value already lives on one of those 1001
levels) -- compact, and resumable per (dataset, attention, mode) condition: see
PVALUE_CACHE_DIR below, written after every condition so a rerun skips whatever already
has a cache file.

Usage:
    python permtest_pvalue_calibration.py                       # all 38 x 2 x 2 modes
    python permtest_pvalue_calibration.py --datasets S1,S2
    python permtest_pvalue_calibration.py --modes untranslated   # skip the translated mode
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths
from common.permutation_test import add_permutation_pvalues, SIGNIFICANCE_ALPHA, N_PERMUTATIONS
import permtest_calibration as calib

MODES = ["untranslated", "translated"]
P_RESOLUTION_DENOM = N_PERMUTATIONS + 1  # 1001 possible p-value levels: k/1001, k=1..1001

PVALUE_CACHE_DIR = paths.RQ_RESULTS_DIR / "permtest_pvalue_cache"


def numerator_of(p: float) -> int:
    """Recovers the exact integer k (1..P_RESOLUTION_DENOM) from a p-value k/1001 --
    round() rather than int() to absorb float round-trip noise; p-values here are always
    produced by exactly this division, so the rounded result is exact."""
    return int(round(p * P_RESOLUTION_DENOM))


def run_condition_pvalues(dataset_id: str, attention: str, seed: int,
                           translate: bool, translate_seed: int) -> dict:
    """Same pairing/resizing/translation logic as permtest_calibration.run_condition
    (reused via the imported helpers, not reimplemented) -- but returns full p-value
    histograms instead of a single significant-count. Returns
    {"n_images", "object_counts": {k: n}, "unclaimed_counts": {k: n}}."""
    cfg = paths.load_dataset_config(dataset_id)
    segments_dir = paths.dataset_results_dir(cfg, "segments")
    attr_dir = paths.dataset_results_dir(cfg, attention)

    ids = sorted(int(p.stem.replace("_masks", "")) for p in segments_dir.glob("*_masks.npz"))
    ids = [i for i in ids if (attr_dir / f"{i}_attr.npy").exists()]
    if len(ids) < 2:
        return {"n_images": len(ids), "object_counts": {}, "unclaimed_counts": {}}

    sigma = calib.derangement(ids, seed=seed)
    object_counts, unclaimed_counts = {}, {}

    for i in ids:
        masks_arr = calib.load_masks_array(segments_dir, i)
        target_hw = masks_arr.shape[1:]
        if translate:
            rng = np.random.default_rng(translate_seed * 1_000_003 + i)
            objects_i = [{"mask": calib.random_translate_mask(masks_arr[k], rng)}
                         for k in range(masks_arr.shape[0])]
        else:
            objects_i = [{"mask": masks_arr[k]} for k in range(masks_arr.shape[0])]

        saliency_b = calib.load_saliency(attr_dir, sigma[i])
        saliency_b = calib.resize_saliency(saliency_b, target_hw)

        stats_rows = [{} for _ in objects_i] + [{}]  # +1 synthetic "unclaimed" row
        add_permutation_pvalues(stats_rows, saliency_b, objects_i, seed=i)

        for row in stats_rows[:-1]:
            k = numerator_of(row["permutation_pvalue"])
            object_counts[k] = object_counts.get(k, 0) + 1
        k = numerator_of(stats_rows[-1]["permutation_pvalue"])
        unclaimed_counts[k] = unclaimed_counts.get(k, 0) + 1

    return {"n_images": len(ids), "object_counts": object_counts, "unclaimed_counts": unclaimed_counts}


def cache_path(mode: str, dataset_id: str, attention: str) -> Path:
    return PVALUE_CACHE_DIR / mode / f"{dataset_id}__{attention}.json"


def load_or_compute(dataset_id: str, attention: str, mode: str, seed: int, translate_seed: int) -> dict:
    cp = cache_path(mode, dataset_id, attention)
    if cp.exists():
        return json.loads(cp.read_text())
    result = run_condition_pvalues(dataset_id, attention, seed,
                                    translate=(mode == "translated"), translate_seed=translate_seed)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(result))
    return result


def merge_counts(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] = dst.get(int(k), 0) + v


def p_star_from_histogram(counts: dict, alpha: float = SIGNIFICANCE_ALPHA) -> dict:
    """counts: {numerator_k (int): count}. p* is the LARGEST cutoff (among the
    P_RESOLUTION_DENOM discrete levels k/denom) such that the null rejection rate using
    "p < p*" is at most `alpha`. Scans the 1001 levels once ascending, tracking the
    cumulative count strictly below the current candidate -- O(1001), exact (no
    interpolation/approximation), matches np.quantile(..., alpha, method="lower") applied
    to the reconstructed raw array bit-for-bit, since every value already lives on one of
    these 1001 levels."""
    total = sum(counts.values())
    if total == 0:
        return {"p_star": None, "numerator": None, "rejection_rate_at_p_star": None,
                "n_null_trials": 0, "at_resolution_floor": None}
    threshold = alpha * total
    cum_below = 0
    best_k, best_cum_below = 1, 0
    for k in range(1, P_RESOLUTION_DENOM + 1):
        if cum_below <= threshold:
            best_k, best_cum_below = k, cum_below
        cum_below += counts.get(k, 0)
    return {
        "p_star": best_k / P_RESOLUTION_DENOM,
        "numerator": best_k,
        "rejection_rate_at_p_star": best_cum_below / total,
        "n_null_trials": total,
        "at_resolution_floor": best_k == 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None, help="comma-separated dataset ids (default: all 38)")
    ap.add_argument("--modes", default=",".join(MODES), help="comma-separated subset of untranslated,translated")
    ap.add_argument("--seed", type=int, default=calib.DERANGEMENT_SEED)
    ap.add_argument("--translate-seed", type=int, default=calib.TRANSLATE_SEED)
    ap.add_argument("--out", default=None, help="repo-relative or absolute "
                                                 "(default: results/permtest_pvalue_calibration.json)")
    args = ap.parse_args()

    dataset_ids = args.datasets.split(",") if args.datasets else calib.DATASETS
    modes = args.modes.split(",")

    # buckets[mode][row_scope]["pooled" | "family:<F>" | "attention:<A>"] -> counts dict
    buckets = {mode: {scope: {} for scope in ("object", "all")} for mode in modes}

    for mode in modes:
        for dataset_id in dataset_ids:
            fam = calib.family_of(dataset_id)
            for attention in calib.ATTENTIONS:
                r = load_or_compute(dataset_id, attention, mode, args.seed, args.translate_seed)
                obj_counts, unc_counts = r["object_counts"], r["unclaimed_counts"]
                all_counts = dict(obj_counts)
                for k, v in unc_counts.items():
                    all_counts[int(k)] = all_counts.get(int(k), 0) + v

                for scope_name, src in (("object", obj_counts), ("all", all_counts)):
                    for agg_key in ("pooled", f"family:{fam}", f"attention:{attention}"):
                        merge_counts(buckets[mode][scope_name].setdefault(agg_key, {}), src)

            print(f"[{mode}] [{dataset_id}] done", flush=True)
        print(f"===== [{mode}] all {len(dataset_ids)} datasets done =====", flush=True)

    result = {
        "alpha": SIGNIFICANCE_ALPHA,
        "n_permutations_P": N_PERMUTATIONS,
        "p_value_resolution": 1.0 / P_RESOLUTION_DENOM,
        "derangement_seed": args.seed,
        "translate_seed": args.translate_seed,
    }
    for mode in modes:
        result[mode] = {}
        for scope_name in ("object", "all"):
            result[mode][scope_name] = {
                agg_key: p_star_from_histogram(counts)
                for agg_key, counts in buckets[mode][scope_name].items()
            }

    print(f"\n===== p* summary (alpha={SIGNIFICANCE_ALPHA}, P={N_PERMUTATIONS}, "
          f"resolution={1.0/P_RESOLUTION_DENOM:.6f}) =====")
    for mode in modes:
        for scope_name in ("object", "all"):
            pooled = result[mode][scope_name]["pooled"]
            floor_flag = " <-- AT RESOLUTION FLOOR" if pooled["at_resolution_floor"] else ""
            print(f"[{mode}/{scope_name}] pooled p*={pooled['p_star']:.6f} "
                  f"(n={pooled['numerator']}/{P_RESOLUTION_DENOM}) "
                  f"rejection_rate_at_p*={100*pooled['rejection_rate_at_p_star']:.2f}% "
                  f"n_trials={pooled['n_null_trials']}{floor_flag}")

    out_path = Path(args.out) if args.out else paths.RQ_RESULTS_DIR / "permtest_pvalue_calibration.json"
    if not out_path.is_absolute():
        out_path = paths.PRISM_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
