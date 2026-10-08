"""PermTest calibration check (paper §3.2 Eq. 7 / Alg. 1 Line 11; §5 Internal Validity "[TBD]").

Matt's review note: the per-image permutation test (PermTest, Eq. 7) asks whether a
feature's mean attribution beats same-size random windows of the SAME image's own
saliency map. To check its false-positive rate we need cases where the true answer is
"no relationship" -- which a real image never gives us, since every object's mask and
its own saliency map ARE related. So: take the masks discovered for image A and run
PermTest against a DIFFERENT image B's saliency map instead. A and B share nothing, so
the test should come back significant only ~alpha=5% of the time if it's calibrated.
It's expected to run higher than that: the pixel-shuffle null treats pixels as
independent, but real saliency maps are spatially smooth/clustered and masks are
contiguous blobs, so the null underestimates the variance a contiguous mask can land on
just by chance. Per Matt: report the empirical rate, don't redesign the test -- PermTest
(Alg. 1) only screens candidates that go on to the FDR-controlled, dataset-level tests of
Alg. 2, so an inflated per-image rate here means more candidates pass through to Alg. 2,
not more false discoveries in the final, validated output.

This script reuses the EXACT per-image significance code the real pipeline calls
(common.permutation_test.add_permutation_pvalues, Alg. 1 Line 11's implementation) --
never reimplemented -- so the calibration measures the literal code path used
everywhere else in PRISM, not a lookalike. It only reads already-cached pipeline output
(no model inference, no GPU):
  - output/<dataset>/<model>/segments/<id>_masks.npz + <id>_segments.json (masks)
  - output/<dataset>/<model>/{idg,ig2}/<id>_attr.npy (raw attribution -> saliency)

Pairing: for each (dataset, attention) condition, draws ONE fixed-seed derangement sigma
of that condition's cached image ids (sigma(i) != i for every i, so every id is paired
with a genuinely different image, each id used exactly once as a mask source and once as
a saliency source -- N mismatched pairs per condition, not an O(N^2) sweep). Masks from
image i are tested against image sigma(i)'s saliency map, exactly as process_image() in
generate_analysis.py tests image i's masks against image i's OWN saliency map in the
real analysis -- same function, same per-image engine seed convention (seed=i), only the
saliency map fed in is swapped for a mismatched one.

Resizing: masks are always native-resolution (generate_segments.py never resizes).
Attribution can be computed at a smaller working resolution for some datasets
(idg.py/ig2.py's raw_image_size); within a given dataset this working resolution is
fixed, so SGSM and the ImageNet-derived sets (Mimicry/GenAI-TIG/EP-DNN) never need a
resize here. Waterbirds_test is the one condition where it matters: its CUB source
photos vary in native size (spot-checked: 500x335, 500x347, 216x136, ...), so when
sigma(i)'s saliency shape disagrees with i's mask shape, it's bilinear-resized to match
-- the identical resize generate_analysis.py's process_image() already uses to reconcile
idg/ig2 saliency with native mask resolution.

Ablation (--translate): separates how much of the inflated rate comes from (1) local
spatial autocorrelation -- a contiguous mask landing on any locally-smooth/clustered
region of saliency gets an inflated-variance mean regardless of WHERE it sits -- versus
(2) a shared layout prior -- masked objects and a model's attention hot-zones both tend
to sit in similar typical locations across images of one dataset (e.g. frame center), so
even a mismatched B's saliency map runs unusually high right where A's mask happens to
be. With --translate, each mask is rigidly translated to a uniformly random valid
position within its own image's bounds (same shape/size, random placement) before being
tested against B's saliency -- this keeps effect (1) (contiguous shape on a locally
correlated map) but removes effect (2) (the mask no longer sits at its "typical" layout
location). If the rate with --translate drops close to alpha, the layout prior (2)
explains most of the inflation; if it stays near the untranslated rate, local spatial
autocorrelation (1) does.

Usage:
    python permtest_calibration.py                        # all 38 RQ1 conditions x 2 methods
    python permtest_calibration.py --datasets S1,S2        # restrict to specific dataset ids
    python permtest_calibration.py --seed 7                # derangement seed (default 42)
    python permtest_calibration.py --translate             # layout-prior ablation (see above)
    python permtest_calibration.py --translate --translate-seed 7
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths
from common.permutation_test import add_permutation_pvalues, SIGNIFICANCE_ALPHA

DATASETS = (
    [f"S{i}" for i in range(1, 8)]
    + [f"mimicry_{i}" for i in range(10)]
    + [f"biggan_ai_tigs_{i}" for i in range(10)]
    + [f"ep_dnn_{i}" for i in range(10)]
    + ["waterbird_test"]
)
ATTENTIONS = ["idg", "ig2"]
DERANGEMENT_SEED = 42
TRANSLATE_SEED = 123


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


def derangement(ids: list, seed: int) -> dict:
    """A permutation of `ids` with no fixed point (sigma(i) != i for every i), via
    rejection sampling -- cheap at N<=1000 (a uniform random permutation has no fixed
    point with probability ~1/e, so this resamples a handful of times on average)."""
    rng = np.random.default_rng(seed)
    arr = np.array(ids)
    n = len(arr)
    for _ in range(10000):
        perm = rng.permutation(n)
        if np.all(arr[perm] != arr):
            return {int(a): int(b) for a, b in zip(arr, arr[perm])}
    raise RuntimeError(f"failed to find a derangement of {n} ids in 10000 tries")


def load_masks_array(segments_dir: Path, image_id: int) -> np.ndarray:
    return np.load(segments_dir / f"{image_id}_masks.npz")["masks"]  # (n_objects, H, W) bool


def random_translate_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rigidly translates `mask`'s True pixels to a uniformly random valid position
    within its own (H, W) canvas -- same shape/size, just a different (random) top-left
    placement within the canvas's bounding-box slack. A no-op for an empty mask (slack
    computation would be meaningless with no True pixels to bound)."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return mask
    h, w = mask.shape
    min_r, max_r = int(rows.min()), int(rows.max())
    min_c, max_c = int(cols.min()), int(cols.max())
    slack_r = h - (max_r - min_r + 1)
    slack_c = w - (max_c - min_c + 1)
    new_min_r = int(rng.integers(0, slack_r + 1))
    new_min_c = int(rng.integers(0, slack_c + 1))
    dr, dc = new_min_r - min_r, new_min_c - min_c
    out = np.zeros_like(mask)
    out[rows + dr, cols + dc] = True
    return out


def load_saliency(attr_dir: Path, image_id: int) -> np.ndarray:
    attr = np.load(attr_dir / f"{image_id}_attr.npy")
    return np.abs(attr).sum(axis=0)  # Saliency(A), Eq. 1


def resize_saliency(saliency: np.ndarray, target_hw: tuple) -> np.ndarray:
    if saliency.shape == target_hw:
        return saliency
    h, w = target_hw
    resized = Image.fromarray(saliency.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR)
    return np.array(resized)


def run_condition(dataset_id: str, attention: str, seed: int,
                   translate: bool = False, translate_seed: int = TRANSLATE_SEED) -> dict:
    cfg = paths.load_dataset_config(dataset_id)
    segments_dir = paths.dataset_results_dir(cfg, "segments")
    attr_dir = paths.dataset_results_dir(cfg, attention)

    ids = sorted(int(p.stem.replace("_masks", "")) for p in segments_dir.glob("*_masks.npz"))
    ids = [i for i in ids if (attr_dir / f"{i}_attr.npy").exists()]
    if len(ids) < 2:
        return {"n_images": len(ids), "n_trials": 0, "n_significant": 0, "rate": None}

    sigma = derangement(ids, seed=seed)

    n_sig, n_total = 0, 0
    for i in ids:
        masks_arr = load_masks_array(segments_dir, i)
        target_hw = masks_arr.shape[1:]
        if translate:
            # One RNG per image, consumed in object order -- reproducible given
            # (translate_seed, i), independent of attention method (same translated
            # layout tested against both idg's and ig2's mismatched saliency map).
            rng = np.random.default_rng(translate_seed * 1_000_003 + i)
            objects_i = [{"mask": random_translate_mask(masks_arr[k], rng)}
                         for k in range(masks_arr.shape[0])]
        else:
            objects_i = [{"mask": masks_arr[k]} for k in range(masks_arr.shape[0])]

        saliency_b = load_saliency(attr_dir, sigma[i])
        saliency_b = resize_saliency(saliency_b, target_hw)

        stats_rows = [{} for _ in objects_i] + [{}]  # +1 synthetic "unclaimed" row
        add_permutation_pvalues(stats_rows, saliency_b, objects_i, seed=i)

        for row in stats_rows:
            n_total += 1
            if row["is_significant"]:
                n_sig += 1

    return {"n_images": len(ids), "n_trials": n_total, "n_significant": n_sig,
            "rate": (n_sig / n_total) if n_total else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None, help="comma-separated dataset ids (default: all 38)")
    ap.add_argument("--seed", type=int, default=DERANGEMENT_SEED, help="derangement seed")
    ap.add_argument("--translate", action="store_true",
                     help="layout-prior ablation: randomly translate each mask within its own "
                          "image bounds before testing against the mismatched saliency map "
                          "(see module docstring)")
    ap.add_argument("--translate-seed", type=int, default=TRANSLATE_SEED,
                     help="seed for --translate's random mask placement (default 123)")
    ap.add_argument("--out", default=None, help="output json path, repo-relative or absolute "
                                                 "(default: results/permtest_calibration.json, or "
                                                 "results/permtest_calibration_translated.json with "
                                                 "--translate)")
    args = ap.parse_args()

    dataset_ids = args.datasets.split(",") if args.datasets else DATASETS

    per_condition = {}
    totals = {"overall": {"n_trials": 0, "n_significant": 0}}
    totals_by_family = {}
    totals_by_attention = {a: {"n_trials": 0, "n_significant": 0} for a in ATTENTIONS}

    for dataset_id in dataset_ids:
        fam = family_of(dataset_id)
        totals_by_family.setdefault(fam, {"n_trials": 0, "n_significant": 0})
        per_condition[dataset_id] = {}
        for attention in ATTENTIONS:
            r = run_condition(dataset_id, attention, args.seed,
                               translate=args.translate, translate_seed=args.translate_seed)
            per_condition[dataset_id][attention] = r
            for bucket in (totals["overall"], totals_by_family[fam], totals_by_attention[attention]):
                bucket["n_trials"] += r["n_trials"]
                bucket["n_significant"] += r["n_significant"]
            rate_str = "n/a" if r["rate"] is None else f"{100 * r['rate']:.2f}%"
            print(f"[{dataset_id}/{attention}] images={r['n_images']} trials={r['n_trials']} "
                  f"significant={r['n_significant']} rate={rate_str}", flush=True)

    def rate_of(bucket):
        return (bucket["n_significant"] / bucket["n_trials"]) if bucket["n_trials"] else None

    result = {
        "alpha": SIGNIFICANCE_ALPHA,
        "derangement_seed": args.seed,
        "translate": args.translate,
        "translate_seed": args.translate_seed if args.translate else None,
        "overall": {**totals["overall"], "rate": rate_of(totals["overall"])},
        "by_family": {f: {**b, "rate": rate_of(b)} for f, b in totals_by_family.items()},
        "by_attention": {a: {**b, "rate": rate_of(b)} for a, b in totals_by_attention.items()},
        "per_condition": per_condition,
    }

    mode = "TRANSLATED (layout-prior ablation)" if args.translate else "untranslated (original placement)"
    print(f"\n===== PermTest calibration (alpha={SIGNIFICANCE_ALPHA}, {mode}) =====")
    print(f"overall: {result['overall']['n_significant']}/{result['overall']['n_trials']} "
          f"= {100 * result['overall']['rate']:.2f}%")
    for f, b in result["by_family"].items():
        print(f"  {f:12s}: {b['n_significant']:6d}/{b['n_trials']:6d} = {100 * b['rate']:.2f}%")
    for a, b in result["by_attention"].items():
        print(f"  {a:12s}: {b['n_significant']:6d}/{b['n_trials']:6d} = {100 * b['rate']:.2f}%")

    default_name = "permtest_calibration_translated.json" if args.translate else "permtest_calibration.json"
    out_path = Path(args.out) if args.out else paths.RQ_RESULTS_DIR / default_name
    if not out_path.is_absolute():
        out_path = paths.PRISM_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
