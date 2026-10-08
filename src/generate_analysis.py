"""For each image: loads its segments (Generate_segments.py) and its
attention map (ig2.py or idg.py), computes per-object attention share, filters
to objects whose attention is statistically significant (permutation test),
and renders a contact sheet. Then, across every requested dataset: pools and
canonicalizes labels, builds the presence/mass-share/peak-ratio significance
tables, Key Findings, a Random Forest + SHAP feature-combination stage, a
label-to-image backtrack, and a feature_importance.md (primary, significance-filtered) +
feature_importance_full.md (full detail) report pair per dataset.

Usage:
    python generate_analysis.py --datasets S1,S2,genai_tigs_biggan_7 --attention ig2
    python generate_analysis.py --datasets S2_300 --attention both
"""
import argparse
import glob
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths
from common.lc_models import ensure_metadata_csv
from common.object_stats import compute_object_stats
from common.permutation_test import add_permutation_pvalues, SIGNIFICANCE_ALPHA
from common.visualization import overlay_heatmap_on_image, normalize_saliency
from common.drawing import build_contact_sheet
from common.canonicalize import build_canonical_map, write_canonicalization_trace, normalize_label
from common.stats_tables import (build_report, build_label_membership, write_label_membership,
                                  build_feature_importance_report as build_passfail_importance_report)
from common.low_failure_stats import (LOW_FAILURE_THRESHOLD, dataset_fail_rate, dataset_pass_rate,
                                       low_failure_branch_reason, build_confidence_series,
                                       build_low_failure_report, build_feature_importance_report,
                                       compute_label_df, feature_importance_csv_table)
from common.sampling import stratified_sample_counts


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


SAMPLE_SEED = 42   # fixed seed for --sample-json's random draw, same convention as this
                    # project's other fixed seeds (e.g. UMAP random_state=42) -- reproducible.


def resolve_sample_ids(sample_json: str, sample_size: int, dataset_id: str,
                        results_root: str = None, kind_priority: list = None,
                        sample_seed: int = SAMPLE_SEED) -> tuple:
    """--sample-json support (jobs 8-11): PATH is resolved relative to PRISM_ROOT
    if not already absolute (same convention as --out-dir's override handling
    elsewhere in this file). If PATH doesn't exist yet, --sample-size distinct
    image ids are drawn from the pool of images with completed segments for
    this dataset (segments is attention-independent and always computed
    first, so it's the right pool for either/both attention passes) and
    written to PATH as a plain JSON list of ints (not paths -- every
    per-image file in this pipeline is already addressed by bare id, so the
    list needs nothing else, keeping it PRISM-relative-path-free entirely).

    The draw is stratified by pass/fail (see stratified_sample_counts): the
    pool is split into its pass sub-pool and fail sub-pool using the same
    metadata_csv labels load_dataset_for_report() reads, x = sample_size /
    len(pool) is computed, and sample_size is apportioned across the two
    sub-pools proportional to their share of the pool -- so a random sample
    always preserves the dataset's own pass/fail ratio exactly, rather than
    a plain pool-wide random draw which only preserves it in expectation
    (and can skew badly at small sample_size). Each sub-pool is then sampled
    independently (sample_seed) and the two draws are unioned.

    sample_seed (jobs 8-11/16-19/24-27/33-36/42-45 all used the default,
    SAMPLE_SEED=42; repeat runs pass a different value here so a repeat's
    --sample-json, pointed at a not-yet-existing PATH, draws a genuinely
    different random subset instead of reproducing the same draw): the seed
    used for both the plain and kind_priority draw below. Irrelevant whenever
    PATH already exists (id list reused as-is, no draw happens at all).

    kind_priority (jobs 16-19/24-27, mimicry/biggan_ai_tigs only): an
    ordered list of metadata_csv "kind" values (e.g.
    ["perturbed", "original", "duplicate"]). When given, each pass/fail
    stratum is filled by draining higher-priority kinds first (each tier's
    ids sampled randomly via sample_seed before moving to the next tier),
    instead of sampling the whole stratum uniformly at random -- so a
    report prefers the deliberately-generated perturbed/original images and
    only reaches into mimicry_6's 212 duplicate-image ids if a stratum
    can't be filled without them. Stratum SIZES (the pass/fail ratio) are
    unaffected either way -- this only changes which specific ids within a
    stratum get picked. Datasets with no "kind" column, or called without
    this argument, are completely unaffected (original behavior).

    If PATH already exists, its id list is used as-is (--sample-size and the
    stratification above are both skipped). Returns (sample_ids: set[int],
    tag: str) where tag is PATH's stem, used to suffix the analysis/report
    stage directories so different sample sizes (or the unsampled default)
    never collide."""
    sample_path = Path(sample_json)
    if not sample_path.is_absolute():
        sample_path = paths.PRISM_ROOT / sample_path

    if sample_path.exists():
        sample_ids = set(json.loads(sample_path.read_text()))
        log(f"[{dataset_id}] --sample-json {sample_path} exists -- using its "
            f"{len(sample_ids)} image ids as-is (--sample-size ignored if given)")
    else:
        if sample_size is None:
            raise SystemExit(f"--sample-json {sample_path} does not exist yet -- "
                              f"--sample-size is required to create it")
        cfg = paths.load_dataset_config(dataset_id)
        segments_dir = paths.dataset_results_dir(cfg, "segments",
                                                  override=f"{results_root}/segments" if results_root else None)
        pool = sorted(int(Path(p).stem.replace("_segments", ""))
                      for p in glob.glob(str(segments_dir / "*_segments.json")))
        if sample_size > len(pool):
            raise SystemExit(f"--sample-size {sample_size} exceeds the available pool of "
                              f"{len(pool)} images with completed segments for {dataset_id}")

        meta_df = pd.read_csv(cfg["metadata_csv"])
        pf_map = meta_df.drop_duplicates("image_id").set_index("image_id")["label"].to_dict()
        missing = [i for i in pool if pf_map.get(i) not in ("pass", "fail")]
        if missing:
            raise SystemExit(f"resolve_sample_ids: {len(missing)} of {len(pool)} pooled image ids for "
                              f"{dataset_id} have no pass/fail label in {cfg['metadata_csv']} (e.g. "
                              f"{missing[:5]}) -- can't stratify")
        pools_by_class = {"pass": [i for i in pool if pf_map[i] == "pass"],
                           "fail": [i for i in pool if pf_map[i] == "fail"]}
        counts = stratified_sample_counts({k: len(v) for k, v in pools_by_class.items()}, sample_size)
        rng = random.Random(sample_seed)
        sample_ids = set()
        if kind_priority:
            kind_map = meta_df.drop_duplicates("image_id").set_index("image_id")["kind"].to_dict()
            tier_log = []
            for cls, n in counts.items():
                remaining = set(pools_by_class[cls])
                chosen = []
                for kind in kind_priority:
                    if len(chosen) >= n:
                        break
                    tier_ids = sorted(i for i in remaining if kind_map.get(i) == kind)
                    take = rng.sample(tier_ids, min(n - len(chosen), len(tier_ids)))
                    chosen.extend(take)
                    remaining -= set(take)
                if len(chosen) < n:  # safety net: kinds outside kind_priority, if any
                    chosen.extend(rng.sample(sorted(remaining), n - len(chosen)))
                tier_log.append(f"{cls}: " + ", ".join(
                    f"{k}={sum(1 for i in chosen if kind_map.get(i) == k)}" for k in kind_priority))
                sample_ids |= set(chosen)
            log(f"[{dataset_id}] kind-priority draw ({'>'.join(kind_priority)}): " + " | ".join(tier_log))
        else:
            for cls, n in counts.items():
                sample_ids |= set(rng.sample(pools_by_class[cls], n))
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(json.dumps(sorted(sample_ids)))
        log(f"[{dataset_id}] --sample-json {sample_path} did not exist -- drew {counts['pass']} pass + "
            f"{counts['fail']} fail = {sample_size} ids (seed={sample_seed}, stratified from a pool of "
            f"{len(pools_by_class['pass'])} pass / {len(pools_by_class['fail'])} fail), wrote it")

    return sample_ids, sample_path.stem


def load_segments(segments_dir: Path, image_id: int):
    seg = json.loads((segments_dir / f"{image_id}_segments.json").read_text())
    masks = np.load(segments_dir / f"{image_id}_masks.npz")["masks"]
    objects = [{**d, "mask": masks[i]} for i, d in enumerate(seg["detections"])]
    return seg, objects


def process_image(image_id: int, cfg: dict, segments_dir: Path, attr_dir: Path, out_dir: Path,
                   pass_fail_label: str, attention_name: str) -> dict:
    image_path = cfg["images_dir"] / f"{image_id}.png"
    pil_image = Image.open(image_path).convert("RGB")
    seg, objects = load_segments(segments_dir, image_id)
    attr = np.load(attr_dir / f"{image_id}_attr.npy")
    saliency = np.abs(attr).sum(axis=0)
    # Segments' masks are always native-resolution (generate_segments.py never
    # resizes); an attention map can be computed at a smaller bounded working
    # resolution instead (idg.py does this via cfg["raw_image_size"], needed
    # for datasets with large/variable native resolutions like ISIC, to keep
    # IDG's own batched interpolation steps from allocating tens of GB per
    # image). Upsample saliency to match whenever the two disagree -- a
    # generic reconciliation, not specific to any one dataset.
    if saliency.shape != pil_image.size[::-1]:
        saliency = np.array(Image.fromarray(saliency.astype(np.float32), mode="F")
                             .resize(pil_image.size, Image.BILINEAR))

    stats = compute_object_stats(objects, saliency)
    add_permutation_pvalues(stats["rows"], saliency, objects, seed=image_id)

    highlighted = Image.fromarray(overlay_heatmap_on_image(np.array(pil_image), normalize_saliency(saliency)))
    build_contact_sheet(pil_image, highlighted, objects, stats["rows"], seg["qwen_tags"], attention_name,
                         pass_fail_label, SIGNIFICANCE_ALPHA, out_dir / f"{image_id}_contact_sheet.png")

    payload = {
        "image_id": image_id, "qwen_tags": seg["qwen_tags"], "n_ungrounded_tags": seg["n_ungrounded_tags"],
        "global_peak": stats["global_peak"], "overlap_mass_share": stats["overlap_mass_share"],
        "degenerate": stats["degenerate"],
        "objects": [{"label": r["label"], "score": r["score"], "area_share": r["area_share"],
                     "mass_share": r["mass_share"], "peak_ratio": r["peak_ratio"],
                     "permutation_pvalue": r["permutation_pvalue"], "is_significant": r["is_significant"]}
                    for r in stats["rows"]],
    }
    (out_dir / f"{image_id}_stats.json").write_text(json.dumps(payload, indent=2))
    n_sig = sum(1 for r in stats["rows"][:-1] if r["is_significant"])
    log(f"[{image_id}] OK -- {n_sig}/{len(objects)} objects significant (p<{SIGNIFICANCE_ALPHA})")
    return payload


# --- Per-image stage: significance filter + contact sheet -------------------
def run_per_image_stage(dataset_id: str, attention_name: str, results_root: str = None,
                         sample_ids: set = None, stage_suffix: str = "") -> None:
    """results_root (optional): this dataset's results ROOT (e.g. an
    --out-dir given to its segments/ig2/idg SLURM jobs), containing
    segments/<attention_name>/analysis_<attention_name> as siblings --
    NOT a single stage's own directory, since this stage reads two of
    those (segments, attention) and writes a third. Single-dataset only:
    a run over multiple --datasets sharing one results_root would collide.
    Omit to use the default results/<dataset>/<model>/<stage> layout for
    all three, same as every other script here does by default.

    sample_ids/stage_suffix (both optional, from --sample-json -- see
    resolve_sample_ids): when given, restricts this stage to just
    sample_ids (segments/attention themselves are still read from the FULL
    unsampled dataset -- only which images get "collected" into per-image
    stats/contact sheets is restricted) and writes to
    analysis_<attention_name><stage_suffix> instead of the unsuffixed
    default, so different sample sizes (and the unsampled default) never
    collide with each other's per-image output."""
    cfg = paths.load_dataset_config(dataset_id)
    segments_dir = paths.dataset_results_dir(cfg, "segments",
                                              override=f"{results_root}/segments" if results_root else None)
    attr_dir = paths.dataset_results_dir(cfg, attention_name,
                                          override=f"{results_root}/{attention_name}" if results_root else None)
    analysis_stage = f"analysis_{attention_name}{stage_suffix}"
    out_dir = paths.dataset_results_dir(cfg, analysis_stage,
                                         override=f"{results_root}/{analysis_stage}" if results_root else None)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta_df = ensure_metadata_csv(cfg, cfg["images_dir"], cfg["metadata_csv"], device)
    label_by_id = meta_df.set_index("image_id")["label"].to_dict()
    image_ids = sorted(int(Path(p).stem.replace("_segments", ""))
                       for p in glob.glob(str(segments_dir / "*_segments.json")))
    if sample_ids is not None:
        image_ids = [i for i in image_ids if i in sample_ids]
    done_ids = {int(Path(p).stem.replace("_stats", "")) for p in glob.glob(str(out_dir / "*_stats.json"))}

    for image_id in image_ids:
        if image_id in done_ids:
            continue
        attr_path = attr_dir / f"{image_id}_attr.npy"
        if not attr_path.exists():
            log(f"[{image_id}] SKIP -- no {attention_name} attribution computed yet")
            continue
        try:
            process_image(image_id, cfg, segments_dir, attr_dir, out_dir,
                           label_by_id.get(image_id, "unknown"), attention_name)
        except Exception:
            log(f"[{image_id}] ERROR:\n{traceback.format_exc()}")


# --- Cross-dataset stage: canonicalize, stats tables, report -----------------
def load_dataset_for_report(dataset_id: str, attention_name: str, results_root: str = None,
                             stage_suffix: str = ""):
    """Returns (all_images, per_image_meta, chosen_df, confidence_series,
    confidence_desc, all_occurrence_df) -- mirrors the schema the old flat
    objects.json produced, so the stats-table code needs no change to
    consume the new per-stage layout. chosen_df carries area_share alongside
    mass_share/peak_ratio so the low-failure branch (common/low_failure_stats.py)
    can compare attribution share against segment-size share; confidence_series
    is None when this dataset has no continuous output column (that branch
    self-skips its correlation section in that case). all_occurrence_df is
    the UNFILTERED sibling of chosen_df -- one row per (image_id, label) for
    every detection regardless of is_significant, "UNCLAIMED" excluded --
    chosen_df alone can't say how often a label was detected but NOT
    significant, which the low-failure branch's consistency-rate column
    needs (see low_failure_stats.add_consistency)."""
    cfg = paths.load_dataset_config(dataset_id)
    analysis_stage = f"analysis_{attention_name}{stage_suffix}"
    out_dir = paths.dataset_results_dir(cfg, analysis_stage,
                                         override=f"{results_root}/{analysis_stage}" if results_root else None)
    files = sorted(glob.glob(str(out_dir / "*_stats.json")), key=lambda f: int(Path(f).stem.split("_")[0]))

    all_images, per_image_meta, chosen_rows, all_rows = {}, {}, [], []
    for f in files:
        d = json.loads(Path(f).read_text())
        iid = int(d["image_id"])
        per_image_meta[iid] = {"overlap_mass_share": d["overlap_mass_share"], "degenerate": d["degenerate"],
                                "n_chosen": sum(1 for o in d["objects"] if o["label"] != "UNCLAIMED" and o["is_significant"])}
        for o in d["objects"]:
            if o["label"] != "UNCLAIMED":
                all_rows.append({"image_id": iid, "label": o["label"]})
            if o["is_significant"]:
                chosen_rows.append({"image_id": iid, "label": o["label"], "mass_share": o["mass_share"],
                                     "peak_ratio": o["peak_ratio"], "area_share": o["area_share"]})

    meta_df = pd.read_csv(cfg["metadata_csv"])
    pf_map = meta_df.drop_duplicates("image_id").set_index("image_id")["label"].to_dict()
    for iid in per_image_meta:
        if iid in pf_map:
            all_images[iid] = pf_map[iid]
    if not all_images:
        raise RuntimeError(f"Could not recover pass/fail labels for {dataset_id} from {cfg['metadata_csv']}")
    confidence_series, confidence_desc = build_confidence_series(meta_df, cfg)
    return (all_images, per_image_meta, pd.DataFrame(chosen_rows), confidence_series, confidence_desc,
            pd.DataFrame(all_rows))


def run_cross_dataset_stage(dataset_ids: list, attention_name: str, results_root: str = None,
                             stage_suffix: str = "") -> None:
    """results_root: see run_per_image_stage's docstring -- same single-
    dataset-only caveat applies (a shared root across multiple --datasets
    would collide). stage_suffix: see run_per_image_stage's docstring --
    must match whatever it was called with, so this reads the same
    analysis_<attention_name><stage_suffix> directory that stage wrote."""
    loaded = {}
    for dataset_id in dataset_ids:
        log(f"[{dataset_id}] loading {attention_name} analysis...")
        loaded[dataset_id] = load_dataset_for_report(dataset_id, attention_name, results_root, stage_suffix)

    # Pooled across every requested dataset so the same word canonicalizes
    # identically everywhere, not per-dataset. Pulls from BOTH chosen_df and
    # all_occurrence_df -- a raw label string that only ever shows up in a
    # non-significant detection would otherwise be missing from the
    # canonicalization vocabulary and fall back to un-grouped, even though
    # every occurrence (significant or not) of the same string must map to
    # the same canonical label for the consistency-rate math to be correct.
    all_raw_labels = [lbl for _, _, chosen_df, _, _, all_occ_df in loaded.values()
                       for lbl in pd.concat([chosen_df["label"], all_occ_df["label"]], ignore_index=True)]
    canonical_map = build_canonical_map(all_raw_labels)

    for dataset_id, (all_images, per_image_meta, chosen_df, confidence_series, confidence_desc,
                      all_occ_df) in loaded.items():
        chosen_df["canonical_label"] = chosen_df["label"].apply(
            lambda l: canonical_map.get(normalize_label(l), normalize_label(l)))
        if len(all_occ_df):
            all_occ_df["canonical_label"] = all_occ_df["label"].apply(
                lambda l: canonical_map.get(normalize_label(l), normalize_label(l)))

    for dataset_id in dataset_ids:
        all_images, per_image_meta, chosen_df, confidence_series, confidence_desc, all_occ_df = loaded[dataset_id]
        cfg = paths.load_dataset_config(dataset_id)
        # Reports (small: markdown/csv/json) go in their own directory, kept
        # separate from analysis_{attention} (per-image contact sheets +
        # _stats.json -- thousands of files, hard to find a report in).
        report_stage = f"report_{attention_name}{stage_suffix}"
        report_dir = paths.dataset_results_dir(
            cfg, report_stage,
            override=f"{results_root}/{report_stage}" if results_root else None)

        write_canonicalization_trace(all_raw_labels, canonical_map, report_dir)

        fail_rate = dataset_fail_rate(all_images)
        pass_rate = dataset_pass_rate(all_images)
        total_occurrences = None   # only the low-failure branch below has this (all_occ_df-derived)
        # Normal branch (stats_tables' pass-vs-fail contrast tests) only when BOTH classes clear
        # LOW_FAILURE_THRESHOLD; if either is too thin (near-0% failure OR near-100% failure, i.e.
        # near-0% passing), fall back to the low-failure (attention-enrichment) branch, which needs
        # no balanced fail/pass class split at all.
        if fail_rate >= LOW_FAILURE_THRESHOLD and pass_rate >= LOW_FAILURE_THRESHOLD:
            importance_report = build_passfail_importance_report(dataset_id, all_images, per_image_meta, chosen_df)
            (report_dir / "feature_importance.md").write_text(importance_report)
            log(f"[{dataset_id}] -> {report_dir / 'feature_importance.md'}")

            full_report = build_report(dataset_id, all_images, per_image_meta, chosen_df)
            (report_dir / "feature_importance_full.md").write_text(full_report)
            log(f"[{dataset_id}] -> {report_dir / 'feature_importance_full.md'}")
        else:
            log(f"[{dataset_id}] fail rate {fail_rate:.1%}, pass rate {pass_rate:.1%} -- "
                f"{low_failure_branch_reason(fail_rate)} -- using low-failure (attention-enrichment) "
                f"analysis branch")
            label_df = compute_label_df(chosen_df, all_occ_df)
            if "n_total_occurrences" in label_df.columns:
                total_occurrences = label_df.set_index("label")["n_total_occurrences"].to_dict()

            importance_report = build_feature_importance_report(dataset_id, label_df)
            (report_dir / "feature_importance.md").write_text(importance_report)
            feature_importance_csv_table(label_df).to_csv(report_dir / "feature_importance.csv", index=False)
            log(f"[{dataset_id}] -> {report_dir / 'feature_importance.md'} (+ .csv)")

            full_report = build_low_failure_report(dataset_id, all_images, per_image_meta, chosen_df, label_df,
                                                    confidence_series, confidence_desc)
            (report_dir / "feature_importance_full.md").write_text(full_report)
            log(f"[{dataset_id}] -> {report_dir / 'feature_importance_full.md'}")

        def resolve_image_path(image_id, _cfg=cfg):
            p = _cfg["images_dir"] / f"{image_id}.png"
            return str(p) if p.exists() else ""

        membership = build_label_membership(chosen_df, all_images, resolve_image_path)
        write_label_membership(dataset_id, membership, report_dir, total_occurrences)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="comma-separated dataset ids")
    ap.add_argument("--attention", required=True, choices=["ig2", "idg", "both"])
    ap.add_argument("--out-dir", default=None,
                     help="override the default results/<dataset>/<model> ROOT (not a single stage's "
                          "own directory, unlike segments.py/ig2.py/idg.py -- this script reads "
                          "segments/<attention> and writes analysis_<attention>/report_<attention> as "
                          "siblings under it, e.g. --out-dir matching the same root the segments/ig2/idg "
                          "SLURM jobs for this dataset were given). Single-dataset --datasets only.")
    ap.add_argument("--sample-json", default=None,
                     help="path (PRISM-relative, or absolute) to a JSON file listing image ids to "
                          "restrict this analysis run to -- for jobs that want a report over a random "
                          "subset rather than every image (see resolve_sample_ids). If the file doesn't "
                          "exist yet, --sample-size distinct ids are randomly drawn and written there "
                          "first. If it already exists, its id list is used as-is. Omit both flags to "
                          "analyze every image (unchanged default behavior). Single-dataset only.")
    ap.add_argument("--sample-size", type=int, default=None,
                     help="required together with --sample-json only when that file doesn't exist yet.")
    ap.add_argument("--sample-seed", type=int, default=SAMPLE_SEED,
                     help="only meaningful together with --sample-json on a first (file-creating) run "
                          "(default: SAMPLE_SEED=42, the original jobs 8-45 convention). Repeat runs "
                          "pass a different value + a not-yet-existing --sample-json path to draw a "
                          "genuinely distinct random subset instead of reproducing the same draw.")
    ap.add_argument("--kind-priority", default=None,
                     help="comma-separated metadata_csv 'kind' values in preference order (e.g. "
                          "perturbed,original,duplicate) -- see resolve_sample_ids' own docstring. "
                          "Only meaningful together with --sample-json on a first (file-creating) run.")
    ap.add_argument("--report-only", action="store_true",
                     help="skip the per-image stage (contact sheets / _stats.json, which must already "
                          "exist) and only rebuild the cross-dataset report files")
    args = ap.parse_args()
    dataset_ids = args.datasets.split(",")
    attention_names = ["ig2", "idg"] if args.attention == "both" else [args.attention]
    if args.out_dir and len(dataset_ids) != 1:
        ap.error(f"--out-dir takes exactly one dataset (it's a shared results root), got {len(dataset_ids)}: "
                  f"{args.datasets!r}")
    if args.sample_json and len(dataset_ids) != 1:
        ap.error(f"--sample-json takes exactly one dataset, got {len(dataset_ids)}: {args.datasets!r}")

    sample_ids, stage_suffix = None, ""
    if args.sample_json:
        kind_priority = args.kind_priority.split(",") if args.kind_priority else None
        sample_ids, tag = resolve_sample_ids(args.sample_json, args.sample_size, dataset_ids[0], args.out_dir,
                                              kind_priority=kind_priority, sample_seed=args.sample_seed)
        stage_suffix = f"_{tag}"

    for attention_name in attention_names:
        if not args.report_only:
            for dataset_id in dataset_ids:
                run_per_image_stage(dataset_id, attention_name, args.out_dir, sample_ids, stage_suffix)
        run_cross_dataset_stage(dataset_ids, attention_name, args.out_dir, stage_suffix=stage_suffix)


if __name__ == "__main__":
    main()
