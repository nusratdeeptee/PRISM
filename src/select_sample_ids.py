"""Picks which images to run segments/ig2/idg over in the first place, from a
dataset that hasn't been processed at all yet (job 37's role for
waterbird_test -- distinct from generate_analysis.py's --sample-json/
resolve_sample_ids, which subsamples an ALREADY-computed pool for a smaller
report). Draws --sample-size distinct image ids straight from the dataset's
own metadata_csv, stratified by pass/fail (see common.sampling) so the chosen
subset keeps the same pass/fail ratio as the full dataset, and writes them to
a plain JSON list of ints under output/<dataset>/<model>/samples/ -- the same
file format/location convention resolve_sample_ids uses, and (deliberately)
no new image folder or copy of the images themselves, per the "keep the list,
not a folder" instruction.

Usage:
    python select_sample_ids.py --dataset waterbird_test --sample-size 1000
    # -> writes output/waterbird_test/finetuned_classifier/samples/sample_1000.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths
from common.sampling import stratified_sample_counts

SAMPLE_SEED = 42   # same fixed seed convention as generate_analysis.py's resolve_sample_ids


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample-size", type=int, required=True)
    ap.add_argument("--out-name", default=None,
                     help="output file stem (default: sample_<size>, matching resolve_sample_ids' "
                          "own naming so the two stay visually interchangeable)")
    args = ap.parse_args()

    cfg = paths.load_dataset_config(args.dataset)
    out_name = args.out_name or f"sample_{args.sample_size}"
    out_path = paths.dataset_results_dir(cfg, "samples") / f"{out_name}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists -- delete it first if you want to redraw "
                          f"(resolve_sample_ids-style scripts reuse an existing file as-is rather "
                          f"than silently overwriting it, and so does this one)")

    meta_df = pd.read_csv(cfg["metadata_csv"])
    if meta_df["image_id"].duplicated().any():
        raise SystemExit(f"{cfg['metadata_csv']} has duplicate image_id rows -- refusing to sample")
    pf_map = meta_df.set_index("image_id")["label"].to_dict()
    bad = {v for v in pf_map.values()} - {"pass", "fail"}
    if bad:
        raise SystemExit(f"{cfg['metadata_csv']}'s label column has non-pass/fail values: {bad}")

    pools_by_class = {"pass": sorted(i for i, l in pf_map.items() if l == "pass"),
                       "fail": sorted(i for i, l in pf_map.items() if l == "fail")}
    total = len(pf_map)
    if args.sample_size > total:
        raise SystemExit(f"--sample-size {args.sample_size} exceeds the dataset's {total} images")

    counts = stratified_sample_counts({k: len(v) for k, v in pools_by_class.items()}, args.sample_size)

    import random
    rng = random.Random(SAMPLE_SEED)
    sample_ids = set()
    for cls, n in counts.items():
        sample_ids |= set(rng.sample(pools_by_class[cls], n))

    out_path.write_text(json.dumps(sorted(sample_ids)))
    fail_rate = len(pools_by_class["fail"]) / total
    log(f"[{args.dataset}] pool={total} (pass={len(pools_by_class['pass'])}, "
        f"fail={len(pools_by_class['fail'])}, fail_rate={fail_rate:.1%}) -> drew {counts['pass']} pass "
        f"+ {counts['fail']} fail = {args.sample_size} ids (seed={SAMPLE_SEED}), wrote {out_path}")


if __name__ == "__main__":
    main()
