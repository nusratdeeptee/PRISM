"""Single entry point over the 4 pipeline scripts, so each stage can be
launched as its own SLURM job. Every mode just dispatches to the matching
script(s) as a subprocess (using this same interpreter) -- no pipeline logic
lives here.

Modes:
  segments   generate_segments.py only (Qwen + Grounding DINO + SAM2)
  ig2        ig2.py only
  idg        idg.py only
  analysis   generate_analysis.py only (--dataset may be a comma-separated
             list here, for pooled canonicalization across datasets)
  full-ig2   segments -> ig2 -> analysis(ig2), one dataset
  full-idg   segments -> idg -> analysis(idg), one dataset

Usage:
    python main.py --mode ig2 --dataset S2_300
    python main.py --mode analysis --dataset S1,S2,genai_tigs_biggan_7 --attention both
    python main.py --mode full-ig2 --dataset S2_300
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SINGLE_DATASET_MODES = {"segments", "ig2", "idg", "full-ig2", "full-idg"}


def run(script: str, args: list) -> None:
    cmd = [sys.executable, str(HERE / script), *args]
    print(f"+ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                     choices=["segments", "ig2", "idg", "analysis", "full-ig2", "full-idg"])
    ap.add_argument("--dataset", required=True,
                     help="dataset id; comma-separated list allowed only for --mode analysis")
    ap.add_argument("--attention", choices=["ig2", "idg", "both"],
                     help="required for --mode analysis")
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--image-ids-file", default=None,
                     help="segments/ig2/idg modes only: restrict to exactly these image ids -- see "
                          "the underlying scripts' own --image-ids-file docstring. Mutually exclusive "
                          "with --limit-images.")
    ap.add_argument("--shard-index", type=int, default=None,
                     help="0-based shard index for splitting a dataset's images across an "
                          "array job (segments/ig2/idg modes only); requires --num-shards")
    ap.add_argument("--num-shards", type=int, default=None,
                     help="total number of shards; this task processes images "
                          "[shard_index::num_shards] of the sorted image list")
    ap.add_argument("--out-dir", default=None,
                     help="override the default results/<dataset>/<model>/<stage> layout "
                          "(segments/ig2/idg modes only)")
    ap.add_argument("--sample-json", default=None,
                     help="--mode analysis only: restrict the report to a random image subset -- "
                          "see generate_analysis.py's own --sample-json docstring")
    ap.add_argument("--sample-size", type=int, default=None,
                     help="--mode analysis only: required together with --sample-json only when "
                          "that file doesn't exist yet")
    ap.add_argument("--sample-seed", type=int, default=None,
                     help="--mode analysis only: seed for --sample-json's random draw when that file "
                          "doesn't exist yet (default: generate_analysis.py's own SAMPLE_SEED=42). Pass "
                          "a different value together with a not-yet-existing --sample-json path for a "
                          "repeat run, so it draws a distinct random subset instead of reproducing the "
                          "original draw.")
    ap.add_argument("--kind-priority", default=None,
                     help="--mode analysis only: comma-separated metadata_csv 'kind' values in "
                          "preference order -- see generate_analysis.py's --kind-priority docstring")
    ap.add_argument("--report-only", action="store_true",
                     help="--mode analysis only: skip the per-image stage and only rebuild the "
                          "report files from the already-existing analysis_<attention> outputs")
    args = ap.parse_args()
    if (args.shard_index is None) != (args.num_shards is None):
        ap.error("--shard-index and --num-shards must be given together")

    datasets = args.dataset.split(",")
    if args.mode in SINGLE_DATASET_MODES and len(datasets) != 1:
        ap.error(f"--mode {args.mode} takes exactly one dataset, got {len(datasets)}: {args.dataset!r}")

    if args.limit_images and args.image_ids_file:
        ap.error("--limit-images and --image-ids-file are mutually exclusive")
    limit_args = ["--limit-images", str(args.limit_images)] if args.limit_images else []
    ids_file_args = ["--image-ids-file", args.image_ids_file] if args.image_ids_file else []
    shard_args = (["--shard-index", str(args.shard_index), "--num-shards", str(args.num_shards)]
                  if args.num_shards else [])
    out_dir_args = ["--out-dir", args.out_dir] if args.out_dir else []

    if args.mode == "segments":
        run("generate_segments.py", ["--dataset", datasets[0], *limit_args, *ids_file_args,
                                      *shard_args, *out_dir_args])

    elif args.mode == "ig2":
        run("ig2.py", ["--dataset", datasets[0], *limit_args, *ids_file_args, *out_dir_args])

    elif args.mode == "idg":
        run("idg.py", ["--dataset", datasets[0], *limit_args, *ids_file_args, *shard_args, *out_dir_args])

    elif args.mode == "analysis":
        if not args.attention:
            ap.error("--mode analysis requires --attention")
        # --out-dir here means generate_analysis.py's results ROOT (it reads
        # segments/<attention> and writes analysis_<attention>/report_<attention>
        # as siblings under it), not a single stage's own directory like the
        # segments/ig2/idg modes above -- only valid with a single --dataset.
        sample_args = ["--sample-json", args.sample_json] if args.sample_json else []
        if args.sample_size is not None:
            sample_args += ["--sample-size", str(args.sample_size)]
        if args.sample_seed is not None:
            sample_args += ["--sample-seed", str(args.sample_seed)]
        if args.kind_priority is not None:
            sample_args += ["--kind-priority", args.kind_priority]
        run("generate_analysis.py", ["--datasets", args.dataset, "--attention", args.attention,
                                      *out_dir_args, *sample_args,
                                      *(["--report-only"] if args.report_only else [])])

    elif args.mode == "full-ig2":
        run("generate_segments.py", ["--dataset", datasets[0], *limit_args])
        run("ig2.py", ["--dataset", datasets[0], *limit_args])
        run("generate_analysis.py", ["--datasets", datasets[0], "--attention", "ig2"])

    elif args.mode == "full-idg":
        run("generate_segments.py", ["--dataset", datasets[0], *limit_args])
        run("idg.py", ["--dataset", datasets[0], *limit_args])
        run("generate_analysis.py", ["--datasets", datasets[0], "--attention", "idg"])


if __name__ == "__main__":
    main()
