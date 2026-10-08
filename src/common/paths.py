"""Repo-relative path/config resolution -- the one place that knows where
config, data, output, and RQ results live, so every script can be run from
anywhere."""
import json
from pathlib import Path

PRISM_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PRISM_ROOT / "config"
DATA_DIR = PRISM_ROOT / "data"
# All generated PIPELINE output (per-dataset segments/ig2/idg/analysis, plus
# model-repair jobs under output/model_repair/) lives under PRISM/output
# (2026-09-22 reorg -- previously called "results"). See RQ_RESULTS_DIR below
# for the separate, narrower convention introduced the same day.
OUTPUT_DIR = PRISM_ROOT / "output"
# RQ_RESULTS_DIR is reserved specifically for research-question-level graphs
# (cross-dataset/cross-run comparison figures, e.g. a SEFAR-rq_plot.py-style
# script) -- NOT the per-dataset pipeline output above. Nothing writes here
# yet (2026-09-22): no PRISM rq.py/rq_plot.py script exists in this repo yet,
# only the convention/location is established now, for whenever one is
# built. Use rq_results_dir() below rather than this path directly, so a
# subdirectory always gets created on demand the same way dataset_results_dir
# does for OUTPUT_DIR.
RQ_RESULTS_DIR = PRISM_ROOT / "results"


def load_attention_config() -> dict:
    return json.loads((CONFIG_DIR / "attention.json").read_text())


def load_dataset_config(dataset_id: str) -> dict:
    """Reads config/datasets/<dataset_id>.json and resolves its paths
    relative to the repo root. `model_name` (used in the results/ layout)
    is the classifier name for imagenet_classifier datasets, or the lc_type
    itself for sgsm_resnet34 (there is only one such model)."""
    cfg = json.loads((CONFIG_DIR / "datasets" / f"{dataset_id}.json").read_text())
    cfg["dataset_id"] = dataset_id
    cfg["images_dir"] = PRISM_ROOT / cfg["images_dir"]
    cfg["metadata_csv"] = PRISM_ROOT / cfg["metadata_csv"]
    # Path(abs) / rel returns abs unchanged, so datasets whose lc_model_path
    # is a genuinely external/shared checkpoint (outside PRISM_ROOT, e.g. the
    # SGSM driving_model.ckpt under ai-toolkit/) still resolve correctly here.
    if cfg.get("lc_model_path"):
        cfg["lc_model_path"] = PRISM_ROOT / cfg["lc_model_path"]
    cfg["model_name"] = cfg["classifier_name"] if cfg["lc_type"] == "imagenet_classifier" else cfg["lc_type"]
    return cfg


def shard(items: list, shard_index: int | None, num_shards: int | None) -> list:
    """Splits `items` into `num_shards` contiguous, roughly-equal blocks and
    returns block `shard_index` (0-based) -- for array jobs that split one
    dataset's images across many parallel tasks. No-op (returns `items`
    unchanged) if shard_index/num_shards aren't both given."""
    if shard_index is None or num_shards is None:
        return items
    n = len(items)
    base, extra = divmod(n, num_shards)
    # first `extra` shards get one extra item, so sizes differ by at most 1.
    start = shard_index * base + min(shard_index, extra)
    end = start + base + (1 if shard_index < extra else 0)
    return items[start:end]


def dataset_results_dir(cfg: dict, stage: str, override: str = None) -> Path:
    """output/<dataset>/<model>/<stage> by default, stage one of
    "segments"/"ig2"/"idg"/"analysis_ig2"/"analysis_idg" -- or, if `override`
    is given (repo-relative or absolute), that exact directory instead, for
    callers that need a different output layout than the standard one."""
    d = Path(override) if override else OUTPUT_DIR / cfg["dataset_id"] / cfg["model_name"] / stage
    if not d.is_absolute():
        d = PRISM_ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def filter_by_ids_file(image_paths: list, ids_file: str) -> list:
    """--image-ids-file support (job 37-40 style: restrict segments/ig2/idg
    to a pre-chosen subset of a dataset's images, e.g. select_sample_ids.py's
    output, rather than "first N" (--limit-images) or "every image" (the
    default)). ids_file is resolved relative to PRISM_ROOT if not already
    absolute, same convention as generate_analysis.py's --sample-json. Errors
    out if any id in the file isn't among image_paths, rather than silently
    processing fewer images than requested."""
    p = Path(ids_file)
    if not p.is_absolute():
        p = PRISM_ROOT / p
    wanted = set(json.loads(p.read_text()))
    by_id = {int(path.stem): path for path in image_paths}
    missing = wanted - by_id.keys()
    if missing:
        raise SystemExit(f"--image-ids-file {p}: {len(missing)} requested id(s) not found among this "
                          f"dataset's images, e.g. {sorted(missing)[:5]}")
    return [by_id[i] for i in sorted(wanted)]


def rq_results_dir(*subdirs: str) -> Path:
    """results/<subdirs...> -- the one place any RQ-graph-generating script
    should write to (see RQ_RESULTS_DIR above for why this is separate from
    dataset_results_dir/OUTPUT_DIR). Creates the directory on demand, same
    convention as dataset_results_dir."""
    d = RQ_RESULTS_DIR.joinpath(*subdirs)
    d.mkdir(parents=True, exist_ok=True)
    return d
