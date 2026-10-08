"""Computes IDG (Integrated Decision Gradients) attention for every image in
a dataset, one attribution map per image, saved to disk. Config-driven, same
role as ig2.py -- the only real difference is the attribution algorithm
itself; IDG needs no separate precompute pass (no reference sampling), it
attributes fresh per image against a fixed zero baseline.

Usage:
    python idg.py --dataset S2_300 [--limit-images 5]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths
from common.lc_models import load_wrapped_lc, load_input_tensor
from third_party.idg_algorithm import IDG as idg_attribute


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--image-ids-file", default=None,
                     help="restrict to exactly these image ids (JSON list of ints, PRISM-relative or "
                          "absolute) -- e.g. select_sample_ids.py's output. Mutually exclusive with "
                          "--limit-images; applied before sharding.")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--out-dir", default=None, help="override the default results/<dataset>/<model>/idg layout")
    args = ap.parse_args()
    if args.limit_images and args.image_ids_file:
        ap.error("--limit-images and --image-ids-file are mutually exclusive")

    attn_cfg = paths.load_attention_config()["idg"]
    cfg = paths.load_dataset_config(args.dataset)
    out_dir = paths.dataset_results_dir(cfg, "idg", override=args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # No metadata.csv dependency here -- both target_class_sources this
    # script's configs actually use ("fixed:N", or "metadata:COLUMN" on a
    # dataset that already ships its own metadata.csv) need nothing IDG
    # itself would have to compute; ig2.py/generate_analysis.py are the
    # scripts that create/require it.
    image_paths = sorted(cfg["images_dir"].glob("*.png"), key=lambda p: int(p.stem))
    if args.image_ids_file:
        image_paths = paths.filter_by_ids_file(image_paths, args.image_ids_file)
    image_paths = paths.shard(image_paths, args.shard_index, args.num_shards)
    if args.limit_images:
        image_paths = image_paths[:args.limit_images]

    done_ids = {int(p.stem.replace("_attr", "")) for p in out_dir.glob("*_attr.npy")}
    t0 = time.time()
    for path in image_paths:
        # image_id MUST come from the filename, not an enumerate() index --
        # once sharding takes a middle slice of image_paths, enumerate()
        # would restart at 0 per shard, colliding output filenames and
        # target_class lookups across shards.
        image_id = int(path.stem)
        if image_id in done_ids:
            continue
        wrapped_lc, target_class = load_wrapped_lc(cfg, image_id, cfg["metadata_csv"], device)
        x = load_input_tensor(cfg, path, device)
        # baseline=0, matching the official IDG repo's own ImageNet-eval
        # convention (evalOnImageNet.py). target_class=0 here indexes into
        # the wrapper's own single output column, not the LC's raw class
        # space -- the wrapper already selected the right class/head.
        grads = idg_attribute(x, wrapped_lc, attn_cfg["steps"], attn_cfg["batch_size"], 0, device, 0)
        attr = grads.detach().cpu().numpy()
        np.save(out_dir / f"{image_id}_attr.npy", attr.astype(np.float32))
        del wrapped_lc
        torch.cuda.empty_cache()
        log(f"[{image_id}] OK -- target_class={target_class}")
    log(f"done in {time.time() - t0:.1f}s ({len(image_paths)} images)")


if __name__ == "__main__":
    main()
