"""Runs Qwen2.5-VL -> Grounding DINO -> SAM2 once per image to discover and
mask every visible object. Independent of attention method (IG2/IDG), so its
output is reused by both -- this is the one pass over Qwen/DINO/SAM2 the
whole pipeline needs, not one per attention method.

Usage:
    python generate_segments.py --dataset S2_300 [--limit-images 5]
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths, qwen_dino_sam as qds
from common.drawing import draw_boxes, draw_masks


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def process_image(image_id: int, image_path: Path, models: dict, cfg: dict, out_dir: Path) -> None:
    pil_image = Image.open(image_path).convert("RGB")
    tags, raw_qwen = qds.generate_qwen_objects(models["qwen_model"], models["qwen_proc"], models["device"],
                                                pil_image, cfg["max_tags"])
    if not tags:
        log(f"[{image_id}] Qwen returned zero parseable objects -- skipping detection")
        return

    detections = qds.detect_objects(models["dino_model"], models["dino_proc"], models["device"], pil_image,
                                     tags, cfg["box_threshold"], cfg["text_threshold"])
    n_ungrounded = qds.count_ungrounded(tags, detections)
    masks = qds.masks_for_boxes(models["sam_model"], models["sam_proc"], models["device"], pil_image,
                                 [d["box"] for d in detections])
    objects = [{**d, "mask": m} for d, m in zip(detections, masks)]

    np.savez_compressed(out_dir / f"{image_id}_masks.npz",
                         masks=np.stack([o["mask"] for o in objects]) if objects
                         else np.zeros((0, *pil_image.size[::-1]), dtype=bool))
    import json
    json.dump({"image_id": image_id, "qwen_tags": tags, "qwen_raw": raw_qwen,
               "n_ungrounded_tags": n_ungrounded,
               "detections": [{"label": d["label"], "score": d["score"], "box": d["box"]} for d in detections]},
              open(out_dir / f"{image_id}_segments.json", "w"), indent=2)

    draw_boxes(pil_image, objects).save(out_dir / f"{image_id}_boxes.png")
    draw_masks(pil_image, objects).save(out_dir / f"{image_id}_masks.png")
    log(f"[{image_id}] OK -- {len(tags)} Qwen object(s) ({n_ungrounded} ungrounded), "
        f"{len(detections)} DINO detection(s)")


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
    ap.add_argument("--out-dir", default=None, help="override the default results/<dataset>/<model>/segments layout")
    args = ap.parse_args()
    if args.limit_images and args.image_ids_file:
        ap.error("--limit-images and --image-ids-file are mutually exclusive")

    attn_cfg = paths.load_attention_config()
    cfg = paths.load_dataset_config(args.dataset)
    out_dir = paths.dataset_results_dir(cfg, "segments", override=args.out_dir)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # No metadata.csv (pass/fail labels) dependency here -- segmentation is
    # attention- and label-independent, so this can run fully in parallel
    # with ig2.py/idg.py, which resolve labels themselves if/when they need them.
    image_paths = sorted(cfg["images_dir"].glob("*.png"), key=lambda p: int(p.stem))
    if args.image_ids_file:
        image_paths = paths.filter_by_ids_file(image_paths, args.image_ids_file)
    image_paths = paths.shard(image_paths, args.shard_index, args.num_shards)
    if args.limit_images:
        image_paths = image_paths[:args.limit_images]
    done_ids = {int(p.stem.replace("_segments", "")) for p in out_dir.glob("*_segments.json")}

    qwen_model, qwen_proc, _ = qds.load_qwen(attn_cfg["qwen_model"])
    dino_model, dino_proc = qds.load_grounding_dino(attn_cfg["grounding_dino_model"], device)
    sam_model, sam_proc = qds.load_sam2(attn_cfg["sam2_model"], device)
    models = {"qwen_model": qwen_model, "qwen_proc": qwen_proc, "dino_model": dino_model,
              "dino_proc": dino_proc, "sam_model": sam_model, "sam_proc": sam_proc, "device": device}
    box_cfg = {"box_threshold": attn_cfg["box_threshold"], "text_threshold": attn_cfg["text_threshold"],
               "max_tags": attn_cfg["max_tags"]}

    t0 = time.time()
    for path in image_paths:
        image_id = int(path.stem)
        if image_id in done_ids:
            continue
        try:
            process_image(image_id, path, models, box_cfg, out_dir)
        except Exception:
            log(f"[{image_id}] ERROR:\n{traceback.format_exc()}")
    log(f"done in {time.time() - t0:.1f}s ({len(image_paths)} images)")


if __name__ == "__main__":
    main()
