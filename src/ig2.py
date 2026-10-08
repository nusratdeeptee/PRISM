"""Computes IG2 (Integrated Gradients on an Iterative Gradient Path) attention
for every image in a dataset, one attribution map per image, saved to disk.
Config-driven: works on any dataset with a config/datasets/<id>.json entry
and an existing lc_type, no code change needed for a new dataset.

Usage:
    python ig2.py --dataset S2_300 [--limit-images 5]
    python ig2.py --dataset waterbird_test --image-ids-file output/waterbird_test/finetuned_classifier/samples/sample_1000.json
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths
from common.lc_models import ensure_metadata_csv, load_lc_for_ig2, get_target_class, image_to_chw_float
from third_party.ig2_algorithm import IG2, REP_LAYER_VALUES, REP_DISTANCE_GRADIENTS
from saliency.core.base import INPUT_OUTPUT_GRADIENTS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_call_model_function(model, normalize_fn, rep_module, extract_scalar_fn, device):
    """IG2 expects raw 0-255 pixel arrays in and out; normalization happens
    here, inside the call, not before -- IG2's own path search perturbs
    pixels directly in that raw space."""

    def to_tensor(x):
        return torch.tensor(np.asarray(x, dtype=np.float32), device=device)

    def call_model_function(images_np, call_model_args=None, expected_keys=None):
        call_model_args = call_model_args or {}
        model.eval()

        if REP_LAYER_VALUES in expected_keys:
            x = normalize_fn(to_tensor(images_np))
            buf = []
            h = rep_module.register_forward_hook(lambda m, i, o: buf.append(o))
            with torch.no_grad():
                model(x)
            h.remove()
            return {REP_LAYER_VALUES: buf[0]}

        if REP_DISTANCE_GRADIENTS in expected_keys:
            # BUG (fixed 2026-09-22): must differentiate w.r.t. the RAW pixel
            # tensor (x_raw, same convention as the INPUT_OUTPUT_GRADIENTS
            # branch below), not w.r.t. `x` = normalize_fn(x_raw) -- when
            # normalize_fn resizes/crops (every lc_type except sgsm_resnet34,
            # whose normalize_fn is a pure elementwise (x/255-mean)/std with
            # no spatial op), autograd.grad(loss, [x]) returns a gradient in
            # the RESIZED/CROPPED tensor's own space, not raw-pixel space --
            # for finetuned_classifier (waterbird, resize 256->crop 224) that
            # literally changes shape and crashes Get_GradPath's
            # `delta = delta + grad * step_size` (delta lives in raw-pixel
            # space). For imagenet_classifier (mimicry/biggan/ep_dnn,
            # RESIZE_SHORT_SIDE=256 -> CROP_SIZE=224 on already-224x224
            # images) the shapes coincidentally both end up 224x224, so it
            # never crashed -- but the gradient values were still wrong
            # (evaluated at a resized+cropped tensor, not the raw pixels
            # Get_GradPath's path search perturbs), silently corrupting the
            # counterfactual path search for every non-SGSM dataset.
            x_raw = to_tensor(images_np).requires_grad_(True)
            x = normalize_fn(x_raw)
            buf = []
            h = rep_module.register_forward_hook(lambda m, i, o: buf.append(o))
            model(x)
            h.remove()
            loss = -F.mse_loss(buf[0], call_model_args["layer_baseline"].to(device))
            (grads,) = torch.autograd.grad(loss, [x_raw])
            return {REP_DISTANCE_GRADIENTS: grads.detach().cpu().numpy(), "loss": loss.item()}

        if INPUT_OUTPUT_GRADIENTS in expected_keys:
            x_raw = to_tensor(images_np).requires_grad_(True)
            out = extract_scalar_fn(model(normalize_fn(x_raw)))
            (grads,) = torch.autograd.grad(out.sum(), [x_raw])
            return {INPUT_OUTPUT_GRADIENTS: grads.detach().cpu().numpy()}

        raise ValueError(f"Unexpected expected_keys: {expected_keys}")

    return call_model_function


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--image-ids-file", default=None,
                     help="restrict to exactly these image ids (JSON list of ints, PRISM-relative or "
                          "absolute) -- e.g. select_sample_ids.py's output. Mutually exclusive with "
                          "--limit-images.")
    ap.add_argument("--out-dir", default=None, help="override the default results/<dataset>/<model>/ig2 layout")
    args = ap.parse_args()
    if args.limit_images and args.image_ids_file:
        ap.error("--limit-images and --image-ids-file are mutually exclusive")

    attn_cfg = paths.load_attention_config()["ig2"]
    cfg = paths.load_dataset_config(args.dataset)
    out_dir = paths.dataset_results_dir(cfg, "ig2", override=args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_df = ensure_metadata_csv(cfg, cfg["images_dir"], cfg["metadata_csv"], device)
    image_paths = sorted(cfg["images_dir"].glob("*.png"), key=lambda p: int(p.stem))
    if args.image_ids_file:
        image_paths = paths.filter_by_ids_file(image_paths, args.image_ids_file)
    if args.limit_images:
        image_paths = image_paths[:args.limit_images]

    # Some datasets' source images aren't all one native resolution (e.g.
    # Waterbirds' original CUB photos) -- image_to_chw_float square-stretch
    # resizes to raw_image_size first so the reference stack below and each
    # per-image x share one shape. Datasets with fixed-size images (SGSM,
    # ImageNet crops) omit this key and get the old unresized behavior.
    raw_size = tuple(cfg["raw_image_size"]) if cfg.get("raw_image_size") else None

    # References for the gradient-descent path search: sampled from PASSING
    # images only (falls back to all images if fewer than 2 pass) -- IG2
    # searches for a counterfactual near a "typical passing" representation.
    # image_id MUST come from the filename (path_by_id), not an enumerate()/
    # range() index -- those only happen to equal the real id for a
    # contiguous 0..N-1 prefix (the full dataset, or --limit-images); once
    # --image-ids-file takes an arbitrary non-contiguous subset they diverge,
    # which would silently look up the wrong labels/target classes and write
    # output under the wrong filenames (see idg.py's identical warning about
    # sharding, same underlying hazard).
    path_by_id = {int(p.stem): p for p in image_paths}
    all_ids = sorted(path_by_id)
    label_by_id = meta_df.set_index("image_id")["label"].to_dict()
    pass_ids = [i for i in all_ids if label_by_id.get(i) == "pass"]
    ref_pool = pass_ids if len(pass_ids) >= 2 else all_ids
    random.seed(42)
    ref_ids = random.sample(ref_pool, min(attn_cfg["n_references"], len(ref_pool)))
    references = np.stack([image_to_chw_float(path_by_id[i], raw_size) for i in ref_ids], axis=0)

    done_ids = {int(p.stem.replace("_attr", "")) for p in out_dir.glob("*_attr.npy")}
    model, normalize_fn, rep_module, extract_scalar_fn_factory = load_lc_for_ig2(cfg, device)
    explainer = IG2()
    t0 = time.time()
    for path in image_paths:
        image_id = int(path.stem)
        if image_id in done_ids:
            continue
        target_class = get_target_class(cfg, image_id, cfg["metadata_csv"])
        extract_scalar_fn = extract_scalar_fn_factory(target_class)
        call_model_fn = make_call_model_function(model, normalize_fn, rep_module, extract_scalar_fn, device)
        x = image_to_chw_float(path, raw_size)
        attr = explainer.GetMask(x, references, call_model_fn, call_model_args={},
                                  steps=attn_cfg["steps"], step_size=attn_cfg["step_size"],
                                  clip_min_max=[0, 255], return_path=False)
        np.save(out_dir / f"{image_id}_attr.npy", attr.astype(np.float32))
        log(f"[{image_id}] OK -- target_class={target_class}")
    log(f"done in {time.time() - t0:.1f}s ({len(image_paths)} images)")


if __name__ == "__main__":
    main()
