#!/usr/bin/env python3
"""
Build a paper_data-style feature panel: one row of original images and one row
of attention-overlay images for a chosen (dataset, attention method, feature)
combination, matching the format already in paper_data/<dataset>_panels/
(see paper_data/readme.md).

Images are selected from output/<dataset>/<model_dir>/report_<attn>/
label_membership.json's `by_label` entries, ranked by attention mass_share.
A "feature" can merge multiple canonical labels -- PRISM's label canonicalizer
sometimes keeps near-synonyms as separate canonical labels (e.g. "road" and
"road pavement" for SGSM), so pass all of them via --labels to pool their
images into one feature bucket.

Usage:
    python src/make_paper_panel.py \
        --dataset S2 --model-dir sgsm_resnet34 --attention idg \
        --labels "road,road pavement" --pass-fail fail --n 5 \
        --feature-name road --out-dir paper_data/s2_panels
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common.visualization import overlay_heatmap_on_image, normalize_saliency  # noqa: E402

PRISM_ROOT = Path(__file__).resolve().parent.parent


def load_membership(dataset, model_dir, attention):
    path = PRISM_ROOT / "output" / dataset / model_dir / f"report_{attention}" / "label_membership.json"
    return json.loads(path.read_text())


def select_images(membership, labels, pass_fail, n):
    """Pool by_label entries across `labels`, keep each image's best (highest
    mass_share) match, and return the top-n sorted by mass_share descending."""
    by_label = membership["by_label"]
    best = {}
    for label in labels:
        for entry in by_label.get(label, []):
            if pass_fail is not None and entry["pass_fail"] != pass_fail:
                continue
            iid = entry["image_id"]
            if iid not in best or entry["mass_share"] > best[iid]["mass_share"]:
                candidate = dict(entry)
                candidate["matched_canonical_label"] = label
                best[iid] = candidate
    return sorted(best.values(), key=lambda e: -e["mass_share"])[:n]


def attr_path_for(dataset, model_dir, attention, image_id):
    return PRISM_ROOT / "output" / dataset / model_dir / attention / f"{image_id}_attr.npy"


def build_row(images):
    """Concatenate same-height thumbnails left-to-right with no gaps, no
    margin -- matches the existing paper_data panel style."""
    w = sum(im.width for im in images)
    h = images[0].height
    row = Image.new("RGB", (w, h))
    x = 0
    for im in images:
        row.paste(im, (x, 0))
        x += im.width
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="e.g. S2")
    ap.add_argument("--model-dir", required=True, help="e.g. sgsm_resnet34")
    ap.add_argument("--attention", required=True, choices=["idg", "ig2"])
    ap.add_argument("--labels", required=True,
                     help="comma-separated canonical labels to pool into one feature")
    ap.add_argument("--pass-fail", default="fail", choices=["pass", "fail", "all"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--feature-name", required=True, help="output file prefix, e.g. 'road'")
    ap.add_argument("--out-dir", required=True, help="relative to the PRISM root, e.g. paper_data/s2_panels")
    args = ap.parse_args()

    labels = [l.strip() for l in args.labels.split(",")]
    pass_fail = None if args.pass_fail == "all" else args.pass_fail

    membership = load_membership(args.dataset, args.model_dir, args.attention)
    chosen = select_images(membership, labels, pass_fail, args.n)
    if not chosen:
        sys.exit(f"No images found for labels={labels} pass_fail={args.pass_fail}")
    if len(chosen) < args.n:
        print(f"! only {len(chosen)}/{args.n} matching images available -- using all of them")

    out_dir = PRISM_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    originals, overlays = [], []
    for entry in chosen:
        image_id = entry["image_id"]
        pil_image = Image.open(entry["image_path"]).convert("RGB")
        attr = np.load(attr_path_for(args.dataset, args.model_dir, args.attention, image_id))
        saliency = np.abs(attr).sum(axis=0)
        if saliency.shape != pil_image.size[::-1]:
            saliency = np.array(Image.fromarray(saliency.astype(np.float32), mode="F")
                                 .resize(pil_image.size, Image.BILINEAR))
        overlay = Image.fromarray(overlay_heatmap_on_image(np.array(pil_image), normalize_saliency(saliency)))
        originals.append(pil_image)
        overlays.append(overlay)
        print(f"  image {image_id}: matched '{entry['matched_canonical_label']}', "
              f"mass_share={entry['mass_share']:.4f}, pass_fail={entry['pass_fail']}")

    original_path = out_dir / f"{args.feature_name}_original.png"
    attn_path = out_dir / f"{args.feature_name}_{args.attention}.png"
    csv_path = out_dir / f"{args.feature_name}_images.csv"

    build_row(originals).save(original_path)
    build_row(overlays).save(attn_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "image_id", "pass_fail",
                          f"{args.attention}_matched_label", f"{args.attention}_mass_share"])
        for i, entry in enumerate(chosen, start=1):
            writer.writerow([i, entry["image_id"], entry["pass_fail"],
                              entry["matched_canonical_label"], round(entry["mass_share"], 4)])

    print(f"\nWrote {original_path}")
    print(f"Wrote {attn_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
