#!/usr/bin/env python3
"""
RQ2 attention-rate dot plot.

Produces a 2×2 panel figure (one panel per dataset).
Each panel: features on the y-axis (sorted by IDG attention rate descending),
attention rate on the x-axis [0, 1], with IDG (filled circle) and IG²
(open circle) as two series, connected by a horizontal line per feature.

Usage:
    python plot_rq2.py                                    # uses default path
    python plot_rq2.py --json path/to/attention_rate.json # custom path
    python plot_rq2.py --out rq2_plot.pdf                 # custom output file
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_JSON = "results/rq2.json"
DEFAULT_OUT  = "results/rq2.png"

# Display names for datasets (key -> panel title)
DATASET_LABELS = {
    "mimicry_5": "Mimicry-5",
    "s1":        "S1",
    "s2":        "S2",
    "waterbird": "Waterbird",
}

# Order panels in this sequence (2×2 grid, row-major)
DATASET_ORDER = ["s1", "s2", "waterbird", "mimicry_5"]

# rq2.py's folder-key dataset segment -> panel key used above
DATASET_ALIASES = {
    "waterbird_test": "waterbird",
}

# Visual style
IDG_COLOR    = "#2563EB"   # blue  – filled marker
IG2_COLOR    = "#DC2626"   # red   – open marker
LINE_COLOR   = "#9CA3AF"   # grey connecting line
MARKER_SIZE  = 7
LINE_WIDTH   = 1.0
GRID_ALPHA   = 0.3


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def normalise(raw: dict) -> dict:
    """
    Return a dict of shape:
        { dataset: { method: { feature: {"attention_rate": float, "presence_ratio": float} } } }

    Handles three JSON layouts:
      A) rq2.py's actual output -- flat "rq2_data/<dataset>/<method>/image_75" keys,
         each mapping to a LIST of {"feature", "attention_rate", "presence_ratio"} dicts
      B) flat "dataset/method" keys -> { "road": {"attention_rate": ..., ...}, ... }
      C) nested "dataset": { "idg": { "road": {...}, ... }, "ig2": {...} }
    """
    first_key = next(iter(raw))
    first_val = raw[first_key]

    # Layout A: folder-key -> list of per-feature dicts
    if isinstance(first_val, list):
        out = {}
        for key, entries in raw.items():
            parts = key.split("/")
            if len(parts) < 2:
                continue
            dataset, method = parts[-3], parts[-2]
            dataset = DATASET_ALIASES.get(dataset, dataset)
            out.setdefault(dataset, {})[method] = {
                entry["feature"]: {
                    "attention_rate": entry["attention_rate"],
                    "presence_ratio": entry.get("presence_ratio"),
                }
                for entry in entries
            }
        return out

    # Layout B: flat "dataset/method" keys
    if "/" in first_key:
        out = {}
        for key, features in raw.items():
            parts = key.rsplit("/", 1)
            if len(parts) != 2:
                continue
            dataset, method = parts
            dataset = DATASET_ALIASES.get(dataset, dataset)
            out.setdefault(dataset, {})[method] = features
        return out

    # Layout C: already nested
    return raw


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel(ax, dataset: str, methods: dict, title: str):
    """
    Draw one panel.

    methods = { "idg": { feature: {"attention_rate": float, ...} },
                "ig2": { feature: {"attention_rate": float, ...} } }
    """
    idg_data = methods.get("idg", {})
    ig2_data = methods.get("ig2", {})

    # Union of features from both methods
    all_features = sorted(
        set(idg_data) | set(ig2_data),
        key=lambda f: idg_data.get(f, {}).get("attention_rate", ig2_data.get(f, {}).get("attention_rate", 0)),
        reverse=False,   # ascending so highest is at the top in the horizontal plot
    )

    y_pos = np.arange(len(all_features))

    for i, feature in enumerate(all_features):
        idg_val = idg_data.get(feature, {}).get("attention_rate", None)
        ig2_val = ig2_data.get(feature, {}).get("attention_rate", None)

        # Connecting line
        vals = [v for v in (idg_val, ig2_val) if v is not None]
        if len(vals) == 2:
            ax.hlines(i, min(vals), max(vals), colors=LINE_COLOR,
                      linewidth=LINE_WIDTH, zorder=1)

        # IDG dot (filled)
        if idg_val is not None:
            ax.scatter(idg_val, i, color=IDG_COLOR, s=MARKER_SIZE**2,
                       zorder=3, marker="o")

        # IG² dot (open / hollow)
        if ig2_val is not None:
            ax.scatter(ig2_val, i, color=IG2_COLOR, s=MARKER_SIZE**2,
                       zorder=3, marker="o",
                       facecolors="none", edgecolors=IG2_COLOR, linewidths=1.5)

    # Axes formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_features, fontsize=8, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Attention rate", fontsize=9, fontweight="bold")
    # Panel title (dataset name) is sized a point above the axis labels/ticks
    # so it reads as a distinct heading rather than blending into them.
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.xaxis.grid(True, alpha=GRID_ALPHA, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bold the x-axis tick numbers too (yticklabels/xlabel/title set above
    # already carry their own bold weight).
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    # Reference line at 0.5
    ax.axvline(0.5, color="#6B7280", linewidth=0.8, linestyle=":", zorder=0)


def build_figure(data: dict, out_path: str):
    datasets = [d for d in DATASET_ORDER if d in data]
    if not datasets:
        sys.exit("No recognised dataset keys found. "
                 f"Expected one of: {DATASET_ORDER}. Got: {list(data)}")

    n = len(datasets)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 2.8 * nrows),
                             constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    for idx, dataset in enumerate(datasets):
        plot_panel(axes_flat[idx], dataset, data[dataset],
                   DATASET_LABELS.get(dataset, dataset))

    # Hide any unused panel slots
    for idx in range(len(datasets), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend
    legend_handles = [
        mlines.Line2D([], [], color=IDG_COLOR, marker="o",
                      markersize=MARKER_SIZE, linestyle="None", label="IDG"),
        mlines.Line2D([], [], color=IG2_COLOR, marker="o",
                      markersize=MARKER_SIZE, linestyle="None",
                      fillstyle="none", label="IG²"),
    ]
    # Legend labels (IDG / IG²) are sized a point above the axis labels/ticks,
    # matching the panel titles, so both stand out from the smaller axis text.
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, prop={"size": 10, "weight": "bold"}, frameon=False,
               bbox_to_anchor=(0.5, -0.03))

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RQ2 attention-rate dot plot")
    parser.add_argument("--json", default=DEFAULT_JSON,
                        help=f"Path to attention_rate.json (default: {DEFAULT_JSON})")
    parser.add_argument("--out",  default=DEFAULT_OUT,
                        help=f"Output file path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    raw  = load_json(args.json)
    data = normalise(raw)
    build_figure(data, args.out)


if __name__ == "__main__":
    main()
