#!/usr/bin/env python3
"""
RQ2 Stage-2 aggregation: turn the per-folder open-coding CSVs (produced by
rq2_claude.py) into a single per-feature importance ranking.

For every (image_folder -> feature_list) entry in the mapping JSON, and for
each feature in that folder:

    value          = count(is_present=1 AND has_attention=1) / count(is_present=1)
    presence_ratio = count(is_present=1) / count(total images)

`value` is the fraction of the times a feature actually appears in a scene
that the model's attention also lands on it -- higher means more important.
Features within each folder are sorted descending by `value`.

Only fully-coded rows (no blank fields, i.e. not a failed image from
rq2_claude.py) are counted; a fully-failed image is excluded from both the
numerator and the denominator rather than silently treated as "absent".

Run from the PRISM root, after rq2_claude.py has produced results/rq2/*.csv:
    python src/rq2.py
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_claude import csv_name_for_folder  # noqa: E402

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

MAPPING_PATH = "results/rq2_75_img_fet_mapping.json"
CODED_CSV_DIR = "results/rq2"
OUTPUT_JSON_PATH = "results/rq2.json"

ROUND_DIGITS = 4


def load_coded_rows(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    return header, rows


def row_is_complete(row):
    return len(row) > 1 and all(value != "" for value in row[1:])


def compute_feature_stats(csv_path, feature_list):
    header, rows = load_coded_rows(csv_path)

    expected_header_len = 1 + 2 * len(feature_list)
    if len(header) != expected_header_len:
        raise ValueError(
            f"{csv_path}: header has {len(header)} columns, expected "
            f"{expected_header_len} for {len(feature_list)} features -- "
            "does this CSV match the current mapping JSON?"
        )

    complete_rows = [row for row in rows if row_is_complete(row)]
    skipped = len(rows) - len(complete_rows)
    if skipped:
        print(f"    ! {csv_path}: skipping {skipped} incomplete/failed row(s) "
              f"(rerun rq2_claude.py to fill them in)")

    total_images = len(complete_rows)
    stats = []
    for idx, feature in enumerate(feature_list):
        present_col = 1 + 2 * idx
        attention_col = 2 + 2 * idx

        present_count = 0
        present_and_attention_count = 0
        for row in complete_rows:
            if row[present_col] == "1":
                present_count += 1
                if row[attention_col] == "1":
                    present_and_attention_count += 1

        value = (present_and_attention_count / present_count) if present_count else 0.0
        presence_ratio = (present_count / total_images) if total_images else 0.0

        stats.append({
            "feature": feature,
            "attention_rate": round(value, ROUND_DIGITS),
            "presence_ratio": round(presence_ratio, ROUND_DIGITS),
        })

    stats.sort(key=lambda entry: entry["attention_rate"], reverse=True)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate RQ2 open-coding CSVs into a feature ranking")
    parser.add_argument("--mapping-path", default=MAPPING_PATH)
    parser.add_argument("--coded-csv-dir", default=CODED_CSV_DIR)
    parser.add_argument("--output-path", default=OUTPUT_JSON_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.mapping_path, "r") as f:
        mapping = json.load(f)

    result = {}
    for folder_key, feature_list in mapping.items():
        csv_path = os.path.join(args.coded_csv_dir, csv_name_for_folder(folder_key))
        print(f"=== {folder_key} -> {csv_path} ===")
        if not os.path.exists(csv_path):
            print(f"    ! missing {csv_path} -- run rq2_claude.py first, skipping")
            continue
        result[folder_key] = compute_feature_stats(csv_path, feature_list)

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {args.output_path} ({len(result)} folders)")


if __name__ == "__main__":
    main()
