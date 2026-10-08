"""RQ1 feature-count aggregation: for every (dataset, attention method, sample size), reads
the already-generated report_{attention}[_sample_{n}[_repeat{2,3,4,5}]] directory and records
two numbers -- "total" (every distinct label PRISM's analysis reports for that run, regardless
of significance) and "validated" (how many of those survive the significance bar) -- plus which
analysis branch that run used ("balanced" or "imbalanced", see below) -- into one JSON file:
results/rq1_feature_counts.json.

Two report formats exist, auto-detected per report directory (never assumed from the dataset's
category -- the same dataset can land in either branch at different sample sizes, since the
branch choice depends on that SPECIFIC run's own pass/fail split, e.g. S1's full 1000-image pool
is 38%/62% pass/fail (balanced branch) while S5's is 98.5%/1.5% (low-failure branch)):

  - Low-failure ("imbalanced") branch (report_dir/feature_importance.csv exists --
    generate_analysis.py writes this file only on this branch): "total" = number of rows (every
    label seen >= 3 times, no cutoff), "validated" = number of rows with importance
    (enrichment_diff_mean) > 0.1. Recorded as branch = "imbalanced".
  - Balanced pass/fail branch (no .csv; report_dir/feature_importance_full.md's three tables --
    Presence, Mass Share, Peak Ratio -- built from stats_tables.build_report(), which run on the
    SAME unfiltered label set): "total" = row count of any one of those three tables (identical
    label set across all three, verified at parse time). "validated" = number of DISTINCT labels
    where at least one of the three tables' own FDR p column is < 0.1 (a label can clear the bar
    on, say, Presence without clearing it on Mass Share -- this counts it once either way).
    Recorded as branch = "balanced".

Each (dataset, attention, sample size) entry gets a "branch" field alongside "total"/"validated",
same shape as those two (a list of 5 repeat-run branches for n < 1000, a single string for
n = 1000) -- None wherever that run's report_dir was missing/unparseable, same as total/validated
being None there. The branch is a property of that SPECIFIC run's own pass/fail split (see above),
not of the dataset as a whole, so two repeats of the same (dataset, n) can in principle land in
different branches if that draw's sampled pass/fail ratio happened to cross LOW_FAILURE_THRESHOLD.

Usage:
    python rq1.py                  # every dataset, both attentions, all 5 sample sizes
    python rq1.py --datasets S1,S2 # restrict to specific datasets (comma-separated)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths

ATTENTIONS = ["idg", "ig2"]
SAMPLE_SIZES = [100, 300, 500, 700, 1000]
REPEATS = [None, 2, 3, 4, 5]   # None = the original (seed 42) run, matching sample_<n>.json's own naming

IMPORTANCE_THRESHOLD = 0.1
FDR_ALPHA = 0.1

DATASETS = (
    [f"S{i}" for i in range(1, 8)]
    + [f"mimicry_{i}" for i in range(10)]
    + [f"biggan_ai_tigs_{i}" for i in range(10)]
    + [f"ep_dnn_{i}" for i in range(10)]
    + ["waterbird_test"]
)


def report_dir_name(attention: str, n: int, repeat: int | None) -> str:
    if n == 1000:
        return f"report_{attention}"
    tag = f"report_{attention}_sample_{n}"
    return tag if repeat is None else f"{tag}_repeat{repeat}"


def _split_md_table_row(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator_row(cells: list) -> bool:
    return all(re.fullmatch(r"-+", c) for c in cells)


def _parse_md_section_table(lines: list, start_idx: int) -> tuple:
    """From start_idx (a '## <Section>' heading line), finds the next pipe-table and returns
    (header_cells, data_rows, next_section_idx) -- data_rows is a list of cell-lists. Returns
    (None, [], next_section_idx) if the section has no table (label_df was empty for this run,
    e.g. "(no label met the threshold)")."""
    i = start_idx + 1
    n = len(lines)
    header = None
    rows = []
    while i < n and not lines[i].startswith("## "):
        line = lines[i].strip()
        if line.startswith("|"):
            cells = _split_md_table_row(line)
            if header is None:
                header = cells
            elif not _is_separator_row(cells):
                rows.append(cells)
        i += 1
    return header, rows, i


def count_from_full_md(full_md_path: Path) -> tuple:
    """Balanced-branch counting: parse feature_importance_full.md's Presence/Mass Share/Peak
    Ratio tables (stats_tables.build_report()'s output). Returns (total, validated).

    validated = count of DISTINCT labels where at least one of the three tables' own FDR p
    clears FDR_ALPHA -- a label significant on 2 or 3 of the tests still counts once, same as a
    label significant on only 1."""
    text = full_md_path.read_text()
    lines = text.splitlines()

    section_tables = {}   # "presence"/"mass_share"/"peak_ratio" -> {label: fdr_p or None}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## Presence"):
            _, rows, i = _parse_md_section_table(lines, i)
            section_tables["presence"] = {r[0]: (float(r[-1]) if r[-1] not in ("", "nan") else None)
                                           for r in rows}
            continue
        if line.startswith("## Mass Share"):
            _, rows, i = _parse_md_section_table(lines, i)
            section_tables["mass_share"] = {r[0]: (float(r[-1]) if r[-1] not in ("", "nan") else None)
                                             for r in rows}
            continue
        if line.startswith("## Peak Ratio"):
            _, rows, i = _parse_md_section_table(lines, i)
            section_tables["peak_ratio"] = {r[0]: (float(r[-1]) if r[-1] not in ("", "nan") else None)
                                             for r in rows}
            continue
        i += 1

    label_sets = [set(t) for t in section_tables.values() if t]
    if not label_sets:
        return 0, 0   # label_df was empty for this run -- no label cleared MIN_OCCURRENCES at all

    all_labels = label_sets[0]
    if any(s != all_labels for s in label_sets[1:]):
        # Should never happen (all three tables come from the same unfiltered label_df) -- fall
        # back to the union rather than silently under/over-counting if it ever does.
        all_labels = set().union(*label_sets)
    total = len(all_labels)

    validated = 0
    for label in all_labels:
        fdrs = [t.get(label) for t in section_tables.values()]
        if any(p is not None and p < FDR_ALPHA for p in fdrs):
            validated += 1
    return total, validated


def count_from_csv(csv_path: Path) -> tuple:
    """Low-failure-branch counting: every row of feature_importance.csv, importance >
    IMPORTANCE_THRESHOLD."""
    df = pd.read_csv(csv_path)
    total = len(df)
    validated = int((df["importance"] > IMPORTANCE_THRESHOLD).sum())
    return total, validated


def count_report_dir(report_dir: Path) -> tuple:
    """Returns (total, validated, branch) -- branch is "imbalanced" (low-failure branch,
    feature_importance.csv present), "balanced" (pass/fail contrast branch, parsed from
    feature_importance_full.md), or None if report_dir has neither (missing/unparseable, same
    condition that already made total/validated None)."""
    csv_path = report_dir / "feature_importance.csv"
    if csv_path.exists():
        total, validated = count_from_csv(csv_path)
        return total, validated, "imbalanced"
    full_md_path = report_dir / "feature_importance_full.md"
    if not full_md_path.exists():
        return None, None, None
    total, validated = count_from_full_md(full_md_path)
    return total, validated, "balanced"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None, help="comma-separated dataset ids (default: all 38)")
    ap.add_argument("--out", default=None, help="output json path, repo-relative or absolute "
                                                 "(default: results/rq1_feature_counts.json)")
    args = ap.parse_args()

    dataset_ids = args.datasets.split(",") if args.datasets else DATASETS

    result = {}
    for dataset_id in dataset_ids:
        cfg = paths.load_dataset_config(dataset_id)
        model_name = cfg["model_name"]
        dataset_root = paths.OUTPUT_DIR / dataset_id / model_name
        result[dataset_id] = {}

        for attention in ATTENTIONS:
            result[dataset_id][attention] = {}

            for n in SAMPLE_SIZES:
                if n == 1000:
                    d = dataset_root / report_dir_name(attention, n, None)
                    total, validated, branch = count_report_dir(d) if d.exists() else (None, None, None)
                    if total is None:
                        print(f"[{dataset_id}/{attention}/1000] MISSING: {d}", file=sys.stderr)
                    result[dataset_id][attention]["1000"] = {
                        "total": total, "validated": validated, "branch": branch}
                    continue

                totals, valids, branches = [], [], []
                for repeat in REPEATS:
                    d = dataset_root / report_dir_name(attention, n, repeat)
                    total, validated, branch = count_report_dir(d) if d.exists() else (None, None, None)
                    if total is None:
                        print(f"[{dataset_id}/{attention}/{n}/repeat={repeat}] MISSING: {d}", file=sys.stderr)
                    totals.append(total)
                    valids.append(validated)
                    branches.append(branch)
                result[dataset_id][attention][str(n)] = {
                    "total": totals, "validated": valids, "branch": branches}

        print(f"[{dataset_id}] done", flush=True)

    out_path = Path(args.out) if args.out else paths.rq_results_dir() / "rq1_feature_counts.json"
    if not out_path.is_absolute():
        out_path = paths.PRISM_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
