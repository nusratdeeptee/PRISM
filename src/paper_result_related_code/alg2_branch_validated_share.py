"""Per-family, per-attention, per-branch breakdown of Alg. 2's branch choice and
validated share.

The paper's Threats to Validity section (Internal validity) flags that the validated
share reported in Table 1 mixes two statistically different procedures -- Alg. 2's
balanced branch (FDR-controlled ContrastTest, Line 4) and its imbalanced branch
(effect-size-thresholded EnrichmentTest, Line 9) -- selected per Line 3's
Balanced({y_j}, tau) check, and promises the share reported per branch. This script
fills that in, reusing rq1.py's already-computed per-run numbers rather than
recomputing them:

    results/rq1_feature_counts.json has, for every (dataset, attention, N) triple
    PRISM's RQ1 sweep covers, that run's |L| ("total"), |V| ("validated"), and which
    of Alg. 2's two branches it took ("branch": "balanced"/"imbalanced") -- a list of
    5 independent-draw values for N < 1000, a single value for N = 1000 (one draw,
    the full pool).

Table 1 (and the first version of this script) pools IDG and IG2 together. That hides
that Alg. 2's branch choice is a property of a run's pass/fail label counts only (Line
3's Balanced({y_j}, tau)), not of the attribution method -- so IDG and IG2 branch
census numbers happen to agree in the current outputs, but the two attribution methods
produce DIFFERENT |L|/|V| per run (different attribution maps -> different per-image
significance in Alg. 1 -> different discovered/validated feature sets in Alg. 2), so
their validated shares differ and should not be averaged together. This script
reports IDG and IG2 separately throughout: every family has two independent reports,
one per attention method.

This script does two things per (family, attention) -- families are SGSM = S1-S7,
Mimicry = mimicry_0-9, GenAI-TIG = biggan_ai_tigs_0-9, EP-DNN = ep_dnn_0-9,
Waterbirds_test = waterbird_test; attentions are idg and ig2:

  1. Branch census: how many of the family's conditions take each branch, under that
     attention method's own runs. Balanced() depends only on a run's own pass/fail
     label counts, not on the random N-subsample drawn, so in principle a condition
     could take different branches at different N or draws (rq1.py's docstring notes
     this explicitly) -- this script does not assume that invariant: for each
     (family, attention) it verifies the condition's branch is the same at every N and
     repeat, and prints a warning (without erroring) for any run where it isn't,
     rather than silently reporting a single number that hides disagreement. The
     reported per-condition branch label uses that condition's N=1000 (full pool,
     single draw) run under that attention method.

  2. Validated share (Eq. 16): ValidatedShare = sum_u |V_u| / sum_u |L_u|, pooling
     over runs u. Computed once per (family, attention, branch, N-level), where
     N-level is "100" (5 draws per condition), "1000" (1 draw per condition), or
     "All N" (every draw at every sampled size in {100,300,500,700,1000}, matching
     Table 1's "All N" column) -- the pooled runs u are restricted to that attention
     method's own runs whose OWN branch equals the target branch, so a condition that
     flipped branch at some N (if one ever does) contributes its runs to whichever
     branch each run actually took.

Usage:
    python alg2_branch_validated_share.py
    python alg2_branch_validated_share.py --counts-json /path/to/rq1_feature_counts.json
    python alg2_branch_validated_share.py --out results/alg2_branch_validated_share.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths

SAMPLE_SIZES = [100, 300, 500, 700, 1000]
ATTENTIONS = ["idg", "ig2"]
BRANCHES = ["balanced", "imbalanced"]

FAMILIES = {
    "SGSM": [f"S{i}" for i in range(1, 8)],
    "Mimicry": [f"mimicry_{i}" for i in range(10)],
    "GenAI-TIG": [f"biggan_ai_tigs_{i}" for i in range(10)],
    "EP-DNN": [f"ep_dnn_{i}" for i in range(10)],
    "Waterbirds_test": ["waterbird_test"],
}


def iter_runs(counts: dict, dataset_id: str, attention: str):
    """Yields (N, repeat_index, total, validated, branch) for every run recorded for
    dataset_id under the given attention method. repeat_index is None for the single
    N=1000 draw."""
    by_n = counts.get(dataset_id, {}).get(attention, {})
    for n in SAMPLE_SIZES:
        rec = by_n.get(str(n))
        if rec is None:
            continue
        if n == 1000:
            if rec["total"] is None:
                continue
            yield n, None, rec["total"], rec["validated"], rec["branch"]
        else:
            for i, (total, validated, branch) in enumerate(
                    zip(rec["total"], rec["validated"], rec["branch"])):
                if total is None:
                    continue
                yield n, i, total, validated, branch


def condition_branch(counts: dict, dataset_id: str, attention: str) -> str | None:
    """The condition's branch label under this attention method, taken from its
    N=1000 (full pool) draw. Returns None if that run is missing."""
    rec = counts.get(dataset_id, {}).get(attention, {}).get("1000")
    return rec["branch"] if rec else None


def check_branch_invariance(counts: dict, dataset_id: str, attention: str, label: str) -> list:
    """Returns a list of human-readable warnings for any run of dataset_id (under this
    attention method) whose branch differs from `label` (the condition's assigned
    branch) -- empty if every run agrees."""
    warnings = []
    for n, repeat, _total, _validated, branch in iter_runs(counts, dataset_id, attention):
        if branch is not None and branch != label:
            where = f"N={n}" if repeat is None else f"N={n} repeat={repeat}"
            warnings.append(
                f"{dataset_id}/{attention}: {where} took branch '{branch}', "
                f"differs from condition label '{label}' (from N=1000)")
    return warnings


def validated_share(runs: list) -> dict:
    """runs: list of (total, validated) pairs already filtered to one branch/N-level.
    Returns {"validated_share_pct", "numerator", "denominator", "n_runs"} -- share is
    None if there were no runs (empty denominator)."""
    total_sum = sum(t for t, _ in runs)
    validated_sum = sum(v for _, v in runs)
    share = (100.0 * validated_sum / total_sum) if total_sum > 0 else None
    return {
        "validated_share_pct": share,
        "numerator_validated": validated_sum,
        "denominator_total": total_sum,
        "n_runs": len(runs),
    }


def analyze_family(counts: dict, dataset_ids: list, attention: str) -> dict:
    branch_census = {b: [] for b in BRANCHES}
    unlabeled = []
    all_warnings = []

    runs_by_branch_and_level = {
        b: {"100": [], "1000": [], "All N": []} for b in BRANCHES
    }

    for dataset_id in dataset_ids:
        label = condition_branch(counts, dataset_id, attention)
        if label is None:
            unlabeled.append(dataset_id)
        else:
            branch_census[label].append(dataset_id)
            all_warnings.extend(check_branch_invariance(counts, dataset_id, attention, label))

        for n, _repeat, total, validated, branch in iter_runs(counts, dataset_id, attention):
            if branch not in BRANCHES:
                continue
            pair = (total, validated)
            runs_by_branch_and_level[branch]["All N"].append(pair)
            if n == 100:
                runs_by_branch_and_level[branch]["100"].append(pair)
            elif n == 1000:
                runs_by_branch_and_level[branch]["1000"].append(pair)

    validated_share_table = {
        branch: {level: validated_share(runs_by_branch_and_level[branch][level])
                 for level in ("100", "1000", "All N")}
        for branch in BRANCHES
    }

    return {
        "n_conditions": len(dataset_ids),
        "branch_census": {b: {"n_conditions": len(ids), "conditions": ids}
                           for b, ids in branch_census.items()},
        "unlabeled_conditions": unlabeled,
        "branch_invariance_warnings": all_warnings,
        "validated_share": validated_share_table,
    }


def print_report(results: dict) -> None:
    for family, by_attention in results.items():
        print(f"\n=== {family} ===")
        for attention in ATTENTIONS:
            info = by_attention[attention]
            print(f"  --- {attention.upper()} ({info['n_conditions']} conditions) ---")
            for b in BRANCHES:
                census = info["branch_census"][b]
                print(f"    {b}: {census['n_conditions']} condition(s) -> {census['conditions']}")
            if info["unlabeled_conditions"]:
                print(f"    UNLABELED (no N=1000 run found): {info['unlabeled_conditions']}")
            if info["branch_invariance_warnings"]:
                print(f"    WARNINGS ({len(info['branch_invariance_warnings'])}):")
                for w in info["branch_invariance_warnings"]:
                    print(f"      - {w}")

            print(f"    {'branch':<12}{'N=100':>12}{'N=1000':>12}{'All N':>12}")
            for b in BRANCHES:
                row = info["validated_share"][b]

                def fmt(level):
                    v = row[level]["validated_share_pct"]
                    return "n/a" if v is None else f"{v:.1f}%"

                print(f"    {b:<12}{fmt('100'):>12}{fmt('1000'):>12}{fmt('All N'):>12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-json", default=None,
                     help="path to rq1_feature_counts.json (default: results/rq1_feature_counts.json)")
    ap.add_argument("--out", default=None,
                     help="output json path, repo-relative or absolute "
                          "(default: results/alg2_branch_validated_share.json)")
    args = ap.parse_args()

    counts_path = Path(args.counts_json) if args.counts_json else paths.rq_results_dir() / "rq1_feature_counts.json"
    if not counts_path.is_absolute():
        counts_path = paths.PRISM_ROOT / counts_path
    counts = json.loads(counts_path.read_text())

    results = {
        family: {attention: analyze_family(counts, dataset_ids, attention)
                 for attention in ATTENTIONS}
        for family, dataset_ids in FAMILIES.items()
    }

    print_report(results)

    out_path = Path(args.out) if args.out else paths.rq_results_dir() / "alg2_branch_validated_share.json"
    if not out_path.is_absolute():
        out_path = paths.PRISM_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
