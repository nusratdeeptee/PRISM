"""Second analysis branch for datasets whose failure rate (or, symmetrically,
whose PASSING rate) is too low for the pass/fail contrast tests in
stats_tables.py to say anything (at 0%/100% failure every AUC/odds-ratio/
p-value in that module comes back None -- see label_stats_table()'s guards).
Whenever fail_rate or pass_rate drops below LOW_FAILURE_THRESHOLD,
generate_analysis switches to this module instead, which never needs a
balanced "pass" vs "fail" class split at all:

Each canonical label is scored against baselines derived from the images
themselves, not from a second class:
  1. Mass-share enrichment -- is this label's mean attribution mass_share
     bigger than its mean area_share (the share you'd expect if attention
     were spread evenly per pixel)? Tested per label with a one-sample
     Wilcoxon signed-rank test on (mass_share - area_share) across the
     images where the label occurs, + a bootstrap CI on the mean diff.
     This is the primary table -- it has a real, self-contained baseline
     (area), unlike a plain mean-mass_share ranking.
  2. Peak-ratio baseline check -- object_stats.py already defines peak_ratio
     as this label's own top-k saliency mean divided by the whole image's
     top-k mean, so its own no-signal value is 1.0 by construction (not
     area_share, since a single peak pixel isn't an area-proportional
     quantity). Tested the same way, against 1.0 instead of 0. Weaker/
     noisier than (1) for small-area objects (a small region's top-k mean
     has higher sampling variance) -- reported as a secondary/confirmatory
     column, not the primary ranking.
  3. Attribution-vs-continuous-output correlation -- even at 0% failure the
     model's own continuous output (whatever this dataset already stores:
     an LC's raw scalar output, or a class-margin) varies image to image.
     Spearman correlation between each label's mass_share (0 when absent)
     and that value, across every image. Self-skips (documented, not
     silent) for any dataset with no such column -- e.g. Waterbirds, which
     only stores a discrete predicted_class.
  4. Frequency x consistency -- a plain descriptive sort key (occurrence
     count x mean/std of mass_share), no test attached, used to break ties
     between similarly-significant labels.
  5. Bootstrap rank stability -- resamples images with replacement and
     recomputes the top-K enrichment ranking each time, reporting how often
     each of the point-estimate's top-K labels stays in the resampled
     top-K. Validates the RANKING as a whole (labels can each individually
     look significant yet reorder on resampling); dataset-agnostic, no
     external labels of any kind.

Nothing here is specific to any one dataset/model -- no hardcoded label
names, no reference to any dataset's own known group/confound labels.

Two reports come out of this module (2026-09-14, in response to the full
detail report being too dense to scan with 100+ labels), both built from
ONE shared label_df (compute_label_df(chosen_df, all_occurrence_df),
computed once by the caller so the two reports can't disagree):
  - build_feature_importance_report() -- ONE table, every label that clears
    MIN_OCCURRENCES (no top-K cap), five columns: rank, label, n_occurrences,
    importance (enrichment_diff_mean, the one effect-size metric used to
    rank), consistency (% of this label's occurrences that individually
    passed the per-image permutation test, "82% (41/50)" style). This is
    the "pick your own cutoff" table -- nothing to cross-reference.
  - build_low_failure_report() -- the same underlying stats, full detail
    (every raw number, both tests, CIs, correlation, rank stability) for
    anyone who wants to audit *why* a label ranked where it did.
consistency needs the TOTAL occurrence count (significant or not) per
label, which chosen_df alone can't give you -- chosen_df is already
pre-filtered to significant occurrences by generate_analysis.py, so
compute_label_df()/add_consistency() take a second, unfiltered
all_occurrence_df (one row per detection, significant or not).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from common.stats_tables import (BOOTSTRAP_SEED, MIN_OCCURRENCES, N_BOOTSTRAP, SIGNIFICANCE_ALPHA,
                                  _bh_fdr, df_to_markdown)

LOW_FAILURE_THRESHOLD = 0.05   # this module is used instead of stats_tables.build_report whenever
                                # fail_rate or pass_rate (1 - fail_rate) drops below this -- either
                                # class being this thin starves the pass-vs-fail contrast tests
TOP_K_SUMMARY = 20      # rows fed into the Key Findings prose section (full report)
TOP_K_STABILITY = 20    # rows checked for bootstrap rank stability
N_BOOTSTRAP_RANK = 500
RANK_STABILITY_CANDIDATE_POOL = 30   # label_df rows fed into the resampling matrix, well above TOP_K_STABILITY

# Columns checked, in order, for a usable continuous "how strong was this
# decision" signal -- generic names already used by multiple existing
# dataset configs (sgsm_resnet34's output_key, genai_tigs' class margin),
# not tied to any one dataset. Absent everywhere here -> section 3 self-skips.
CONFIDENCE_COLUMN_CANDIDATES = ("output_value", "acceleration_value")


def _fmt_p(p) -> str:
    """Display formatter for FDR-corrected p-values. Plain round(4) makes
    every strongly-significant label print as an identical "0.0" once
    sample sizes get large, which hides WHY the (still perfectly real,
    deterministic) sort order picked one label over another -- switches to
    scientific notation below 0.001 so the actual ordering is verifiable
    from the table itself instead of looking arbitrary."""
    if pd.isna(p):
        return ""
    if p == 0:
        return "0"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def dataset_fail_rate(all_images: dict) -> float:
    if not all_images:
        return 0.0
    return sum(1 for l in all_images.values() if l == "fail") / len(all_images)


def dataset_pass_rate(all_images: dict) -> float:
    """1 - dataset_fail_rate(), broken out as its own function (rather than
    inlined at call sites) so the symmetric low-failure/low-pass branch
    condition in generate_analysis.py reads the same way on both sides."""
    if not all_images:
        return 0.0
    return 1.0 - dataset_fail_rate(all_images)


def low_failure_branch_reason(fail_rate: float) -> str:
    """One-line description of which side of the pass/fail split is too thin
    to support the pass-vs-fail contrast branch (stats_tables.build_report)
    -- this branch originally only ever fired for a low fail_rate; it now
    also fires symmetrically for a low PASS rate (fail_rate above
    1 - LOW_FAILURE_THRESHOLD), so every hardcoded explanatory string below
    routes through this instead of assuming "failure rate is low"."""
    if fail_rate < LOW_FAILURE_THRESHOLD:
        return f"failure rate ({fail_rate:.1%}) is below the {LOW_FAILURE_THRESHOLD:.0%} threshold"
    return f"passing rate ({1 - fail_rate:.1%}) is below the {LOW_FAILURE_THRESHOLD:.0%} threshold"


def find_confidence_column(meta_df: pd.DataFrame):
    return next((c for c in CONFIDENCE_COLUMN_CANDIDATES
                 if c in meta_df.columns and pd.api.types.is_numeric_dtype(meta_df[c])), None)


def build_confidence_series(meta_df: pd.DataFrame, cfg: dict):
    """Returns (series indexed by image_id, human-readable description) or
    (None, None) if this dataset has no usable column. When the dataset also
    has a postcondition threshold (sgsm_resnet34-style configs), the raw
    value is converted to a distance-from-threshold -- closer to an actual
    confidence than the raw signed output. Otherwise the raw column is used
    as-is (already margin-like for e.g. genai_tigs_biggan_7)."""
    col = find_confidence_column(meta_df)
    if col is None:
        return None, None
    series = meta_df.drop_duplicates("image_id").set_index("image_id")[col].astype(float)
    postcondition = cfg.get("postcondition")
    if postcondition and "threshold" in postcondition:
        series = (series - postcondition["threshold"]).abs()
        desc = f"|{col} - {postcondition['threshold']}| (distance from this dataset's own decision threshold)"
    else:
        desc = f"{col} (this dataset's own continuous decision-relevant output value)"
    return series, desc


# --- one-sample helpers (generic: test any array's values against a center) -
def _wilcoxon_p(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2 or np.allclose(values, 0.0):
        return None
    try:
        _, p = wilcoxon(values)
        return float(p)
    except ValueError:
        return None


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _per_image_label_table(chosen_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """mass_share/area_share/peak_ratio per image for one canonical label,
    max-aggregated when a label appears via more than one detected object in
    the same image -- same independent-per-column max convention
    stats_tables.label_stats_table() already uses for mass_share/peak_ratio."""
    g = chosen_df[chosen_df["canonical_label"] == label]
    return g.groupby("image_id").agg(mass_share=("mass_share", "max"), area_share=("area_share", "max"),
                                      peak_ratio=("peak_ratio", "max"))


# --- Stage 1+2: mass-share enrichment + peak-ratio baseline -----------------
def label_enrichment_table(chosen_df: pd.DataFrame, min_occurrences: int = MIN_OCCURRENCES) -> pd.DataFrame:
    rows = []
    if len(chosen_df) == 0:
        return pd.DataFrame(rows)
    for label in chosen_df["canonical_label"].unique():
        per_img = _per_image_label_table(chosen_df, label)
        if len(per_img) < min_occurrences:
            # Threshold on distinct IMAGES, not raw chosen_df rows -- a
            # single image can hold multiple detected objects that
            # canonicalize to the same label (e.g. 3 separate boxes all
            # tagged "helicopter" in one image), which would otherwise
            # clear a row-count threshold while still being one real
            # per-image data point (every stat here is per-image).
            continue
        diff = (per_img["mass_share"] - per_img["area_share"]).to_numpy()
        peak = per_img["peak_ratio"].to_numpy()

        mean_mass, mean_area = float(per_img["mass_share"].mean()), float(per_img["area_share"].mean())
        mean_peak = float(np.nanmean(peak)) if len(peak) else None
        mass_lo, mass_hi = _bootstrap_mean_ci(diff)
        peak_lo, peak_hi = _bootstrap_mean_ci(peak)

        rows.append({
            "label": label, "n_occurrences": len(per_img),
            "mean_mass_share": mean_mass, "mean_area_share": mean_area,
            "enrichment_ratio": (mean_mass / mean_area) if mean_area > 0 else None,
            "enrichment_diff_mean": float(diff.mean()),
            "enrichment_diff_ci_lo": mass_lo, "enrichment_diff_ci_hi": mass_hi,
            "mass_share_pvalue": _wilcoxon_p(diff),
            "mean_peak_ratio": mean_peak,
            "peak_ratio_ci_lo": peak_lo, "peak_ratio_ci_hi": peak_hi,
            "peak_ratio_pvalue": _wilcoxon_p(peak - 1.0),
        })
    df = pd.DataFrame(rows)
    if len(df):
        # Pooled FDR across both test types, same "one jointly-reported
        # family" convention as stats_tables.label_stats_table().
        pooled_p = pd.concat([df["mass_share_pvalue"], df["peak_ratio_pvalue"]], ignore_index=True)
        pooled_fdr = _bh_fdr(pooled_p.to_numpy())
        n = len(df)
        df["mass_share_pvalue_fdr"] = pooled_fdr[:n]
        df["peak_ratio_pvalue_fdr"] = pooled_fdr[n:2 * n]
        df["peak_ratio_abs_effect"] = (df["mean_peak_ratio"] - 1.0).abs()
        df = sort_by_mass_share(df)
    return df


# Named, documented multi-key sort chains -- one per table, since "the
# leading metric" differs per table (mass_share's own significance test for
# the mass_share-led views, peak_ratio's own for the peak_ratio-led view).
# Every level after the first only breaks ties left by the level above it;
# spelling all of them out (rather than a single primary key) is what makes
# the resulting order verifiable from the table itself instead of looking
# arbitrary once many labels round to the same displayed p-value.
#
# IMPORTANT: the leading key in each chain is an EFFECT-SIZE metric (how big
# the departure from baseline is), never a p-value or n_occurrences.
# p-values measure how CONFIDENT we are that an effect is real, and for a
# fixed effect size a p-value shrinks as n_occurrences grows -- so sorting
# by p-value first silently reorders by sample size, not by importance to
# the model's decision. That's a real bug an earlier version of this report
# had (labels with more occurrences floated to the top regardless of effect
# size); direction/effect-size metrics come first here for exactly that
# reason. p-values only appear as a low-priority final tiebreak (in
# practice this almost never fires -- two labels essentially never share
# the exact same effect size to float precision).
MASS_SHARE_SORT_KEYS = ["enrichment_diff_mean", "enrichment_ratio", "mass_share_pvalue_fdr"]
MASS_SHARE_SORT_DIRS = [False, False, True]   # bigger enrichment = more important -> descending
MASS_SHARE_SORT_EXPLANATION = ("1) mass_share effect size `enrichment_diff_mean` (descending -- bigger "
                               "departure from the area-share baseline = more important, so biggest first), "
                               "2) `enrichment_ratio` (descending, same direction -- breaks ties left by (1)), "
                               "3) mass_share FDR p (ascending, lowest priority -- only breaks a remaining "
                               "exact tie in effect size, which is rare)")

PEAK_RATIO_SORT_KEYS = ["peak_ratio_abs_effect", "peak_ratio_pvalue_fdr"]
PEAK_RATIO_SORT_DIRS = [False, True]   # bigger departure from 1.0, either direction = more important -> descending
PEAK_RATIO_SORT_EXPLANATION = ("1) how far `mean_peak_ratio` sits from its own 1.0 baseline in EITHER "
                               "direction (descending -- peak_ratio is two-sided, so both far-above-1 and "
                               "far-below-1 are equally \"important,\" only the raw value's distance from 1.0 "
                               "matters, not its sign), 2) peak_ratio FDR p (ascending, lowest priority "
                               "tiebreak)")


def sort_by_mass_share(label_df: pd.DataFrame) -> pd.DataFrame:
    """The report's master ordering -- used by the Top Contributing Features
    leaderboard and the Mass-Share Enrichment table (mass_share is the
    PRIMARY test, see module docstring). See MASS_SHARE_SORT_EXPLANATION for
    the full tiebreak chain."""
    return label_df.sort_values(MASS_SHARE_SORT_KEYS, ascending=MASS_SHARE_SORT_DIRS)


def sort_by_peak_ratio(label_df: pd.DataFrame) -> pd.DataFrame:
    """The Peak-Ratio table's OWN ordering -- deliberately independent of
    sort_by_mass_share() above, since a reader of this table wants to know
    which labels' peak_ratio is most significant/most extreme, not which
    ones happen to lead on the other table's metric. See
    PEAK_RATIO_SORT_EXPLANATION for the full tiebreak chain."""
    return label_df.sort_values(PEAK_RATIO_SORT_KEYS, ascending=PEAK_RATIO_SORT_DIRS)


# --- Stage 3: attribution vs. continuous output correlation ----------------
def confidence_correlation_table(chosen_df: pd.DataFrame, all_image_ids: list, confidence_series,
                                  min_occurrences: int = MIN_OCCURRENCES):
    """Returns None if confidence_series is None (caller renders a "skipped"
    note) -- otherwise a DataFrame, one row per label that clears
    min_occurrences AND has enough non-null confidence coverage to test."""
    if confidence_series is None:
        return None
    rows = []
    conf = confidence_series.reindex(all_image_ids)
    for label, g in chosen_df.groupby("canonical_label"):
        per_img = g.groupby("image_id")["mass_share"].max()
        if len(per_img) < min_occurrences:   # distinct images, same reasoning as label_enrichment_table
            continue
        scores = pd.Series(0.0, index=all_image_ids)
        scores.loc[per_img.index] = per_img
        valid = conf.notna()
        if valid.sum() < 3 or scores[valid].nunique() <= 1:
            continue
        rho, p = spearmanr(scores[valid], conf[valid])
        rows.append({"label": label, "n_occurrences": len(per_img),
                      "spearman_rho": float(rho), "spearman_pvalue": float(p)})
    df = pd.DataFrame(rows)
    if len(df):
        df["spearman_pvalue_fdr"] = _bh_fdr(df["spearman_pvalue"].to_numpy())
        df["abs_spearman_rho"] = df["spearman_rho"].abs()
        # Effect-size metric (|rho|, how strong the relationship is) leads,
        # descending -- same reasoning as MASS_SHARE_SORT_KEYS above: a
        # p-value-first sort would silently reorder by sample size instead
        # of by strength of relationship. FDR p is a low-priority tiebreak.
        df = df.sort_values(["abs_spearman_rho", "spearman_pvalue_fdr"], ascending=[False, True])
        df = df.drop(columns="abs_spearman_rho")
    return df


# --- Stage 4: frequency x consistency (descriptive tiebreaker only) --------
def add_frequency_consistency(label_df: pd.DataFrame, chosen_df: pd.DataFrame) -> pd.DataFrame:
    if len(label_df) == 0:
        return label_df
    scores = {}
    for label in label_df["label"]:
        per_img = _per_image_label_table(chosen_df, label)["mass_share"]
        mean_, std_ = per_img.mean(), per_img.std(ddof=0)
        scores[label] = len(per_img) * mean_ / (std_ + 1e-6)
    label_df = label_df.copy()
    label_df["frequency_consistency_score"] = label_df["label"].map(scores)
    return label_df


# --- Stage 4b: consistency (needs the UNFILTERED occurrence count) ---------
def add_consistency(label_df: pd.DataFrame, all_occurrence_df: pd.DataFrame) -> pd.DataFrame:
    """label_df's own "n_occurrences" column is already the count of
    SIGNIFICANT occurrences (it's built from chosen_df, which
    generate_analysis.py already filtered to is_significant==True) -- that
    count is the CONSISTENCY NUMERATOR, not an independent total. The
    denominator (every time this label was detected, significant or not)
    has to come from a second, unfiltered table -- all_occurrence_df,
    one row per (image_id, canonical_label) for every detection, no
    is_significant filter, "UNCLAIMED" excluded (it isn't a real object).
    Adds n_total_occurrences and consistency_pct; a no-op (returns label_df
    unchanged) if all_occurrence_df is None, e.g. an older caller that
    hasn't been updated to supply it yet."""
    if len(label_df) == 0 or all_occurrence_df is None or len(all_occurrence_df) == 0:
        return label_df
    totals = all_occurrence_df.groupby("canonical_label")["image_id"].nunique()
    label_df = label_df.copy()
    # fillna guard: a label present in label_df should always be in totals
    # too (every significant occurrence is also a total occurrence), but
    # fall back to n_occurrences (100% consistency) rather than NaN/crash
    # if something upstream ever disagrees.
    label_df["n_total_occurrences"] = (label_df["label"].map(totals)
                                        .fillna(label_df["n_occurrences"]).astype(int))
    label_df["consistency_pct"] = label_df["n_occurrences"] / label_df["n_total_occurrences"]
    return label_df


def compute_label_df(chosen_df: pd.DataFrame, all_occurrence_df: pd.DataFrame = None) -> pd.DataFrame:
    """The one label_df both reports are built from: mass-share/peak-ratio
    enrichment (label_enrichment_table), the frequency x consistency
    tiebreak (add_frequency_consistency), and -- if all_occurrence_df is
    given -- the consistency-rate columns (add_consistency). Call this ONCE
    per dataset and pass the result to both build_feature_importance_report()
    and build_low_failure_report() -- each recomputing its own would risk
    the two reports silently disagreeing."""
    label_df = label_enrichment_table(chosen_df)
    label_df = add_frequency_consistency(label_df, chosen_df)
    label_df = add_consistency(label_df, all_occurrence_df)
    return label_df


# --- Primary report: feature importance ranking (all labels, no cap) -------
def sort_by_importance_then_consistency(label_df: pd.DataFrame) -> pd.DataFrame:
    """The feature-importance report's OWN ordering (2026-09-14, by request):
    importance (enrichment_diff_mean) leads, descending; consistency_pct
    breaks ties within the same importance value, also descending. This is
    deliberately independent of sort_by_mass_share()'s tiebreak chain
    (enrichment_ratio, then FDR p) -- the full-detail report's Mass-Share
    Enrichment table, Key Findings, and Rank Stability still use that one
    unchanged. Falls back to importance alone if consistency_pct isn't
    present (all_occurrence_df wasn't supplied to compute_label_df())."""
    if "consistency_pct" in label_df.columns:
        return label_df.sort_values(["enrichment_diff_mean", "consistency_pct"], ascending=[False, False])
    return label_df.sort_values("enrichment_diff_mean", ascending=False)


def format_feature_importance_table(label_df: pd.DataFrame) -> pd.DataFrame:
    """Re-sorts label_df itself (sort_by_importance_then_consistency) --
    the caller does not need to pre-sort it. One row per label that
    cleared MIN_OCCURRENCES -- every one of them, no top-K cap; the reader
    picks their own cutoff by how far down the table they read."""
    label_df = sort_by_importance_then_consistency(label_df)
    has_consistency = "consistency_pct" in label_df.columns
    rows = []
    for i, (_, row) in enumerate(label_df.iterrows(), start=1):
        p_fdr = row["mass_share_pvalue_fdr"]
        star = "*" if pd.notna(p_fdr) and p_fdr < SIGNIFICANCE_ALPHA else ""
        if has_consistency and pd.notna(row.get("consistency_pct")):
            n_occ = int(row["n_total_occurrences"])
            consistency = f"{row['consistency_pct']:.0%} ({int(row['n_occurrences'])}/{n_occ})"
        else:
            n_occ = int(row["n_occurrences"])
            consistency = "n/a"
        rows.append({"rank": i, "label": row["label"], "n_occurrences": n_occ,
                      "importance (mass_share vs area_share)": f"{row['enrichment_diff_mean']:+.3f}{star}",
                      "consistency": consistency})
    return pd.DataFrame(rows)


def feature_importance_csv_table(label_df: pd.DataFrame) -> pd.DataFrame:
    """Same ranking as format_feature_importance_table() (re-sorts label_df
    itself the same way, so the .md and .csv can't disagree on order),
    fully split into typed numeric columns (no combined display strings)
    -- for opening in a spreadsheet and sorting/filtering directly, rather
    than reading top to bottom."""
    if len(label_df) == 0:
        return pd.DataFrame(columns=["rank", "label", "n_total_occurrences", "n_significant_occurrences",
                                      "consistency_pct", "importance", "fdr_pvalue", "fdr_significant"])
    label_df = sort_by_importance_then_consistency(label_df)
    has_consistency = "consistency_pct" in label_df.columns
    out = pd.DataFrame({
        "label": label_df["label"],
        "n_total_occurrences": label_df["n_total_occurrences"] if has_consistency else label_df["n_occurrences"],
        "n_significant_occurrences": label_df["n_occurrences"],
        "consistency_pct": label_df["consistency_pct"] if has_consistency else np.nan,
        "importance": label_df["enrichment_diff_mean"],
        "fdr_pvalue": label_df["mass_share_pvalue_fdr"],
    })
    out.insert(0, "rank", range(1, len(out) + 1))
    out["fdr_significant"] = out["fdr_pvalue"] < SIGNIFICANCE_ALPHA
    return out


FEATURE_IMPORTANCE_EXPLANATION = (
    "**How to read this table**\n\n"
    "One row per label that was detected at least "
    f"{MIN_OCCURRENCES} times, ranked by **importance** -- no top-K cutoff, read as far down as "
    "you want.\n\n"
    "**importance** -- mean, per image, of (mass_share - area_share): how much more attention this "
    "label gets than its own footprint (segment size) would predict, if attention were spread "
    "evenly per pixel. Positive = gets MORE attention than its size predicts; negative = LESS. "
    "This is the one effect-size number the ranking is sorted by. `*` marks FDR-corrected p < "
    f"{SIGNIFICANCE_ALPHA} (one-sample Wilcoxon signed-rank test on mass_share - area_share, "
    "Benjamini-Hochberg-corrected across every label tested).\n\n"
    "**consistency** -- of every time this label was detected (the second number), what fraction "
    "individually beat a per-image chance baseline (the permutation test: is this object's mean "
    "saliency higher than a random same-size region of the same image, tested separately per "
    "detection). A label at 90% (180/200) is a reliable, repeatable driver; a label at 20% (4/20) "
    "is occasionally spiking, not consistently used -- read this alongside importance, since a big "
    "importance number built on a low consistency rate is more likely a few noisy images than a "
    "real pattern.\n\n"
    "See `feature_importance_full.md` (same folder) for every other statistic behind this ranking "
    "-- CIs, peak_ratio, the confidence correlation, and rank stability."
)


def build_feature_importance_report(run_id: str, label_df: pd.DataFrame) -> str:
    """The primary report: one uncapped table, five columns, nothing to
    cross-reference. label_df must already have consistency columns
    (add_consistency()) -- generate_analysis.py always supplies these for
    the low-failure branch."""
    lines = [f"# Feature importance ranking: {run_id}", "",
             "Every label ranked by how much more (or less) attention it draws than a baseline "
             "derived from the images themselves -- no fail class needed (see "
             "`feature_importance_full.md` for why). Pick your own cutoff by reading as far down "
             "this table as you want.", "",
             FEATURE_IMPORTANCE_EXPLANATION, ""]
    table = format_feature_importance_table(label_df)
    lines.append(df_to_markdown(table) if len(table) else "(no label met the threshold)")
    return "\n".join(lines)


# --- Stage 5: bootstrap rank stability --------------------------------------
def bootstrap_rank_stability(chosen_df: pd.DataFrame, point_ranking_labels: list,
                              top_k: int = TOP_K_STABILITY, n_boot: int = N_BOOTSTRAP_RANK,
                              seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """point_ranking_labels: labels in point-estimate rank order (best
    first) -- typically label_enrichment_table()'s top ~20 by
    enrichment_diff_mean, kept small so the O(n_boot * n_images * n_labels)
    resampling stays fast; this checks stability of the *ranking*, not any
    one label's own estimate (that's what enrichment_diff's CI is for)."""
    if not point_ranking_labels:
        return pd.DataFrame(columns=["label", "point_rank", "pct_in_top_k"])

    wide = pd.DataFrame({label: _per_image_label_table(chosen_df, label)["mass_share"]
                          - _per_image_label_table(chosen_df, label)["area_share"]
                          for label in point_ranking_labels})
    n = len(wide)
    if n == 0:
        return pd.DataFrame(columns=["label", "point_rank", "pct_in_top_k"])

    point_rank = {label: i + 1 for i, label in enumerate(point_ranking_labels)}
    top_k_labels = point_ranking_labels[:top_k]
    counts = {label: 0 for label in top_k_labels}

    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means = wide.iloc[idx].mean(skipna=True)   # per-label mean over drawn rows where present, NaN elsewhere
        resample_top_k = set(means.dropna().sort_values(ascending=False).index[:top_k])
        for label in top_k_labels:
            if label in resample_top_k:
                counts[label] += 1

    rows = [{"label": label, "point_rank": point_rank[label], "pct_in_top_k": counts[label] / n_boot}
            for label in top_k_labels]
    return pd.DataFrame(rows).sort_values("point_rank")


# --- Key Findings (deterministic) -------------------------------------------
def generate_low_failure_key_findings(label_df: pd.DataFrame, fail_rate: float) -> str:
    if len(label_df) == 0:
        return "No label met the minimum occurrence threshold -- no findings to report."

    # Two separate groups, not one interleaved list -- mass_share diffs and
    # peak_ratio deviations are different units, so a single combined
    # magnitude ranking across both would compare apples to oranges. Each
    # group is sorted by ITS OWN effect magnitude (distance from that
    # test's own center), descending -- never by p-value or n_occurrences,
    # same reasoning as the sort chains above.
    specs = [("mass_share", "mass_share_pvalue_fdr", "enrichment_diff_mean", 0.0, "mass_share vs. area_share",
              "enrichment_diff_ci_lo", "enrichment_diff_ci_hi"),
             ("peak_ratio", "peak_ratio_pvalue_fdr", "mean_peak_ratio", 1.0,
              "peak_ratio vs. its own no-signal baseline of 1.0", "peak_ratio_ci_lo", "peak_ratio_ci_hi")]
    groups = {"mass_share": [], "peak_ratio": []}
    for _, row in label_df.iterrows():
        for group_key, p_col, effect_col, center, desc, lo_col, hi_col in specs:
            p_fdr, effect = row[p_col], row[effect_col]
            if pd.isna(effect) or pd.isna(p_fdr) or p_fdr >= SIGNIFICANCE_ALPHA:
                continue
            lo, hi = row[lo_col], row[hi_col]
            ci = "" if pd.isna(lo) or pd.isna(hi) else f" (95% CI [{lo:.3f}, {hi:.3f}])"
            direction = ("ENRICHED (gets MORE attention than its footprint/no-signal baseline predicts)"
                         if effect > center else
                         "DEPLETED (gets LESS attention than its footprint/no-signal baseline predicts)")
            groups[group_key].append({"label": row["label"], "test": desc, "effect": effect, "ci": ci,
                                       "p": p_fdr, "direction": direction, "n": row["n_occurrences"],
                                       "magnitude": abs(effect - center)})

    if not groups["mass_share"] and not groups["peak_ratio"]:
        return (f"No label's mass_share (vs. its own area_share) or peak_ratio (vs. its own no-signal "
                f"baseline of 1.0) showed a statistically significant departure (FDR-corrected p < "
                f"{SIGNIFICANCE_ALPHA}) in this run -- no pattern is reported here rather than inventing "
                f"one from noise.")

    lines = []
    titles = {"mass_share": "By mass_share enrichment (biggest departure from area_share first)",
              "peak_ratio": "By peak_ratio departure (biggest departure from 1.0 first, either direction)"}
    for group_key in ("mass_share", "peak_ratio"):
        findings = groups[group_key]
        if not findings:
            continue
        # real float precision on magnitude -- _fmt_p below only affects DISPLAY, p is only a tiebreak
        findings.sort(key=lambda f: (-f["magnitude"], f["p"]))
        lines.append(f"**{titles[group_key]}** (FDR-corrected p < {SIGNIFICANCE_ALPHA}):")
        for f in findings:
            lines.append(f"- **{f['label']}**: {f['direction']}, effect={f['effect']:.3f}{f['ci']}, "
                          f"FDR p={_fmt_p(f['p'])}, seen in {f['n']} image(s)")
        lines.append("")

    lines.append(f"These compare each label's own attention share against a baseline derived from the "
                 f"images themselves (segment area, or this label's own no-signal ratio) -- not against a "
                 f"failing class, since this dataset's {low_failure_branch_reason(fail_rate)} for the "
                 f"pass-vs-fail comparison this needs. Still an attention-based association, not a causal "
                 f"test of what drives the decision.")
    return "\n".join(lines)


# --- Report assembly ---------------------------------------------------------
def branch_explanation(fail_rate: float) -> str:
    """"Why this report looks different" blurb. Takes fail_rate (rather than
    being a fixed string) so it names whichever class is actually too thin --
    this branch fires both for a low fail_rate (few/no failing images) and,
    symmetrically, for a low pass_rate (few/no passing images)."""
    return (
        "**Why this report looks different**\n\n"
        f"This dataset's {low_failure_branch_reason(fail_rate)}, so the usual pass-vs-fail "
        "contrast tables (odds ratio, Mann-Whitney AUC, Random Forest + SHAP -- see stats_tables.py) would "
        "have nothing to contrast against: every one of those tests requires a real sample of BOTH classes, "
        "and with too few (or zero) images in one class they degrade to `None`/skipped without erroring, "
        "which looks indistinguishable from a genuine null result. This branch instead scores each label "
        "against a baseline computed from the images themselves -- how much attribution it gets vs. how "
        "much its own segment size or its own no-signal ratio would predict -- so it still produces a real "
        "ranking without needing a balanced pass/fail split at all. Nothing below refers to any "
        "dataset-specific label or known group -- the same logic applies to any dataset that lands in this "
        "branch."
    )

ENRICHMENT_EXPLANATION = (
    "**How to read this table, in plain terms**\n\n"
    "For each label, `mass_share` is what fraction of the image's total attribution landed on it; "
    "`area_share` is what fraction of the image's *pixels* it occupies. If attention were spread evenly "
    "per pixel, the two would be equal. A label whose `mass_share` consistently exceeds its `area_share` "
    "is getting disproportionate attention relative to its size -- evidence of decision-relevance rather "
    "than just being large/prominent.\n\n"
    "**enrichment_ratio** -- mean mass_share / mean area_share. 1.0 = proportional to size (unremarkable). "
    ">1 = more attention than size predicts. <1 = less.\n\n"
    "**enrichment_diff_mean / 95% CI** -- the mean, per image, of (mass_share - area_share), with a "
    "bootstrap 95% CI (resample the images where this label occurs, 2,000 times). A CI that excludes 0 "
    "means the label reliably gets more (or less) attention than its size alone predicts.\n\n"
    "**mass_share FDR p** -- a one-sample Wilcoxon signed-rank test on (mass_share - area_share) per "
    "image, Benjamini-Hochberg-corrected jointly with the peak_ratio test below (every label x both "
    "metrics tested at once). Smaller = stronger evidence this isn't just noise. Shown in scientific "
    "notation below 0.001 rather than rounded to 0.000, so the sort order below is verifiable rather than "
    "opaque once many labels are all far past the significance threshold.\n\n"
    f"**Sort order**: {MASS_SHARE_SORT_EXPLANATION}."
)

PEAK_EXPLANATION = (
    "**How to read this table, in plain terms**\n\n"
    "`peak_ratio` is already a self-normalized quantity (this label's strongest attention pixels, "
    "divided by the whole image's strongest attention pixels) -- so its own no-signal value is 1.0 by "
    "construction, not related to area. **Secondary/confirmatory column**: a small-area label's top-k "
    "mean has higher sampling variance, so peak_ratio can drift above 1.0 by chance more easily for "
    "small objects than mass_share's area-based enrichment does -- treat agreement between the two "
    "tables as stronger evidence than either alone.\n\n"
    "**mean_peak_ratio / 95% CI** -- mean across images where the label occurs, with a bootstrap CI.\n\n"
    "**peak_ratio FDR p** -- one-sample Wilcoxon test against 1.0, pooled into the same FDR correction "
    "as the mass_share test above. Shown in scientific notation below 0.001 (see note on the table above) "
    "so the sort order is verifiable.\n\n"
    "**This table's own sort order is independent of the Mass-Share table above** -- it leads with "
    f"peak_ratio's own effect size, not mass_share's: {PEAK_RATIO_SORT_EXPLANATION}."
)

CONFIDENCE_EXPLANATION_TMPL = (
    "**How to read this table, in plain terms**\n\n"
    "Even with (near-)zero failures, this dataset's own continuous output value -- **{desc}** -- still "
    "varies image to image. This checks whether a label's attribution (mass_share, 0 when absent, across "
    "every image) correlates with that value: a label consistently associated with a stronger/weaker "
    "output is more likely load-bearing for the decision than one with no relationship.\n\n"
    "**spearman_rho** -- rank correlation (robust to non-linear monotonic relationships). Positive: "
    "higher attribution on this label goes with a higher output value. Negative: the opposite.\n\n"
    "**FDR p** -- Benjamini-Hochberg-corrected across every label tested here, shown in scientific "
    "notation below 0.001 so the sort order is verifiable.\n\n"
    "**Sort order**: 1) `|spearman_rho|` (descending -- strongest relationship first, regardless of "
    "direction), 2) FDR p (ascending, low-priority tiebreak)."
)

STABILITY_EXPLANATION = (
    "**How to read this table, in plain terms**\n\n"
    f"For the top {TOP_K_STABILITY} labels by enrichment_diff_mean above, this resamples the dataset's "
    f"images with replacement {N_BOOTSTRAP_RANK} times and recomputes the ranking each time -- checking "
    "whether the SAME labels stay near the top, not just whether each one's own estimate looks "
    "significant in isolation (that's what the CI above already checks). `pct_in_top_k` = fraction of "
    "resamples in which this label stayed in the (resampled) top-K. Low values mean the ranking itself "
    "is unstable even if individual labels' CIs look fine."
)


def build_low_failure_report(run_id: str, all_images: dict, per_image_meta: dict, chosen_df: pd.DataFrame,
                              label_df: pd.DataFrame, confidence_series=None, confidence_desc=None) -> str:
    """`chosen_df` must already carry a canonical_label column and (unlike
    stats_tables.build_report) an area_share column. `label_df` must come
    from compute_label_df(chosen_df, all_occurrence_df) -- computed once by
    the caller and shared with build_feature_importance_report() so the two
    reports can't silently disagree about the ranking.

    This is the full-detail report -- see build_feature_importance_report()
    for the primary, uncapped ranking table meant to be read on its own."""
    from common.stats_tables import image_summary_table   # local import: avoids a module-load cycle risk

    fail_rate = dataset_fail_rate(all_images)
    image_df = image_summary_table(run_id, all_images, per_image_meta)

    lines = [f"# Analysis report (full detail): {run_id}", "",
              f"(low-failure branch -- fail rate {fail_rate:.1%}, this dataset's "
              f"{low_failure_branch_reason(fail_rate)})",
              "", branch_explanation(fail_rate)]

    lines.append("\n## Feature importance ranking\n")
    lines.append("See **`feature_importance.md`** (same folder) for the primary, uncapped ranking "
                 "table this whole report backs -- one row per label, sorted by effect size, nothing "
                 "to cross-reference. Everything below is the full statistical detail behind that "
                 "ranking: every raw number, both tests, confidence intervals, the correlation check, "
                 "and rank stability.")

    lines.append("\n## Per-image summary\n")
    lines.append(df_to_markdown(image_df))

    lines.append(f"\n## Mass-Share Enrichment -- full detail, all {len(label_df)} label(s) seen >= "
                 f"{MIN_OCCURRENCES} times\n")
    lines.append(ENRICHMENT_EXPLANATION + "\n")
    has_consistency = "consistency_pct" in label_df.columns
    enrichment_cols = ["label", "n_occurrences"]
    if has_consistency:
        enrichment_cols += ["n_total_occurrences", "consistency_pct"]
    enrichment_cols += ["mean_mass_share", "mean_area_share", "enrichment_ratio",
                        "enrichment_diff_mean", "enrichment_diff_ci_lo", "enrichment_diff_ci_hi",
                        "mass_share_pvalue_fdr", "frequency_consistency_score"]
    if len(label_df):
        rounded = label_df[enrichment_cols].round(4)
        rounded["mass_share_pvalue_fdr"] = label_df["mass_share_pvalue_fdr"].map(_fmt_p)
    else:
        rounded = pd.DataFrame(columns=enrichment_cols)
    lines.append(df_to_markdown(rounded) if len(rounded) else "(no label met the threshold)")

    lines.append(f"\n## Peak-Ratio Baseline Check -- full detail, secondary metric, own sort order\n")
    lines.append(PEAK_EXPLANATION + "\n")
    peak_cols = ["label", "n_occurrences", "mean_peak_ratio", "peak_ratio_ci_lo", "peak_ratio_ci_hi",
                 "peak_ratio_pvalue_fdr"]
    if len(label_df):
        peak_sorted = sort_by_peak_ratio(label_df)
        rounded_peak = peak_sorted[peak_cols].round(4)
        rounded_peak["peak_ratio_pvalue_fdr"] = peak_sorted["peak_ratio_pvalue_fdr"].map(_fmt_p)
    else:
        rounded_peak = pd.DataFrame(columns=peak_cols)
    lines.append(df_to_markdown(rounded_peak) if len(rounded_peak) else "(no label met the threshold)")

    lines.append(f"\n## Key Findings (from the top {TOP_K_SUMMARY} labels by mass_share effect size above -- "
                 "see full tables above for every other significant label)\n")
    lines.append(generate_low_failure_key_findings(label_df.head(TOP_K_SUMMARY), fail_rate))

    lines.append(f"\n## Attribution vs. Continuous Output Correlation\n")
    if confidence_series is None:
        lines.append("(skipped: this dataset's metadata.csv has no continuous decision-relevant output "
                      f"column among {CONFIDENCE_COLUMN_CANDIDATES} -- only a discrete predicted class, so "
                      "there is no continuous signal to correlate attribution against)")
    else:
        lines.append(CONFIDENCE_EXPLANATION_TMPL.format(desc=confidence_desc) + "\n")
        conf_df = confidence_correlation_table(chosen_df, sorted(all_images), confidence_series)
        if conf_df is not None and len(conf_df):
            conf_rounded = conf_df.round(4)
            conf_rounded["spearman_pvalue_fdr"] = conf_df["spearman_pvalue_fdr"].map(_fmt_p)
            lines.append(df_to_markdown(conf_rounded))
        else:
            lines.append("(no label had enough occurrences + confidence coverage to test)")

    lines.append(f"\n## Rank Stability (bootstrap, top {TOP_K_STABILITY})\n")
    lines.append(STABILITY_EXPLANATION + "\n")
    # Only the top candidates go into the resampling matrix (bounds cost
    # regardless of how many labels clear MIN_OCCURRENCES -- a dataset with
    # e.g. ~190 canonical labels should not build a ~190-column bootstrap
    # matrix just to report stability of the top 10).
    top_labels = list(label_df["label"])[:RANK_STABILITY_CANDIDATE_POOL] if len(label_df) else []
    stability_df = bootstrap_rank_stability(chosen_df, top_labels)
    lines.append(df_to_markdown(stability_df.round(3)) if len(stability_df) else "(no label met the threshold)")

    lines.append("\n## Feature Combinations (Random Forest + SHAP)\n")
    lines.append(f"(not run in this branch -- that stage needs both classes present with a real sample "
                 f"size in each; this dataset's {low_failure_branch_reason(fail_rate)}. The Mass-Share "
                 "Enrichment ranking + Rank Stability above serve the same role -- which labels the model "
                 "actually leans on -- without needing a balanced second class.)")
    return "\n".join(lines)
