"""Per-label statistics, deterministic Key Findings, and a Random Forest +
SHAP feature-combination stage, assembled into one markdown report per
dataset.

Three independent significance tests per canonical label: presence (does it
show up more often in one class -- Fisher's exact test + odds ratio),
mass_share and peak_ratio (when present, is attention stronger in one class
-- Mann-Whitney AUC). All three get a bootstrap 95% CI and are pooled into
one Benjamini-Hochberg FDR correction (testing dozens of labels/metrics per
dataset at once needs that correction, not a per-test threshold).
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, rankdata

MIN_OCCURRENCES = 3        # skip labels seen fewer than this many times total
SIGNIFICANCE_ALPHA = 0.1   # FDR-corrected p-value threshold for Key Findings
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


def df_to_markdown(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "(no data)"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


# --- AUC + bootstrap CI ------------------------------------------------------
def _mannwhitney_auc(pass_vals, fail_vals):
    """AUC = P(pass_score > fail_score), via Mann-Whitney U, + two-sided p."""
    if len(pass_vals) == 0 or len(fail_vals) == 0 or (pass_vals.nunique() <= 1 and fail_vals.nunique() <= 1):
        return None, None
    try:
        u, p = mannwhitneyu(pass_vals, fail_vals, alternative="two-sided")
        return float(u / (len(pass_vals) * len(fail_vals))), float(p)
    except ValueError:
        return None, None


def _bootstrap_auc_ci(pass_vals, fail_vals, n_boot: int = N_BOOTSTRAP, rng=None):
    n_pass, n_fail = len(pass_vals), len(fail_vals)
    if n_pass < 2 or n_fail < 2:
        return None, None
    rng = rng or np.random.default_rng(BOOTSTRAP_SEED)
    pass_arr, fail_arr = np.asarray(pass_vals), np.asarray(fail_vals)

    # AUC via the rank-sum identity (average ranks handle ties as 0.5), which
    # equals the pairwise wins + 0.5*ties mean but is O(n log n) per resample
    # instead of O(n_pass*n_fail). Chunk size (and hence RNG draw order) is
    # kept identical to the earlier pairwise implementation so results match.
    chunk_budget_bytes = 1.5 * 1024 ** 3
    chunk = max(1, int(chunk_budget_bytes // (n_pass * n_fail * 8)))
    boot_auc = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        pass_mat = pass_arr[rng.integers(0, n_pass, size=(b, n_pass))]
        fail_mat = fail_arr[rng.integers(0, n_fail, size=(b, n_fail))]
        ranks = rankdata(np.concatenate([pass_mat, fail_mat], axis=1), axis=1)
        rank_sum_pass = ranks[:, :n_pass].sum(axis=1)
        boot_auc[start:start + b] = (rank_sum_pass - n_pass * (n_pass + 1) / 2) / (n_pass * n_fail)
    lo, hi = np.percentile(boot_auc, [2.5, 97.5])
    return float(lo), float(hi)


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values. Controls the expected
    proportion of false positives across the whole batch of tests, not the
    risk of any one test in isolation."""
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    adjusted = np.full_like(p, np.nan)
    if valid.sum() == 0:
        return adjusted
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


# --- Presence/absence test --------------------------------------------------
def _fisher_presence_test(n_pass_present, n_pass_total, n_fail_present, n_fail_total):
    """Fisher's exact (not chi-squared, since many labels are present in very
    few images) over the 2x2 presence table. odds_ratio > 1 means more
    likely present in PASS. +0.5 Haldane-Anscombe correction on every cell
    for the point estimate (a raw odds ratio is undefined at 0); the
    p-value itself is scipy's exact, uncorrected calculation."""
    n_pass_absent = n_pass_total - n_pass_present
    n_fail_absent = n_fail_total - n_fail_present
    if n_pass_total == 0 or n_fail_total == 0:
        return None, None
    _, p = fisher_exact([[n_pass_present, n_pass_absent], [n_fail_present, n_fail_absent]])
    a, b = n_pass_present + 0.5, n_pass_absent + 0.5
    c, d = n_fail_present + 0.5, n_fail_absent + 0.5
    return float((a * d) / (b * c)), float(p)


def _bootstrap_odds_ratio_ci(pass_present, fail_present, n_boot: int = N_BOOTSTRAP, rng=None):
    n_pass, n_fail = len(pass_present), len(fail_present)
    if n_pass < 2 or n_fail < 2:
        return None, None
    rng = rng or np.random.default_rng(BOOTSTRAP_SEED)
    pass_arr, fail_arr = np.asarray(pass_present, dtype=bool), np.asarray(fail_present, dtype=bool)
    pass_draws = pass_arr[rng.integers(0, n_pass, size=(n_boot, n_pass))]
    fail_draws = fail_arr[rng.integers(0, n_fail, size=(n_boot, n_fail))]
    a = pass_draws.sum(axis=1).astype(np.float64) + 0.5
    b = (n_pass - pass_draws.sum(axis=1)).astype(np.float64) + 0.5
    c = fail_draws.sum(axis=1).astype(np.float64) + 0.5
    d = (n_fail - fail_draws.sum(axis=1)).astype(np.float64) + 0.5
    boot_or = (a * d) / (b * c)
    lo, hi = np.percentile(boot_or, [2.5, 97.5])
    return float(lo), float(hi)


# --- Stage: per-label statistics --------------------------------------------
def label_stats_table(run_id: str, all_images: dict, chosen_df: pd.DataFrame) -> pd.DataFrame:
    pass_ids = {i for i, l in all_images.items() if l == "pass"}
    fail_ids = {i for i, l in all_images.items() if l == "fail"}
    n_pass, n_fail = len(pass_ids), len(fail_ids)

    rows = []
    if len(chosen_df) == 0:
        return pd.DataFrame(rows)
    for label, g in chosen_df.groupby("canonical_label"):
        if len(g) < MIN_OCCURRENCES:
            continue
        per_img = g.groupby("image_id").agg(mass_share=("mass_share", "max"), peak_ratio=("peak_ratio", "max"))
        present_pass = set(per_img.index) & pass_ids
        present_fail = set(per_img.index) & fail_ids

        mass_scores = pd.Series(0.0, index=sorted(all_images))
        mass_scores.loc[per_img.index] = per_img["mass_share"]
        peak_scores = pd.Series(0.0, index=sorted(all_images))
        peak_scores.loc[per_img.index] = per_img["peak_ratio"]
        y = pd.Series({i: all_images[i] for i in mass_scores.index})

        rng = np.random.default_rng(BOOTSTRAP_SEED)  # fresh per label, order-independent

        def auc_p_ci(scores):
            p_vals, f_vals = scores[y == "pass"], scores[y == "fail"]
            auc, p = _mannwhitney_auc(p_vals, f_vals)
            lo, hi = (_bootstrap_auc_ci(p_vals, f_vals, rng=rng) if auc is not None else (None, None))
            return auc, p, lo, hi

        mass_auc, mass_p, mass_lo, mass_hi = auc_p_ci(mass_scores)
        peak_auc, peak_p, peak_lo, peak_hi = auc_p_ci(peak_scores)

        presence_or, presence_p = _fisher_presence_test(len(present_pass), n_pass, len(present_fail), n_fail)
        pass_bool = [i in present_pass for i in sorted(pass_ids)]
        fail_bool = [i in present_fail for i in sorted(fail_ids)]
        presence_lo, presence_hi = (_bootstrap_odds_ratio_ci(pass_bool, fail_bool, rng=rng)
                                     if presence_or is not None else (None, None))

        rows.append({
            "run_id": run_id, "label": label,
            "n_pass_present": len(present_pass), "n_pass_total": n_pass,
            "n_fail_present": len(present_fail), "n_fail_total": n_fail,
            "pass_rate": len(present_pass) / n_pass if n_pass else None,
            "fail_rate": len(present_fail) / n_fail if n_fail else None,
            "presence_odds_ratio": presence_or, "presence_pvalue": presence_p,
            "presence_ci_lo": presence_lo, "presence_ci_hi": presence_hi,
            "mean_mass_share_pass": per_img.loc[list(present_pass), "mass_share"].mean() if present_pass else None,
            "mean_mass_share_fail": per_img.loc[list(present_fail), "mass_share"].mean() if present_fail else None,
            "mean_peak_ratio_pass": per_img.loc[list(present_pass), "peak_ratio"].mean() if present_pass else None,
            "mean_peak_ratio_fail": per_img.loc[list(present_fail), "peak_ratio"].mean() if present_fail else None,
            "mass_share_auc": mass_auc, "mass_share_pvalue": mass_p,
            "mass_share_ci_lo": mass_lo, "mass_share_ci_hi": mass_hi,
            "peak_ratio_auc": peak_auc, "peak_ratio_pvalue": peak_p,
            "peak_ratio_ci_lo": peak_lo, "peak_ratio_ci_hi": peak_hi,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["presence_rate_diff"] = df["pass_rate"] - df["fail_rate"]
        # Pooled across all three test types -- one jointly-reported family,
        # matching what Key Findings evaluates as one set of candidates.
        pooled_p = pd.concat([df["mass_share_pvalue"], df["peak_ratio_pvalue"], df["presence_pvalue"]],
                              ignore_index=True)
        pooled_fdr = _bh_fdr(pooled_p.to_numpy())
        n = len(df)
        df["mass_share_pvalue_fdr"] = pooled_fdr[:n]
        df["peak_ratio_pvalue_fdr"] = pooled_fdr[n:2 * n]
        df["presence_pvalue_fdr"] = pooled_fdr[2 * n:3 * n]
        df = df.sort_values("presence_rate_diff", ascending=False)
    return df


def image_summary_table(run_id: str, all_images: dict, per_image_meta: dict) -> pd.DataFrame:
    rows = []
    for label in ["pass", "fail"]:
        ids = [i for i, l in all_images.items() if l == label]
        if not ids:
            continue
        n_chosen = [per_image_meta[i]["n_chosen"] for i in ids if i in per_image_meta]
        overlap = [per_image_meta[i]["overlap_mass_share"] for i in ids if i in per_image_meta]
        degen = [per_image_meta[i]["degenerate"] for i in ids if i in per_image_meta]
        rows.append({"run_id": run_id, "pass_fail": label, "n_images": len(ids),
                      "mean_n_chosen_objects": float(np.mean(n_chosen)) if n_chosen else None,
                      "mean_overlap_mass_share": float(np.mean(overlap)) if overlap else None,
                      "degenerate_rate": float(np.mean(degen)) if degen else None})
    return pd.DataFrame(rows)


# --- Key Findings (deterministic -- no model call) --------------------------
# One entry per test type: (metric key, effect column, "no difference"
# center, effect label, verb phrase). presence's center is 1.0 (odds ratio)
# instead of 0.5 (AUC).
_FINDING_METRICS = [
    ("mass_share", "mass_share_auc", 0.5, "AUC", "significantly higher in"),
    ("peak_ratio", "peak_ratio_auc", 0.5, "AUC", "significantly higher in"),
    ("presence", "presence_odds_ratio", 1.0, "odds ratio", "significantly more likely to be present in"),
]


def _format_ci(lo, hi) -> str:
    return "" if pd.isna(lo) or pd.isna(hi) else f"[{lo:.0%}, {hi:.0%}]"


def _format_or_ci(lo, hi) -> str:
    return "" if pd.isna(lo) or pd.isna(hi) else f"[{lo:.2f}, {hi:.2f}]"


def generate_key_findings(label_df: pd.DataFrame) -> str:
    """States, for every (label, test) pair whose FDR-corrected p-value
    clears SIGNIFICANCE_ALPHA, the effect size + CI + direction + sample
    sizes DIRECTLY from label_stats_table()'s own output -- no separate
    retrieval step exists to get a number wrong. Split into "well-supported"
    (both classes have >= MIN_OCCURRENCES present) vs "hypothesis only"."""
    if len(label_df) == 0:
        return "No label met the minimum occurrence threshold -- no findings to report."

    findings = []
    for _, row in label_df.iterrows():
        for metric, effect_col, center, effect_label, phrase in _FINDING_METRICS:
            effect, p_fdr = row[effect_col], row[f"{metric}_pvalue_fdr"]
            if pd.isna(effect) or pd.isna(p_fdr) or p_fdr >= SIGNIFICANCE_ALPHA:
                continue
            ci = (_format_or_ci(row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"]) if metric == "presence"
                  else _format_ci(row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"]))
            findings.append({
                "label": row["label"], "metric": metric, "effect": effect, "effect_label": effect_label,
                "phrase": phrase, "p": p_fdr, "ci": ci,
                "direction": "PASSING" if effect > center else "FAILING",
                "n_pass_present": row["n_pass_present"], "n_fail_present": row["n_fail_present"],
                "well_supported": row["n_pass_present"] >= MIN_OCCURRENCES and row["n_fail_present"] >= MIN_OCCURRENCES,
            })

    if not findings:
        return (f"No feature's mass_share, peak_ratio, or presence showed a statistically "
                f"significant difference between passing and failing images (FDR-corrected p < "
                f"{SIGNIFICANCE_ALPHA}) in this run -- no pattern is reported here rather than "
                f"inventing one from noise.")

    def _fmt(f):
        ci_clause = f" (95% CI {f['ci']})" if f["ci"] else ""
        return (f"- **{f['label']}** ({f['metric']}): {f['phrase']} **{f['direction']}** images "
                f"({f['effect_label']}={f['effect']:.3f}{ci_clause}, FDR p={f['p']:.3f}; "
                f"present in {f['n_pass_present']} pass / {f['n_fail_present']} fail images)")

    well_supported = sorted((f for f in findings if f["well_supported"]), key=lambda f: f["p"])
    hypothesis_only = sorted((f for f in findings if not f["well_supported"]), key=lambda f: f["p"])
    lines = []
    if well_supported:
        lines.append(f"**Well-supported** (FDR-corrected p < {SIGNIFICANCE_ALPHA}, both classes have "
                      f">= {MIN_OCCURRENCES} present):")
        lines += [_fmt(f) for f in well_supported]
    if hypothesis_only:
        if lines:
            lines.append("")
        lines.append(f"**Hypothesis only** (FDR-corrected p < {SIGNIFICANCE_ALPHA}, but one class has "
                      f"fewer than {MIN_OCCURRENCES} present -- treat as worth checking on more data, "
                      f"not a conclusion):")
        lines += [_fmt(f) for f in hypothesis_only]
    lines.append("\nThese are attention-based statistics (does the model's attention correlate "
                  "with pass/fail) -- not a causal test of what drives the outcome.")
    return "\n".join(lines)


# --- Tables for the report ---------------------------------------------------
def _format_presence(n_present, n_total) -> str:
    if n_total == 0:
        return "0/0"
    return f"{int(n_present)}/{int(n_total)} ({n_present / n_total:.0%})"


def _format_diff(diff) -> str:
    return f"{'+' if diff >= 0 else ''}{diff:.1%}"


PRESENCE_EXPLANATION = (
    "**How to read this table, in plain terms**\n\n"
    "This table asks one question: *does this label show up at all more often in one class "
    "than the other?* It does NOT ask how strong the attention is when the label is present -- "
    "that's what the Mass Share and Peak Ratio tables below check instead. A label can appear "
    "equally often in both classes but be far more intense in one (caught below, not here), or "
    "appear far more often in one class while being similarly intense whenever it does appear "
    "(caught here, not fully there).\n\n"
    "**Odds ratio** -- plain meaning: how many times more likely this label is to appear in a "
    "PASS image than a FAIL image.\n"
    "- Formula: with `a` = # pass images with the label, `b` = # pass images without it, `c` = "
    "# fail images with it, `d` = # fail images without it: `odds ratio = (a/b) / (c/d) = "
    "(a*d) / (b*c)`. (A small +0.5 is added to a/b/c/d first so this never divides by zero -- "
    "the \"Haldane-Anscombe correction\".)\n"
    "- What the value means: **1.0** = no difference. **>1** = more common in PASS (e.g. 5.8 "
    "roughly means \"6x more common in pass\"). **<1** = more common in FAIL (e.g. 0.2 roughly "
    "means \"5x more common in fail\", since 1/0.2 = 5).\n\n"
    "**95% CI** -- plain meaning: \"if this exact check were rerun on a fresh random sample, how "
    "much would the odds-ratio estimate bounce around?\" It is a margin of error on the estimate "
    "**from this sample alone**, not a claim about the true odds ratio.\n"
    "- Formula (bootstrap): resample the pass images and fail images (with replacement) 2,000 "
    "times, recompute the odds ratio each time, report the 2.5th/97.5th percentile of those "
    "2,000 numbers.\n"
    "- What the value means: a wider range means a less trustworthy estimate (usually because "
    "very few images have this label). A range that includes 1.0 means \"can't rule out no real "
    "difference.\" **This range does NOT account for the fact that dozens of other labels were "
    "also being tested -- see FDR p below for that.**\n\n"
    "**FDR p** -- plain meaning: \"after admitting that dozens of labels were checked at once, "
    "how likely is it that this particular result is just noise?\"\n"
    "- Formula: start from the raw Fisher's exact test p-value for this label's 2x2 table, then "
    "apply Benjamini-Hochberg correction across every p-value in this report (all labels x "
    "presence/mass share/peak ratio combined) -- the more things tested, the stricter the bar.\n"
    f"- What the value means: smaller = stronger evidence; **{SIGNIFICANCE_ALPHA}** is this "
    "report's cutoff. A value near 1.0 means \"statistically indistinguishable from chance once "
    "every other test in the report is accounted for\" -- even if the odds ratio/CI alone looked "
    "convincing.\n\n"
    "**Why the CI and FDR p can seem to disagree**: the CI only judges this ONE label on its "
    "own; FDR p judges it after admitting ~50 things were tested at once. A label built on very "
    "few images (e.g. only 4-6 total occurrences) can easily look convincing alone while failing "
    "that stricter bar -- that's the correction working as intended, not a contradiction."
)

METRIC_EXPLANATION = (
    "**How to read this table, in plain terms**\n\n"
    "Unlike the Presence table above, this table only looks at images where the label **is "
    "present**, and asks: *when it does show up, is the model's attention on it typically "
    "stronger in PASS images or FAIL images?* (\"mass_share\" = what fraction of the image's "
    "total attribution landed on this label; \"peak_ratio\" = how strong this label's single "
    "most-attended pixel is relative to the image's overall peak.)\n\n"
    "**mean mass_share / mean peak_ratio (pass/fail)** -- plain meaning: the plain average "
    "value of this metric across images (in that class) where the label was found. Descriptive "
    "only, not itself a test.\n\n"
    "**AUC (Mann-Whitney)** -- plain meaning: if you picked one random PASS image and one "
    "random FAIL image (both containing this label), how often would the PASS image have the "
    "higher value?\n"
    "- Formula: `AUC = (# pass/fail pairs where pass > fail + 0.5 x # ties) / (total # pairs)` "
    "-- equivalent to the Mann-Whitney U statistic divided by (n_pass x n_fail).\n"
    "- What the value means: **1.0** = pass always higher. **0.0** = fail always higher. "
    "**0.5** = coin flip, no real difference. Read 0.9 and 0.1 the same way -- both \"far from "
    "chance,\" just opposite direction.\n\n"
    "**95% CI** -- same bootstrap idea as the Presence table: resample pass/fail images 2,000 "
    "times, recompute AUC each time, report the 2.5th/97.5th percentile. A range that includes "
    "0.5 means \"can't be told apart from chance.\" Same caveat as the Presence table: **this "
    "does NOT account for how many other labels/metrics were tested.**\n\n"
    "**FDR p** -- same idea as the Presence table: raw Mann-Whitney p-value for this label+"
    "metric, then Benjamini-Hochberg-corrected across every p-value in this report (all labels "
    "x mass_share/peak_ratio/presence combined). Smaller = stronger evidence; "
    f"**{SIGNIFICANCE_ALPHA}** is this report's cutoff. As above, a convincing-looking CI can "
    "still carry a large FDR p -- that means the effect doesn't survive once every other test "
    "in the report is accounted for."
)


def _presence_table(label_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "label": row["label"], "pass": _format_presence(row["n_pass_present"], row["n_pass_total"]),
        "fail": _format_presence(row["n_fail_present"], row["n_fail_total"]),
        "diff": _format_diff(row["presence_rate_diff"]), "odds ratio": row["presence_odds_ratio"],
        "95% CI": _format_or_ci(row["presence_ci_lo"], row["presence_ci_hi"]),
        "FDR p": row["presence_pvalue_fdr"],
    } for _, row in label_df.iterrows()])


DISPLAY_COLS = ["label", "n_pass_present", "n_pass_total", "n_fail_present", "n_fail_total",
                "pass_rate", "fail_rate", "presence_rate_diff",
                "presence_odds_ratio", "presence_pvalue", "presence_ci_lo", "presence_ci_hi",
                "presence_pvalue_fdr", "mean_mass_share_pass", "mean_mass_share_fail",
                "mean_peak_ratio_pass", "mean_peak_ratio_fail",
                "mass_share_auc", "mass_share_pvalue", "mass_share_ci_lo", "mass_share_ci_hi",
                "mass_share_pvalue_fdr", "peak_ratio_auc", "peak_ratio_pvalue",
                "peak_ratio_ci_lo", "peak_ratio_ci_hi", "peak_ratio_pvalue_fdr"]


def _significant_sorted(label_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Subset of label_df where <metric>'s own FDR-corrected p-value clears
    SIGNIFICANCE_ALPHA, sorted by that FDR-corrected p-value ascending --
    strongest evidence first, the row closest to the SIGNIFICANCE_ALPHA
    cutoff last. Per explicit request (2026-09-22): feature_importance.md
    orders by p-value, not effect size (feature_importance_full.md remains
    the unfiltered, diff-sorted full reference). NOT presence_rate_diff
    (label_stats_table()'s default sort) either, for the same reason this
    used effect size before switching to p-value -- a raw percentage-point
    difference is distorted by pass/fail class-size imbalance."""
    fdr_col = f"{metric}_pvalue_fdr"
    sig = label_df[label_df[fdr_col] < SIGNIFICANCE_ALPHA].copy()
    return sig.sort_values(fdr_col, ascending=True)


FEATURE_IMPORTANCE_INTRO = (
    "Only labels/tests that cleared FDR-corrected significance (p < {alpha}) appear below, each "
    "table sorted by FDR-corrected p-value ascending -- lowest (strongest) p-value first, the row "
    "closest to the {alpha} cutoff last. See **`feature_importance_full.md`** (same folder) for "
    "every label regardless of significance, full methodology, Key Findings, and the per-image "
    "summary."
)


def build_feature_importance_report(run_id: str, all_images: dict, per_image_meta: dict,
                                     chosen_df: pd.DataFrame) -> str:
    """The primary report for the normal pass/fail branch (see
    common.low_failure_stats.build_feature_importance_report for the
    below-5%-fail-rate analog): short, filtered to what's actually
    significant, each of the three tests ranked by its own effect size.
    `chosen_df` must already carry a canonical_label column."""
    label_df = label_stats_table(run_id, all_images, chosen_df)
    lines = [f"# Feature importance: {run_id}", "",
             FEATURE_IMPORTANCE_INTRO.format(alpha=SIGNIFICANCE_ALPHA)]

    presence_sig = _significant_sorted(label_df, "presence")
    lines.append(f"\n## Presence -- significant only ({len(presence_sig)} of {len(label_df)} label(s))\n")
    lines.append(df_to_markdown(_presence_table(presence_sig[DISPLAY_COLS].round(3)))
                 if len(presence_sig) else "(no label was significant on this test)")

    for metric, title in [("mass_share", "Mass Share"), ("peak_ratio", "Peak Ratio")]:
        sig = _significant_sorted(label_df, metric)
        lines.append(f"\n## {title} -- significant only ({len(sig)} of {len(label_df)} label(s))\n")
        lines.append(df_to_markdown(_metric_table(sig[DISPLAY_COLS].round(3), metric))
                     if len(sig) else "(no label was significant on this test)")

    lines.append("\n## Feature Combinations (Random Forest + SHAP)\n")
    feature_X, feature_y = build_feature_matrix(all_images, chosen_df, label_df)
    lines.append(_format_combination_section(random_forest_shap_analysis(feature_X, feature_y)))
    return "\n".join(lines)


def _metric_table(label_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "label": row["label"], "pass": _format_presence(row["n_pass_present"], row["n_pass_total"]),
        "fail": _format_presence(row["n_fail_present"], row["n_fail_total"]),
        "diff": _format_diff(row["presence_rate_diff"]),
        f"mean {metric} (pass)": row[f"mean_{metric}_pass"], f"mean {metric} (fail)": row[f"mean_{metric}_fail"],
        "AUC": row[f"{metric}_auc"], "95% CI": _format_ci(row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"]),
        "FDR p": row[f"{metric}_pvalue_fdr"],
    } for _, row in label_df.iterrows()])


# --- Random Forest + SHAP feature-combination analysis -----------------------
RF_N_ESTIMATORS = 500
RF_RANDOM_STATE = 42
RF_MIN_IMAGES = 20
RF_FALLBACK_MAX_DEPTH = 30   # see random_forest_shap_analysis
RF_PROBE_ROWS = 200

COMBINATION_EXPLANATION = (
    "**How to read this section, in plain terms**\n\n"
    "Every table above tests ONE label at a time. This section instead asks: *if a model uses "
    "ALL labels together, can it separate pass from fail -- and which labels is it actually "
    "relying on?*\n\n"
    "**Cross-validated ROC-AUC** -- plain meaning: train a Random Forest on most of the images "
    "(every label's mass_share as input), test it on held-out images it never saw, repeat "
    "across folds. Same 0.5=chance / 1.0=perfect scale as AUC above.\n"
    "- Formula: standard ROC-AUC on out-of-fold predictions, averaged across up to 5 "
    "StratifiedKFold folds.\n"
    "- What the value means: **read this FIRST -- it's a gate.** Near 0.5 means the model "
    "cannot actually separate pass/fail, so the SHAP numbers below are explaining noise, not a "
    "finding, no matter how large they look. Only trust the rest below if this is comfortably "
    "above 0.5.\n\n"
    "**Feature importance (mean |SHAP|)** -- plain meaning: on average, how much does this "
    "label's value push the prediction away from baseline (in either direction)?\n"
    "- Formula: SHAP assigns each feature a signed contribution per image; this is the average "
    "of the *absolute value* of that contribution across all images. Large = the model leans on "
    "it a lot; small = barely used.\n\n"
    "**direction** -- plain meaning: does this label push predictions toward PASS or FAIL "
    "overall?\n"
    "- Formula: Pearson correlation between the feature's own value and its own per-image SHAP "
    "value, across all images. Positive -> PASSING, negative -> FAILING. (Not just the average "
    "SHAP sign -- that can mislead when a feature's push is much bigger on one side, e.g. when "
    "absent, than the other.)"
)


def build_feature_matrix(all_images: dict, chosen_df: pd.DataFrame, label_df: pd.DataFrame):
    """One row per image, one column per canonical label that cleared
    MIN_OCCURRENCES (the same label universe as every table above). Cell =
    that label's mass_share (max if duplicated), 0 if absent."""
    labels = list(label_df["label"]) if len(label_df) else []
    image_ids = sorted(all_images)
    X = pd.DataFrame(0.0, index=image_ids, columns=labels)
    if labels and len(chosen_df):
        sub = chosen_df[chosen_df["canonical_label"].isin(labels)]
        per_cell = sub.groupby(["image_id", "canonical_label"])["mass_share"].max()
        for (iid, label), val in per_cell.items():
            if iid in X.index:
                X.loc[iid, label] = val
    return X, pd.Series({i: all_images[i] for i in image_ids})


def random_forest_shap_analysis(X: pd.DataFrame, y: pd.Series) -> dict:
    n_images, n_features = X.shape
    if n_images < RF_MIN_IMAGES or n_features == 0 or y.nunique() < 2:
        return {"skipped": True, "reason": (
            f"only {n_images} image(s) / {n_features} feature(s) available (need >= "
            f"{RF_MIN_IMAGES} images, >= 1 feature, both classes present)")}

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import shap
    from shap.utils._exceptions import ExplainerError

    y_bin = (y == "pass").astype(int)

    # Unlimited-depth trees are the default. On large/sparse matrices (celeba: 19.9k
    # images x 288 features -> trees 100+ deep) shap's TreeExplainer loses numerical
    # accuracy with depth and its additivity check raises ExplainerError; only then
    # refit with a depth cap (verified exact to ~1e-11 at 30). Datasets that never hit
    # the error keep the uncapped forest, so their reports are unchanged.
    max_depth_used = None
    for max_depth in (None, RF_FALLBACK_MAX_DEPTH):
        clf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE,
                                      class_weight="balanced", min_samples_leaf=2,
                                      max_depth=max_depth)
        n_splits = min(5, int(y_bin.value_counts().min()))
        cv_auc_mean = cv_auc_std = None
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RF_RANDOM_STATE)
            scores = cross_val_score(clf, X.values, y_bin.values, cv=cv, scoring="roc_auc")
            cv_auc_mean, cv_auc_std = float(scores.mean()), float(scores.std())

        clf.fit(X.values, y_bin.values)
        explainer = shap.TreeExplainer(clf)
        try:
            # Probe a small row sample first so a failure surfaces in ~1-2 min, not after the
            # full pass (celeba's full pass ran ~2h before the check tripped). Same rows,
            # same forest -> the full pass below is unaffected when the probe passes.
            probe = np.random.default_rng(RF_RANDOM_STATE).choice(
                n_images, size=min(RF_PROBE_ROWS, n_images), replace=False)
            explainer.shap_values(X.values[probe])
            sv = explainer.shap_values(X.values)
        except ExplainerError:
            if max_depth is not None:
                raise
            print(f"[rf-shap] additivity check failed on unlimited-depth forest (max tree depth "
                  f"{max(e.get_depth() for e in clf.estimators_)}); refitting with "
                  f"max_depth={RF_FALLBACK_MAX_DEPTH}", flush=True)
            continue
        max_depth_used = max_depth
        break
    # shap's binary-classifier convention: a list of two arrays, one per
    # class, exact negations of each other -- class 1 ("pass") loses nothing.
    sv_pass = sv[1] if isinstance(sv, list) else sv
    mean_abs = np.abs(sv_pass).mean(axis=0)

    # Direction: the sign of the Pearson correlation between each feature's
    # own value and its own per-image SHAP value (what a SHAP beeswarm plot
    # visualizes) -- NOT the plain mean signed SHAP value. The plain mean is
    # unreliable when a feature's SHAP swing is asymmetric between present
    # and absent (verified case: a feature that is overwhelmingly a PASS
    # signal by its own presence/mass_share tests was scored FAILING by
    # plain mean SHAP, because its absence-side swing was larger).
    n_features = X.shape[1]
    X_arr = X.values
    with np.errstate(invalid="ignore"):
        direction_corr = np.array([
            np.corrcoef(X_arr[:, i], sv_pass[:, i])[0, 1]
            if np.std(X_arr[:, i]) > 0 and np.std(sv_pass[:, i]) > 0 else 0.0
            for i in range(n_features)
        ])
    direction_corr = np.nan_to_num(direction_corr, nan=0.0)
    importance_df = pd.DataFrame({"label": X.columns, "mean_abs_shap": mean_abs,
                                   "direction": ["PASSING" if v > 0 else ("FAILING" if v < 0 else "neutral")
                                                 for v in direction_corr]}
                                  ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return {"skipped": False, "n_images": n_images, "n_features": n_features,
            "cv_auc_mean": cv_auc_mean, "cv_auc_std": cv_auc_std,
            "max_depth_used": max_depth_used,
            "feature_importance": importance_df}


def _format_combination_section(result: dict) -> str:
    if result["skipped"]:
        return f"(skipped: {result['reason']})"
    lines = [COMBINATION_EXPLANATION, ""]
    if result["cv_auc_mean"] is not None:
        lines.append(f"**Cross-validated ROC-AUC**: {result['cv_auc_mean']:.3f} "
                      f"(+/- {result['cv_auc_std']:.3f} across folds), predicting pass/fail from "
                      f"all {result['n_features']} feature(s) together over {result['n_images']} "
                      f"images.\n")
    else:
        lines.append(f"(too few images in the smaller class for cross-validation -- the "
                      f"importance below is fit on all {result['n_images']} images "
                      f"directly with no held-out check; treat with extra caution)\n")

    imp = result["feature_importance"].copy()
    imp["mean_abs_shap"] = imp["mean_abs_shap"].round(4)
    lines.append(f"**Feature importance** (all {len(imp)} feature(s), sorted highest to lowest by "
                 "mean |SHAP|; direction is which class this feature's presence/intensity typically "
                 "pushes the prediction toward, jointly with every other feature):\n")
    # column names avoid literal "|" -- df_to_markdown() doesn't escape pipes
    lines.append(df_to_markdown(imp[["label", "mean_abs_shap", "direction"]]
                                  .rename(columns={"mean_abs_shap": "mean abs SHAP"})))
    return "\n".join(lines)


# --- Report assembly ---------------------------------------------------------
def build_report(run_id: str, all_images: dict, per_image_meta: dict, chosen_df: pd.DataFrame) -> str:
    """The full-detail report -- see build_feature_importance_report() for
    the primary, significance-filtered, effect-size-sorted report meant to
    be read on its own. `chosen_df` must already carry a canonical_label
    column."""
    label_df = label_stats_table(run_id, all_images, chosen_df)
    image_df = image_summary_table(run_id, all_images, per_image_meta)

    lines = [f"# Analysis report (full detail): {run_id}", "",
             "See **`feature_importance.md`** (same folder) for the primary, significance-filtered "
             "report this backs -- each test's labels ranked by their own real effect size, plus the "
             "Random Forest + SHAP feature-combination stage. Everything below is the full "
             "statistical detail: every label regardless of significance, full methodology.",
             "\n## Per-image summary\n", df_to_markdown(image_df)]

    rounded = label_df[DISPLAY_COLS].round(3) if len(label_df) else pd.DataFrame(columns=DISPLAY_COLS)

    lines.append(f"\n## Presence (labels seen >= {MIN_OCCURRENCES} times)\n")
    lines.append(PRESENCE_EXPLANATION + "\n")
    lines.append(df_to_markdown(_presence_table(rounded)) if len(rounded) else "(no label met the threshold)")

    for metric, title in [("mass_share", "Mass Share"), ("peak_ratio", "Peak Ratio")]:
        lines.append(f"\n## {title} (labels seen >= {MIN_OCCURRENCES} times)\n")
        lines.append(METRIC_EXPLANATION + "\n")
        lines.append(df_to_markdown(_metric_table(rounded, metric)) if len(rounded) else "(no label met the threshold)")

    lines.append("\n## Key Findings\n")
    lines.append(generate_key_findings(rounded))
    return "\n".join(lines)


# --- Label -> image backtrack -------------------------------------------------
def build_label_membership(chosen_df: pd.DataFrame, all_images: dict, resolve_image_path) -> dict:
    by_label = defaultdict(list)
    for _, row in chosen_df.iterrows():
        iid = row["image_id"]
        by_label[row["canonical_label"]].append({
            "image_id": iid, "pass_fail": all_images.get(iid, "?"), "raw_label": row["label"],
            "mass_share": round(float(row["mass_share"]), 4), "peak_ratio": round(float(row["peak_ratio"]), 4),
            "image_path": resolve_image_path(iid),
        })
    for members in by_label.values():
        members.sort(key=lambda m: -m["mass_share"])

    by_image = defaultdict(list)
    for label, members in by_label.items():
        for m in members:
            by_image[m["image_id"]].append({"label": label, "mass_share": m["mass_share"],
                                              "peak_ratio": m["peak_ratio"], "pass_fail": m["pass_fail"]})
    for members in by_image.values():
        members.sort(key=lambda m: -m["mass_share"])
    return {"by_label": dict(by_label), "by_image": dict(by_image)}


MEMBERSHIP_TOP_N = 5


def write_label_membership(run_id: str, membership: dict, out_dir, total_occurrences: dict = None) -> None:
    """total_occurrences (optional): {canonical_label: n_total_detections},
    including non-significant ones -- the low-failure branch already has
    this (label_df's n_total_occurrences, see low_failure_stats.add_consistency)
    and passes it through so each label's header can show "X significant
    out of Y total" instead of a bare count that reads as "every occurrence".
    None (the normal pass/fail branch, which has no such total) just omits
    the "out of Y total" part -- the SIGNIFICANT-ONLY caveat in the intro
    text still applies either way, since membership itself is always built
    from chosen_df, which is always pre-filtered to is_significant==True."""
    import json
    (out_dir / "label_membership.json").write_text(json.dumps(membership, indent=2))

    lines = [f"# Label -> Image backtrack: {run_id}", "",
              "**Only SIGNIFICANT occurrences are listed here** -- every entry (in this file and in "
              "label_membership.json) individually passed the per-image permutation-test significance "
              "filter (is an object's attention higher than a random same-size region of that same "
              "image, tested separately per detection). An image can contain a label without "
              "appearing here AT ALL, if that particular detection didn't individually clear that bar "
              "-- this is not a list of every image the label was ever detected in.\n\n"
              "For each label (including \"Unknown Background\" -- attention that landed outside any "
              f"detected object), this file shows only the top {MEMBERSHIP_TOP_N} (by mass_share) -- "
              "open image_path directly to manually inspect what's actually there. Every SIGNIFICANT "
              "occurrence (not capped at top 5, but still significant-only, per the caveat above) is "
              "in label_membership.json alongside this file.", ""]
    for label, members in sorted(membership["by_label"].items(), key=lambda kv: -len(kv[1])):
        # Distinct images, not raw member rows -- an image can hold 2+
        # separate significant boxes of the same label, and total_occurrences
        # (from label_df, see add_consistency) is already an image count, not
        # a row count. Comparing row-count-significant against image-count-
        # total would print nonsense like "4408 significant out of 4170
        # total" (real bug this replaced) -- both sides need the same unit.
        n_sig_images = len({m["image_id"] for m in members})
        total = total_occurrences.get(label) if total_occurrences else None
        count_desc = (f"{n_sig_images} significant out of {total} total image(s)" if total is not None
                      else f"{n_sig_images} significant image(s) ({len(members)} detection(s))")
        lines.append(f"## {label} ({count_desc})\n")
        lines.append("| image_id | pass/fail | mass_share | peak_ratio | image_path |")
        lines.append("|---|---|---|---|---|")
        for m in members[:MEMBERSHIP_TOP_N]:
            lines.append(f"| {m['image_id']} | {m['pass_fail']} | {m['mass_share']} | "
                          f"{m['peak_ratio']} | `{m['image_path'] or 'NOT FOUND'}` |")
        lines.append("")
    (out_dir / "label_membership.md").write_text("\n".join(lines))
