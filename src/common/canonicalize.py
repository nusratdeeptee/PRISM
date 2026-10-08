"""Label canonicalization: groups near-duplicate object labels (e.g. "traffic
light"/"traffic signal", "car"/"car car") into one canonical name, fully
deterministic and dataset-agnostic.

Two merge rules only, deliberately narrow -- broader alternatives (embedding
clustering, substring containment, checking every WordNet sense, an LLM
merge-refinement pass, an NLI cross-encoder) were tried and each produced
real false merges on live data (e.g. embedding clustering merged "chicken"
into "rooster"; substring containment merged "car" into "car bumper";
checking every WordNet sense merged "shadow" into "tail" via an idiomatic
sense neither word is actually used for here). These two rules produce no
known false merge on any dataset tested so far:
  1. Same PRIMARY WordNet noun sense (synsets()[0] -- a word's single most
     common meaning, not any sense it happens to carry).
  2. Same SET of distinct words, order/repetition aside ("car"/"car car").
The real, accepted cost: domain-specific near-duplicates with no WordNet
entry (e.g. "coca-cola truck"/"coca-cola van") are not merged.
"""
import re
from collections import Counter, defaultdict


def normalize_label(label: str) -> str:
    """Formatting only (not synonym grouping) -- collapses whitespace and
    stray hyphen-spacing artifacts from the detector's phrase decoding."""
    s = label.strip().lower()
    s = re.sub(r"\s*-\s*", "-", s)
    return re.sub(r"\s+", " ", s)


def wordnet_primary_synset(label: str):
    from nltk.corpus import wordnet as wn
    synsets = wn.synsets(label.replace(" ", "_"), pos=wn.NOUN)
    return synsets[0].name() if synsets else None


def same_word_set(a: str, b: str) -> bool:
    return set(a.split()) == set(b.split())


def _union_find_groups(items: list, should_merge) -> dict:
    parent = {x: x for x in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(items):
        for b in items[i + 1:]:
            merge, reason = should_merge(a, b)
            if merge:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
                print(f"  [{reason}] {a!r} <-> {b!r}")
    return {x: find(x) for x in items}


def build_canonical_map(raw_labels: list) -> dict:
    """Call once with labels pooled across every dataset in a run (not
    per-dataset), so the same word canonicalizes identically everywhere.
    "unclaimed" (background attention outside any detected object) is
    renamed to "Unknown Background" for display only, after normal
    canonicalization -- it still has to clear the same significance/
    occurrence bar as any other label to reach the final report."""
    normalized = [normalize_label(l) for l in raw_labels]
    unique_labels = sorted(set(normalized))
    occurrence_counts = Counter(normalized)
    if len(unique_labels) <= 1:
        return {l: l for l in unique_labels}

    primary_synset = {l: wordnet_primary_synset(l) for l in unique_labels}

    def should_merge(a, b):
        if primary_synset[a] is not None and primary_synset[a] == primary_synset[b]:
            return True, f"wordnet ({primary_synset[a]})"
        if same_word_set(a, b):
            return True, "dup-wordset"
        return False, None

    roots = _union_find_groups(unique_labels, should_merge)
    groups = defaultdict(list)
    for l, root in roots.items():
        groups[root].append(l)

    canonical_map = {}
    for members in groups.values():
        canonical = max(members, key=lambda l: occurrence_counts[l])
        for l in members:
            canonical_map[l] = canonical
    if "unclaimed" in canonical_map:
        for l in groups[roots["unclaimed"]]:
            canonical_map[l] = "Unknown Background"
    return canonical_map


def write_canonicalization_trace(all_raw_labels: list, canonical_map: dict, out_dir) -> None:
    """One row per distinct raw label, sorted by canonical name so merged
    labels sit next to each other -- lets a reader verify Stage 1's grouping
    by eye."""
    import pandas as pd
    normalized = [normalize_label(l) for l in all_raw_labels]
    occurrence_counts = Counter(normalized)
    rows = [{"raw_label_before_canonicalization": raw, "occurrences": occurrence_counts[raw],
             "canonical_label": canonical_map[raw]} for raw in sorted(set(normalized))]
    df = pd.DataFrame(rows).sort_values(["canonical_label", "raw_label_before_canonicalization"])
    df.to_csv(out_dir / "label_canonicalization.csv", index=False)

    from .stats_tables import df_to_markdown
    md = ["# Label canonicalization trace", "",
          "One row per distinct raw label seen across every requested dataset, before/after "
          "grouping -- rows sort so labels that end up grouped together sit next to each other.",
          "", df_to_markdown(df)]
    (out_dir / "label_canonicalization.md").write_text("\n".join(md))
