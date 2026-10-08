"""Shared stratified-sampling helper. Used by generate_analysis.py's
resolve_sample_ids() (subsampling an already-computed pool for a smaller
report -- jobs 8-11/16-19/24-27/33-36/42-45 style) and by
select_sample_ids.py (picking which images to compute segments/ig2/idg over
in the first place, from a dataset that hasn't been processed at all yet --
job 37 style). Both need the same "N images, same pass/fail ratio as the
source pool" logic, so it lives here once rather than twice."""


def stratified_sample_counts(pool_size_by_class: dict, sample_size: int) -> dict:
    """Apportions sample_size across classes (here: "pass"/"fail") proportional
    to each class's share of the pool -- x = sample_size / total_pool, each
    class gets round(x * its own pool size) -- via the largest-remainder
    method (floor each class's exact share, then hand out the leftover +1s to
    whichever classes have the largest fractional remainder). This guarantees
    sum(result.values()) == sample_size exactly (a plain independent round()
    per class can drift by +-1 off sample_size when the fractions don't
    happen to round the same way), and is deterministic given the pool sizes."""
    total_pool = sum(pool_size_by_class.values())
    exact = {k: sample_size * v / total_pool for k, v in pool_size_by_class.items()}
    counts = {k: int(v) for k, v in exact.items()}   # floor
    remainder = sample_size - sum(counts.values())
    # Hand out the leftover +1s to the classes with the largest fractional part, largest first.
    order = sorted(pool_size_by_class, key=lambda k: exact[k] - counts[k], reverse=True)
    for k in order[:remainder]:
        counts[k] += 1
    for k, n in counts.items():
        if n > pool_size_by_class[k]:
            raise SystemExit(f"stratified_sample_counts: class {k!r} needs {n} images but its pool "
                              f"only has {pool_size_by_class[k]} -- sample_size {sample_size} can't be "
                              f"met while keeping the pass/fail ratio")
    return counts
