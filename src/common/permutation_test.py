"""Per-object significance test: is an object's mean saliency higher than a
random same-size region of the same image would get by chance?"""
import numpy as np

N_PERMUTATIONS = 1000
SIGNIFICANCE_ALPHA = 0.05


class PermutationEngine:
    """Shuffle the image's flattened saliency ONCE and take its cumulative
    sum -- after that single O(H*W) setup, the mean of any random contiguous
    window of any length is an O(1) lookup, so testing every object in an
    image (each a different mask size) only costs one shuffle total, not one
    per object per permutation sample."""

    def __init__(self, saliency: np.ndarray, seed: int):
        flat = np.asarray(saliency, dtype=np.float64).ravel()
        rng = np.random.default_rng(seed)
        self.shuffled = rng.permutation(flat)
        self.cumsum = np.concatenate([[0.0], np.cumsum(self.shuffled)])
        self.n = flat.size
        self.rng = rng

    def pvalue(self, real_mean: float, size: int) -> float:
        if size <= 0 or size >= self.n or self.cumsum[-1] <= 0.0:
            return 1.0
        starts = self.rng.integers(0, self.n - size + 1, size=N_PERMUTATIONS)
        window_means = (self.cumsum[starts + size] - self.cumsum[starts]) / size
        return float((np.sum(window_means >= real_mean) + 1) / (N_PERMUTATIONS + 1))


def add_permutation_pvalues(stats_rows: list, saliency: np.ndarray, objects: list, seed: int) -> None:
    """Mutates stats_rows in place, adding permutation_pvalue/is_significant
    to every row including the last (synthetic UNCLAIMED) row -- background
    attention is tested the same way as any named object, not skipped."""
    engine = PermutationEngine(saliency, seed)
    claimed = np.zeros(saliency.shape, dtype=bool)
    for row, obj in zip(stats_rows, objects):
        mask = obj["mask"]
        claimed |= mask
        real_mean = float(saliency[mask].mean()) if mask.any() else 0.0
        p = engine.pvalue(real_mean, int(mask.sum()))
        row["permutation_pvalue"] = p
        row["is_significant"] = p < SIGNIFICANCE_ALPHA

    unclaimed_mask = ~claimed
    real_mean = float(saliency[unclaimed_mask].mean()) if unclaimed_mask.any() else 0.0
    p = engine.pvalue(real_mean, int(unclaimed_mask.sum()))
    stats_rows[-1]["permutation_pvalue"] = p
    stats_rows[-1]["is_significant"] = p < SIGNIFICANCE_ALPHA
