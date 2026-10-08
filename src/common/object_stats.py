"""Per-object attention statistics: given object masks and a saliency map for
the same image, compute how much of the model's attention landed on each
object."""
from typing import Optional

import numpy as np

TOP_K_PEAK = 20  # pixels averaged for a "robust peak" instead of a single
                 # max pixel, which can be an interpolation/gradient artifact


def _top_k_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return 0.0
    k = min(k, values.size)
    return float(np.partition(values, -k)[-k:].mean())


def compute_object_stats(objects: list, saliency: np.ndarray, top_k_peak: int = TOP_K_PEAK) -> dict:
    """objects: list of {"label", "score", "mask": bool array same shape as
    saliency}. Returns "rows" (one dict per object, in order, plus a final
    synthetic "UNCLAIMED" row for saliency outside every object's mask) each
    with area_share, mass_share (share of total attribution mass), peak_ratio
    (this object's robust peak / the whole image's robust peak). Also returns
    global_peak, overlap_mass_share (mass under 2+ overlapping masks -- since
    mass_share double-counts overlaps, this flags how much), and degenerate
    (True if the saliency map sums to 0, in which case every mass/peak value
    is None rather than a misleading number)."""
    saliency = np.asarray(saliency, dtype=np.float64)
    h, w = saliency.shape
    total_px = h * w
    total_mass = float(saliency.sum())
    degenerate = total_mass <= 0.0
    global_peak = 0.0 if degenerate else _top_k_mean(saliency.ravel(), top_k_peak)

    for obj in objects:
        if obj["mask"].shape != saliency.shape:
            raise ValueError(f"mask shape {obj['mask'].shape} != saliency shape {saliency.shape} "
                              f"for object {obj.get('label')!r}")

    def _row(label: str, score: Optional[float], mask: np.ndarray) -> dict:
        mask = mask.astype(bool)
        area_px = int(mask.sum())
        area_share = area_px / total_px if total_px > 0 else 0.0
        if degenerate:
            mass, mass_share, peak, peak_ratio = None, None, None, None
        else:
            mass = float(saliency[mask].sum())
            mass_share = mass / total_mass
            peak = _top_k_mean(saliency[mask], top_k_peak) if area_px > 0 else 0.0
            peak_ratio = (peak / global_peak) if global_peak > 0 else None
        return {"label": label, "score": score, "area_px": area_px, "area_share": area_share,
                "mass": mass, "mass_share": mass_share, "peak": peak, "peak_ratio": peak_ratio}

    rows = [_row(o["label"], o.get("score"), o["mask"]) for o in objects]

    if objects:
        stacked = np.stack([o["mask"].astype(bool) for o in objects], axis=0)
        union_mask = stacked.any(axis=0)
        overlap_mask = stacked.sum(axis=0) >= 2
    else:
        union_mask = np.zeros_like(saliency, dtype=bool)
        overlap_mask = np.zeros_like(saliency, dtype=bool)

    rows.append(_row("UNCLAIMED", None, ~union_mask))
    overlap_mass_share = (0.0 if degenerate or not objects
                           else float(saliency[overlap_mask].sum()) / total_mass)

    return {"rows": rows, "total_mass": total_mass, "total_px": total_px,
            "global_peak": global_peak, "overlap_mass_share": overlap_mass_share,
            "degenerate": degenerate}
