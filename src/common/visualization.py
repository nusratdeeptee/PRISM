"""Attribution heatmap rendering."""
import numpy as np
import matplotlib.cm as cm


def overlay_heatmap_on_image(image_np: np.ndarray, attribution_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """attribution_map must already be normalized to [0, 1]. Returns a uint8
    (H, W, 3) array of the image blended with a jet-colormap heatmap."""
    heatmap = (cm.jet(attribution_map)[:, :, :3] * 255).astype(np.uint8)
    overlay = ((1 - alpha) * image_np.astype(np.float32) + alpha * heatmap.astype(np.float32))
    return overlay.clip(0, 255).astype(np.uint8)


def normalize_saliency(saliency: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Collapses raw saliency to [0, 1] by clipping at a high percentile
    (not the max) so one outlier pixel doesn't wash out the whole map."""
    hi = float(np.percentile(saliency, percentile))
    if hi < 1e-8:
        return np.zeros_like(saliency, dtype=np.float32)
    return np.clip(saliency / hi, 0.0, 1.0).astype(np.float32)
