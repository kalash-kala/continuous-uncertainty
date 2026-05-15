"""Attention heatmap overlays for individual frames."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


def _load_attn_map(path, layer=None):
    """Load attention .pt file and average over chosen layer(s)."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if layer is not None and layer in obj:
            arr = obj[layer]
        else:
            arr = torch.stack([v.float() for v in obj.values()], dim=0).mean(dim=0)
    else:
        arr = obj
    return np.asarray(arr.float().numpy() if hasattr(arr, "numpy") else arr)


def render_overlay(image, attn_1d, out_path, grid_shape=None,
                   colormap="jet", alpha=0.45, draw_patch_boundaries=True,
                   title=None):
    """Save an overlay PNG of attention on top of the image."""
    img = np.asarray(image.convert("RGB"))
    H, W = img.shape[:2]

    a = np.asarray(attn_1d, dtype=np.float64).ravel()
    n = a.size
    if grid_shape is None:
        from ..models.token_index_utils import infer_grid_shape
        grid_shape = infer_grid_shape(n)
    h, w = grid_shape
    if h * w != n:
        h = w = int(round(np.sqrt(n)))
        a = a[: h * w]
    grid = a.reshape(h, w)
    mn, mx = grid.min(), grid.max()
    if mx > mn:
        grid_n = (grid - mn) / (mx - mn)
    else:
        grid_n = np.zeros_like(grid)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.imshow(grid_n, cmap=colormap, alpha=alpha, extent=(0, W, H, 0), interpolation="bilinear")
    if draw_patch_boundaries and h > 0 and w > 0:
        for i in range(1, h):
            ax.axhline(i * H / h, color="white", linewidth=0.3, alpha=0.4)
        for j in range(1, w):
            ax.axvline(j * W / w, color="white", linewidth=0.3, alpha=0.4)
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def render_overlay_from_record(record, attention_layer=None,
                               colormap="jet", alpha=0.45,
                               draw_patch_boundaries=True, out_path=None):
    attn_path = record.get("feature_paths", {}).get("attention_map")
    if not attn_path:
        return None
    attn = _load_attn_map(attn_path, layer=attention_layer)
    image = Image.open(record["image_path"])
    title = (f"frame {record['frame_index']} | ans={record.get('parsed_answer','?')} | "
             f"margin={record.get('logit_margin_yes_no',float('nan')):.2f} | "
             f"H={record.get('binary_entropy', float('nan')):.2f}")
    grid_shape = tuple(record.get("image_grid_shape") or [])
    if len(grid_shape) != 2:
        grid_shape = None
    return render_overlay(image, attn, out_path,
                          grid_shape=grid_shape, colormap=colormap, alpha=alpha,
                          draw_patch_boundaries=draw_patch_boundaries, title=title)
