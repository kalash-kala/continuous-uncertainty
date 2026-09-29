"""Per-question-token attention overlays.

For each frame, render a 1-row x N-column grid where each subplot shows the
attention map of one selected question token overlaid on the frame image.
Uses the head-averaged matrix saved in `frame_XX_per_token_attention.pt`.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from ..models.token_index_utils import infer_grid_shape


def _to_np(x):
    if hasattr(x, "float") and hasattr(x, "numpy"):
        return x.float().cpu().numpy()
    return np.asarray(x)


def load_per_token_payload(path):
    """Load the per-token attention .pt file."""
    return torch.load(path, map_location="cpu")


def get_token_attention_vector(payload, position, layer=None):
    """Return a 1D np.ndarray of length num_image_patches for the given token row.

    layer: if None, average across all saved layers; else pick that layer.
    """
    mats = payload.get("per_layer_matrices_head_avg") or {}
    if not mats:
        return None
    if layer is not None and layer in mats:
        m = _to_np(mats[layer])
    else:
        stacked = np.stack([_to_np(v) for v in mats.values()], axis=0)
        m = stacked.mean(axis=0)
    if position < 0 or position >= m.shape[0]:
        return None
    return m[position]


def _overlay_axis(ax, img_arr, attn_1d, grid_shape, colormap, alpha, draw_boundaries=False):
    H, W = img_arr.shape[:2]
    a = np.asarray(attn_1d, dtype=np.float64).ravel()
    n = a.size
    if grid_shape is None or grid_shape[0] * grid_shape[1] != n:
        grid_shape = infer_grid_shape(n)
    h, w = grid_shape
    if h * w != n:
        h = w = int(round(np.sqrt(n)))
        a = a[: h * w]
    g = a.reshape(h, w)
    mn, mx = g.min(), g.max()
    g_n = (g - mn) / (mx - mn) if mx > mn else np.zeros_like(g)
    ax.imshow(img_arr)
    ax.imshow(g_n, cmap=colormap, alpha=alpha, extent=(0, W, H, 0), interpolation="bilinear")
    if draw_boundaries:
        for i in range(1, h):
            ax.axhline(i * H / h, color="white", linewidth=0.3, alpha=0.4)
        for j in range(1, w):
            ax.axvline(j * W / w, color="white", linewidth=0.3, alpha=0.4)
    ax.axis("off")


def render_per_frame_token_grid(image, payload, selected_tokens, out_path,
                                layer=None, colormap="jet", alpha=0.45,
                                draw_patch_boundaries=False, title=None):
    """One frame, one PNG: row of subplots, one per selected token.

    selected_tokens: iterable of objects with `.position`, `.text`, `.pos` attrs
        (or dicts with the same keys).
    """
    if not selected_tokens:
        return None
    img_arr = np.asarray(image.convert("RGB"))
    grid_shape = tuple(payload.get("image_grid_shape") or [])
    if len(grid_shape) != 2:
        grid_shape = None

    n = len(selected_tokens)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = np.array([axes])

    for ax, tok in zip(axes, selected_tokens):
        pos = getattr(tok, "position", None) if not isinstance(tok, dict) else tok.get("position")
        text = getattr(tok, "text", "?") if not isinstance(tok, dict) else tok.get("text", "?")
        pos_tag = getattr(tok, "pos", None) if not isinstance(tok, dict) else tok.get("pos")
        a = get_token_attention_vector(payload, pos, layer=layer)
        if a is None:
            ax.imshow(img_arr); ax.axis("off")
            ax.set_title(f"{text} (no data)", fontsize=9)
            continue
        _overlay_axis(ax, img_arr, a, grid_shape, colormap, alpha, draw_patch_boundaries)
        ttl = f"{text}" + (f" [{pos_tag}]" if pos_tag else "")
        ax.set_title(ttl, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92] if title else None)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
