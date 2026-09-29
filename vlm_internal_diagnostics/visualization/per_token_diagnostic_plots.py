"""Diagnostic plots: temporal variance, center-vs-clear difference, head specialization."""
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


def _stack_token_attention_across_frames(payloads, position, layer=None):
    """Return np.ndarray [F, P] for one token across frames (avg across layers if layer is None)."""
    out = []
    for p in payloads:
        mats = p.get("per_layer_matrices_head_avg") or {}
        if not mats:
            continue
        if layer is not None and layer in mats:
            m = _to_np(mats[layer])
        else:
            m = np.stack([_to_np(v) for v in mats.values()], axis=0).mean(axis=0)
        if 0 <= position < m.shape[0]:
            out.append(m[position])
    if not out:
        return None
    return np.stack(out, axis=0)


def _reshape_to_grid(vec, grid_shape):
    n = vec.size
    if grid_shape is None or grid_shape[0] * grid_shape[1] != n:
        grid_shape = infer_grid_shape(n)
    h, w = grid_shape
    if h * w != n:
        h = w = int(round(np.sqrt(n)))
        vec = vec[: h * w]
    return vec.reshape(h, w), (h, w)


def _normalize(g):
    mn, mx = g.min(), g.max()
    return (g - mn) / (mx - mn) if mx > mn else np.zeros_like(g)


def _center_image(seq_records, max_amb_idx):
    for r in seq_records:
        if r["frame_index"] == max_amb_idx:
            try:
                return np.asarray(Image.open(r["image_path"]).convert("RGB"))
            except FileNotFoundError:
                break
    # fallback: middle frame
    mid = seq_records[len(seq_records) // 2]
    return np.asarray(Image.open(mid["image_path"]).convert("RGB"))


def render_temporal_variance_map(seq_records, payloads, selected_tokens, out_path,
                                 layer=None, colormap="viridis", alpha=0.55):
    """Per-patch variance across frames, averaged across selected tokens.

    Overlaid on the center frame image.
    """
    if not selected_tokens:
        return None
    max_amb = seq_records[0]["max_ambiguity_index"]
    img = _center_image(seq_records, max_amb)
    grid_shape = tuple(seq_records[0].get("image_grid_shape") or [])
    if len(grid_shape) != 2:
        grid_shape = None

    per_token_var = []
    for tok in selected_tokens:
        pos = getattr(tok, "position", None) if not isinstance(tok, dict) else tok.get("position")
        stack = _stack_token_attention_across_frames(payloads, pos, layer=layer)
        if stack is None or stack.shape[0] < 2:
            continue
        per_token_var.append(stack.var(axis=0))  # [P]
    if not per_token_var:
        return None
    mean_var = np.mean(np.stack(per_token_var, axis=0), axis=0)

    grid, (h, w) = _reshape_to_grid(mean_var, grid_shape)
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.imshow(_normalize(grid), cmap=colormap, alpha=alpha,
              extent=(0, W, H, 0), interpolation="bilinear")
    tok_names = ", ".join(
        (getattr(t, "text", None) if not isinstance(t, dict) else t.get("text"))
        for t in selected_tokens
    )
    ax.set_title(f"temporal variance over selected tokens [{tok_names}]", fontsize=9)
    ax.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def render_center_vs_clear_diff(seq_records, payloads, selected_tokens, out_path,
                                num_farthest_frames=4, layer=None,
                                colormap="seismic", alpha=0.55):
    """Difference map: attention at center frame minus mean attention at the
    `num_farthest_frames` frames furthest from max_ambiguity_index, averaged
    across selected tokens. Overlaid on center frame image.
    """
    if not selected_tokens:
        return None
    max_amb = seq_records[0]["max_ambiguity_index"]

    # Indices of frames in seq_records sorted by distance from center (descending)
    frame_indices = [r["frame_index"] for r in seq_records]
    dists = [abs(fi - max_amb) for fi in frame_indices]
    sorted_by_far = sorted(range(len(seq_records)), key=lambda i: -dists[i])
    far_idx = sorted_by_far[:num_farthest_frames]
    center_pos = None
    for i, r in enumerate(seq_records):
        if r["frame_index"] == max_amb:
            center_pos = i
            break
    if center_pos is None:
        return None

    img = _center_image(seq_records, max_amb)
    grid_shape = tuple(seq_records[0].get("image_grid_shape") or [])
    if len(grid_shape) != 2:
        grid_shape = None

    diff_per_token = []
    for tok in selected_tokens:
        pos = getattr(tok, "position", None) if not isinstance(tok, dict) else tok.get("position")
        # center attention
        c_payload = payloads[center_pos]
        c_mats = c_payload.get("per_layer_matrices_head_avg") or {}
        if not c_mats:
            continue
        if layer is not None and layer in c_mats:
            c_attn = _to_np(c_mats[layer])
        else:
            c_attn = np.stack([_to_np(v) for v in c_mats.values()], axis=0).mean(axis=0)
        if pos >= c_attn.shape[0]:
            continue
        c_vec = c_attn[pos]

        far_vecs = []
        for fi in far_idx:
            p = payloads[fi]
            mats = p.get("per_layer_matrices_head_avg") or {}
            if not mats:
                continue
            if layer is not None and layer in mats:
                m = _to_np(mats[layer])
            else:
                m = np.stack([_to_np(v) for v in mats.values()], axis=0).mean(axis=0)
            if pos < m.shape[0]:
                far_vecs.append(m[pos])
        if not far_vecs:
            continue
        clear_mean = np.mean(np.stack(far_vecs, axis=0), axis=0)
        diff_per_token.append(c_vec - clear_mean)

    if not diff_per_token:
        return None
    diff = np.mean(np.stack(diff_per_token, axis=0), axis=0)

    grid, (h, w) = _reshape_to_grid(diff, grid_shape)
    # symmetric normalization around 0 for seismic colormap
    m = max(abs(grid.min()), abs(grid.max()))
    grid_n = grid / m if m > 0 else grid

    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.imshow(grid_n, cmap=colormap, alpha=alpha, vmin=-1, vmax=1,
              extent=(0, W, H, 0), interpolation="bilinear")
    ax.set_title(f"center - mean(top-{num_farthest_frames} farthest)  (red=center higher)",
                 fontsize=9)
    ax.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def render_head_specialization_heatmap(payloads, selected_tokens, out_path,
                                       max_amb_idx=None, frame_records=None):
    """For each selected token, plot per-head entropy across layers as a small heatmap.

    Uses the per-head stats computed at extraction time. Aggregates by averaging
    across frames (so it's one heatmap per token: rows = heads, cols = layers).
    """
    if not selected_tokens:
        return None

    # Stack per-head entropy across frames for each token: [F, L, H]
    layers = None
    stacks = []
    for tok in selected_tokens:
        pos = getattr(tok, "position", None) if not isinstance(tok, dict) else tok.get("position")
        per_frame = []
        for p in payloads:
            stats = p.get("per_head_stats") or {}
            ent = stats.get("entropy")
            if ent is None:
                continue
            ent_np = _to_np(ent)  # [L, H, T]
            if layers is None:
                layers = p.get("layers")
            if pos >= ent_np.shape[2]:
                per_frame.append(None)
            else:
                per_frame.append(ent_np[:, :, pos])  # [L, H]
        per_frame = [x for x in per_frame if x is not None]
        if not per_frame:
            stacks.append(None)
            continue
        stacks.append(np.stack(per_frame, axis=0).mean(axis=0))  # [L, H]

    if not any(s is not None for s in stacks):
        return None

    n_tok = len(selected_tokens)
    fig, axes = plt.subplots(1, n_tok, figsize=(3.0 * n_tok, 3.2), squeeze=False)
    for j, (tok, s) in enumerate(zip(selected_tokens, stacks)):
        ax = axes[0, j]
        text = getattr(tok, "text", "?") if not isinstance(tok, dict) else tok.get("text", "?")
        if s is None:
            ax.axis("off")
            ax.set_title(f"{text}\n(no data)", fontsize=9)
            continue
        # s is [L, H]; transpose to put heads on y so x=layers reads naturally
        im = ax.imshow(s.T, aspect="auto", cmap="magma")
        ax.set_title(text, fontsize=9)
        ax.set_xlabel("layer idx (in saved list)")
        ax.set_ylabel("head")
        if layers is not None:
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([str(l) for l in layers], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Per-head attention entropy (rows=heads, cols=layers; avg over frames)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
