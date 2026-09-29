"""6 row x 10 col sequence grid: image / attention overlay / answer summary / per-token rows.

Rows 1-3 are unchanged (frame thumbnail, head-avg attention overlay, text).
Rows 4..(3+K) show per-token attention overlays for the top-K selected tokens
(K controlled by visualization config). If selected_tokens is None or empty,
only the original 3 rows are rendered.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from .attention_overlay import _load_attn_map
from .per_token_attention_overlay import load_per_token_payload, get_token_attention_vector
from ..models.token_index_utils import infer_grid_shape


def render_sequence_grid(seq_records, out_path, mark_ambiguity_center=True,
                         colormap="jet", alpha=0.45,
                         selected_tokens=None, top_k_token_rows=0,
                         per_token_layer=None):
    """Render the sequence grid.

    selected_tokens: list of SelectedToken (or dicts) — same selection across all frames.
    top_k_token_rows: number of extra rows to add (top-K from selected_tokens).
    """
    n = len(seq_records)
    extra = min(top_k_token_rows, len(selected_tokens)) if selected_tokens else 0
    rows = 3 + extra
    fig, axes = plt.subplots(rows, n, figsize=(2.0 * n, 2.0 * rows + 0.5))
    if n == 1:
        axes = np.array(axes).reshape(rows, 1)
    if rows == 1:
        axes = axes.reshape(1, -1)
    max_idx = seq_records[0].get("max_ambiguity_index")

    # Preload per-token payloads if we'll need them
    per_token_payloads = [None] * n
    if extra > 0:
        for j, rec in enumerate(seq_records):
            p_path = rec.get("feature_paths", {}).get("per_token_attention_map")
            if p_path:
                try:
                    per_token_payloads[j] = load_per_token_payload(p_path)
                except Exception:
                    per_token_payloads[j] = None

    for j, rec in enumerate(seq_records):
        try:
            img = np.asarray(Image.open(rec["image_path"]).convert("RGB"))
        except FileNotFoundError:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        H, W = img.shape[:2]

        # Row 0: thumbnail
        ax_img = axes[0, j]
        ax_img.imshow(img); ax_img.axis("off")
        ax_img.set_title(f"f{rec['frame_index']}", fontsize=8)
        if mark_ambiguity_center and rec["frame_index"] == max_idx:
            for spine in ax_img.spines.values():
                spine.set_visible(True); spine.set_edgecolor("red"); spine.set_linewidth(3)

        # Row 1: head-averaged (legacy) attention overlay
        ax_attn = axes[1, j]
        attn_path = rec.get("feature_paths", {}).get("attention_map")
        if attn_path:
            try:
                a = _load_attn_map(attn_path)
                grid_shape = tuple(rec.get("image_grid_shape") or [])
                if len(grid_shape) != 2:
                    grid_shape = infer_grid_shape(a.size)
                h, w = grid_shape
                if h * w == a.size:
                    g = a.reshape(h, w)
                    mn, mx = g.min(), g.max()
                    g = (g - mn) / (mx - mn) if mx > mn else g
                    ax_attn.imshow(img)
                    ax_attn.imshow(g, cmap=colormap, alpha=alpha,
                                   extent=(0, W, H, 0), interpolation="bilinear")
            except Exception:
                ax_attn.imshow(img)
        else:
            ax_attn.imshow(img)
        ax_attn.axis("off")
        if j == 0:
            ax_attn.set_ylabel("avg attn", fontsize=8)

        # Row 2: answer / margin / entropy text
        ax_txt = axes[2, j]
        ax_txt.axis("off")
        ans = rec.get("parsed_answer", "?")
        margin = rec.get("logit_margin_yes_no", float("nan"))
        h_ent = rec.get("binary_entropy", float("nan"))
        ax_txt.text(0.5, 0.7, f"ans={ans}", ha="center", va="center", fontsize=8)
        ax_txt.text(0.5, 0.4, f"m={margin:.2f}", ha="center", va="center", fontsize=8)
        ax_txt.text(0.5, 0.1, f"H={h_ent:.2f}", ha="center", va="center", fontsize=8)

        # Rows 3..3+extra-1: per-token attention overlays
        for k in range(extra):
            ax_tk = axes[3 + k, j]
            tok = selected_tokens[k]
            payload = per_token_payloads[j]
            if payload is None:
                ax_tk.imshow(img); ax_tk.axis("off")
                continue
            pos = getattr(tok, "position", None) if not isinstance(tok, dict) else tok.get("position")
            vec = get_token_attention_vector(payload, pos, layer=per_token_layer)
            if vec is None:
                ax_tk.imshow(img); ax_tk.axis("off")
                continue
            grid_shape = tuple(rec.get("image_grid_shape") or [])
            if len(grid_shape) != 2 or grid_shape[0] * grid_shape[1] != vec.size:
                grid_shape = infer_grid_shape(vec.size)
            h, w = grid_shape
            g = vec[: h * w].reshape(h, w)
            mn, mx = g.min(), g.max()
            g = (g - mn) / (mx - mn) if mx > mn else np.zeros_like(g)
            ax_tk.imshow(img)
            ax_tk.imshow(g, cmap=colormap, alpha=alpha,
                         extent=(0, W, H, 0), interpolation="bilinear")
            ax_tk.axis("off")
            if j == 0:
                text = getattr(tok, "text", "?") if not isinstance(tok, dict) else tok.get("text", "?")
                ax_tk.set_ylabel(text, fontsize=8, rotation=0, labelpad=24, va="center")

    fig.suptitle(f"{seq_records[0]['sequence_id']}  | {seq_records[0].get('question','')}",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
