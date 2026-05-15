"""3 row x 10 col sequence grid: image / attention overlay / answer summary."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from .attention_overlay import _load_attn_map
from ..models.token_index_utils import infer_grid_shape


def render_sequence_grid(seq_records, out_path, mark_ambiguity_center=True,
                         colormap="jet", alpha=0.45):
    n = len(seq_records)
    fig, axes = plt.subplots(3, n, figsize=(2.0 * n, 6.5))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)
    max_idx = seq_records[0].get("max_ambiguity_index")

    for j, rec in enumerate(seq_records):
        try:
            img = np.asarray(Image.open(rec["image_path"]).convert("RGB"))
        except FileNotFoundError:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        H, W = img.shape[:2]

        ax_img = axes[0, j]
        ax_img.imshow(img); ax_img.axis("off")
        ax_img.set_title(f"f{rec['frame_index']}", fontsize=8)
        if mark_ambiguity_center and rec["frame_index"] == max_idx:
            for spine in ax_img.spines.values():
                spine.set_visible(True); spine.set_edgecolor("red"); spine.set_linewidth(3)

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
                    ax_attn.imshow(g, cmap=colormap, alpha=alpha, extent=(0, W, H, 0), interpolation="bilinear")
            except Exception:
                ax_attn.imshow(img)
        else:
            ax_attn.imshow(img)
        ax_attn.axis("off")

        ax_txt = axes[2, j]
        ax_txt.axis("off")
        ans = rec.get("parsed_answer", "?")
        margin = rec.get("logit_margin_yes_no", float("nan"))
        h_ent = rec.get("binary_entropy", float("nan"))
        ax_txt.text(0.5, 0.7, f"ans={ans}", ha="center", va="center", fontsize=8)
        ax_txt.text(0.5, 0.4, f"m={margin:.2f}", ha="center", va="center", fontsize=8)
        ax_txt.text(0.5, 0.1, f"H={h_ent:.2f}", ha="center", va="center", fontsize=8)

    fig.suptitle(f"{seq_records[0]['sequence_id']}  | {seq_records[0].get('question','')}", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
