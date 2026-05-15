"""Per-sequence trajectory plots: margin, entropy, attention metrics vs frame."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def render_trajectory_plot(seq_records, out_path):
    fi = [r["frame_index"] for r in seq_records]
    margins = [r.get("logit_margin_yes_no", float("nan")) for r in seq_records]
    bin_ent = [r.get("binary_entropy", float("nan")) for r in seq_records]
    max_idx = seq_records[0].get("max_ambiguity_index")
    attn_ent_first = []
    for r in seq_records:
        a = r.get("attention_entropy", {})
        if a:
            attn_ent_first.append(list(a.values())[0])
        else:
            attn_ent_first.append(float("nan"))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(fi, margins, "-o"); axes[0, 0].set_title("yes/no margin")
    axes[0, 0].axhline(0, color="gray", linestyle="--", lw=0.5)
    axes[0, 1].plot(fi, bin_ent, "-o", color="orange"); axes[0, 1].set_title("binary entropy")
    axes[1, 0].plot(fi, attn_ent_first, "-o", color="green"); axes[1, 0].set_title("attention entropy")
    axes[1, 1].plot(fi, [abs(i - max_idx) for i in fi], "-o", color="purple")
    axes[1, 1].set_title("ambiguity distance")
    for ax in axes.ravel():
        ax.axvline(max_idx, color="red", linestyle=":", lw=1, alpha=0.6)
        ax.set_xlabel("frame_index")
    fig.suptitle(seq_records[0]["sequence_id"])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
