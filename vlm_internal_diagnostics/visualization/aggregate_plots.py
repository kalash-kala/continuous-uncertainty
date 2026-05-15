"""Aggregate plots: layerwise, category-wise scatter/bar plots."""
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def _cat_bar(ax, by_cat, title, ylabel):
    cats = sorted(by_cat.keys())
    vals = [np.nanmean(by_cat[c]) if by_cat[c] else float("nan") for c in cats]
    ax.bar(cats, vals); ax.set_title(title); ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)


def render_aggregate_plots(seq_summaries, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    by_cat_flip = defaultdict(list)
    by_cat_bndy = defaultdict(list)
    by_cat_attn = defaultdict(list)
    vision_jumps, proj_jumps = [], []
    layer_spearman = []

    for s in seq_summaries:
        c = s.get("category", "unknown")
        if "answer_flip_rate" in s: by_cat_flip[c].append(s["answer_flip_rate"])
        if "boundary_error" in s: by_cat_bndy[c].append(s["boundary_error"])
        if "mean_attention_jump" in s: by_cat_attn[c].append(s["mean_attention_jump"])
        if "mean_vision_adjacent_jump" in s and "mean_projector_adjacent_jump" in s:
            vision_jumps.append(s["mean_vision_adjacent_jump"])
            proj_jumps.append(s["mean_projector_adjacent_jump"])
        if "best_hidden_layer_by_spearman" in s and "best_hidden_center_distance_spearman" in s:
            layer_spearman.append((s["best_hidden_layer_by_spearman"], s["best_hidden_center_distance_spearman"]))

    fig, ax = plt.subplots(figsize=(6, 4))
    _cat_bar(ax, by_cat_flip, "Category-wise answer flip rate", "flip_rate")
    fig.tight_layout(); fig.savefig(out_dir / "cat_answer_flip_rate.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    _cat_bar(ax, by_cat_bndy, "Category-wise boundary error", "boundary_error")
    fig.tight_layout(); fig.savefig(out_dir / "cat_boundary_error.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    _cat_bar(ax, by_cat_attn, "Category-wise mean attention jump", "attn_jump")
    fig.tight_layout(); fig.savefig(out_dir / "cat_attention_jump.png", dpi=120); plt.close(fig)

    if vision_jumps and proj_jumps:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(vision_jumps, proj_jumps)
        lo = min(min(vision_jumps), min(proj_jumps)); hi = max(max(vision_jumps), max(proj_jumps))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.5)
        ax.set_xlabel("vision mean adjacent jump"); ax.set_ylabel("projector mean adjacent jump")
        ax.set_title("Vision vs projector smoothness")
        fig.tight_layout(); fig.savefig(out_dir / "vision_vs_projector_scatter.png", dpi=120); plt.close(fig)

    if layer_spearman:
        layers, corrs = zip(*layer_spearman)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(layers, corrs)
        ax.set_xlabel("best layer"); ax.set_ylabel("center-distance Spearman")
        ax.set_title("Best hidden layer by ambiguity Spearman")
        fig.tight_layout(); fig.savefig(out_dir / "best_layer_spearman.png", dpi=120); plt.close(fig)

    return str(out_dir)
