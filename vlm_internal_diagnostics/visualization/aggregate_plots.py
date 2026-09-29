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


def _aggregate_by_normalized_index(per_frame_by_seq, value_key, use_abs=False):
    """Build {normalized_index: [values]} across all sequences.

    Normalized index = frame_index - max_ambiguity_index (so center=0).
    """
    by_norm = defaultdict(list)
    for seq_id, records in per_frame_by_seq.items():
        if not records:
            continue
        center = records[0].get("max_ambiguity_index")
        if center is None:
            continue
        for r in records:
            v = r.get(value_key)
            if v is None:
                continue
            if use_abs:
                v = abs(v)
            norm_idx = int(r["frame_index"]) - int(center)
            by_norm[norm_idx].append(float(v))
    return by_norm


def _plot_normalized_aggregate(by_norm_dict_or_dict_of_dicts, title, ylabel, out_path,
                               by_category=False):
    """Plot mean value vs normalized index. If by_category, plot one line per category."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if by_category:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, (cat, by_norm) in enumerate(sorted(by_norm_dict_or_dict_of_dicts.items())):
            if not by_norm:
                continue
            indices = sorted(by_norm.keys())
            means = [np.mean(by_norm[idx]) for idx in indices]
            ax.plot(indices, means, "-o", label=f"{cat} (n_seq varies)",
                    color=colors[i % len(colors)])
        ax.legend(fontsize=9, loc="best")
    else:
        by_norm = by_norm_dict_or_dict_of_dicts
        indices = sorted(by_norm.keys())
        means = [np.mean(by_norm[idx]) for idx in indices]
        counts = [len(by_norm[idx]) for idx in indices]
        ax.plot(indices, means, "-o", color="#1f77b4", label="mean")
        for idx, mn, ct in zip(indices, means, counts):
            ax.annotate(f"n={ct}", (idx, mn), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, alpha=0.6)
    ax.axvline(0, color="red", linestyle=":", lw=1, alpha=0.6, label="center (ambiguity)")
    ax.set_xlabel("normalized frame index (frame_index − max_ambiguity_index)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def render_aggregate_plots(seq_summaries, out_dir, per_frame_by_seq=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    by_cat_flip = defaultdict(list)
    by_cat_bndy = defaultdict(list)
    by_cat_attn = defaultdict(list)
    by_cat_margin_cds = defaultdict(list)
    by_cat_binent_cds = defaultdict(list)
    vision_jumps, proj_jumps = [], []
    layer_spearman = []

    for s in seq_summaries:
        c = s.get("category", "unknown")
        if "answer_flip_rate" in s: by_cat_flip[c].append(s["answer_flip_rate"])
        if "boundary_error" in s: by_cat_bndy[c].append(s["boundary_error"])
        if "mean_attention_jump" in s: by_cat_attn[c].append(s["mean_attention_jump"])
        if "yes_no_margin_center_distance_spearman" in s and s["yes_no_margin_center_distance_spearman"] is not None:
            v = s["yes_no_margin_center_distance_spearman"]
            if not (isinstance(v, float) and np.isnan(v)):
                by_cat_margin_cds[c].append(v)
        if "binary_entropy_center_distance_spearman" in s and s["binary_entropy_center_distance_spearman"] is not None:
            v = s["binary_entropy_center_distance_spearman"]
            if not (isinstance(v, float) and np.isnan(v)):
                by_cat_binent_cds[c].append(v)
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

    if by_cat_margin_cds:
        fig, ax = plt.subplots(figsize=(6, 4))
        _cat_bar(ax, by_cat_margin_cds,
                 "Category-wise |yes/no margin| center-distance Spearman", "spearman")
        fig.tight_layout()
        fig.savefig(out_dir / "cat_yes_no_margin_center_distance_spearman.png", dpi=120)
        plt.close(fig)

    if by_cat_binent_cds:
        fig, ax = plt.subplots(figsize=(6, 4))
        _cat_bar(ax, by_cat_binent_cds,
                 "Category-wise binary entropy center-distance Spearman", "spearman")
        fig.tight_layout()
        fig.savefig(out_dir / "cat_binary_entropy_center_distance_spearman.png", dpi=120)
        plt.close(fig)

    if per_frame_by_seq:
        margin_all = _aggregate_by_normalized_index(
            per_frame_by_seq, "logit_margin_yes_no", use_abs=True)
        binent_all = _aggregate_by_normalized_index(
            per_frame_by_seq, "binary_entropy", use_abs=False)

        if margin_all:
            _plot_normalized_aggregate(
                margin_all,
                title="Mean |yes/no margin| vs normalized frame index (all videos)",
                ylabel="mean |margin|",
                out_path=out_dir / "normalized_abs_margin_combined.png",
                by_category=False,
            )
        if binent_all:
            _plot_normalized_aggregate(
                binent_all,
                title="Mean binary entropy vs normalized frame index (all videos)",
                ylabel="mean binary entropy",
                out_path=out_dir / "normalized_binary_entropy_combined.png",
                by_category=False,
            )

        by_cat_records = defaultdict(dict)
        for seq_id, records in per_frame_by_seq.items():
            if not records:
                continue
            cat = records[0].get("category", "unknown")
            by_cat_records[cat][seq_id] = records

        margin_by_cat = {
            cat: _aggregate_by_normalized_index(seqs, "logit_margin_yes_no", use_abs=True)
            for cat, seqs in by_cat_records.items()
        }
        binent_by_cat = {
            cat: _aggregate_by_normalized_index(seqs, "binary_entropy", use_abs=False)
            for cat, seqs in by_cat_records.items()
        }

        if any(margin_by_cat.values()):
            _plot_normalized_aggregate(
                margin_by_cat,
                title="Mean |yes/no margin| vs normalized frame index (by category)",
                ylabel="mean |margin|",
                out_path=out_dir / "normalized_abs_margin_by_category.png",
                by_category=True,
            )
        if any(binent_by_cat.values()):
            _plot_normalized_aggregate(
                binent_by_cat,
                title="Mean binary entropy vs normalized frame index (by category)",
                ylabel="mean binary entropy",
                out_path=out_dir / "normalized_binary_entropy_by_category.png",
                by_category=True,
            )

    return str(out_dir)
