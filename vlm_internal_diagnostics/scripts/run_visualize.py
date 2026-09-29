"""Render sequence grids, trajectory plots, aggregate plots, per-token attention plots."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image

from _common import load_yaml, read_jsonl
from vlm_internal_diagnostics.visualization import (
    render_overlay_from_record, render_sequence_grid,
    render_trajectory_plot, render_aggregate_plots,
    render_per_frame_token_grid, load_per_token_payload,
    render_temporal_variance_map, render_center_vs_clear_diff,
    render_head_specialization_heatmap,
)


def _rehydrate_selected(summary):
    """Build a lightweight list of token dicts from the summary fields."""
    positions = summary.get("selected_token_positions") or []
    texts = summary.get("selected_token_texts") or []
    tags = summary.get("selected_token_pos_tags") or []
    scores = summary.get("selected_token_scores") or [None] * len(positions)
    out = []
    for i, pos in enumerate(positions):
        out.append({
            "position": pos,
            "text": texts[i] if i < len(texts) else None,
            "pos": tags[i] if i < len(tags) else None,
            "score": scores[i] if i < len(scores) else None,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--visualization_config", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    viz_cfg_root = load_yaml(args.visualization_config)
    viz_cfg = viz_cfg_root.get("visualization", {})
    overlay_cfg = viz_cfg_root.get("attention_overlay", {})
    grid_cfg = viz_cfg_root.get("sequence_grid", {})
    pt_cfg = viz_cfg_root.get("per_token_attention", {})
    diag_cfg = viz_cfg_root.get("diagnostic_maps", {})

    per_frame = read_jsonl(run_dir / "per_frame_outputs.jsonl")
    by_seq = defaultdict(list)
    for r in per_frame:
        by_seq[r["sequence_id"]].append(r)
    for v in by_seq.values():
        v.sort(key=lambda r: r["frame_index"])

    # Load per-sequence summary (contains selected_token_* fields)
    summary_path = run_dir / "per_sequence_summary.jsonl"
    summaries_by_seq = {}
    if summary_path.exists():
        for s in read_jsonl(summary_path):
            summaries_by_seq[s["sequence_id"]] = s

    figures_dir = run_dir / "figures"
    grids_dir = figures_dir / "sequence_grids"
    traj_dir = figures_dir / "trajectory_plots"
    agg_dir = figures_dir / "aggregate"
    per_token_dir = figures_dir / "per_token_attention"
    temporal_var_dir = figures_dir / "temporal_variance"
    center_vs_clear_dir = figures_dir / "center_vs_clear"
    head_spec_dir = figures_dir / "head_specialization"

    # ============================================================
    # DISABLED: per-frame attention overlays under figures/attention_overlays/
    # The averaged attention is shown in the sequence_grids (row 2) and the
    # per-token attention plots provide much finer-grained information. Set
    # `create_attention_overlays: true` to re-enable this block.
    # ============================================================
    # if viz_cfg.get("create_attention_overlays", False):
    #     overlays_dir = figures_dir / "attention_overlays"
    #     for seq_id, records in by_seq.items():
    #         for r in records:
    #             out = overlays_dir / seq_id / f"frame_{r['frame_index']:02d}.png"
    #             try:
    #                 render_overlay_from_record(
    #                     r, colormap=overlay_cfg.get("colormap", "jet"),
    #                     alpha=overlay_cfg.get("alpha", 0.45),
    #                     draw_patch_boundaries=overlay_cfg.get("draw_patch_boundaries", True),
    #                     out_path=out,
    #                 )
    #             except Exception as e:
    #                 print(f"[viz] overlay fail {seq_id} f{r['frame_index']}: {e}")

    # DISABLED: sequence grids (attention-map heavy; re-enable when needed)
    # if viz_cfg.get("create_sequence_grids", True):
    #     top_k_rows = int(grid_cfg.get("show_top_k_tokens", 3))
    #     per_token_layer = pt_cfg.get("layer", None)
    #     for seq_id, records in by_seq.items():
    #         selected = _rehydrate_selected(summaries_by_seq.get(seq_id, {}))
    #         try:
    #             render_sequence_grid(
    #                 records, grids_dir / f"{seq_id}.png",
    #                 mark_ambiguity_center=grid_cfg.get("mark_ambiguity_center", True),
    #                 colormap=overlay_cfg.get("colormap", "jet"),
    #                 alpha=overlay_cfg.get("alpha", 0.45),
    #                 selected_tokens=selected if top_k_rows > 0 else None,
    #                 top_k_token_rows=top_k_rows,
    #                 per_token_layer=per_token_layer,
    #             )
    #         except Exception as e:
    #             print(f"[viz] grid fail {seq_id}: {e}")

    if viz_cfg.get("create_trajectory_plots", True):
        for seq_id, records in by_seq.items():
            try:
                render_trajectory_plot(records, traj_dir / f"{seq_id}.png")
            except Exception as e:
                print(f"[viz] trajectory fail {seq_id}: {e}")

    if viz_cfg.get("create_aggregate_plots", True):
        if summary_path.exists():
            summaries = read_jsonl(summary_path)
            render_aggregate_plots(summaries, agg_dir, per_frame_by_seq=by_seq)

    # DISABLED: per-token attention overlays + diagnostic maps (re-enable when needed)
    # if pt_cfg.get("enabled", True):
    #     per_token_layer = pt_cfg.get("layer", None)
    #     colormap = overlay_cfg.get("colormap", "jet")
    #     alpha = overlay_cfg.get("alpha", 0.45)
    #     for seq_id, records in by_seq.items():
    #         selected = _rehydrate_selected(summaries_by_seq.get(seq_id, {}))
    #         if not selected:
    #             continue
    #         payloads = []
    #         for r in records:
    #             p_path = r.get("feature_paths", {}).get("per_token_attention_map")
    #             if p_path:
    #                 try:
    #                     payloads.append(load_per_token_payload(p_path))
    #                 except Exception:
    #                     payloads.append(None)
    #             else:
    #                 payloads.append(None)
    #
    #         for r, payload in zip(records, payloads):
    #             if payload is None:
    #                 continue
    #             out_path = per_token_dir / seq_id / f"frame_{r['frame_index']:02d}.png"
    #             try:
    #                 image = Image.open(r["image_path"]).convert("RGB")
    #                 title = (f"f{r['frame_index']} | ans={r.get('parsed_answer','?')} | "
    #                          f"m={r.get('logit_margin_yes_no', float('nan')):.2f} | "
    #                          f"H={r.get('binary_entropy', float('nan')):.2f}")
    #                 render_per_frame_token_grid(
    #                     image, payload, selected, out_path,
    #                     layer=per_token_layer, colormap=colormap, alpha=alpha,
    #                     title=title,
    #                 )
    #             except Exception as e:
    #                 print(f"[viz] per-token overlay fail {seq_id} f{r['frame_index']}: {e}")
    #
    #         valid_payloads = [p for p in payloads if p is not None]
    #         valid_records = [r for r, p in zip(records, payloads) if p is not None]
    #         if not valid_payloads or not valid_records:
    #             continue
    #
    #         n_far = int(diag_cfg.get("num_farthest_frames", 4))
    #         if diag_cfg.get("temporal_variance", True):
    #             try:
    #                 render_temporal_variance_map(
    #                     valid_records, valid_payloads, selected,
    #                     temporal_var_dir / f"{seq_id}.png",
    #                     layer=per_token_layer,
    #                 )
    #             except Exception as e:
    #                 print(f"[viz] temporal_variance fail {seq_id}: {e}")
    #         if diag_cfg.get("center_vs_clear", True):
    #             try:
    #                 render_center_vs_clear_diff(
    #                     valid_records, valid_payloads, selected,
    #                     center_vs_clear_dir / f"{seq_id}.png",
    #                     num_farthest_frames=n_far,
    #                     layer=per_token_layer,
    #                 )
    #             except Exception as e:
    #                 print(f"[viz] center_vs_clear fail {seq_id}: {e}")
    #         if diag_cfg.get("head_specialization", True):
    #             try:
    #                 render_head_specialization_heatmap(
    #                     valid_payloads, selected,
    #                     head_spec_dir / f"{seq_id}.png",
    #                 )
    #             except Exception as e:
    #                 print(f"[viz] head_specialization fail {seq_id}: {e}")

    print(f"[viz] figures written under {figures_dir}")


if __name__ == "__main__":
    main()
