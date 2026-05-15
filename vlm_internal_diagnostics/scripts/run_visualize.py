"""Render attention overlays, sequence grids, trajectory plots, aggregate plots."""
import argparse
from collections import defaultdict
from pathlib import Path

from _common import load_yaml, read_jsonl
from vlm_internal_diagnostics.visualization import (
    render_overlay_from_record, render_sequence_grid,
    render_trajectory_plot, render_aggregate_plots,
)


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

    per_frame = read_jsonl(run_dir / "per_frame_outputs.jsonl")
    by_seq = defaultdict(list)
    for r in per_frame:
        by_seq[r["sequence_id"]].append(r)
    for v in by_seq.values():
        v.sort(key=lambda r: r["frame_index"])

    figures_dir = run_dir / "figures"
    overlays_dir = figures_dir / "attention_overlays"
    grids_dir = figures_dir / "sequence_grids"
    traj_dir = figures_dir / "trajectory_plots"
    agg_dir = figures_dir / "aggregate"

    if viz_cfg.get("create_attention_overlays", True):
        for seq_id, records in by_seq.items():
            for r in records:
                out = overlays_dir / seq_id / f"frame_{r['frame_index']:02d}.png"
                try:
                    render_overlay_from_record(
                        r, colormap=overlay_cfg.get("colormap", "jet"),
                        alpha=overlay_cfg.get("alpha", 0.45),
                        draw_patch_boundaries=overlay_cfg.get("draw_patch_boundaries", True),
                        out_path=out,
                    )
                except Exception as e:
                    print(f"[viz] overlay fail {seq_id} f{r['frame_index']}: {e}")

    if viz_cfg.get("create_sequence_grids", True):
        for seq_id, records in by_seq.items():
            try:
                render_sequence_grid(records, grids_dir / f"{seq_id}.png",
                                     mark_ambiguity_center=grid_cfg.get("mark_ambiguity_center", True),
                                     colormap=overlay_cfg.get("colormap", "jet"),
                                     alpha=overlay_cfg.get("alpha", 0.45))
            except Exception as e:
                print(f"[viz] grid fail {seq_id}: {e}")

    if viz_cfg.get("create_trajectory_plots", True):
        for seq_id, records in by_seq.items():
            try:
                render_trajectory_plot(records, traj_dir / f"{seq_id}.png")
            except Exception as e:
                print(f"[viz] trajectory fail {seq_id}: {e}")

    if viz_cfg.get("create_aggregate_plots", True):
        summary_path = run_dir / "per_sequence_summary.jsonl"
        if summary_path.exists():
            summaries = read_jsonl(summary_path)
            render_aggregate_plots(summaries, agg_dir)

    print(f"[viz] figures written under {figures_dir}")


if __name__ == "__main__":
    main()
