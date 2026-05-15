"""Aggregate per-frame outputs into per-sequence summaries + aggregate CSV."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from _common import read_jsonl, write_jsonl
from vlm_internal_diagnostics.metrics import aggregate_sequence


AGG_COLUMNS = [
    "run_name", "model_name", "sequence_id", "category", "num_frames", "max_ambiguity_index",
    "accuracy", "answer_flip_rate",
    "mean_vision_adjacent_jump", "max_vision_adjacent_jump", "vision_path_length",
    "vision_center_distance_spearman",
    "mean_projector_adjacent_jump", "max_projector_adjacent_jump", "projector_path_length",
    "projector_center_distance_spearman", "smoothness_drop_projector_minus_vision",
    "best_hidden_layer_by_spearman", "best_hidden_center_distance_spearman",
    "mean_hidden_adjacent_jump_best_layer",
    "mean_attention_entropy", "mean_attention_jump",
    "attention_entropy_ambiguity_spearman", "attention_center_distance_spearman",
    "boundary_index", "boundary_error", "center_abs_margin", "mean_abs_margin",
]


def _load_pt(path):
    if not path:
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def _to_vec(t):
    if t is None:
        return None
    if hasattr(t, "float"):
        t = t.float()
    return np.asarray(t.numpy() if hasattr(t, "numpy") else t).ravel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    per_frame = read_jsonl(run_dir / "per_frame_outputs.jsonl")

    by_seq = defaultdict(list)
    for r in per_frame:
        by_seq[r["sequence_id"]].append(r)

    summaries = []
    for seq_id, records in by_seq.items():
        records.sort(key=lambda r: r["frame_index"])

        vision_feats = []
        projector_feats = []
        hidden_by_layer = defaultdict(list)
        attention_maps = []
        for r in records:
            fp = r.get("feature_paths", {})
            v = _load_pt(fp.get("vision_feature"));
            if v is not None: vision_feats.append(_to_vec(v))
            pj = _load_pt(fp.get("projected_feature"))
            if pj is not None: projector_feats.append(_to_vec(pj))
            hd = _load_pt(fp.get("hidden_states"))
            if isinstance(hd, dict):
                fp_layer = hd.get("final_prompt", {})
                for layer, vec in fp_layer.items():
                    hidden_by_layer[int(layer)].append(_to_vec(vec))
            am = _load_pt(fp.get("attention_map"))
            if isinstance(am, dict) and am:
                first = next(iter(am.values()))
                attention_maps.append(_to_vec(first))

        # Drop layers that don't have all frames
        n = len(records)
        hidden_by_layer = {l: v for l, v in hidden_by_layer.items() if len(v) == n}

        summary = aggregate_sequence(
            records,
            vision_feats=vision_feats if len(vision_feats) == n else None,
            projector_feats=projector_feats if len(projector_feats) == n else None,
            hidden_by_layer=hidden_by_layer or None,
            attention_maps=attention_maps if len(attention_maps) == n else None,
        )
        summary["run_name"] = records[0].get("run_name")
        summary["model_name"] = records[0].get("model_name")
        summaries.append(summary)

    write_jsonl(summaries, run_dir / "per_sequence_summary.jsonl")

    csv_path = run_dir / "aggregate_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGG_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k, "") for k in AGG_COLUMNS})
    print(f"[metrics] wrote {csv_path}")


if __name__ == "__main__":
    main()
