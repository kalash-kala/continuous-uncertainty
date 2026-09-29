"""Aggregate per-frame outputs into per-sequence summaries + aggregate CSV.

Also runs the per-question-token selection + metrics if enabled.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from _common import load_yaml, read_jsonl, write_jsonl
from vlm_internal_diagnostics.metrics import aggregate_sequence, compute_per_token_metrics


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
    "yes_no_margin_center_distance_spearman", "binary_entropy_center_distance_spearman",
    # Representation-based boundary errors (vision)
    "vision_chord_boundary_index", "vision_chord_boundary_error",
    "vision_angle_boundary_index", "vision_angle_boundary_error",
    "vision_local_var_boundary_index", "vision_local_var_boundary_error",
    "vision_anchor_product_boundary_index", "vision_anchor_product_boundary_error",
    # Representation-based boundary errors (projector)
    "projector_chord_boundary_index", "projector_chord_boundary_error",
    "projector_angle_boundary_index", "projector_angle_boundary_error",
    "projector_local_var_boundary_index", "projector_local_var_boundary_error",
    "projector_anchor_product_boundary_index", "projector_anchor_product_boundary_error",
    # Representation-based boundary errors (hidden, best layer by spearman)
    "hidden_chord_boundary_index", "hidden_chord_boundary_error",
    "hidden_angle_boundary_index", "hidden_angle_boundary_error",
    "hidden_local_var_boundary_index", "hidden_local_var_boundary_error",
    "hidden_anchor_product_boundary_index", "hidden_anchor_product_boundary_error",
    # Per-question-token aggregates
    "mean_attention_entropy_content_tokens",
    "mean_attention_jump_content_tokens",
    "mean_center_distance_spearman_content_tokens",
    "mean_temporal_variance_content_tokens",
    "mean_center_vs_clear_diff_content_tokens",
    "mean_head_entropy_spread_content_tokens",
    "mean_head_image_mass_spread_content_tokens",
    "mean_top_head_vs_mean_divergence_content_tokens",
]


def _load_pt(path):
    if not path:
        return None
    candidates = [path]
    cleaned = "".join(ch for ch in str(path) if ord(ch) < 128)
    if cleaned != path:
        candidates.append(cleaned)
    for p in candidates:
        try:
            return torch.load(p, map_location="cpu")
        except Exception:
            continue
    return None


def _to_vec(t):
    if t is None:
        return None
    if hasattr(t, "float"):
        t = t.float()
    return np.asarray(t.numpy() if hasattr(t, "numpy") else t).ravel()


def _per_token_payload_to_dict(p):
    """Convert a torch-saved per-token payload (with tensors) to numpy-friendly dict."""
    if p is None or not isinstance(p, dict):
        return None
    out = dict(p)
    mats = out.get("per_layer_matrices_head_avg") or {}
    out["per_layer_matrices_head_avg"] = {
        int(k): (v.float().cpu().numpy() if hasattr(v, "float") else np.asarray(v))
        for k, v in mats.items()
    }
    stats = out.get("per_head_stats") or {}
    out["per_head_stats"] = {
        k: (v.float().cpu().numpy() if hasattr(v, "float") else np.asarray(v))
        for k, v in stats.items()
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--visualization_config", required=False, default=None,
                   help="Used to read per-token selection settings.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    per_frame = read_jsonl(run_dir / "per_frame_outputs.jsonl")

    # Token-selection / per-token-metrics settings (from visualization_config if provided)
    vc = {}
    if args.visualization_config:
        vc = load_yaml(args.visualization_config) or {}
    selection_cfg = vc.get("per_token_attention", {}) or {}
    pt_enabled = selection_cfg.get("enabled", True)
    pt_strategy = selection_cfg.get("selection_strategy", "strategy_d")
    pt_top_k = int(selection_cfg.get("top_k", 5))
    pt_pos = selection_cfg.get("pos_tags_to_keep",
                                ["NOUN", "PROPN", "VERB", "ADJ", "ADV"])
    pt_layer = selection_cfg.get("layer", None)
    n_far = int((vc.get("diagnostic_maps", {}) or {}).get("num_farthest_frames", 4))

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
        per_token_payloads = []
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
            ptp = _load_pt(fp.get("per_token_attention_map"))
            per_token_payloads.append(_per_token_payload_to_dict(ptp))

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

        # ----- Per-question-token selection + metrics -----
        if pt_enabled and any(p is not None for p in per_token_payloads):
            # Token selection — use the first available payload to get the question tokens
            first_payload = next((p for p in per_token_payloads if p is not None), None)
            llava_tokens = list(first_payload.get("question_tokens") or [])
            question_text = records[0].get("question", "")
            # Build frame-aligned head-averaged matrices for strategy_d scoring
            # (one [T, P] per frame, averaged across layers if layer is None)
            per_frame_mats = []
            for p in per_token_payloads:
                if p is None:
                    continue
                mats = p.get("per_layer_matrices_head_avg") or {}
                if not mats:
                    continue
                if pt_layer is not None and pt_layer in mats:
                    per_frame_mats.append(np.asarray(mats[pt_layer]))
                else:
                    stk = np.stack([np.asarray(v) for v in mats.values()], axis=0).mean(axis=0)
                    per_frame_mats.append(stk)

            try:
                from vlm_internal_diagnostics.analysis import select_tokens
                selected = select_tokens(
                    llava_tokens=llava_tokens,
                    question=question_text,
                    strategy=pt_strategy,
                    top_k=pt_top_k,
                    pos_tags_to_keep=pt_pos,
                    per_frame_matrices=per_frame_mats if pt_strategy == "strategy_d" else None,
                )
            except Exception as e:
                print(f"[metrics] token-selection failed for {seq_id}: {e}")
                selected = []

            if selected:
                pt_result = compute_per_token_metrics(
                    payloads=[p for p in per_token_payloads if p is not None],
                    selected_tokens=selected,
                    frame_records=[r for r, p in zip(records, per_token_payloads) if p is not None],
                    num_farthest_frames=n_far,
                    layer=pt_layer,
                )
                summary["per_token_attention_metrics"] = {
                    "selection_strategy": pt_strategy,
                    "selected_tokens": pt_result["selected_tokens"],
                }
                summary.update(pt_result.get("aggregate", {}))
                # Persist the selection separately for visualization
                summary["selected_token_positions"] = [t.position for t in selected]
                summary["selected_token_texts"] = [t.text for t in selected]
                summary["selected_token_pos_tags"] = [t.pos for t in selected]
                summary["selected_token_scores"] = [t.score for t in selected]

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
