"""Aggregate per-sequence metric records from per-frame outputs."""
import numpy as np
from .smoothness_metrics import smoothness_block, ambiguity_spearman
from .attention_metrics import attention_block
from .logit_metrics import boundary_index, boundary_error, answer_flip_rate


def accuracy(parsed_answers, ground_truths):
    correct = sum(1 for p, g in zip(parsed_answers, ground_truths)
                  if p is not None and g is not None and str(p).lower() == str(g).lower())
    return float(correct / len(parsed_answers)) if parsed_answers else float("nan")


def aggregate_sequence(frame_records,
                       vision_feats=None,
                       projector_feats=None,
                       hidden_by_layer=None,
                       attention_maps=None):
    """Build the per-sequence summary dict (schema in Section 11.2)."""
    frame_indices = [r["frame_index"] for r in frame_records]
    center_index = frame_records[0]["max_ambiguity_index"]
    parsed = [r.get("parsed_answer") for r in frame_records]
    gts = [r.get("ground_truth") for r in frame_records]
    margins = [r.get("logit_margin_yes_no") for r in frame_records]
    margins_arr = [m for m in margins if m is not None]

    out = {
        "sequence_id": frame_records[0]["sequence_id"],
        "category": frame_records[0]["category"],
        "num_frames": len(frame_records),
        "max_ambiguity_index": center_index,
        "accuracy": accuracy(parsed, gts),
        "answer_flip_rate": answer_flip_rate(parsed),
    }

    if vision_feats is not None and len(vision_feats) >= 2:
        sb = smoothness_block(vision_feats, frame_indices, center_index)
        out.update({
            "mean_vision_adjacent_jump": sb["mean_adjacent_jump"],
            "max_vision_adjacent_jump": sb["max_adjacent_jump"],
            "vision_path_length": sb["path_length"],
            "vision_center_distance_spearman": sb["center_distance_spearman"],
        })

    if projector_feats is not None and len(projector_feats) >= 2:
        sb = smoothness_block(projector_feats, frame_indices, center_index)
        out.update({
            "mean_projector_adjacent_jump": sb["mean_adjacent_jump"],
            "max_projector_adjacent_jump": sb["max_adjacent_jump"],
            "projector_path_length": sb["path_length"],
            "projector_center_distance_spearman": sb["center_distance_spearman"],
        })
        if "mean_vision_adjacent_jump" in out:
            out["smoothness_drop_projector_minus_vision"] = (
                out["mean_projector_adjacent_jump"] - out["mean_vision_adjacent_jump"]
            )

    if hidden_by_layer:
        best_layer = None
        best_corr = -np.inf
        layer_corrs = {}
        for layer, feats in hidden_by_layer.items():
            if len(feats) < 2:
                continue
            sb = smoothness_block(feats, frame_indices, center_index)
            layer_corrs[layer] = sb
            if not np.isnan(sb["center_distance_spearman"]) and sb["center_distance_spearman"] > best_corr:
                best_corr = sb["center_distance_spearman"]
                best_layer = layer
        if best_layer is not None:
            out["best_hidden_layer_by_spearman"] = int(best_layer)
            out["best_hidden_center_distance_spearman"] = float(best_corr)
            out[f"mean_hidden_adjacent_jump_layer_{best_layer}"] = layer_corrs[best_layer]["mean_adjacent_jump"]
            out["mean_hidden_adjacent_jump_best_layer"] = layer_corrs[best_layer]["mean_adjacent_jump"]

    if attention_maps is not None and len(attention_maps) >= 2:
        ab = attention_block(attention_maps, frame_indices, center_index)
        out.update({
            "mean_attention_entropy": ab["mean_entropy"],
            "mean_attention_jump": ab["mean_jump"],
            "attention_entropy_ambiguity_spearman": ab["entropy_ambiguity_spearman"],
            "attention_center_distance_spearman": ab["center_distance_spearman"],
        })

    if margins_arr:
        bi = boundary_index(margins_arr)
        out["boundary_index"] = bi
        out["boundary_error"] = boundary_error(margins_arr, center_index)
        if 0 <= center_index < len(margins_arr):
            out["center_abs_margin"] = float(abs(margins_arr[center_index]))
        out["mean_abs_margin"] = float(np.mean(np.abs(margins_arr)))

    return out
