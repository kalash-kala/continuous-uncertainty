"""Top-level extraction loop. For each sequence/frame: run forward, save outputs."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from ..models.llava_loader import load_llava, build_prompt
from ..models.token_index_utils import get_token_indices, infer_grid_shape
from ..models.hook_utils import capture_vision_and_projector
from .extract_vision_features import pool_vision_features
from .extract_projector_features import pool_projector_features
from .extract_hidden_states import collect_hidden_states, hidden_state_norms
from .extract_attentions import slice_query_to_image
from .extract_logits import compute_yes_no_logits
from ..metrics.attention_metrics import attention_entropy


_TORCH_DTYPE = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def _save_tensor(t, path, dtype=torch.float16):
    if t is None:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(t, torch.Tensor):
        torch.save(t.to(dtype), path)
    else:
        torch.save(t, path)
    return path


def extract_all(model, processor, sequences, configs, output_dir):
    """Run extraction for all sequences. Writes per_frame_outputs.jsonl.

    sequences: iterable of list-of-frame-records.
    configs: dict with keys 'model', 'extraction', 'run'.
    """
    extraction_cfg = configs["extraction"]
    model_cfg = configs["model"]
    run_cfg = configs["run"]
    prompt_cfg = model_cfg["prompt"]

    save_dtype = _TORCH_DTYPE.get(extraction_cfg.get("storage", {}).get("tensor_dtype", "float16"),
                                  torch.float16)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = out_dir / "features"
    features_dir.mkdir(exist_ok=True)
    per_frame_path = out_dir / "per_frame_outputs.jsonl"

    layers_cfg = extraction_cfg.get("hidden_states", {}).get("layers", "all")
    attn_cfg = extraction_cfg.get("attentions", {})
    logits_cfg = extraction_cfg.get("logits", {})
    storage_cfg = extraction_cfg.get("storage", {})
    save_large = storage_cfg.get("save_large_tensors", True)

    yes_variants = logits_cfg.get("yes_variants", ["yes", "Yes"])
    no_variants = logits_cfg.get("no_variants", ["no", "No"])

    tokenizer = processor.tokenizer
    device = next(model.parameters()).device

    fout = open(per_frame_path, "w", encoding="utf-8")

    try:
        for seq in sequences:
            seq_id = seq[0]["sequence_id"]
            seq_feat_dir = features_dir / seq_id
            seq_feat_dir.mkdir(parents=True, exist_ok=True)
            for rec in seq:
                try:
                    image = Image.open(rec["image_path"]).convert("RGB")
                except FileNotFoundError:
                    print(f"[warn] missing image: {rec['image_path']}")
                    continue

                prompt = build_prompt(processor, rec["question"], prompt_cfg)
                token_meta = get_token_indices(processor, model, image, rec["question"], prompt)

                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                if inputs["pixel_values"].dtype != next(model.parameters()).dtype:
                    inputs["pixel_values"] = inputs["pixel_values"].to(next(model.parameters()).dtype)

                with torch.no_grad(), capture_vision_and_projector(model) as hooks:
                    outputs = model(
                        **inputs,
                        output_hidden_states=True,
                        output_attentions=True,
                        return_dict=True,
                    )

                vision_pooled = pool_vision_features(hooks["vision"]) \
                    if extraction_cfg.get("save_vision_features", True) else None
                projector_pooled = pool_projector_features(hooks["projector"]) \
                    if extraction_cfg.get("save_projector_features", True) else None

                # Hidden states
                hidden = collect_hidden_states(outputs, token_meta, layers=layers_cfg) \
                    if extraction_cfg.get("save_hidden_states", True) else {}
                hs_norms = hidden_state_norms(hidden) if hidden else {}

                # Attentions
                per_layer_attn, merged_attn = ({}, None)
                if extraction_cfg.get("save_attentions", True) and attn_cfg.get("enabled", True):
                    per_layer_attn, merged_attn = slice_query_to_image(
                        outputs, token_meta,
                        query_source=attn_cfg.get("query_source", "final_prompt"),
                        layers=tuple(attn_cfg.get("layers", [8, 16, 24, 31])),
                        head_aggregation=attn_cfg.get("aggregate_heads", "mean"),
                    )
                attn_entropies = {f"layer_{l}_{attn_cfg.get('query_source','final_prompt')}":
                                  float(attention_entropy(v)) for l, v in per_layer_attn.items()}

                # Logits
                logit_info = None
                if extraction_cfg.get("save_logits", True) and logits_cfg.get("compute_yes_no_margin", True):
                    logit_info = compute_yes_no_logits(
                        outputs, tokenizer, yes_variants, no_variants,
                        first_answer_position=token_meta["final_prompt_token_index"],
                    )

                # Save tensors
                feat_paths = {}
                stem = f"frame_{rec['frame_index']:02d}"
                if save_large:
                    if vision_pooled is not None:
                        feat_paths["vision_feature"] = str(_save_tensor(vision_pooled, seq_feat_dir / f"{stem}_vision.pt", save_dtype))
                    if projector_pooled is not None:
                        feat_paths["projected_feature"] = str(_save_tensor(projector_pooled, seq_feat_dir / f"{stem}_projected.pt", save_dtype))
                    if hidden:
                        feat_paths["hidden_states"] = str(_save_tensor(
                            {src: {int(l): v for l, v in d.items()} for src, d in hidden.items()},
                            seq_feat_dir / f"{stem}_hidden.pt", save_dtype))
                    if per_layer_attn:
                        feat_paths["attention_map"] = str(_save_tensor(
                            {int(l): torch.tensor(v) for l, v in per_layer_attn.items()},
                            seq_feat_dir / f"{stem}_attention.pt", save_dtype))
                    if logit_info is not None:
                        feat_paths["logits"] = str(_save_tensor(
                            outputs.logits[0, token_meta["final_prompt_token_index"]].detach().cpu(),
                            seq_feat_dir / f"{stem}_logits.pt", save_dtype))

                record = {
                    "run_name": run_cfg["run_name"],
                    "model_name": model_cfg["name"],
                    "sequence_id": seq_id,
                    "frame_index": rec["frame_index"],
                    "frame_number": rec["frame_number"],
                    "image_path": rec["image_path"],
                    "question": rec["question"],
                    "category": rec["category"],
                    "max_ambiguity_index": rec["max_ambiguity_index"],
                    "relative_ambiguity_distance": rec["relative_ambiguity_distance"],
                    "ground_truth": rec["ground_truth"],
                    "num_image_tokens": token_meta["num_image_tokens"],
                    "image_grid_shape": list(token_meta["image_grid_shape"]),
                    "final_prompt_token_index": int(token_meta["final_prompt_token_index"]),
                    "first_answer_token_index": token_meta["first_answer_token_index"],
                    "attention_entropy": attn_entropies,
                    "hidden_state_norms": hs_norms,
                    "feature_paths": feat_paths,
                }
                if logit_info is not None:
                    record.update({
                        "logit_yes": logit_info["logit_yes"],
                        "logit_no": logit_info["logit_no"],
                        "logit_margin_yes_no": logit_info["logit_margin_yes_no"],
                        "binary_entropy": logit_info["binary_entropy"],
                        "parsed_answer": logit_info["parsed_answer"],
                        "generated_answer": logit_info["parsed_answer"],
                        "is_correct": (logit_info["parsed_answer"] == rec["ground_truth"]),
                    })
                fout.write(json.dumps(record) + "\n")
                fout.flush()
    finally:
        fout.close()

    return str(per_frame_path)
