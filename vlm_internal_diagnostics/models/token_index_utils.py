"""Identify image / question / final-prompt / first-answer token positions.

LLaVA-1.5 HF expands the single <image> token in input_ids into many visual
tokens inside the LM forward. We locate them by matching the model's
image_token_id (config.image_token_index) in input_ids and inferring the
expansion count from input embeddings vs raw input length.
"""
import math
import torch


def _image_token_id(model):
    cfg = model.config
    for attr in ("image_token_index", "image_token_id"):
        v = getattr(cfg, attr, None)
        if v is not None:
            return int(v)
    return None


def infer_grid_shape(num_image_tokens):
    """Square grid if possible, else closest factor pair."""
    n = int(num_image_tokens)
    if n <= 0:
        return (0, 0)
    r = int(round(math.sqrt(n)))
    if r * r == n:
        return (r, r)
    for d in range(r, 0, -1):
        if n % d == 0:
            return (d, n // d)
    return (1, n)


def get_token_indices(processor, model, image, question, prompt, generated_ids=None):
    """Compute the token-index metadata dict described in Section 5.

    Returns a dict with input_ids/attention_mask/pixel_values (CPU tensors),
    image_token_indices (in the *expanded* LM sequence), question_token_indices,
    final_prompt_token_index, first_answer_token_index, image_grid_shape,
    num_image_tokens, decoded_prompt_tokens.
    """
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    pixel_values = inputs.get("pixel_values")

    img_tok = _image_token_id(model)

    # Figure out how many visual tokens the LM sees.
    num_image_tokens = _estimate_num_image_tokens(model, pixel_values)

    n_sentinel = int((input_ids == img_tok).sum()) if img_tok is not None else 0

    if img_tok is not None and n_sentinel == num_image_tokens:
        # Processor pre-expanded image tokens — each patch already has its own
        # token id in input_ids. No further expansion needed.
        image_token_indices = (input_ids == img_tok).nonzero(as_tuple=True)[0].tolist()
        expanded_len = input_ids.shape[0]
        last_img = image_token_indices[-1] if image_token_indices else -1
        question_token_indices = list(range(last_img + 1, expanded_len))
    else:
        # Single sentinel case: LLaVA-1.5 puts one <image> token in input_ids
        # and the model internally expands it to num_image_tokens visual tokens.
        raw_image_positions = (input_ids == img_tok).nonzero(as_tuple=True)[0].tolist() \
            if img_tok is not None else []
        image_token_indices = []
        expanded_len = input_ids.shape[0]
        offset = 0
        raw_to_expanded = []
        for i, tid in enumerate(input_ids.tolist()):
            raw_to_expanded.append(i + offset)
            if img_tok is not None and tid == img_tok:
                start = i + offset
                image_token_indices.extend(range(start, start + num_image_tokens))
                offset += (num_image_tokens - 1)
                expanded_len += (num_image_tokens - 1)
        last_img = raw_image_positions[-1] if raw_image_positions else -1
        question_raw_positions = list(range(last_img + 1, input_ids.shape[0]))
        question_token_indices = [raw_to_expanded[i] for i in question_raw_positions]

    final_prompt_token_index = expanded_len - 1

    first_answer_token_index = None
    if generated_ids is not None:
        # generated_ids is the full sequence returned by model.generate.
        # The first answer token in the LM sequence is at position expanded_len.
        first_answer_token_index = expanded_len

    decoded_prompt_tokens = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
    grid = infer_grid_shape(num_image_tokens)

    return {
        "input_ids": input_ids,
        "attention_mask": inputs.get("attention_mask"),
        "pixel_values": pixel_values,
        "image_token_indices": image_token_indices,
        "question_token_indices": question_token_indices,
        "final_prompt_token_index": final_prompt_token_index,
        "first_answer_token_index": first_answer_token_index,
        "image_grid_shape": grid,
        "num_image_tokens": num_image_tokens,
        "decoded_prompt_tokens": decoded_prompt_tokens,
        "expanded_seq_len": expanded_len,
    }


def _estimate_num_image_tokens(model, pixel_values):
    """Use the vision tower config to compute the number of visual tokens."""
    try:
        vt_cfg = model.config.vision_config
        image_size = getattr(vt_cfg, "image_size", 336)
        patch_size = getattr(vt_cfg, "patch_size", 14)
        n_patches = (image_size // patch_size) ** 2
        # LLaVA-1.5 drops the CLS token in projector.
        feat_strategy = getattr(model.config, "vision_feature_select_strategy", "default")
        if feat_strategy == "full":
            return n_patches + 1
        return n_patches
    except Exception:
        return 576
