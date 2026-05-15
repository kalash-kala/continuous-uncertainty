"""Slice attention from query-token rows to image-token columns."""
import numpy as np
import torch


def slice_query_to_image(outputs, token_indices, query_source="final_prompt",
                        layers=(8, 16, 24, 31), head_aggregation="mean"):
    """Returns dict: layer -> 1D np.ndarray over image tokens (averaged over heads).

    Also returns merged map: average over the chosen layers, normalized to sum=1.
    """
    attns = outputs.attentions  # tuple of [B, heads, S, S]
    if attns is None or len(attns) == 0:
        return {}, None

    img_idx = token_indices["image_token_indices"]
    if not img_idx:
        return {}, None

    if query_source == "final_prompt":
        q_idx = token_indices["final_prompt_token_index"]
    elif query_source == "first_answer":
        q_idx = token_indices["first_answer_token_index"] or token_indices["final_prompt_token_index"]
    elif query_source == "question_mean":
        q_idx = token_indices["question_token_indices"]
    else:
        q_idx = token_indices["final_prompt_token_index"]

    num_layers = len(attns)
    layer_iter = [l for l in layers if l < num_layers] if layers else range(num_layers)

    per_layer = {}
    for l in layer_iter:
        A = attns[l][0]  # [heads, S, S]
        seq_len = A.shape[-1]
        ii = torch.tensor([i for i in img_idx if i < seq_len], dtype=torch.long, device=A.device)
        if isinstance(q_idx, list):
            qi = torch.tensor([q for q in q_idx if q < seq_len], dtype=torch.long, device=A.device)
            if qi.numel() == 0:
                continue
            row = A[:, qi, :].mean(dim=1)  # [heads, S]
        else:
            qi_v = min(q_idx, seq_len - 1)
            row = A[:, qi_v, :]  # [heads, S]
        sub = row[:, ii]  # [heads, num_img]
        if head_aggregation == "mean":
            vec = sub.mean(dim=0)
        elif head_aggregation == "max":
            vec, _ = sub.max(dim=0)
        else:
            vec = sub.mean(dim=0)
        per_layer[l] = vec.detach().cpu().to(torch.float32).numpy()

    if per_layer:
        stacked = np.stack(list(per_layer.values()), axis=0).mean(axis=0)
        s = stacked.sum()
        merged = stacked / s if s > 0 else stacked
    else:
        merged = None
    return per_layer, merged
