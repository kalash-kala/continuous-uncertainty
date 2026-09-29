"""Per-question-token attention extraction.

Preserves the full [num_question_tokens x num_image_patches] attention matrix
per layer (head-averaged for visualization). Also computes lightweight per-head
summary statistics so downstream analysis can detect head specialization.
"""
import numpy as np
import torch


def extract_per_token_attention(outputs, token_indices, layers=(8, 16, 24, 31),
                                head_aggregation="mean"):
    """Build per-token attention tensors + per-head statistics.

    Returns a dict with:
        per_layer_matrices_head_avg: {layer_id: np.ndarray [T, P]} (head-averaged)
        per_head_stats: {
            "entropy":            np.ndarray [L, H, T]  per-head, per-token entropy over patches
            "image_mass":         np.ndarray [L, H, T]  sum of attention on image patches
            "top1_concentration": np.ndarray [L, H, T]  max single-patch weight
        }
        layers: list of layer ids actually used (filtered to be < num_layers)
        num_heads: int
        num_question_tokens: int
        num_image_patches: int
    Returns None values when attention is missing.
    """
    attns = outputs.attentions
    if attns is None or len(attns) == 0:
        return None
    img_idx = token_indices.get("image_token_indices") or []
    q_idx = token_indices.get("question_token_indices") or []
    if not img_idx or not q_idx:
        return None

    num_layers_total = len(attns)
    layer_iter = [l for l in layers if l < num_layers_total] if layers else list(range(num_layers_total))
    if not layer_iter:
        return None

    sample_A = attns[layer_iter[0]][0]
    num_heads = sample_A.shape[0]
    seq_len = sample_A.shape[-1]

    q_clipped = [q for q in q_idx if q < seq_len]
    i_clipped = [i for i in img_idx if i < seq_len]
    if not q_clipped or not i_clipped:
        return None

    T = len(q_clipped)
    P = len(i_clipped)
    L = len(layer_iter)

    qi = torch.tensor(q_clipped, dtype=torch.long, device=sample_A.device)
    ii = torch.tensor(i_clipped, dtype=torch.long, device=sample_A.device)

    per_layer_matrices_head_avg = {}
    per_head_entropy = np.zeros((L, num_heads, T), dtype=np.float32)
    per_head_image_mass = np.zeros((L, num_heads, T), dtype=np.float32)
    per_head_top1 = np.zeros((L, num_heads, T), dtype=np.float32)

    eps = 1e-12
    for li, l in enumerate(layer_iter):
        A = attns[l][0]  # [heads, S, S]
        rows_full = A[:, qi, :]              # [heads, T, S]  (full row, all keys)
        rows_img = rows_full[:, :, ii]        # [heads, T, P]
        rows_img_f = rows_img.detach().to(torch.float32).cpu().numpy()
        rows_full_f = rows_full.detach().to(torch.float32).cpu().numpy()

        # head-averaged matrix for visualization
        if head_aggregation == "mean":
            head_avg = rows_img_f.mean(axis=0)  # [T, P]
        elif head_aggregation == "max":
            head_avg = rows_img_f.max(axis=0)
        else:
            head_avg = rows_img_f.mean(axis=0)
        per_layer_matrices_head_avg[int(l)] = head_avg

        # per-head stats
        # Image-mass = sum of attention to image patches (per head, per token)
        per_head_image_mass[li] = rows_img_f.sum(axis=2)                     # [H, T]
        # Top1 concentration = max single-patch weight (per head, per token)
        per_head_top1[li] = rows_img_f.max(axis=2)                            # [H, T]
        # Per-head entropy over the image-patch distribution (renormalized)
        sums = rows_img_f.sum(axis=2, keepdims=True)                          # [H, T, 1]
        p = rows_img_f / (sums + eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(p * np.log(p + eps)).sum(axis=2)                           # [H, T]
        per_head_entropy[li] = ent

    return {
        "per_layer_matrices_head_avg": per_layer_matrices_head_avg,
        "per_head_stats": {
            "entropy": per_head_entropy,
            "image_mass": per_head_image_mass,
            "top1_concentration": per_head_top1,
        },
        "layers": [int(l) for l in layer_iter],
        "num_heads": int(num_heads),
        "num_question_tokens": int(T),
        "num_image_patches": int(P),
        "question_token_indices_used": list(q_clipped),
    }
