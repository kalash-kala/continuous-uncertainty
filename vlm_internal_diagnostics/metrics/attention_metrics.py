"""Attention entropy, attention jumps, attention-ambiguity correlations."""
import numpy as np
from .distance_utils import cosine_similarity
from .smoothness_metrics import ambiguity_spearman


def attention_entropy(attn, eps=1e-12):
    a = np.asarray(attn, dtype=np.float64).ravel()
    s = a.sum()
    if s <= 0:
        return float("nan")
    a = a / (s + eps)
    return float(-(a * np.log(a + eps)).sum())


def attention_jump(a, b):
    return 1.0 - cosine_similarity(a, b)


def adjacent_attention_jumps(attns):
    return [attention_jump(attns[i], attns[i + 1]) for i in range(len(attns) - 1)]


def attention_center_distances(attns, center_index):
    c = attns[center_index]
    return [1.0 - cosine_similarity(a, c) for a in attns]


def attention_block(attns, frame_indices, center_index):
    entropies = [attention_entropy(a) for a in attns]
    jumps = adjacent_attention_jumps(attns)
    center_d = attention_center_distances(attns, center_index)
    return {
        "entropies": entropies,
        "mean_entropy": float(np.nanmean(entropies)) if entropies else float("nan"),
        "jumps": jumps,
        "mean_jump": float(np.mean(jumps)) if jumps else float("nan"),
        "entropy_ambiguity_spearman": ambiguity_spearman(entropies, frame_indices, center_index),
        "center_distance_spearman": ambiguity_spearman(center_d, frame_indices, center_index),
    }
