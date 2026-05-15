"""Adjacent jumps, path length, center distances, ambiguity Spearman."""
import numpy as np
from scipy.stats import spearmanr
from .distance_utils import cosine_distance


def adjacent_jumps(features):
    return [cosine_distance(features[i], features[i + 1]) for i in range(len(features) - 1)]


def path_length(features):
    return float(sum(adjacent_jumps(features)))


def center_distances(features, center_index):
    center = features[center_index]
    return [cosine_distance(f, center) for f in features]


def ambiguity_spearman(values, frame_indices, center_index):
    ambiguity_distances = [abs(int(i) - int(center_index)) for i in frame_indices]
    if len(set(values)) < 2 or len(set(ambiguity_distances)) < 2:
        return float("nan")
    res = spearmanr(values, ambiguity_distances)
    corr = getattr(res, "correlation", None)
    if corr is None:
        corr = res[0]
    return float(corr)


def smoothness_block(features, frame_indices, center_index):
    """Return dict with mean/max adjacent jump, path length, center spearman."""
    jumps = adjacent_jumps(features)
    cds = center_distances(features, center_index)
    return {
        "adjacent_jumps": jumps,
        "mean_adjacent_jump": float(np.mean(jumps)) if jumps else float("nan"),
        "max_adjacent_jump": float(np.max(jumps)) if jumps else float("nan"),
        "path_length": float(sum(jumps)),
        "center_distances": cds,
        "center_distance_spearman": ambiguity_spearman(cds, frame_indices, center_index),
    }
