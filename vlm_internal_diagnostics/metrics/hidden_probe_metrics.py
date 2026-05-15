"""Linear probe: predict ambiguity distance from hidden states.

Trains a per-layer ridge regression with train/test split by sequence,
reports R^2 and Spearman correlation.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit


def train_layerwise_probe(hidden_by_layer, ambiguity_distances, sequence_ids,
                          test_frac=0.3, alpha=1.0, seed=42):
    """hidden_by_layer: dict {layer -> np.ndarray [N, D]}.

    Returns {layer -> {r2, spearman, n_train, n_test}}.
    """
    ambiguity_distances = np.asarray(ambiguity_distances)
    sequence_ids = np.asarray(sequence_ids)
    if len(np.unique(sequence_ids)) < 2:
        return {layer: {"r2": float("nan"), "spearman": float("nan")}
                for layer in hidden_by_layer}

    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(gss.split(np.zeros(len(ambiguity_distances)),
                                         ambiguity_distances, sequence_ids))
    results = {}
    for layer, X in hidden_by_layer.items():
        X = np.asarray(X, dtype=np.float32)
        try:
            reg = Ridge(alpha=alpha)
            reg.fit(X[train_idx], ambiguity_distances[train_idx])
            pred = reg.predict(X[test_idx])
            r2 = float(reg.score(X[test_idx], ambiguity_distances[test_idx]))
            sp = spearmanr(pred, ambiguity_distances[test_idx])
            corr = getattr(sp, "correlation", None)
            if corr is None:
                corr = sp[0]
            results[layer] = {
                "r2": r2,
                "spearman": float(corr),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        except Exception as e:
            results[layer] = {"r2": float("nan"), "spearman": float("nan"), "error": str(e)}
    return results
