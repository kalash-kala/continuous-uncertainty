"""Cosine distance / similarity utilities, numpy + torch friendly."""
import numpy as np


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def cosine_similarity(a, b, eps=1e-8):
    a = _to_np(a); b = _to_np(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + eps))


def cosine_distance(a, b, eps=1e-8):
    return 1.0 - cosine_similarity(a, b, eps=eps)
