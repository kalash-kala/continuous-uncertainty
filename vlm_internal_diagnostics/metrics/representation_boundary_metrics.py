"""Boundary estimation from internal representations (vision / projector / hidden).

Four geometric methods to locate the ambiguity boundary in feature space:
  A) chord        – perpendicular distance from the f_0→f_9 chord
  B) angle        – discrete turning angle (curvature) at each interior frame
  E) local_var    – bilateral sum of adjacent cosine distances
  F) anchor_prod  – product of cosine distances to both anchor frames
"""
import math
import numpy as np
from .distance_utils import cosine_distance


# ---------------------------------------------------------------------------
# Method A: Perpendicular distance from the chord (f_0 → f_{N-1})
# ---------------------------------------------------------------------------

def chord_boundary(feats):
    """Return (boundary_index, per-frame scores).

    Score = Euclidean length of the component of (f_i - f_0) perpendicular
    to the chord connecting f_0 and f_{N-1}.
    """
    f0 = np.asarray(feats[0], dtype=np.float64)
    fn = np.asarray(feats[-1], dtype=np.float64)
    chord = fn - f0
    chord_dot = np.dot(chord, chord)

    scores = []
    for f in feats:
        fi = np.asarray(f, dtype=np.float64)
        diff = fi - f0
        if chord_dot < 1e-12:
            # Degenerate: anchors are identical, use raw distance from f0
            scores.append(float(np.linalg.norm(diff)))
        else:
            t = np.dot(diff, chord) / chord_dot
            perp = diff - t * chord
            scores.append(float(np.linalg.norm(perp)))

    boundary = int(np.argmax(scores))
    return boundary, scores


# ---------------------------------------------------------------------------
# Method B: Turning angle (discrete curvature) at interior frames
# ---------------------------------------------------------------------------

def turning_angle_boundary(feats):
    """Return (boundary_index, per-frame scores).

    For interior frames 1..N-2, compute the angle between incoming and
    outgoing direction vectors.  Frames 0 and N-1 get score 0.
    """
    n = len(feats)
    scores = [0.0] * n  # endpoints get 0

    for i in range(1, n - 1):
        vec_in = np.asarray(feats[i], dtype=np.float64) - np.asarray(feats[i - 1], dtype=np.float64)
        vec_out = np.asarray(feats[i + 1], dtype=np.float64) - np.asarray(feats[i], dtype=np.float64)
        norm_in = np.linalg.norm(vec_in)
        norm_out = np.linalg.norm(vec_out)
        if norm_in < 1e-12 or norm_out < 1e-12:
            scores[i] = 0.0
            continue
        cos_angle = np.dot(vec_in, vec_out) / (norm_in * norm_out)
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        scores[i] = math.acos(cos_angle)

    boundary = int(np.argmax(scores))
    return boundary, scores


# ---------------------------------------------------------------------------
# Method E: Local variation (bilateral adjacent-jump sum), interior only
# ---------------------------------------------------------------------------

def local_variation_boundary(feats):
    """Return (boundary_index, per-frame scores).

    For interior frames 1..N-2:
        score_i = cos_dist(f_{i-1}, f_i) + cos_dist(f_i, f_{i+1})
    Frames 0 and N-1 get score 0.
    """
    n = len(feats)
    scores = [0.0] * n

    for i in range(1, n - 1):
        scores[i] = cosine_distance(feats[i - 1], feats[i]) + \
                     cosine_distance(feats[i], feats[i + 1])

    boundary = int(np.argmax(scores))
    return boundary, scores


# ---------------------------------------------------------------------------
# Method F: Product of distances to both anchors
# ---------------------------------------------------------------------------

def anchor_product_boundary(feats):
    """Return (boundary_index, per-frame scores).

    score_i = cos_dist(f_i, f_0) * cos_dist(f_i, f_{N-1})
    """
    scores = []
    f0 = feats[0]
    fn = feats[-1]
    for f in feats:
        scores.append(cosine_distance(f, f0) * cosine_distance(f, fn))

    boundary = int(np.argmax(scores))
    return boundary, scores


# ---------------------------------------------------------------------------
# Public API: compute all four boundary errors for a feature list
# ---------------------------------------------------------------------------

ALL_METHODS = {
    "chord": chord_boundary,
    "angle": turning_angle_boundary,
    "local_var": local_variation_boundary,
    "anchor_product": anchor_product_boundary,
}


def representation_boundary_block(feats, center_index):
    """Compute boundary index and error for all four methods.

    Parameters
    ----------
    feats : list[np.ndarray]
        One pooled feature vector per frame (length N).
    center_index : int
        Ground-truth max-ambiguity frame index.

    Returns
    -------
    dict  with keys ``{method}_boundary_index`` and ``{method}_boundary_error``
          for each of the four methods.
    """
    out = {}
    for name, fn in ALL_METHODS.items():
        bi, _scores = fn(feats)
        out[f"{name}_boundary_index"] = int(bi)
        out[f"{name}_boundary_error"] = int(abs(bi - center_index))
    return out
