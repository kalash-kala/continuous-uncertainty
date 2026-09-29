"""Per-question-token attention metrics and head-specialization metrics.

Operates on the per-frame per-token attention payloads (head-averaged matrices
plus per-head summary stats) and the list of `SelectedToken` returned by the
token selector. Yields:

- Per-token (per-sequence) metrics:
    attention_entropy_per_frame  [N_frames]
    adjacent_jumps               [N_frames - 1]
    center_distance_spearman     scalar
    temporal_variance            scalar (mean over patches of per-patch variance)
    center_vs_clear_diff_norm    scalar (L2 norm of center-minus-mean-farthest)

- Per-token head specialization (per-layer):
    js_divergence_by_layer                   list of len(L)
    entropy_spread_by_layer                  list of len(L)
    image_mass_spread_by_layer               list of len(L)
    top_head_vs_mean_divergence_by_layer     list of len(L)
    best_image_mass_head_by_layer            list of len(L)

- Aggregate (mean across selected tokens) — for CSV.
"""
import numpy as np

from .distance_utils import cosine_similarity
from .smoothness_metrics import ambiguity_spearman


_EPS = 1e-12


def _to_np(x):
    if hasattr(x, "float") and hasattr(x, "numpy"):
        return x.float().cpu().numpy()
    return np.asarray(x)


def _resolve_layer_index(payload, layer):
    """Return position in layers list, or None to mean 'average all layers'."""
    layers = payload.get("layers")
    if layer is None or layers is None:
        return None
    if layer in layers:
        return layers.index(layer)
    return None


def _get_token_attention_across_frames(payloads, position, layer=None):
    """Returns np.ndarray [F, P] for one token across frames (avg across layers if layer is None)."""
    vecs = []
    for p in payloads:
        mats = p.get("per_layer_matrices_head_avg") or {}
        if not mats:
            continue
        if layer is not None and layer in mats:
            m = _to_np(mats[layer])
        else:
            m = np.stack([_to_np(v) for v in mats.values()], axis=0).mean(axis=0)
        if 0 <= position < m.shape[0]:
            vecs.append(m[position])
    if not vecs:
        return None
    return np.stack(vecs, axis=0)


def _attention_entropy(vec):
    s = vec.sum()
    if s <= 0:
        return float("nan")
    p = vec / (s + _EPS)
    return float(-(p * np.log(p + _EPS)).sum())


def _adjacent_jumps(stack):
    return [float(1.0 - cosine_similarity(stack[i], stack[i + 1]))
            for i in range(stack.shape[0] - 1)]


def _center_distances(stack, center_index):
    c = stack[center_index]
    return [float(1.0 - cosine_similarity(stack[i], c)) for i in range(stack.shape[0])]


def _l2(x):
    return float(np.sqrt((x * x).sum()))


def _jensen_shannon_pairwise_mean(P):
    """P: [H, K] non-negative; rows are distributions (will be renormalized).
    Returns mean pairwise JS divergence across rows (in nats)."""
    H, K = P.shape
    if H < 2:
        return float("nan")
    s = P.sum(axis=1, keepdims=True)
    P = P / (s + _EPS)
    out = 0.0
    cnt = 0
    for i in range(H):
        for j in range(i + 1, H):
            m = 0.5 * (P[i] + P[j])
            kl_im = (P[i] * (np.log(P[i] + _EPS) - np.log(m + _EPS))).sum()
            kl_jm = (P[j] * (np.log(P[j] + _EPS) - np.log(m + _EPS))).sum()
            out += 0.5 * (kl_im + kl_jm)
            cnt += 1
    return float(out / cnt) if cnt > 0 else float("nan")


def _compute_head_specialization_for_token(payloads, position):
    """Aggregate head-specialization metrics across frames for one token.

    For each layer we average the per-head distributions across frames and
    then compute the divergence metrics on that averaged distribution.
    """
    # Collect per-frame per-head attention vectors at this token position.
    # We need the full A[heads, P] for this token, but we only saved the
    # head-averaged matrix. So we use per-head stats (entropy/mass/top1)
    # for the spread metrics, and use the per-head distributions reconstructed
    # via... actually we don't have them. Solution: compute the metrics that
    # are computable from the saved stats only.
    #
    # That gives us:
    #   entropy_spread_by_layer        from per_head_stats["entropy"]
    #   image_mass_spread_by_layer     from per_head_stats["image_mass"]
    #   best_image_mass_head_by_layer  argmax over heads of image_mass
    # The JS-divergence and top-head-vs-mean require the full [H, P] which we
    # didn't save. To keep storage tiny we approximate "head agreement" using
    # the spread of attention concentration (top1) across heads, normalized.

    layers_global = None
    F_collected = []
    for p in payloads:
        stats = p.get("per_head_stats") or {}
        ent = stats.get("entropy")
        mass = stats.get("image_mass")
        top1 = stats.get("top1_concentration")
        if ent is None or mass is None or top1 is None:
            continue
        ent_np = _to_np(ent)        # [L, H, T]
        mass_np = _to_np(mass)      # [L, H, T]
        top1_np = _to_np(top1)      # [L, H, T]
        if position >= ent_np.shape[2]:
            continue
        F_collected.append((ent_np[..., position], mass_np[..., position], top1_np[..., position]))
        layers_global = p.get("layers")

    if not F_collected:
        return None

    # Stack across frames: -> [F, L, H]
    ent_stack = np.stack([f[0] for f in F_collected], axis=0)
    mass_stack = np.stack([f[1] for f in F_collected], axis=0)
    top1_stack = np.stack([f[2] for f in F_collected], axis=0)

    # Mean across frames (for stable per-layer metrics)
    ent_mean = ent_stack.mean(axis=0)    # [L, H]
    mass_mean = mass_stack.mean(axis=0)  # [L, H]
    top1_mean = top1_stack.mean(axis=0)  # [L, H]

    L = ent_mean.shape[0]
    entropy_spread = ent_mean.std(axis=1).tolist()           # [L]
    image_mass_spread = mass_mean.std(axis=1).tolist()       # [L]
    top1_spread = top1_mean.std(axis=1).tolist()             # [L]
    best_head = np.argmax(mass_mean, axis=1).tolist()        # [L]

    # As a proxy for "top head vs head-avg" we use:
    #   |top1[best_head] - mean(top1 across heads)| / (mean + eps)
    # This tells us whether the most image-focused head has a markedly different
    # concentration pattern than the average head.
    top_head_vs_mean_div = []
    for li in range(L):
        bh = best_head[li]
        peak = float(top1_mean[li, bh])
        avg = float(top1_mean[li].mean())
        denom = abs(avg) + _EPS
        top_head_vs_mean_div.append(float(abs(peak - avg) / denom))

    return {
        "layers": list(layers_global) if layers_global else list(range(L)),
        "entropy_spread_by_layer": entropy_spread,
        "image_mass_spread_by_layer": image_mass_spread,
        "top1_concentration_spread_by_layer": top1_spread,
        "top_head_vs_mean_divergence_by_layer": top_head_vs_mean_div,
        "best_image_mass_head_by_layer": [int(x) for x in best_head],
        # JS divergence requires per-head [H, P] distributions which aren't saved;
        # we report None so downstream code knows.
        "js_divergence_by_layer": [None] * L,
    }


def compute_per_token_metrics(payloads, selected_tokens, frame_records,
                              num_farthest_frames=4, layer=None):
    """Main entry: returns nested per-token dict + sequence-level aggregate dict."""
    if not selected_tokens or not payloads:
        return {"selected_tokens": [], "aggregate": {}}

    frame_indices = [r["frame_index"] for r in frame_records]
    center_index = frame_records[0]["max_ambiguity_index"]
    # Find position of the center frame in our payloads list
    center_pos = None
    for i, r in enumerate(frame_records):
        if r["frame_index"] == center_index:
            center_pos = i
            break

    # Determine the farthest-frame positions (by index distance from center)
    dists = [abs(fi - center_index) for fi in frame_indices]
    far_positions = sorted(range(len(frame_records)), key=lambda i: -dists[i])[:num_farthest_frames]

    per_token_out = []
    agg_entropy_per_frame = []
    agg_jump = []
    agg_center_spear = []
    agg_temporal_var = []
    agg_center_vs_clear = []

    agg_entropy_spread_layers = []
    agg_image_mass_spread_layers = []
    agg_top_head_vs_mean_layers = []

    for tok in selected_tokens:
        pos = getattr(tok, "position", None) if not isinstance(tok, dict) else tok.get("position")
        stack = _get_token_attention_across_frames(payloads, pos, layer=layer)
        if stack is None:
            continue

        entropies = [_attention_entropy(stack[i]) for i in range(stack.shape[0])]
        jumps = _adjacent_jumps(stack)
        center_dists = _center_distances(stack, center_pos) if center_pos is not None else []
        center_spear = (ambiguity_spearman(center_dists, frame_indices, center_index)
                        if center_dists else float("nan"))

        per_patch_var = stack.var(axis=0)  # [P]
        temporal_var = float(per_patch_var.mean())

        if center_pos is not None and far_positions:
            clear_mean = stack[far_positions].mean(axis=0)
            center_minus_clear = stack[center_pos] - clear_mean
            cvc_norm = _l2(center_minus_clear)
        else:
            cvc_norm = float("nan")

        head_spec = _compute_head_specialization_for_token(payloads, pos)

        tok_dict = {
            "position": int(pos),
            "text": getattr(tok, "text", None) if not isinstance(tok, dict) else tok.get("text"),
            "pos": getattr(tok, "pos", None) if not isinstance(tok, dict) else tok.get("pos"),
            "spacy_text": (getattr(tok, "spacy_text", None) if not isinstance(tok, dict)
                           else tok.get("spacy_text")),
            "is_stop": (getattr(tok, "is_stop", False) if not isinstance(tok, dict)
                        else tok.get("is_stop", False)),
            "score": (getattr(tok, "score", None) if not isinstance(tok, dict)
                      else tok.get("score")),
            "attention_entropy_per_frame": [float(x) for x in entropies],
            "adjacent_jumps": [float(x) for x in jumps],
            "center_distance_spearman": float(center_spear) if center_spear == center_spear else None,
            "temporal_variance": float(temporal_var),
            "center_vs_clear_diff_norm": float(cvc_norm) if cvc_norm == cvc_norm else None,
            "head_specialization": head_spec,
        }
        per_token_out.append(tok_dict)

        if not np.isnan(np.nanmean(entropies)):
            agg_entropy_per_frame.append(float(np.nanmean(entropies)))
        if jumps:
            agg_jump.append(float(np.mean(jumps)))
        if not np.isnan(center_spear):
            agg_center_spear.append(float(center_spear))
        agg_temporal_var.append(temporal_var)
        if cvc_norm == cvc_norm:
            agg_center_vs_clear.append(cvc_norm)
        if head_spec is not None:
            agg_entropy_spread_layers.append(np.mean(head_spec["entropy_spread_by_layer"]))
            agg_image_mass_spread_layers.append(np.mean(head_spec["image_mass_spread_by_layer"]))
            agg_top_head_vs_mean_layers.append(np.mean(head_spec["top_head_vs_mean_divergence_by_layer"]))

    aggregate = {}
    if agg_entropy_per_frame:
        aggregate["mean_attention_entropy_content_tokens"] = float(np.mean(agg_entropy_per_frame))
    if agg_jump:
        aggregate["mean_attention_jump_content_tokens"] = float(np.mean(agg_jump))
    if agg_center_spear:
        aggregate["mean_center_distance_spearman_content_tokens"] = float(np.mean(agg_center_spear))
    if agg_temporal_var:
        aggregate["mean_temporal_variance_content_tokens"] = float(np.mean(agg_temporal_var))
    if agg_center_vs_clear:
        aggregate["mean_center_vs_clear_diff_content_tokens"] = float(np.mean(agg_center_vs_clear))
    if agg_entropy_spread_layers:
        aggregate["mean_head_entropy_spread_content_tokens"] = float(np.mean(agg_entropy_spread_layers))
    if agg_image_mass_spread_layers:
        aggregate["mean_head_image_mass_spread_content_tokens"] = float(np.mean(agg_image_mass_spread_layers))
    if agg_top_head_vs_mean_layers:
        aggregate["mean_top_head_vs_mean_divergence_content_tokens"] = float(np.mean(agg_top_head_vs_mean_layers))

    return {"selected_tokens": per_token_out, "aggregate": aggregate}
