#!/usr/bin/env python
"""Phase 1 & 2: Representation Geometry Validation & Characterization.

Validates whether internal representations of 4 VLMs encode visual ambiguity
through orthogonal structure (decision axis + uncertainty manifold).

Five tests:
  1. PCA alignment with decision axis
  2. Orthogonal variance peaks at center
  3. Frame order recovery from distances
  4. Curvature & smoothness of trajectory
  5. Effective dimensionality (manifold)

Usage:
    conda run -n continuous-uncertainty python scripts/phase_1_2_analysis.py
"""

import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# ── Configuration ───────────────────────────────────────────────────────────

# Machine-local paths live in configs/paths.yaml (or the CU_* env vars).
# See scripts/_paths.py; run `python scripts/_paths.py` to check your setup.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import outputs_root  # noqa: E402

BASE_OUTPUT = outputs_root()

# Run directory per model. These names are the `run_name` used at extraction
# time (run_config.yaml); change them here only if you re-extracted under
# different names.
#
# NOTE on phi4: two run dirs exist on the lab server,
# `phi4_internal_diag_v1_question_mean_new` (used below, and the one every
# published number comes from) and the older `phi4_multimodal_...`. Use the
# former. Likewise for llava, `..._question_mean_new` supersedes the two
# earlier `llava_internal_diag_v1*` runs.
MODELS = {
    "llava":   BASE_OUTPUT / "llava_internal_diag_v1_question_mean_new",
    "phi4":    BASE_OUTPUT / "phi4_internal_diag_v1_question_mean_new",
    "pixtral": BASE_OUTPUT / "pixtral_internal_diag_v1_question_mean_new",
    "qwen":    BASE_OUTPUT / "qwen2_5_vl_internal_diag_v1_question_mean_new",
}

STAGES = ["vision", "projector", "hidden"]

RESULT_DIR = BASE_OUTPUT / "phase_1_2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 10
HIDDEN_AGG = "final_prompt"  # aggregation type for hidden states


# ── Data loading ────────────────────────────────────────────────────────────

def load_tensor(path):
    """Load a .pt file to CPU, return None on failure."""
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def to_vec(t):
    """Convert tensor to flat float64 numpy array."""
    if t is None:
        return None
    if hasattr(t, "float"):
        t = t.float()
    arr = np.asarray(t.numpy() if hasattr(t, "numpy") else t, dtype=np.float64)
    return arr.ravel()


def load_sequence_features(feat_dir, best_hidden_layer):
    """Load vision, projector, hidden features for one sequence.

    Returns dict: {stage: np.ndarray of shape (NUM_FRAMES, d)} or None per stage.
    """
    vision_list, proj_list, hidden_list = [], [], []

    for fi in range(NUM_FRAMES):
        stem = f"frame_{fi:02d}"

        # Vision
        v = load_tensor(feat_dir / f"{stem}_vision.pt")
        vec = to_vec(v)
        if vec is not None:
            vision_list.append(vec)

        # Projector
        p = load_tensor(feat_dir / f"{stem}_projected.pt")
        vec = to_vec(p)
        if vec is not None:
            proj_list.append(vec)

        # Hidden (best layer from final_prompt aggregation)
        h = load_tensor(feat_dir / f"{stem}_hidden.pt")
        if isinstance(h, dict) and HIDDEN_AGG in h:
            layer_dict = h[HIDDEN_AGG]
            layer_key = best_hidden_layer
            # Try int key first, then str
            if layer_key in layer_dict:
                vec = to_vec(layer_dict[layer_key])
            elif int(layer_key) in layer_dict:
                vec = to_vec(layer_dict[int(layer_key)])
            elif str(layer_key) in layer_dict:
                vec = to_vec(layer_dict[str(layer_key)])
            else:
                vec = None
            if vec is not None:
                hidden_list.append(vec)

    out = {}
    if len(vision_list) == NUM_FRAMES:
        out["vision"] = np.stack(vision_list)
    if len(proj_list) == NUM_FRAMES:
        out["projector"] = np.stack(proj_list)
    if len(hidden_list) == NUM_FRAMES:
        out["hidden"] = np.stack(hidden_list)

    return out


# ── Test 1: PCA Alignment with Decision Axis ───────────────────────────────

def test_pca_alignment(reps, center_idx):
    """PC1 alignment with chord, monotonicity, variance ratio."""
    reps_centered = reps - reps.mean(axis=0)
    U, S, Vt = np.linalg.svd(reps_centered, full_matrices=False)
    pcs = U * S  # PC scores, shape (10, k)

    var_ratio = S ** 2 / (S ** 2).sum()

    # Decision axis: chord from frame 0 to frame 9
    v_chord = reps[-1] - reps[0]
    chord_norm = np.linalg.norm(v_chord)
    if chord_norm < 1e-12:
        return {
            "pc1_alignment_with_chord": 0.0,
            "pc1_monotonicity_violations": NUM_FRAMES - 1,
            "pc1_variance_ratio": float(var_ratio[0]) if len(var_ratio) > 0 else 0.0,
            "decision_axis_magnitude": 0.0,
            "test1_pass": 0,
        }
    v_chord_norm = v_chord / chord_norm

    # Alignment: cosine between PC1 loading and chord direction
    pc1_loading = Vt[0, :]
    cos_angle = abs(np.dot(pc1_loading, v_chord_norm))

    # Monotonicity of PC1 scores
    pc1_scores = pcs[:, 0]
    violations = sum(
        1 for i in range(NUM_FRAMES - 1)
        if pc1_scores[i] > pc1_scores[i + 1]
    )
    # Check if reverse direction is more monotonic
    rev_violations = sum(
        1 for i in range(NUM_FRAMES - 1)
        if pc1_scores[i] < pc1_scores[i + 1]
    )
    violations = min(violations, rev_violations)

    pc1_var = float(var_ratio[0])

    # Pass criteria
    pass_alignment = cos_angle > 0.7
    pass_mono = violations <= 1
    pass_var = pc1_var > 0.5
    test1_pass = int(pass_alignment and pass_mono and pass_var)

    return {
        "pc1_alignment_with_chord": float(cos_angle),
        "pc1_monotonicity_violations": int(violations),
        "pc1_variance_ratio": pc1_var,
        "decision_axis_magnitude": float(chord_norm),
        "test1_pass": test1_pass,
    }


# ── Test 2: Orthogonal Variance Peaks at Center ────────────────────────────

def test_orthogonal_variance(reps, center_idx):
    """Decision axis projection, orthogonal component, Spearman correlation."""
    reps_centered = reps - reps.mean(axis=0)

    v_chord = reps[-1] - reps[0]
    chord_norm = np.linalg.norm(v_chord)
    if chord_norm < 1e-12:
        return {
            "decision_axis_monotonic": False,
            "decision_axis_variance": 0.0,
            "orthogonal_variance_mean": 0.0,
            "orthogonal_variance_peak_frame": -1,
            "orthogonal_variance_center_spearman": 0.0,
            "orthogonal_variance_center_p_value": 1.0,
            "orthogonal_variance_shows_correct_peak": False,
            "test2_pass": 0,
        }
    v_chord_unit = v_chord / chord_norm

    # Scalar projection of each frame onto decision axis
    lambda_i = reps @ v_chord_unit

    # Check monotonicity
    is_mono_inc = all(lambda_i[i] <= lambda_i[i + 1] for i in range(NUM_FRAMES - 1))
    is_mono_dec = all(lambda_i[i] >= lambda_i[i + 1] for i in range(NUM_FRAMES - 1))
    is_monotonic = is_mono_inc or is_mono_dec

    # Orthogonal component
    reps_proj = np.outer(lambda_i, v_chord_unit)
    reps_ortho = reps_centered - reps_proj
    ortho_var = np.linalg.norm(reps_ortho, axis=1) ** 2

    # Check peak at center
    distance_from_center = np.abs(np.arange(NUM_FRAMES) - center_idx)
    rho, pval = spearmanr(ortho_var, distance_from_center)
    if np.isnan(rho):
        rho = 0.0
        pval = 1.0

    peak_frame = int(np.argmax(ortho_var))
    correct_peak = abs(peak_frame - center_idx) <= 1

    # Pass criteria (2/3 required)
    passes = [is_monotonic, rho < -0.5, correct_peak]
    test2_pass = int(sum(passes) >= 2)

    return {
        "decision_axis_monotonic": bool(is_monotonic),
        "decision_axis_variance": float(np.var(lambda_i)),
        "orthogonal_variance_mean": float(ortho_var.mean()),
        "orthogonal_variance_peak_frame": peak_frame,
        "orthogonal_variance_center_spearman": float(rho),
        "orthogonal_variance_center_p_value": float(pval),
        "orthogonal_variance_shows_correct_peak": bool(correct_peak),
        "test2_pass": test2_pass,
    }


# ── Test 3: Frame Order Recovery ───────────────────────────────────────────

def longest_increasing_subseq(arr):
    """Length of longest increasing subsequence."""
    n = len(arr)
    if n == 0:
        return 0
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def test_frame_order_recovery(reps):
    """Nearest-neighbor chain from frame 0, check order recovery."""
    distances = squareform(pdist(reps, metric="cosine"))

    nn_order = [0]
    visited = {0}
    current = 0

    for _ in range(NUM_FRAMES - 1):
        dists = distances[current].copy()
        dists[list(visited)] = np.inf
        next_frame = int(np.argmin(dists))
        nn_order.append(next_frame)
        visited.add(next_frame)
        current = next_frame

    nn_arr = np.array(nn_order)
    perfect = np.array_equal(nn_arr, np.arange(NUM_FRAMES))
    lis_len = longest_increasing_subseq(nn_arr)
    lis_ratio = lis_len / NUM_FRAMES

    test3_pass = int(lis_ratio > 0.5)

    return {
        "nn_order_perfect_match": bool(perfect),
        "nn_order_lis_ratio": float(lis_ratio),
        "nn_order": nn_arr.tolist(),
        "test3_pass": test3_pass,
    }


# ── Test 4: Curvature & Smoothness ─────────────────────────────────────────

def test_curvature(reps, center_idx):
    """Discrete curvature at edges, peak location, Spearman with center distance."""
    velocities = np.diff(reps, axis=0)  # (9, d)
    curvatures = []

    for i in range(len(velocities) - 1):
        v1, v2 = velocities[i], velocities[i + 1]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            curvatures.append(0.0)
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        curvatures.append(float(math.acos(cos_a)))

    curvatures = np.array(curvatures)  # (8,)

    # Acceleration magnitudes
    accelerations = np.diff(velocities, axis=0)  # (8, d)
    smoothness = np.linalg.norm(accelerations, axis=1)

    # Edge positions mapped to frame space
    edge_positions = np.arange(len(curvatures)) + 0.5
    dist_from_center = np.abs(edge_positions - center_idx)

    rho_curv, p_curv = spearmanr(curvatures, dist_from_center)
    if np.isnan(rho_curv):
        rho_curv = 0.0
        p_curv = 1.0

    max_curv_idx = int(np.argmax(curvatures))
    max_curv_pos = max_curv_idx + 0.5
    dist_to_center = abs(max_curv_pos - center_idx)

    # Pass criteria (2/3)
    passes = [
        float(curvatures.mean()) > 0.15,
        dist_to_center <= 1.5,
        rho_curv < -0.3,
    ]
    test4_pass = int(sum(passes) >= 2)

    return {
        "mean_curvature_radians": float(curvatures.mean()),
        "max_curvature_radians": float(curvatures.max()),
        "std_curvature_radians": float(curvatures.std()),
        "curvature_peak_distance_to_center": float(dist_to_center),
        "curvature_center_spearman": float(rho_curv),
        "curvature_center_p_value": float(p_curv),
        "curvature_peaks_near_center": bool(dist_to_center <= 1.5),
        "mean_smoothness": float(smoothness.mean()),
        "max_smoothness": float(smoothness.max()),
        "test4_pass": test4_pass,
    }


# ── Test 5: Effective Dimensionality ───────────────────────────────────────

def test_effective_dimensionality(reps):
    """SVD-based effective dimensionality and compression ratio."""
    reps_centered = reps - reps.mean(axis=0)
    U, S, Vt = np.linalg.svd(reps_centered, full_matrices=False)

    var_explained = S ** 2
    total_var = var_explained.sum()
    if total_var < 1e-12:
        return {
            "original_dimension": int(reps.shape[1]),
            "effective_dim_90pct": int(reps.shape[1]),
            "effective_dim_95pct": int(reps.shape[1]),
            "effective_dim_99pct": int(reps.shape[1]),
            "compression_ratio_95pct": 1.0,
            "variance_by_top_10_pcs": 0.0,
            "is_low_dimensional": False,
            "test5_pass": 0,
        }

    cumsum_var = np.cumsum(var_explained) / total_var
    original_dim = reps.shape[1]

    # +1 because argmax returns 0-based index of first True
    eff_90 = int(np.argmax(cumsum_var >= 0.90)) + 1
    eff_95 = int(np.argmax(cumsum_var >= 0.95)) + 1
    eff_99 = int(np.argmax(cumsum_var >= 0.99)) + 1

    compression_95 = original_dim / max(eff_95, 1)

    top_10_idx = min(9, len(cumsum_var) - 1)
    top_10_var = float(cumsum_var[top_10_idx])

    is_low_dim = eff_95 < original_dim / 10

    test5_pass = int(is_low_dim)

    return {
        "original_dimension": int(original_dim),
        "effective_dim_90pct": eff_90,
        "effective_dim_95pct": eff_95,
        "effective_dim_99pct": eff_99,
        "compression_ratio_95pct": float(compression_95),
        "variance_by_top_10_pcs": top_10_var,
        "is_low_dimensional": bool(is_low_dim),
        "test5_pass": test5_pass,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_pipeline():
    all_results = []
    skipped = []

    for model_name, model_dir in MODELS.items():
        csv_path = model_dir / "aggregate_metrics.csv"
        if not csv_path.exists():
            print(f"[SKIP] No aggregate_metrics.csv for {model_name}")
            continue

        df = pd.read_csv(csv_path)
        feat_base = model_dir / "features"
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}  ({len(df)} sequences)")
        print(f"{'='*70}")

        for idx, row in df.iterrows():
            seq_id = row["sequence_id"]
            center_idx = int(row["max_ambiguity_index"])
            category = row["category"]
            best_layer = int(row["best_hidden_layer_by_spearman"]) if pd.notna(row.get("best_hidden_layer_by_spearman")) else 0

            # Find the sequence feature directory
            seq_feat_dir = feat_base / seq_id
            if not seq_feat_dir.exists():
                skipped.append((model_name, seq_id, "feature dir not found"))
                continue

            # Load features
            feats = load_sequence_features(seq_feat_dir, best_layer)

            for stage in STAGES:
                if stage not in feats:
                    skipped.append((model_name, seq_id, f"{stage} features missing"))
                    continue

                reps = feats[stage]

                # Check for NaN/Inf
                if not np.all(np.isfinite(reps)):
                    skipped.append((model_name, seq_id, f"{stage} has NaN/Inf"))
                    continue

                # Run all 5 tests
                t1 = test_pca_alignment(reps, center_idx)
                t2 = test_orthogonal_variance(reps, center_idx)
                t3 = test_frame_order_recovery(reps)
                t4 = test_curvature(reps, center_idx)
                t5 = test_effective_dimensionality(reps)

                combined = {
                    "model": model_name,
                    "stage": stage,
                    "sequence_id": seq_id,
                    "category": category,
                    "center_idx": center_idx,
                    **t1, **t2, **t3, **t4, **t5,
                }
                all_results.append(combined)

            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{len(df)} sequences")

        print(f"  Done: {len(df)} sequences processed for {model_name}")

    if skipped:
        print(f"\n[INFO] Skipped {len(skipped)} (model, seq, stage) entries:")
        for m, s, reason in skipped[:20]:
            print(f"  {m} / {s}: {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    return pd.DataFrame(all_results)


# ── Summary statistics ──────────────────────────────────────────────────────

def generate_summary(results_df):
    """Generate human-readable summary statistics."""
    lines = []
    lines.append("PHASE 1 & 2 VALIDATION SUMMARY")
    lines.append("=" * 30)
    lines.append("")

    total = len(results_df)
    n_models = results_df["model"].nunique()
    n_seqs = results_df.groupby("model")["sequence_id"].nunique().iloc[0]
    lines.append(f"Dataset: {n_seqs} sequences, {n_models} models, {len(STAGES)} stages each = {total} total tests")
    lines.append("")

    for stage in STAGES:
        sdf = results_df[results_df["stage"] == stage]
        n = len(sdf)
        if n == 0:
            continue

        stage_label = "HIDDEN (BEST LAYER)" if stage == "hidden" else stage.upper()
        lines.append(f"{'═'*3} STAGE: {stage_label} {'═' * (60 - len(stage_label))}")
        lines.append("")

        # Test 1
        t1_align = (sdf["pc1_alignment_with_chord"] > 0.7).sum()
        t1_mono = (sdf["pc1_monotonicity_violations"] <= 1).sum()
        t1_var = (sdf["pc1_variance_ratio"] > 0.5).sum()
        t1_all = sdf["test1_pass"].sum()
        mark = "✓" if t1_all / n > 0.7 else ("?" if t1_all / n > 0.5 else "✗")

        lines.append("Test 1: PCA Alignment with Decision Axis")
        lines.append(f"  PC1 alignment > 0.7:              {t1_align}/{n} ({100*t1_align/n:.1f}%)")
        lines.append(f"  Monotonicity violations ≤ 1:      {t1_mono}/{n} ({100*t1_mono/n:.1f}%)")
        lines.append(f"  PC1 variance ratio > 0.5:         {t1_var}/{n} ({100*t1_var/n:.1f}%)")
        lines.append(f"  → Sequences passing ALL:          {t1_all}/{n} ({100*t1_all/n:.1f}%) {mark}")
        lines.append(f"")
        lines.append(f"  Mean PC1 alignment: {sdf['pc1_alignment_with_chord'].mean():.3f} ± {sdf['pc1_alignment_with_chord'].std():.3f}")
        lines.append(f"  Mean PC1 variance ratio: {sdf['pc1_variance_ratio'].mean():.3f} ± {sdf['pc1_variance_ratio'].std():.3f}")
        lines.append(f"  Mean monotonicity violations: {sdf['pc1_monotonicity_violations'].mean():.2f} ± {sdf['pc1_monotonicity_violations'].std():.2f}")
        lines.append("")

        # Test 2
        t2_mono = sdf["decision_axis_monotonic"].sum()
        t2_spear = (sdf["orthogonal_variance_center_spearman"] < -0.5).sum()
        t2_peak = sdf["orthogonal_variance_shows_correct_peak"].sum()
        t2_all = sdf["test2_pass"].sum()
        mark = "✓" if t2_all / n > 0.7 else ("?" if t2_all / n > 0.5 else "✗")

        lines.append("Test 2: Orthogonal Variance Peak at Center")
        lines.append(f"  Decision axis monotonic:          {t2_mono}/{n} ({100*t2_mono/n:.1f}%)")
        lines.append(f"  Orthogonal variance Spearman < -0.5: {t2_spear}/{n} ({100*t2_spear/n:.1f}%)")
        lines.append(f"  Peak at correct frame:            {t2_peak}/{n} ({100*t2_peak/n:.1f}%)")
        lines.append(f"  → Sequences passing ALL:          {t2_all}/{n} ({100*t2_all/n:.1f}%) {mark}")
        lines.append(f"")
        lines.append(f"  Mean orthogonal variance center Spearman: {sdf['orthogonal_variance_center_spearman'].mean():.3f} ± {sdf['orthogonal_variance_center_spearman'].std():.3f}")
        lines.append("")

        # Test 3
        t3_50 = (sdf["nn_order_lis_ratio"] > 0.5).sum()
        t3_70 = (sdf["nn_order_lis_ratio"] > 0.7).sum()
        t3_perf = sdf["nn_order_perfect_match"].sum()
        mark = "✓" if t3_50 / n > 0.7 else ("?" if t3_50 / n > 0.5 else "✗")

        lines.append("Test 3: Frame Order Recovery")
        lines.append(f"  LIS ratio > 0.5:                  {t3_50}/{n} ({100*t3_50/n:.1f}%)")
        lines.append(f"  LIS ratio > 0.7:                  {t3_70}/{n} ({100*t3_70/n:.1f}%)")
        lines.append(f"  → Sequences passing (>0.5):       {t3_50}/{n} ({100*t3_50/n:.1f}%) {mark}")
        lines.append(f"")
        lines.append(f"  Mean LIS ratio: {sdf['nn_order_lis_ratio'].mean():.3f} ± {sdf['nn_order_lis_ratio'].std():.3f}")
        lines.append(f"  Perfect matches (LIS=1.0):        {t3_perf}/{n} ({100*t3_perf/n:.1f}%)")
        lines.append("")

        # Test 4
        t4_curv = (sdf["mean_curvature_radians"] > 0.15).sum()
        t4_peak = sdf["curvature_peaks_near_center"].sum()
        t4_spear = (sdf["curvature_center_spearman"] < -0.3).sum()
        t4_all = sdf["test4_pass"].sum()
        mark = "✓" if t4_all / n > 0.7 else ("?" if t4_all / n > 0.5 else "✗")

        lines.append("Test 4: Trajectory Curvature")
        lines.append(f"  Mean curvature > 0.15 rad:        {t4_curv}/{n} ({100*t4_curv/n:.1f}%)")
        lines.append(f"  Curvature peaks near center:      {t4_peak}/{n} ({100*t4_peak/n:.1f}%)")
        lines.append(f"  Spearman < -0.3:                  {t4_spear}/{n} ({100*t4_spear/n:.1f}%)")
        lines.append(f"  → Sequences passing ALL:          {t4_all}/{n} ({100*t4_all/n:.1f}%) {mark}")
        lines.append(f"")
        mean_curv_deg = math.degrees(sdf['mean_curvature_radians'].mean())
        max_curv_deg = math.degrees(sdf['max_curvature_radians'].mean())
        lines.append(f"  Mean curvature: {sdf['mean_curvature_radians'].mean():.3f} ± {sdf['mean_curvature_radians'].std():.3f} radians (~{mean_curv_deg:.0f}°)")
        lines.append(f"  Max curvature: {sdf['max_curvature_radians'].mean():.3f} ± {sdf['max_curvature_radians'].std():.3f} radians (~{max_curv_deg:.0f}°)")
        lines.append("")

        # Test 5
        t5_comp = (sdf["compression_ratio_95pct"] > 10).sum()
        t5_dim = (sdf["effective_dim_95pct"] < 50).sum()
        t5_all = sdf["test5_pass"].sum()
        mark = "✓" if t5_all / n > 0.7 else ("?" if t5_all / n > 0.5 else "✗")

        lines.append("Test 5: Effective Dimensionality")
        lines.append(f"  Compression ratio > 10x:          {t5_comp}/{n} ({100*t5_comp/n:.1f}%)")
        lines.append(f"  Effective dim < 50:               {t5_dim}/{n} ({100*t5_dim/n:.1f}%)")
        lines.append(f"  → Sequences passing ALL:          {t5_all}/{n} ({100*t5_all/n:.1f}%) {mark}")
        lines.append(f"")
        lines.append(f"  Mean compression ratio: {sdf['compression_ratio_95pct'].mean():.1f}x ± {sdf['compression_ratio_95pct'].std():.1f}x")
        lines.append(f"  Mean effective dim (95% var): {sdf['effective_dim_95pct'].mean():.1f} ± {sdf['effective_dim_95pct'].std():.1f}")
        lines.append(f"  Median effective dim: {sdf['effective_dim_95pct'].median():.0f}")
        lines.append("")

    # Cross-model comparison
    lines.append("=" * 70)
    lines.append("CROSS-MODEL COMPARISON")
    lines.append("=" * 70)
    lines.append("")

    for model in MODELS:
        mdf = results_df[results_df["model"] == model]
        if len(mdf) == 0:
            continue
        nm = len(mdf)
        lines.append(f"Model: {model}")
        for ti in range(1, 6):
            col = f"test{ti}_pass"
            rate = mdf[col].mean() * 100
            mark = "✓" if rate > 70 else ("?" if rate > 50 else "✗")
            lines.append(f"  Test {ti} pass rate (all stages):    {rate:.1f}% {mark}")
        lines.append("")

    # Category breakdown
    lines.append("=" * 70)
    lines.append("BREAKDOWN BY CATEGORY")
    lines.append("=" * 70)
    lines.append("")

    for cat in sorted(results_df["category"].unique()):
        cdf = results_df[results_df["category"] == cat]
        nc = len(cdf)
        n_seq = cdf["sequence_id"].nunique()
        lines.append(f"Category: {cat.upper()} ({n_seq} sequences, all stages = {nc} tests)")
        for ti in range(1, 6):
            col = f"test{ti}_pass"
            rate = cdf[col].mean() * 100
            mark = "✓" if rate > 70 else ("?" if rate > 50 else "✗")
            lines.append(f"  Test {ti} pass rate: {rate:.1f}% {mark}")
        lines.append("")

    return "\n".join(lines)


# ── Validation report ───────────────────────────────────────────────────────

def generate_validation_report(results_df):
    """Decision tree and Phase 3 recommendations."""
    lines = []
    lines.append("PHASE 1 & 2 VALIDATION REPORT")
    lines.append("=" * 30)
    lines.append("")

    # Compute overall pass rates per test (across all models/stages)
    rates = {}
    for ti in range(1, 6):
        rates[ti] = results_df[f"test{ti}_pass"].mean() * 100

    lines.append("Hypothesis: Representations encode ambiguity through orthogonal structure")
    lines.append("            (decision axis + orthogonal uncertainty subspace)")
    lines.append("")
    lines.append("Evidence:")
    lines.append("")

    test_names = {
        1: "PC1 alignment",
        2: "Orthogonal variance",
        3: "Frame order",
        4: "Curvature",
        5: "Dimensionality",
    }
    test_supports = {
        1: "Decision axis is discoverable via PCA",
        2: "Uncertainty manifests orthogonal to decision axis",
        3: "Most sequences have exploitable order",
        4: "Trajectory deviates from chord at ambiguity point",
        5: "Low-dimensional manifold structure exists",
    }

    for ti in range(1, 6):
        r = rates[ti]
        if r > 70:
            verdict = "✓ PASS"
        elif r > 50:
            verdict = "? MARGINAL"
        else:
            verdict = "✗ FAIL"
        lines.append(f"Test {ti} ({test_names[ti]}):")
        lines.append(f"  {verdict} across all stages ({r:.1f}% average)")
        lines.append(f"  → {'Supports' if r > 50 else 'Does not support'}: {test_supports[ti]}")
        lines.append("")

    # Decision tree
    lines.append("=" * 70)
    lines.append("VALIDATION DECISION")
    lines.append("=" * 70)
    lines.append("")

    t1_pass = rates[1] > 70
    t2_pass = rates[2] > 70
    t3_pass = rates[3] > 50
    t4_pass = rates[4] > 70
    t4_weak = rates[4] < 60
    t5_pass = rates[5] > 70

    # Mean effective dim
    mean_eff_dim = results_df["effective_dim_95pct"].mean()

    if t1_pass and t2_pass and t4_pass:
        lines.append("OVERALL RESULT: ✓✓✓ ORTHOGONAL STRUCTURE HYPOTHESIS IS CONFIRMED")
        lines.append("  → Decision: Proceed to Phase 3 (new geometric methods)")
        lines.append('  → Paper section: "Representations encode ambiguity orthogonally"')
    elif t1_pass and t2_pass and t4_weak:
        lines.append("OVERALL RESULT: ✓ STRUCTURE EXISTS BUT TRAJECTORY IS LESS CURVED")
        lines.append("  → Decision: Skip manifold methods (G3), focus on simpler methods (G1, G2, G4)")
        lines.append('  → Paper section: "Geometric structure is evident but subtle"')
    elif t1_pass and t2_pass:
        lines.append("OVERALL RESULT: ✓✓ STRUCTURE EXISTS")
        lines.append("  → Decision: Proceed to Phase 3 with caution")
    else:
        lines.append("OVERALL RESULT: ? STRUCTURE IS WEAK OR ABSENT")
        lines.append("  → Decision: Re-examine representations and methodology")
    lines.append("")

    if not t3_pass:
        lines.append("⚠ Test 3 (frame order recovery) MARGINAL/FAILED (<50%)")
        lines.append("  → Manifold structure not strong enough for Isomap")
        lines.append("  → Decision: Skip method G3 (geodesic distance)")
        lines.append("")

    if mean_eff_dim < 5:
        lines.append("★ Very low-dimensional manifold discovered (mean eff dim < 5)")
        lines.append("  → Decision: HIGHLY encourage method G3 (Isomap)")
        lines.append("")

    # Counting passes
    num_strong = sum(1 for ti in range(1, 6) if rates[ti] > 70)
    lines.append(f"Tests strongly passing (>70%): {num_strong} / 5")
    lines.append("")

    # Phase 3 recommendations
    lines.append("=" * 70)
    lines.append("RECOMMENDED NEXT STEPS (PHASE 3)")
    lines.append("=" * 70)
    lines.append("")

    if rates[4] > 70:
        lines.append("✓ IMPLEMENT METHOD G1 (Curvature-weighted):")
        lines.append("  Rationale: Test 4 shows curvature reliably peaks at center")
    else:
        lines.append("? CONSIDER METHOD G1 (Curvature-weighted):")
        lines.append("  Rationale: Test 4 results are marginal")
    lines.append("")

    if rates[2] > 70:
        lines.append("✓ IMPLEMENT METHOD G2 (Orthogonal variance peak):")
        lines.append("  Rationale: Test 2 shows orthogonal variance encoded cleanly")
    else:
        lines.append("? CONSIDER METHOD G2 (Orthogonal variance peak):")
        lines.append("  Rationale: Test 2 results are not fully conclusive")
    lines.append("")

    if t3_pass and t5_pass and mean_eff_dim < 50:
        lines.append("✓ IMPLEMENT METHOD G3 (Geodesic/Isomap):")
        lines.append(f"  Rationale: Test 5 shows {results_df['compression_ratio_95pct'].mean():.0f}x dimensionality reduction possible")
    else:
        lines.append("? CONSIDER METHOD G3 (Geodesic/Isomap) with caution:")
        lines.append("  Rationale: Frame order recovery or dimensionality results are weak")
    lines.append("")

    lines.append("? CONSIDER METHOD G4 (Density anomaly):")
    lines.append("  Rationale: Not explicitly tested, but compatible with manifold structure")
    lines.append("")

    # Key findings
    lines.append("=" * 70)
    lines.append("KEY FINDINGS FOR PAPER")
    lines.append("=" * 70)
    lines.append("")

    lines.append("Finding 1: Decision Axis is Principal Component")
    lines.append(f"  - PC1 aligns with chord direction (cos = {results_df['pc1_alignment_with_chord'].mean():.2f} ± {results_df['pc1_alignment_with_chord'].std():.2f})")
    lines.append(f"  - Explains ~{results_df['pc1_variance_ratio'].mean()*100:.0f}% of variance on average")
    lines.append("")

    lines.append("Finding 2: Orthogonal Encoding of Uncertainty")
    lines.append(f"  - Spearman correlation: ρ = {results_df['orthogonal_variance_center_spearman'].mean():.2f} ± {results_df['orthogonal_variance_center_spearman'].std():.2f}")
    lines.append("")

    lines.append("Finding 3: Low-Dimensional Manifold")
    lines.append(f"  - ~{results_df['compression_ratio_95pct'].mean():.0f}x compression from original to effective dimensionality")
    lines.append(f"  - {results_df['effective_dim_95pct'].mean():.0f} dimensions (vs {results_df['original_dimension'].mean():.0f} original) capture 95% variance")
    lines.append("")

    lines.append("Finding 4: Curvature Concentration")
    mean_curv_deg = math.degrees(results_df['mean_curvature_radians'].mean())
    lines.append(f"  - Trajectory curvature ~{mean_curv_deg:.0f}° on average")
    pct_near = results_df['curvature_peaks_near_center'].mean() * 100
    lines.append(f"  - Highest curvature within 1.5 frames of center in {pct_near:.0f}% of sequences")
    lines.append("")

    lines.append("Finding 5: Cross-Architecture Consistency")
    for model in MODELS:
        mdf = results_df[results_df["model"] == model]
        if len(mdf) == 0:
            continue
        strong = sum(1 for ti in range(1, 6) if mdf[f"test{ti}_pass"].mean() > 0.7)
        lines.append(f"  - {model}: {strong}/5 tests strongly pass")
    lines.append("")

    return "\n".join(lines)


# ── Visualizations ──────────────────────────────────────────────────────────

def generate_visualizations(results_df):
    """Generate all Phase 1&2 visualizations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[WARN] matplotlib not available, skipping visualizations")
        return

    fig_dir = RESULT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── 1. Heatmap: pass rates (tests × models × stages) ───────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    models_list = [m for m in MODELS if m in results_df["model"].values]
    col_labels = []
    data_matrix = []

    for ti in range(1, 6):
        row = []
        for model in models_list:
            for stage in STAGES:
                sub = results_df[(results_df["model"] == model) & (results_df["stage"] == stage)]
                if len(sub) == 0:
                    row.append(np.nan)
                else:
                    row.append(sub[f"test{ti}_pass"].mean() * 100)
                if ti == 1:
                    col_labels.append(f"{model}\n{stage}")
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)
    im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"Test {i}" for i in range(1, 6)])

    # Annotate cells
    for i in range(5):
        for j in range(len(col_labels)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 40 or val > 85 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Pass Rate (%)")
    ax.set_title("Phase 1 & 2: Pass Rates by Test × Model × Stage")
    plt.tight_layout()
    plt.savefig(fig_dir / "heatmap_pass_rates.png", dpi=150)
    plt.close()

    # ── 2. Histogram: PC1 alignment ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for i, stage in enumerate(STAGES):
        sdf = results_df[results_df["stage"] == stage]
        axes[i].hist(sdf["pc1_alignment_with_chord"], bins=25, alpha=0.7,
                      edgecolor="black", color="steelblue")
        axes[i].axvline(0.7, color="red", linestyle="--", label="Threshold (0.7)")
        axes[i].set_title(f"{stage.capitalize()}")
        axes[i].set_xlabel("cos(angle) with chord")
        axes[i].legend(fontsize=8)
    axes[0].set_ylabel("Count")
    fig.suptitle("Test 1: PC1 Alignment with Decision Axis", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "hist_pc1_alignment.png", dpi=150)
    plt.close()

    # ── 3. Histogram: Orthogonal variance Spearman ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for i, stage in enumerate(STAGES):
        sdf = results_df[results_df["stage"] == stage]
        axes[i].hist(sdf["orthogonal_variance_center_spearman"], bins=25,
                      alpha=0.7, edgecolor="black", color="darkorange")
        axes[i].axvline(-0.5, color="red", linestyle="--", label="Threshold (-0.5)")
        axes[i].set_title(f"{stage.capitalize()}")
        axes[i].set_xlabel("Spearman ρ")
        axes[i].legend(fontsize=8)
    axes[0].set_ylabel("Count")
    fig.suptitle("Test 2: Orthogonal Variance Center Spearman ρ", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "hist_orthogonal_spearman.png", dpi=150)
    plt.close()

    # ── 4. Histogram: LIS ratio ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for i, stage in enumerate(STAGES):
        sdf = results_df[results_df["stage"] == stage]
        axes[i].hist(sdf["nn_order_lis_ratio"], bins=15, alpha=0.7,
                      edgecolor="black", color="seagreen")
        axes[i].axvline(0.5, color="red", linestyle="--", label="Threshold (0.5)")
        axes[i].set_title(f"{stage.capitalize()}")
        axes[i].set_xlabel("LIS Ratio")
        axes[i].legend(fontsize=8)
    axes[0].set_ylabel("Count")
    fig.suptitle("Test 3: Frame Order Recovery (Longest Increasing Subsequence)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "hist_lis_ratio.png", dpi=150)
    plt.close()

    # ── 5. Scatter: curvature vs distance from center ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, stage in enumerate(STAGES):
        sdf = results_df[results_df["stage"] == stage]
        axes[i].scatter(sdf["curvature_peak_distance_to_center"],
                         sdf["max_curvature_radians"],
                         alpha=0.4, s=15, color="purple")
        axes[i].set_title(f"{stage.capitalize()}")
        axes[i].set_xlabel("Peak curvature distance to center")
        axes[i].set_ylabel("Max curvature (radians)")
        axes[i].axvline(1.5, color="red", linestyle="--", alpha=0.5)
    fig.suptitle("Test 4: Max Curvature vs Distance from Center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "scatter_curvature_vs_center.png", dpi=150)
    plt.close()

    # ── 6. Summary bar chart ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(5)
    width = 0.2
    offsets = np.arange(len(STAGES)) - (len(STAGES) - 1) / 2
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for si, stage in enumerate(STAGES):
        sdf = results_df[results_df["stage"] == stage]
        rates = [sdf[f"test{ti}_pass"].mean() * 100 for ti in range(1, 6)]
        ax.bar(x + offsets[si] * width, rates, width, label=stage.capitalize(),
               color=colors[si], edgecolor="black", linewidth=0.5)

    ax.axhline(70, color="red", linestyle="--", alpha=0.5, label="70% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Test {i}" for i in range(1, 6)])
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.set_title("Phase 1 & 2: Pass Rates by Test and Stage (All Models)")
    plt.tight_layout()
    plt.savefig(fig_dir / "bar_pass_rates.png", dpi=150)
    plt.close()

    # ── 7. Per-model bar chart ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for mi, model in enumerate(models_list):
        ax = axes[mi]
        mdf = results_df[results_df["model"] == model]
        for si, stage in enumerate(STAGES):
            sdf = mdf[mdf["stage"] == stage]
            rates = [sdf[f"test{ti}_pass"].mean() * 100 for ti in range(1, 6)]
            ax.bar(x + offsets[si] * width, rates, width, label=stage.capitalize(),
                   color=colors[si], edgecolor="black", linewidth=0.5)
        ax.axhline(70, color="red", linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{i}" for i in range(1, 6)])
        ax.set_ylabel("Pass Rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"{model}")
        ax.legend(fontsize=7)

    fig.suptitle("Phase 1 & 2: Pass Rates by Model", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / "bar_pass_rates_per_model.png", dpi=150)
    plt.close()

    print(f"\n[VIS] Saved {7} figures to {fig_dir}/")


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    print("Phase 1 & 2: Representation Geometry Validation")
    print("=" * 50)

    # Run pipeline
    results_df = run_pipeline()

    if len(results_df) == 0:
        print("[ERROR] No results generated!")
        sys.exit(1)

    print(f"\nTotal results: {len(results_df)} rows")
    print(f"Models: {results_df['model'].unique().tolist()}")
    print(f"Stages: {results_df['stage'].unique().tolist()}")

    # Drop nn_order column (list, not CSV-friendly) for the CSV
    csv_df = results_df.drop(columns=["nn_order"], errors="ignore")
    csv_path = RESULT_DIR / "phase1_phase2_results.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    # Summary statistics
    summary = generate_summary(results_df)
    summary_path = RESULT_DIR / "phase1_phase2_summary_stats.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"[SAVED] {summary_path}")
    print()
    print(summary)

    # Validation report
    report = generate_validation_report(results_df)
    report_path = RESULT_DIR / "validation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n[SAVED] {report_path}")
    print()
    print(report)

    # Visualizations
    generate_visualizations(results_df)

    print("\n[DONE] Phase 1 & 2 analysis complete.")


if __name__ == "__main__":
    main()
