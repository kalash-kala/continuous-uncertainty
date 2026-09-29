#!/usr/bin/env python
"""Phase 3: G3 (Geodesic/Isomap) + G4 (Density anomaly) + G3+G4 hybrid.

Implements two new geometric boundary-detection methods motivated by Phase 1&2
findings: Test 3 (frame-order recovery, 76.9%) and Test 5 (low-dim manifold,
100%) both pass strongly — both are local/manifold properties that G3 and G4
exploit.

Outputs:
    outputs/phase_3/g3_g4_results.csv         (raw G3/G4/hybrid per stage)
    outputs/phase_3/g3_g4_with_baselines.csv  (joined with 4 baselines + logit)
    outputs/phase_3/g3_g4_summary.txt         (comparison table + Wilcoxon)

Usage:
    conda run -n continuous-uncertainty python scripts/phase_3_g3_g4.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors

# Reuse loaders from Phase 1&2
sys.path.insert(0, str(Path(__file__).parent))
from phase_1_2_analysis import (
    MODELS, STAGES, NUM_FRAMES, BASE_OUTPUT, load_sequence_features,
)

warnings.filterwarnings("ignore", category=UserWarning)

RESULT_DIR = BASE_OUTPUT / "phase_3"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

K_CANDIDATES = [4, 3, 5]
N_COMPONENTS = 3
EPS = 1e-6

BASELINE_METHODS = ["chord", "angle", "local_var", "anchor_product"]


# ── G3: Geodesic perpendicular via Isomap ───────────────────────────────────

def _g3_with_k(reps, k):
    """Returns (boundary_idx, h_array, D_geo) or None on failure."""
    try:
        iso = Isomap(n_neighbors=k, n_components=N_COMPONENTS, metric="cosine")
        iso.fit(reps)
        D = iso.dist_matrix_
    except Exception:
        return None

    off_diag = D[~np.eye(NUM_FRAMES, dtype=bool)]
    if np.any(np.isinf(off_diag)) or np.any(np.isnan(off_diag)):
        return None
    if D[0, -1] < EPS:
        return None

    c = D[0, -1]
    h = np.zeros(NUM_FRAMES)
    for i in range(NUM_FRAMES):
        a = D[0, i]
        b = D[i, -1]
        proj = (a * a + c * c - b * b) / (2.0 * c)
        h[i] = np.sqrt(max(0.0, a * a - proj * proj))

    return int(np.argmax(h)), h, D


def run_g3(reps, anchor_product_fallback_idx):
    """Try k=4, 3, 5. Returns dict with index, k_used, fallback flag, h_array, D_geo."""
    for k in K_CANDIDATES:
        result = _g3_with_k(reps, k)
        if result is not None:
            idx, h, D = result
            return {
                "boundary_index": idx,
                "k_used": k,
                "isomap_fallback": False,
                "h_array": h,
                "D_geo": D,
            }
    return {
        "boundary_index": int(anchor_product_fallback_idx)
                          if anchor_product_fallback_idx is not None else 0,
        "k_used": -1,
        "isomap_fallback": True,
        "h_array": None,
        "D_geo": None,
    }


# ── G4: Density anomaly + chord perpendicular ───────────────────────────────

def run_g4(reps):
    """Cosine kNN density anomaly + Euclidean chord perpendicular."""
    nbrs = NearestNeighbors(n_neighbors=4, metric="cosine").fit(reps)
    dists, _ = nbrs.kneighbors(reps)
    # exclude self (column 0), use next 3
    mean_knn = dists[:, 1:4].mean(axis=1)
    density = 1.0 / (mean_knn + EPS)
    anomaly = 1.0 - density / (density.max() + EPS)

    v = reps[-1] - reps[0]
    vn = np.linalg.norm(v)
    if vn < EPS:
        perp = np.zeros(NUM_FRAMES)
    else:
        v_hat = v / vn
        proj_scalar = reps @ v_hat
        residuals = reps - np.outer(proj_scalar, v_hat)
        perp = np.linalg.norm(residuals, axis=1)

    perp_norm = perp / (perp.max() + EPS)
    score = 0.6 * perp_norm + 0.4 * anomaly
    return int(np.argmax(score))


# ── G3+G4 hybrid (geodesic perp + geodesic density) ─────────────────────────

def run_g3g4_hybrid(g3_info):
    if g3_info["isomap_fallback"]:
        return g3_info["boundary_index"]

    D = g3_info["D_geo"].copy()
    np.fill_diagonal(D, np.inf)
    knn_geo = np.sort(D, axis=1)[:, :3].mean(axis=1)
    density_geo = 1.0 / (knn_geo + EPS)
    anomaly_geo = 1.0 - density_geo / (density_geo.max() + EPS)

    h = g3_info["h_array"]
    h_norm = h / (h.max() + EPS)

    score = 0.6 * h_norm + 0.4 * anomaly_geo
    return int(np.argmax(score))


# ── Main pipeline ───────────────────────────────────────────────────────────

def process_sequence(reps, center_idx, anchor_product_fallback_idx):
    if not np.all(np.isfinite(reps)):
        return None

    g3 = run_g3(reps, anchor_product_fallback_idx)
    g4_idx = run_g4(reps)
    hybrid_idx = run_g3g4_hybrid(g3)

    return {
        "g3_boundary_index": g3["boundary_index"],
        "g3_boundary_error": int(abs(g3["boundary_index"] - center_idx)),
        "g3_k_used": g3["k_used"],
        "g3_isomap_fallback": g3["isomap_fallback"],
        "g4_boundary_index": g4_idx,
        "g4_boundary_error": int(abs(g4_idx - center_idx)),
        "g3g4_boundary_index": hybrid_idx,
        "g3g4_boundary_error": int(abs(hybrid_idx - center_idx)),
    }


def run_pipeline():
    rows = []
    skipped = []

    for model_name, model_dir in MODELS.items():
        csv_path = model_dir / "aggregate_metrics.csv"
        if not csv_path.exists():
            print(f"[SKIP] No aggregate_metrics.csv for {model_name}")
            continue

        df = pd.read_csv(csv_path)
        feat_base = model_dir / "features"
        print(f"\n{'='*70}\nMODEL: {model_name}  ({len(df)} sequences)\n{'='*70}")

        for idx, row in df.iterrows():
            seq_id = row["sequence_id"]
            center_idx = int(row["max_ambiguity_index"])
            category = row["category"]
            best_layer = (int(row["best_hidden_layer_by_spearman"])
                          if pd.notna(row.get("best_hidden_layer_by_spearman"))
                          else 0)

            seq_feat_dir = feat_base / seq_id
            if not seq_feat_dir.exists():
                skipped.append((model_name, seq_id, "feature dir missing"))
                continue

            feats = load_sequence_features(seq_feat_dir, best_layer)

            out = {
                "model": model_name,
                "sequence_id": seq_id,
                "category": category,
                "max_ambiguity_index": center_idx,
            }

            for stage in STAGES:
                if stage not in feats:
                    skipped.append((model_name, seq_id, f"{stage} missing"))
                    continue

                fallback_idx = row.get(f"{stage}_anchor_product_boundary_index")
                if pd.isna(fallback_idx):
                    fallback_idx = None

                result = process_sequence(feats[stage], center_idx, fallback_idx)
                if result is None:
                    skipped.append((model_name, seq_id, f"{stage} NaN/Inf"))
                    continue

                for k, v in result.items():
                    out[f"{stage}_{k}"] = v

            rows.append(out)

            if (idx + 1) % 25 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        print(f"  Done: {model_name}")

    if skipped:
        print(f"\n[INFO] Skipped {len(skipped)} entries (first 10):")
        for s in skipped[:10]:
            print(f"  {s}")

    return pd.DataFrame(rows)


# ── Join with existing baselines ────────────────────────────────────────────

def join_with_baselines(results_df):
    """Left-join G3/G4 results with the per-model aggregate_metrics.csv baselines."""
    baseline_frames = []
    baseline_cols_per_stage = [
        f"{stage}_{m}_boundary_{kind}"
        for stage in STAGES for m in BASELINE_METHODS for kind in ("index", "error")
    ]
    logit_cols = ["boundary_index", "boundary_error"]
    keep_cols = ["sequence_id", "model"] + logit_cols + baseline_cols_per_stage

    for model_name, model_dir in MODELS.items():
        csv_path = model_dir / "aggregate_metrics.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["model"] = model_name
        # Rename the logit baseline so we can distinguish it
        df = df.rename(columns={
            "boundary_index": "logit_boundary_index",
            "boundary_error": "logit_boundary_error",
        })
        cols_present = ["sequence_id", "model", "logit_boundary_index", "logit_boundary_error"]
        for c in baseline_cols_per_stage:
            if c in df.columns:
                cols_present.append(c)
        baseline_frames.append(df[cols_present])

    baselines = pd.concat(baseline_frames, ignore_index=True)
    merged = results_df.merge(baselines, on=["model", "sequence_id"], how="left")
    return merged


# ── Summary + Wilcoxon ──────────────────────────────────────────────────────

def fmt_mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return "  n/a   "
    return f"{arr.mean():.2f}±{arr.std():.2f}"


def generate_summary(merged):
    lines = []
    lines.append("PHASE 3: G3 / G4 / G3+G4 vs BASELINES")
    lines.append("=" * 70)
    lines.append("")

    # Note on logit baseline
    lines.append(f"Total sequences (rows): {len(merged)}")
    lines.append(f"Models: {sorted(merged['model'].unique().tolist())}")
    lines.append("")

    methods = ["logit"] + BASELINE_METHODS + ["g3", "g4", "g3g4"]

    def be_col(stage, method):
        if method == "logit":
            return "logit_boundary_error"
        return f"{stage}_{method}_boundary_error"

    # Per-model, per-stage table
    for model in sorted(merged["model"].unique()):
        mdf = merged[merged["model"] == model]
        lines.append("─" * 70)
        lines.append(f"Model: {model}  ({len(mdf)} sequences)")
        lines.append("─" * 70)
        header = f"{'stage':<10}" + "".join(f"{m:>12}" for m in methods)
        lines.append(header)
        for stage in STAGES:
            row_means = [f"{stage:<10}"]
            for m in methods:
                col = be_col(stage, m)
                if col in mdf.columns:
                    row_means.append(f"{fmt_mean_std(mdf[col]):>12}")
                else:
                    row_means.append(f"{'n/a':>12}")
            lines.append("".join(row_means))
        # exact-match counts
        lines.append("")
        lines.append("Exact match counts (boundary_error == 0):")
        hdr = f"{'stage':<10}" + "".join(f"{m:>12}" for m in methods)
        lines.append(hdr)
        for stage in STAGES:
            row = [f"{stage:<10}"]
            for m in methods:
                col = be_col(stage, m)
                if col in mdf.columns:
                    em = int((mdf[col] == 0).sum())
                    n = mdf[col].notna().sum()
                    row.append(f"{em}/{n:>4}".rjust(12))
                else:
                    row.append(f"{'n/a':>12}")
            lines.append("".join(row))
        lines.append("")

    # Pooled-all-models view
    lines.append("=" * 70)
    lines.append("ALL MODELS POOLED — mean BE ± std")
    lines.append("=" * 70)
    header = f"{'stage':<10}" + "".join(f"{m:>12}" for m in methods)
    lines.append(header)
    for stage in STAGES:
        row = [f"{stage:<10}"]
        for m in methods:
            col = be_col(stage, m)
            if col in merged.columns:
                row.append(f"{fmt_mean_std(merged[col]):>12}")
            else:
                row.append(f"{'n/a':>12}")
        lines.append("".join(row))
    lines.append("")

    # Wilcoxon: G3, G4, G3+G4 vs chord per stage (pooled)
    lines.append("=" * 70)
    lines.append("WILCOXON SIGNED-RANK (paired) — boundary error vs chord baseline")
    lines.append("(negative median diff means new method has LOWER error → better)")
    lines.append("=" * 70)
    for stage in STAGES:
        chord_col = f"{stage}_chord_boundary_error"
        if chord_col not in merged.columns:
            continue
        lines.append(f"\nStage: {stage}")
        for new in ["g3", "g4", "g3g4"]:
            new_col = f"{stage}_{new}_boundary_error"
            if new_col not in merged.columns:
                continue
            sub = merged[[chord_col, new_col]].dropna()
            diff = sub[new_col] - sub[chord_col]
            try:
                stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                p_str = f"{p:.4g}"
            except ValueError:
                p_str = "n/a (all zeros)"
            med = float(np.median(diff)) if len(diff) else float("nan")
            mean = float(diff.mean()) if len(diff) else float("nan")
            lines.append(f"  {new:>5} vs chord:  median Δ={med:+.2f}  mean Δ={mean:+.3f}  p={p_str}  n={len(sub)}")
    lines.append("")

    # Per-model Wilcoxon (G3+G4 vs chord, hidden stage as most informative)
    lines.append("=" * 70)
    lines.append("PER-MODEL: G3+G4 vs chord (boundary error)")
    lines.append("=" * 70)
    for model in sorted(merged["model"].unique()):
        mdf = merged[merged["model"] == model]
        lines.append(f"\nModel: {model}")
        for stage in STAGES:
            c1 = f"{stage}_chord_boundary_error"
            c2 = f"{stage}_g3g4_boundary_error"
            if c1 not in mdf.columns or c2 not in mdf.columns:
                continue
            sub = mdf[[c1, c2]].dropna()
            if len(sub) == 0:
                continue
            diff = sub[c2] - sub[c1]
            try:
                _, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                p_str = f"{p:.4g}"
            except ValueError:
                p_str = "n/a"
            lines.append(f"  {stage:<10} median Δ={np.median(diff):+.2f}  mean Δ={diff.mean():+.3f}  p={p_str}")
    lines.append("")

    # G3 diagnostics
    lines.append("=" * 70)
    lines.append("G3 DIAGNOSTICS")
    lines.append("=" * 70)
    for stage in STAGES:
        k_col = f"{stage}_g3_k_used"
        fb_col = f"{stage}_g3_isomap_fallback"
        if k_col not in merged.columns:
            continue
        vc = merged[k_col].value_counts().to_dict()
        fb_rate = merged[fb_col].mean() * 100 if fb_col in merged.columns else float("nan")
        lines.append(f"\nStage: {stage}")
        lines.append(f"  k_used distribution: {vc}")
        lines.append(f"  isomap fallback rate: {fb_rate:.1f}%")
    lines.append("")

    return "\n".join(lines)


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    print("Phase 3: G3 (Geodesic/Isomap) + G4 (Density) + G3+G4 hybrid")
    print("=" * 60)

    results_df = run_pipeline()
    if len(results_df) == 0:
        print("[ERROR] No results.")
        sys.exit(1)

    raw_path = RESULT_DIR / "g3_g4_results.csv"
    results_df.to_csv(raw_path, index=False)
    print(f"\n[SAVED] {raw_path}  ({len(results_df)} rows)")

    merged = join_with_baselines(results_df)
    merged_path = RESULT_DIR / "g3_g4_with_baselines.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[SAVED] {merged_path}")

    summary = generate_summary(merged)
    summary_path = RESULT_DIR / "g3_g4_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"[SAVED] {summary_path}\n")
    print(summary)

    print("\n[DONE] Phase 3 G3/G4 analysis complete.")


if __name__ == "__main__":
    main()