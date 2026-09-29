#!/usr/bin/env python
"""Phase 4: Uncertainty -> error discrimination AUROC for geometric distance methods.

Mirrors the AUROC used in the `semantic_uncertainty` codebase
(uncertainty/utils/eval_utils.py): a single ROC-AUC over a flat array where
  y_true  = the model is WRONG on that unit (1) vs correct (0)
  y_score = an uncertainty signal

In `semantic_uncertainty` the unit is a Q/A example and the score is semantic
entropy. Here the unit is a single FRAME (one image+question -> one VLM yes/no
answer) and the score is the raw per-frame DISTANCE produced by a geometric
boundary method. Those methods currently take argmax(distance) to locate the
ambiguous frame; this script tests whether the distance MAGNITUDE itself is a
usable per-frame uncertainty signal (i.e. does a high distance flag the frames
where the VLM answers incorrectly).

Three distance methods are evaluated, each over three representation stages
(vision / projector / hidden-best-layer):
  anchor      -> anchor_product_boundary  (product of cosine dist to both anchors)
  orthogonal  -> chord_boundary           (perpendicular dist from f0->f9 chord)
  geodesic    -> G3 Isomap geodesic perpendicular (phase_3_g3_g4.run_g3)

Per-frame ground truth (rebuilt here; the pipeline's stored `ground_truth` /
`is_correct` use a constant label and are NOT used):
  Ambiguous index set A:
     single center_frame -> {center_pos}
     dual   center_frames -> {min_pos .. max_pos}  (inclusive range)
  Frames in A are EXCLUDED entirely (no definitive human label there).
  Transition sequences (scalar gt_answer):
     "yes": index < min(A) -> "yes",  index > max(A) -> "no"
     "no" : index < min(A) -> "no" ,  index > max(A) -> "yes"   (symmetric)
  Constant sequences (list gt_answer, e.g. "[yes yes]" / "[no no]"):
     every non-ambiguous frame -> the constant label.
  is_wrong = (parsed_answer.lower() != per_frame_gt)

Frames are POOLED across sequences (the per-frame analogue of
semantic_uncertainty's per-example pooling): one AUROC over all frames, not a
mean of per-sequence AUROCs.

Outputs:
    outputs/phase_4/uncertainty_auroc.csv      (one row per grouping x stage x method)
    outputs/phase_4/uncertainty_auroc_summary.txt

Usage:
    conda run -n continuous-uncertainty python scripts/phase_4_uncertainty_auroc.py
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Make sibling phase scripts + the package importable ──────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))   # phase_1_2_analysis, phase_3_g3_g4
sys.path.insert(0, str(REPO_ROOT))    # vlm_internal_diagnostics package

from phase_1_2_analysis import (  # noqa: E402
    MODELS, STAGES, NUM_FRAMES, BASE_OUTPUT, load_sequence_features,
)
from phase_3_g3_g4 import run_g3  # noqa: E402
from vlm_internal_diagnostics.metrics.representation_boundary_metrics import (  # noqa: E402
    chord_boundary, anchor_product_boundary,
)
from vlm_internal_diagnostics.data.manifest_utils import (  # noqa: E402
    load_manifest_rows, sequence_id_from_video_path, _parse_list_field,
)

# ── Import the semantic_uncertainty AUROC + bootstrap (reuse, don't reimplement)
# Source repo: https://github.com/jlko/semantic_uncertainty
# We import ITS eval_utils so our AUROC and bootstrap CIs are produced by the
# exact same function as the semantic-entropy baselines we compare against.
# Clone it, then point `semantic_uncertainty_utils` in configs/paths.yaml at
# <clone>/semantic_uncertainty/uncertainty/utils (or set CU_SEMUNC_UTILS).
from _paths import get_path  # noqa: E402

SEMUNC_UTILS = str(get_path("semantic_uncertainty_utils"))
sys.path.insert(0, SEMUNC_UTILS)
import eval_utils as semunc_eval  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────

RESULT_DIR = BASE_OUTPUT / "phase_4"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Manifest carrying gt_answer / frames / center_frames (transition structure).
MANIFEST_PATH = str(get_path("manifest_path"))

# Distance method name -> callable returning (boundary_index, per_frame_scores).
DISTANCE_METHODS = {
    "anchor": anchor_product_boundary,
    "orthogonal": chord_boundary,
    # geodesic handled separately (run_g3 has a different signature / fallback).
}

MIN_FRAMES_FOR_AUROC = 10   # need a reasonable pool + both classes
BOOTSTRAP_SEED = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase4")


# ── Per-frame ground-truth construction ──────────────────────────────────────

def build_frame_ground_truth(gt_raw, frames, center_frames):
    """Return (gt_labels, ambiguous_positions).

    gt_labels : list[str|None] length len(frames); None == excluded (ambiguous).
    ambiguous_positions : set[int] of frame positions treated as ambiguous.
    """
    n = len(frames)
    gt_raw = "" if gt_raw is None else str(gt_raw).strip()
    is_constant = gt_raw.startswith("[")

    # Resolve center frame numbers -> positions within `frames`.
    center_positions = []
    for cf in center_frames:
        try:
            center_positions.append(frames.index(cf))
        except ValueError:
            continue
    if not center_positions:
        center_positions = [n // 2]   # fallback: middle frame
    a_min, a_max = min(center_positions), max(center_positions)
    ambiguous = set(range(a_min, a_max + 1))

    if is_constant:
        items = _parse_list_field(gt_raw)
        const = str(items[0]).strip().lower() if items else None
        labels = [None if i in ambiguous else const for i in range(n)]
        return labels, ambiguous

    # Transition: scalar gt_answer names the BEFORE-ambiguity state.
    before = gt_raw.lower()
    after = "no" if before == "yes" else "yes"
    labels = []
    for i in range(n):
        if i in ambiguous:
            labels.append(None)
        elif i < a_min:
            labels.append(before)
        else:  # i > a_max
            labels.append(after)
    return labels, ambiguous


def load_manifest_ground_truth(manifest_path):
    """sequence_id -> (gt_labels, ambiguous_positions, gt_raw)."""
    out = {}
    n_const = n_trans = 0
    for row in load_manifest_rows(manifest_path):
        seq_id = sequence_id_from_video_path(row["video_path"])
        frames = _parse_list_field(row["frames"])
        center_frames = _parse_list_field(row["center_frames"])
        gt_raw = row.get("gt_answer")
        labels, ambiguous = build_frame_ground_truth(gt_raw, frames, center_frames)
        out[seq_id] = (labels, ambiguous, gt_raw)
        if str(gt_raw).strip().startswith("["):
            n_const += 1
        else:
            n_trans += 1
    log.info("Manifest: %d sequences (%d transition, %d constant)",
             len(out), n_trans, n_const)
    return out


# ── Per-frame model predictions ──────────────────────────────────────────────

def load_parsed_answers(jsonl_path):
    """(sequence_id, frame_index) -> parsed_answer (lowercased str) or None."""
    preds = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pa = rec.get("parsed_answer")
            preds[(rec["sequence_id"], int(rec["frame_index"]))] = (
                str(pa).strip().lower() if pa is not None else None
            )
    return preds


# ── Distance computation per stage ───────────────────────────────────────────

def distance_arrays_for_stage(reps):
    """Return {method_name: per_frame_distance_array(len NUM_FRAMES)}.

    `reps` is (NUM_FRAMES, d). Geodesic may be absent (Isomap fallback) -> omitted.
    """
    arrays = {}
    for name, fn in DISTANCE_METHODS.items():
        _bi, scores = fn(reps)
        arrays[name] = np.asarray(scores, dtype=np.float64)

    g3 = run_g3(reps, anchor_product_fallback_idx=None)
    if not g3["isomap_fallback"] and g3["h_array"] is not None:
        arrays["geodesic"] = np.asarray(g3["h_array"], dtype=np.float64)
    return arrays


# ── Build the pooled per-frame table ─────────────────────────────────────────

def build_frame_table():
    """Return a long DataFrame: one row per (model, seq, stage, method, frame)."""
    gt_map = load_manifest_ground_truth(MANIFEST_PATH)

    rows = []
    skip = defaultdict(int)

    for model_name, model_dir in MODELS.items():
        csv_path = model_dir / "aggregate_metrics.csv"
        jsonl_path = model_dir / "per_frame_outputs.jsonl"
        if not csv_path.exists() or not jsonl_path.exists():
            log.warning("[SKIP] %s: missing aggregate_metrics.csv or per_frame_outputs.jsonl", model_name)
            continue

        df = pd.read_csv(csv_path)
        preds = load_parsed_answers(jsonl_path)
        feat_base = model_dir / "features"
        log.info("MODEL %s: %d sequences", model_name, len(df))

        n_seq_used = 0
        for _, row in df.iterrows():
            seq_id = row["sequence_id"]
            center_idx = int(row["max_ambiguity_index"])
            category = row["category"]
            best_layer = (int(row["best_hidden_layer_by_spearman"])
                          if pd.notna(row.get("best_hidden_layer_by_spearman")) else 0)

            if seq_id not in gt_map:
                skip["seq not in manifest"] += 1
                continue
            gt_labels, ambiguous, _gt_raw = gt_map[seq_id]

            seq_feat_dir = feat_base / seq_id
            if not seq_feat_dir.exists():
                skip["feature dir missing"] += 1
                continue
            feats = load_sequence_features(seq_feat_dir, best_layer)

            used_any = False
            for stage in STAGES:
                if stage not in feats:
                    skip[f"{stage} features missing"] += 1
                    continue
                reps = feats[stage]
                if not np.all(np.isfinite(reps)):
                    skip[f"{stage} non-finite feats"] += 1
                    continue

                dist = distance_arrays_for_stage(reps)

                for fi in range(NUM_FRAMES):
                    if fi in ambiguous:
                        continue
                    gt = gt_labels[fi] if fi < len(gt_labels) else None
                    if gt is None:
                        continue
                    parsed = preds.get((seq_id, fi))
                    if parsed is None:
                        continue
                    is_wrong = int(parsed != gt)

                    for method, arr in dist.items():
                        if fi >= len(arr):
                            continue
                        score = float(arr[fi])
                        rows.append({
                            "model": model_name,
                            "sequence_id": seq_id,
                            "category": category,
                            "stage": stage,
                            "method": method,
                            "frame_index": fi,
                            "center_idx": center_idx,
                            "y_true": is_wrong,
                            "y_score_raw": score,
                        })
                        used_any = True
            if used_any:
                n_seq_used += 1

        log.info("MODEL %s: %d sequences contributed frames", model_name, n_seq_used)

    if skip:
        log.info("Skip reasons: %s", dict(skip))

    table = pd.DataFrame(rows)

    # Per-sequence z-score of the distance (distances vary greatly in scale
    # across sequences; z-scoring within each (model, sequence, stage, method)
    # removes that scale/offset so the pooled global ranking is comparable).
    def _zscore(s):
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd < 1e-12:
            return pd.Series(0.0, index=s.index)  # flat profile -> no info
        return (s - mu) / sd

    grp = ["model", "sequence_id", "stage", "method"]
    table["y_score_zscore"] = (
        table.groupby(grp)["y_score_raw"].transform(_zscore)
    )

    log.info("Pooled frame table: %d rows", len(table))
    return table


# Score variants evaluated (column name -> human label).
SCORE_VARIANTS = {
    "y_score_raw": "raw distance",
    "y_score_zscore": "per-sequence z-score",
}


# ── AUROC over a group ───────────────────────────────────────────────────────

def auroc_for_group(sub, rng, score_col):
    """Compute point AUROC + bootstrap CI for a sub-DataFrame on `score_col`.

    Returns dict (auroc, low, high, std_err, n, n_wrong) or None if not computable.
    """
    y_true = sub["y_true"].to_numpy()
    y_score = sub[score_col].to_numpy()
    n = len(y_true)
    n_wrong = int(y_true.sum())
    if n < MIN_FRAMES_FOR_AUROC or n_wrong == 0 or n_wrong == n:
        return {
            "auroc": float("nan"), "auroc_low": float("nan"),
            "auroc_high": float("nan"), "auroc_std_err": float("nan"),
            "n_frames": n, "n_wrong": n_wrong,
            "frac_wrong": (n_wrong / n) if n else float("nan"),
        }
    point = float(semunc_eval.auroc(y_true, y_score))
    try:
        ci = semunc_eval.compatible_bootstrap(semunc_eval.auroc, rng)(y_true, y_score)
        low, high, se = ci["low"], ci["high"], ci["std_err"]
    except Exception as exc:  # single-class resamples etc.
        log.debug("bootstrap failed (n=%d): %s", n, exc)
        low = high = se = float("nan")
    return {
        "auroc": point, "auroc_low": float(low), "auroc_high": float(high),
        "auroc_std_err": float(se), "n_frames": n, "n_wrong": n_wrong,
        "frac_wrong": n_wrong / n,
    }


def compute_all_aurocs(table):
    """Return a results DataFrame across several groupings."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []

    def add(variant, grouping, model, category, stage, method, sub):
        stats = auroc_for_group(sub, rng, variant)
        results.append({
            "score_variant": variant, "grouping": grouping, "model": model,
            "category": category, "stage": stage, "method": method, **stats,
        })

    methods = sorted(table["method"].unique())
    for variant in SCORE_VARIANTS:
        # 1) Headline: pooled over all models, per stage x method
        for stage in STAGES:
            for method in methods:
                sub = table[(table["stage"] == stage) & (table["method"] == method)]
                if len(sub):
                    add(variant, "all_models", "ALL", "ALL", stage, method, sub)
                    log.info("[%s][all_models] %-9s %-10s -> n=%d",
                             variant, stage, method, len(sub))

        # 2) Per model
        for model in sorted(table["model"].unique()):
            for stage in STAGES:
                for method in methods:
                    sub = table[(table["model"] == model) & (table["stage"] == stage)
                                & (table["method"] == method)]
                    if len(sub):
                        add(variant, "per_model", model, "ALL", stage, method, sub)

        # 3) Per category (pooled over models)
        for cat in sorted(table["category"].dropna().unique()):
            for stage in STAGES:
                for method in methods:
                    sub = table[(table["category"] == cat) & (table["stage"] == stage)
                                & (table["method"] == method)]
                    if len(sub):
                        add(variant, "per_category", "ALL", cat, stage, method, sub)

    return pd.DataFrame(results)


# ── Text summary ─────────────────────────────────────────────────────────────

def fmt_auroc(r):
    if pd.isna(r["auroc"]):
        return f"{'n/a':>22}"
    return f"{r['auroc']:.3f} [{r['auroc_low']:.3f},{r['auroc_high']:.3f}] n={int(r['n_frames'])}"


def generate_summary(res, table):
    lines = []
    lines.append("PHASE 4: UNCERTAINTY -> ERROR DISCRIMINATION AUROC")
    lines.append("=" * 70)
    lines.append("y_true = VLM answer wrong on frame (1);  y_score = geometric distance")
    lines.append("Frames pooled across sequences; ambiguous frames excluded.")
    lines.append("AUROC > 0.5 == higher distance flags wrong frames (desired).")
    lines.append("Two score variants reported: raw distance vs per-sequence z-score.")
    lines.append("")
    n_methods = max(1, table["method"].nunique())
    lines.append(f"Unique frames: {len(table) // n_methods}   (frame x method rows: {len(table)})")
    lines.append(f"Overall fraction wrong: {table['y_true'].mean():.3f}")
    lines.append("")

    methods = sorted(table["method"].unique())

    def block(title, sub):
        lines.append("-" * 70)
        lines.append(title)
        lines.append("-" * 70)
        header = f"{'stage':<11}" + "".join(f"{m:>23}" for m in methods)
        lines.append(header)
        for stage in STAGES:
            cells = [f"{stage:<11}"]
            for m in methods:
                r = sub[(sub["stage"] == stage) & (sub["method"] == m)]
                cells.append(f"{fmt_auroc(r.iloc[0]) if len(r) else 'n/a':>23}")
            lines.append("".join(cells))
        lines.append("")

    for variant, label in SCORE_VARIANTS.items():
        vres = res[res["score_variant"] == variant]
        lines.append("")
        lines.append("#" * 70)
        lines.append(f"#  SCORE VARIANT: {label.upper()}  ({variant})")
        lines.append("#" * 70)

        block("ALL MODELS POOLED  (auroc [90% CI] n)",
              vres[vres["grouping"] == "all_models"])

        for model in sorted(vres[vres["grouping"] == "per_model"]["model"].unique()):
            block(f"MODEL: {model}",
                  vres[(vres["grouping"] == "per_model") & (vres["model"] == model)])

        for cat in sorted(vres[vres["grouping"] == "per_category"]["category"].unique()):
            block(f"CATEGORY: {cat}",
                  vres[(vres["grouping"] == "per_category") & (vres["category"] == cat)])

    return "\n".join(lines)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    log.info("Phase 4: uncertainty->error AUROC for geometric distance methods")
    log.info("Reusing AUROC/bootstrap from %s", SEMUNC_UTILS)

    table = build_frame_table()
    if len(table) == 0:
        log.error("No frames pooled; aborting.")
        sys.exit(1)

    table_path = RESULT_DIR / "uncertainty_auroc_frames.csv"
    table.to_csv(table_path, index=False)
    log.info("[SAVED] %s (%d rows)", table_path, len(table))

    res = compute_all_aurocs(table)
    res_path = RESULT_DIR / "uncertainty_auroc.csv"
    res.to_csv(res_path, index=False)
    log.info("[SAVED] %s (%d rows)", res_path, len(res))

    summary = generate_summary(res, table)
    summary_path = RESULT_DIR / "uncertainty_auroc_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    log.info("[SAVED] %s", summary_path)
    print("\n" + summary)

    log.info("[DONE] Phase 4 complete.")


if __name__ == "__main__":
    main()