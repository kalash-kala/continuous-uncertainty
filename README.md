# continuous-uncertainty

Does a VLM's **internal representation geometry** know where a question becomes
genuinely ambiguous — and is that geometry a usable uncertainty signal?

This repo runs the full pipeline: representation extraction from four VLMs,
a hypothesis-validation suite, two new geometric boundary-detection methods,
and an error-prediction AUROC evaluation comparable to the semantic-entropy
literature.

**Headline:** internal geometry localizes the human-labelled ambiguity boundary
**better than the model's own logit confidence** (mean boundary error 1.78 vs
2.48, consistent across 4 architectures). But the same geometric distances are
**at chance for predicting model errors** (AUROC ≈ 0.55). Locating where humans
say it is ambiguous and locating where the model is wrong turn out to be
different problems.

---

## 1. Setup

### Data

92 video sequences, each expanded to **10 frames**, each frame paired with the
**same yes/no question**. Within a sequence the visual evidence slides
continuously from one answer to the other; a human has annotated
`max_ambiguity_index` — the frame where the correct answer flips.

Sequences are labelled by category: **Geometric** (35), **Semantic** (45),
**Compositional** (12).

Images are laid out as `{image_root}/{sequence_id}/frame_{XXXX}.png`, where
`sequence_id` comes from the manifest's `video_path` (`/`→`_`, `.mp4` stripped).
Neither the manifest nor the frames live in this repo — see
[§6 Reproducing](#6-reproducing) for how to get them.

### Models

| Model | HF id | family | preset |
|---|---|---|---|
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | `llava` | `configs/models/llava.yaml` |
| Phi-4-multimodal | `microsoft/Phi-4-multimodal-instruct` | `phi4_multimodal` | `configs/models/phi4_multimodal.yaml` |
| Pixtral-12B | `mistral-community/pixtral-12b` | `pixtral` | `configs/models/pixtral.yaml` |
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen2_5_vl` | `configs/models/qwen2_5_vl.yaml` |

Each is wired in through an adapter in `models/adapters/` (registry:
`models/adapters/__init__.py`), which handles the per-architecture image-token
expansion and forces eager attention.

92 sequences × 4 models = **368 model-sequence rows** in every downstream table.

### Representation stages

Each frame yields a pooled vector at three depths, so each sequence is a
**10-point trajectory** through representation space at each depth:

| Stage | What it is |
|---|---|
| `vision` | mean-pooled vision-encoder (CLIP ViT) output |
| `projector` | mean-pooled multimodal projector (vision→LM adapter) output |
| `hidden` | LM hidden state at the final prompt token; the per-sequence "best layer" by center-distance Spearman is used downstream |

Also extracted: attention (query token → 576 image patches, head-averaged) and
yes/no logits. See `METRICS.md` for every metric and figure the extraction
pipeline produces.

### Reference signals

- **`logit`** — the model's own boundary estimate: `argmin |yes_logit − no_logit|`,
  i.e. the frame where it is least confident.
- **`binary_entropy`** — softmax entropy over {yes, no}.

These are what the geometric methods have to beat.

---

## 2. Methods

### 2.1 Classical geometric boundary methods (baselines)

`vlm_internal_diagnostics/metrics/representation_boundary_metrics.py`. Each
produces a per-frame score; the boundary estimate is `argmax`.

| Method | Score for frame *i* | Intuition |
|---|---|---|
| `chord` | Euclidean length of the component of (f_i − f_0) perpendicular to the f_0→f_9 chord | The trajectory detours off the straight "decision axis" where it is confused |
| `angle` | angle between incoming step (f_i − f_{i−1}) and outgoing step (f_{i+1} − f_i) | The sharp turn is the boundary |
| `local_var` | cos_dist(f_{i−1}, f_i) + cos_dist(f_i, f_{i+1}) | The boundary is where frame-to-frame change is fastest |
| `anchor_product` | cos_dist(f_i, f_0) × cos_dist(f_i, f_9) | The boundary is the point farthest from *both* endpoints — maximally "neither answer" |

**Evaluation metric:** *boundary error* = |predicted index − annotated center|.
Lower is better; 0 = exact hit; uniform random guessing on 10 frames ≈ 3.3.

### 2.2 G3 — geodesic perpendicular

`scripts/phase_3_g3_g4.py`. Same construction as `chord`, but distances measured
**along the manifold** rather than straight through ambient space:

1. Fit Isomap (k=4 neighbours, cosine metric, 3 components) over the 10 frames.
2. Take the geodesic distance matrix `D`.
3. For each frame, law of cosines against the 0→9 geodesic chord gives its
   perpendicular height `h_i`.
4. Boundary = `argmax(h)`.

k=4 succeeded on all 368 × 3 cells; fallback rate 0.0%.

### 2.3 G4 — density anomaly

The ambiguous frame should be a **local-density outlier** (few neighbours,
because it is between two attractors):

1. kNN density (k=3, cosine): `density = 1 / mean_knn_distance`.
2. `anomaly = 1 − density / max(density)`.
3. Blend with the Euclidean chord perpendicular:
   `score = 0.6 × perp_norm + 0.4 × anomaly`.

### 2.4 G3+G4 hybrid

Same 0.6/0.4 blend, but **both** terms computed geodesically on the Isomap
distance matrix.

---

## 3. Results

### 3.1 Phase 1 & 2 — the orthogonal-structure hypothesis

`Representation_geometry_analysis_framework.py`, `scripts/phase_1_2_analysis.py`
→ `outputs/phase_1_2/`

**Hypothesis under test.** A sequence's trajectory decomposes into a *decision
axis* (moving monotonically from one answer to the other) plus an *orthogonal
uncertainty subspace* that bulges at the ambiguous center.

Five tests, 368 rows × 3 stages = 1104 cells:

| Test | What it checks | Pass rate | Verdict |
|---|---|---|---|
| 1. PC1 alignment | PCA top component aligns with the chord, explains most variance, is monotonic | **2.4%** | ✗ |
| 2. Orthogonal variance peaks at center | the core hypothesis | **19.4%** | ✗ |
| 3. Frame-order recovery | true frame order reconstructible from geometry (LIS ratio > 0.5) | **76.9%** | ✓ |
| 4. Curvature at center | curvature highest near the boundary | **48.9%** | ✗ |
| 5. Low-dimensional manifold | 10-frame cloud lives in few dimensions | **100%** | ✓ |

**The hypothesis did not validate.** Diagnostics:

- Test 1 fails on **monotonicity**, not alignment. PC1–chord alignment is
  decent (cos = 0.71 ± 0.25) and explains ~52% of variance, but there are
  ~3.3 monotonicity violations out of 9 steps — the trajectory does not march
  cleanly from one answer to the other.
- Test 2: ρ = −0.26 ± 0.36. Correct sign, far too weak and noisy to build on.
- Test 4 fails because curvature is **huge everywhere** (mean ~1.93 rad ≈ 111°),
  so it carries no localization information.
- Test 5: ~600× compression, **~6–7 effective dimensions** (95% variance) vs
  3000+ ambient.

All four models score 2/5 — the failure is **architecture-independent**, not a
LLaVA quirk. Compositional questions are weakest throughout.

The two tests that *did* pass are both **local / manifold** properties. That is
what motivated G3 and G4.

> ⚠️ **Caveat when reading `validation_report.txt`.** Its "KEY FINDINGS FOR
> PAPER" section states Findings 1, 2 and 4 in positive language for tests the
> same file marks ✗ FAIL — those lines report the *mean effect sizes of failing
> tests*. Do not carry that framing forward.

### 3.2 Phase 3 — boundary localization

`scripts/phase_3_g3_g4.py` → `outputs/phase_3/`

Mean boundary error ± std, pooled over all 4 models (n=368). **Lower is better;
random ≈ 3.3.**

| Method | vision | projector | hidden |
|---|---|---|---|
| `logit` (model's own confidence) | 2.48 ± 1.69 | 2.48 ± 1.69 | 2.48 ± 1.69 |
| `chord` | 1.85 ± 1.23 | 1.81 ± 1.24 | 1.94 ± 1.43 |
| `angle` | 2.07 | 2.14 | 2.32 |
| `local_var` | 2.26 | 2.20 | 2.52 |
| `anchor_product` | **1.78 ± 1.24** | **1.71 ± 1.27** | 2.10 ± 1.45 |
| **G3** (geodesic) | 1.82 ± 1.24 | 1.77 ± 1.28 | 1.97 ± 1.43 |
| **G4** (density) | 3.23 ± 1.63 | 3.28 ± 1.60 | 3.29 ± 1.67 |
| **G3+G4** | 1.90 ± 1.29 | 1.93 ± 1.33 | 2.15 ± 1.46 |

Wilcoxon signed-rank vs the `chord` baseline (negative Δ = better):

| | vision | projector | hidden |
|---|---|---|---|
| G3 | Δ=−0.068, p=0.22 | Δ=−0.103, p=0.12 | Δ=+0.158, **p=0.007** |
| G4 | Δ=+1.383, p=2e-32 | Δ=+1.473, p=2e-35 | Δ=+1.353, p=1e-30 |
| G3+G4 | Δ=+0.057, p=0.54 | Δ=+0.122, p=0.04 | Δ=+0.215, p=0.0003 |

**Read:**

1. **Geometry beats the model's own confidence** — ~1.8 vs 2.48 boundary error,
   consistent across all four architectures. This is the one solid,
   reproducible result in the repo.
2. **G3 ≈ chord ≈ anchor_product.** No significant improvement at vision or
   projector; the only significant cell (hidden) is significant in the *wrong*
   direction. Going geodesic bought nothing over the straight-line version.
3. **G4 is a failure** — worse than random, p ≈ 1e-32, and it drags the hybrid
   below plain G3. Density anomaly actively destroys the signal.
4. **vision and projector beat hidden on every method.** The ambiguity signal is
   strongest early and **degrades as it passes through the LM**.

### 3.3 Phase 4 — uncertainty → error discrimination

`scripts/phase_4_uncertainty_auroc.py` → `outputs/phase_4/`

This pivots from *"can we locate the boundary"* to *"can we predict errors"* —
the question that makes the work comparable to the semantic-entropy literature.
It imports `eval_utils` directly from the `semantic_uncertainty` repo so the
AUROC and bootstrap CIs are the same function.

**Protocol.** `y_true` = the model answered this frame **wrong**; `y_score` =
the **raw per-frame distance** (not the argmax). Annotated-ambiguous frames are
**excluded** (no defensible human label there); the remaining frames get a label
derived from the transition structure in the manifest. Frames are **pooled**
across sequences, not averaged per sequence. 9,624 frames, 38.2% wrong.

Pooled AUROC [90% CI], n=3208 per cell:

| | anchor | geodesic | orthogonal |
|---|---|---|---|
| **raw distance** | | | |
| vision | 0.525 [0.508, 0.541] | 0.520 | 0.533 |
| projector | 0.538 | 0.544 [0.526, 0.561] | 0.532 |
| hidden | 0.505 | 0.520 | 0.512 |
| **per-sequence z-score** | | | |
| vision | **0.562 [0.543, 0.577]** | 0.544 | 0.559 |
| projector | 0.561 | 0.555 | 0.561 |
| hidden | 0.557 | 0.532 | 0.558 |

**This is essentially a null result.** Best cell 0.562; several CIs include
0.50; qwen/hidden is *below* chance at 0.466 raw. Two secondary observations
worth keeping:

- **Per-sequence z-scoring consistently helps** (+0.02–0.03 across the board).
  Distances are only meaningful *relative to their own sequence* — absolute
  magnitude is dominated by sequence-level scale, not by uncertainty. Any
  future score should be sequence-normalized by construction.
- **Category matters**: Geometric reaches 0.58, Semantic ~0.56, Compositional
  is at chance (~0.51). Consistent with the Phase 1 category breakdown.

---

## 4. What is established, and what is dead

**Established**

1. Internal geometry localizes the human ambiguity boundary **better than the
   model's own logit confidence** (1.78 vs 2.48), across 4 architectures.
2. The signal is **strongest at vision/projector and decays through the LM** —
   localizes the loss to the language stack.
3. The representation manifold is genuinely **low-dimensional (~6–7 dims)** and
   frame order is recoverable from it (76.9%).

**Dead — do not re-run these**

4. The **orthogonal decision-axis / uncertainty-subspace hypothesis**: Tests 1,
   2 and 4 all fail, all four models.
5. **G4 / density anomaly**: worse than random (p ≈ 1e-32).
6. **G3 / geodesic**: adds nothing over the straight-line chord. The manifold
   curvature is not where the information is.
7. **Hand-designed distance as an error-prediction score**: at chance
   (AUROC ≈ 0.55). The geometry finds where *humans* said it is ambiguous, not
   where the *model* gets it wrong — which is itself the interesting negative
   result.

**Open / untested**

- *Why* does geometry beat the logits at vision/projector? Is it the same
  information the logits have but badly calibrated, or genuinely extra?
- Phase 4 tested only **unsupervised, hand-picked distance functionals**. The
  obvious next step is a **trained light probe** on the 6–7-dim manifold
  coordinates predicting `is_wrong`, rather than assuming which functional
  carries the signal.
- Everything rests on **92 sequences**, and per-sequence Spearman on 10 points
  is very noisy (see the caveats in `METRICS.md` §7). Effect sizes this small
  need more data before they are publishable in either direction.

---

## 5. Repo layout

```
Representation_geometry_analysis_framework.py   design doc + code sketches for Phase 1/2
llava_internal_diagnostics_instruction.md       original extraction spec
vlm_internal_diagnostics/
  README.md            how to run extraction / metrics / visualization
  METRICS.md           every metric and figure, with formulas and how to read them
  configs/
    paths.yaml         ALL machine-specific paths — ships empty, fill this in first
    models/            one preset per model (llava / qwen2_5_vl / phi4_multimodal / pixtral)
    data_config.yaml   manifest + image root (empty; falls back to paths.yaml)
    run_config.yaml    run_name + debug; output_dir resolves to <outputs_root>/<run_name>
    extraction_config.yaml, visualization_config.yaml
    model_config.yaml  DEPRECATED, kept for back-compat; use configs/models/
  models/              per-architecture loaders, hooks, token indexing
  extraction/          vision / projector / hidden / attention / logits extractors
  metrics/             smoothness, attention, logit, boundary, per-token metrics
  analysis/            aggregation helpers
  visualization/       overlays, sequence grids, trajectory + aggregate plots
  scripts/
    _paths.py        path resolution; run it directly to check your setup
    _common.py       config loading, path backfill, model/run_name guard
    run_extract.py  run_metrics.py  run_visualize.py  run_all.py
    phase_1_2_analysis.py           hypothesis validation (Tests 1–5)
    phase_3_g3_g4.py                G3 / G4 / hybrid vs baselines
    phase_4_uncertainty_auroc.py    error-discrimination AUROC
```

Outputs live under whatever you set as `outputs_root` (on the lab server:
`/data/kalashkala/continuous-uncertainty/outputs/`):

```
{model}_internal_diag_v1_question_mean_new/   per-model extraction + features + figures
phase_1_2/   phase1_phase2_results.csv, summary_stats.txt, validation_report.txt
phase_3/     g3_g4_results.csv, g3_g4_with_baselines.csv, g3_g4_summary.txt
phase_4/     uncertainty_auroc.csv, uncertainty_auroc_frames.csv, summary.txt
```

> ⚠️ **Superseded run directories on the lab server.** Three LLaVA variants
> exist (`llava_internal_diag_v1`, `..._question_mean`, `..._question_mean_new`)
> and two phi4 variants (`phi4_internal_diag_v1_question_mean_new` and the older
> `phi4_multimodal_internal_diag_v1_question_mean_new`). **Every published
> number comes from `llava_internal_diag_v1_question_mean_new` and
> `phi4_internal_diag_v1_question_mean_new`** — the names in the `MODELS` dict.
> The others are earlier runs; do not mix them in.

---

## 6. Reproducing

### Step 1 — configure paths (required, once)

**Every machine-specific path in this repo lives in one file:**
`vlm_internal_diagnostics/configs/paths.yaml`. It is committed with **empty
values on purpose** — fill it in, or set the equivalent env vars. Nothing runs
until you do, and each one fails with a message naming the key, the env var,
what the path should contain, and where the canonical copy lives on the lab
server.

| Key | Env var | What it is | Needed by |
|---|---|---|---|
| `outputs_root` | `CU_OUTPUTS_ROOT` | extraction outputs + `phase_*/` results (~7 GB with all features) | everything |
| `manifest_path` | `CU_MANIFEST_PATH` | the sequence manifest CSV | extraction, Phase 4 |
| `image_root` | `CU_IMAGE_ROOT` | `<image_root>/<sequence_id>/frame_XXXX.png` | extraction only |
| `semantic_uncertainty_utils` | `CU_SEMUNC_UTILS` | `eval_utils.py` from the semantic_uncertainty repo | Phase 4 only |

Check your configuration at any time:

```bash
python vlm_internal_diagnostics/scripts/_paths.py
```

Prefer keeping your values out of `git status`? Copy `configs/paths.yaml` to
`configs/paths.local.yaml` (gitignored) and fill that in instead — it takes
precedence. Env vars beat both.

### Step 2 — get the data

Neither the frames nor the extracted features are in this repo. Two options:

**Copy from the lab server (recommended — extraction is four full model loads
and many GPU-hours):**

```bash
# frames + manifest (needed only if you plan to re-extract)
rsync -av <user>@<server>:/home/kalashkala/acl_rebuttal_form/  ./acl_rebuttal_form/

# already-extracted outputs — this is what the analysis phases read.
# ~7 GB; drop --include of features/ if you only want the CSV summaries.
rsync -av <user>@<server>:/data/kalashkala/continuous-uncertainty/outputs/  /your/outputs_root/
```

Then point `outputs_root` (and `manifest_path` / `image_root` if you copied
those) at your local copies. **If you only want to re-run analysis, you do not
need the frames at all** — the phases read saved tensors from `features/`.

**Phase 4 additionally needs the semantic_uncertainty repo** for its AUROC
implementation — we import theirs rather than reimplementing, so our numbers
are produced by the exact same function as the semantic-entropy baselines:

```bash
git clone https://github.com/jlko/semantic_uncertainty
export CU_SEMUNC_UTILS=$PWD/semantic_uncertainty/semantic_uncertainty/uncertainty/utils
```

### Step 3 — extraction (GPU; skip if you rsynced `outputs/`)

**One run = one model.** Pair a preset from `configs/models/` with its matching
`run_name` in `run_config.yaml`:

| `--model_config` | `run_name` in `run_config.yaml` |
|---|---|
| `configs/models/llava.yaml` | `llava_internal_diag_v1_question_mean_new` |
| `configs/models/phi4_multimodal.yaml` | `phi4_internal_diag_v1_question_mean_new` |
| `configs/models/pixtral.yaml` | `pixtral_internal_diag_v1_question_mean_new` |
| `configs/models/qwen2_5_vl.yaml` | `qwen2_5_vl_internal_diag_v1_question_mean_new` |

Those directory names are looked up **verbatim** by the analysis phases (see
`MODELS` in `scripts/phase_1_2_analysis.py`), so keep them. `output_dir` is
left empty and resolves to `<outputs_root>/<run_name>`.

The loader **refuses** to start if the model family and `run_name` belong to
different models — otherwise switching the preset and forgetting the run_name
silently overwrites another model's extraction.

```bash
cd vlm_internal_diagnostics/scripts
python run_all.py --model_config ../configs/models/llava.yaml \
                  --data_config ../configs/data_config.yaml \
                  --extraction_config ../configs/extraction_config.yaml \
                  --visualization_config ../configs/visualization_config.yaml \
                  --run_config ../configs/run_config.yaml
```

Smoke-test first with `debug.enabled: true` in `run_config.yaml` — 3 sequences
instead of 92, same code path end to end.

(`configs/model_config.yaml` still exists and mirrors the llava preset, but is
deprecated: it used to be hand-edited between runs, which made it impossible to
tell from the repo which model produced which output directory.)

### Step 4 — analysis (CPU; reads the saved tensors)

```bash
cd vlm_internal_diagnostics
conda run -n continuous-uncertainty python scripts/phase_1_2_analysis.py
conda run -n continuous-uncertainty python scripts/phase_3_g3_g4.py
conda run -n continuous-uncertainty python scripts/phase_4_uncertainty_auroc.py
```

Phases 3 and 4 have been verified to reproduce the committed results
**byte-for-byte** against `outputs_root` as shipped.

**Known gotchas**

- Attention extraction forces `attn_implementation="eager"` (and overwrites
  `model.config._attn_implementation`) because SDPA/FlashAttention does not
  return attention weights.
- LLaVA-1.5 stores **one** `<image>` sentinel in `input_ids` and expands it
  internally to 576 visual tokens; `models/token_index_utils.py` infers the
  expansion from `vision_config` to compute LM-sequence positions.
- The pipeline's stored `ground_truth` / `is_correct` fields use a constant
  label and are **not** used by Phase 4, which rebuilds per-frame labels from
  the manifest's transition structure.
- `phase_3_g3_g4.py` emits many `SparseEfficiencyWarning`s from scipy via
  Isomap. Harmless.

---

## 7. Caveats for anyone comparing against this

- Attention is reported as a **grounding diagnostic**, not a causal
  explanation. Stronger claims need intervention experiments (patch masking,
  activation patching) — not implemented here.
- Per-sequence Spearman on 10 points is noisy; trust population trends over
  individual sequences.
- "Best hidden layer" is chosen **per sequence** by highest center-distance
  Spearman, so it is a selected quantity — aggregate across sequences before
  claiming a population-level best layer.
- Head-specialization metrics are **surrogates** (spreads of per-head summary
  stats); full per-head [H, P] matrices are not saved, so JS-divergence is
  reported as `null`.
