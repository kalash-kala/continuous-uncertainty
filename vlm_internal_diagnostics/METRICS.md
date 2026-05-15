# Metrics & Plots Reference

Reference for every metric the pipeline computes and every figure it generates. Read this alongside `outputs/{run_name}/per_sequence_summary.jsonl`, `aggregate_metrics.csv`, and `figures/`.

The pipeline analyzes **sequences of 10 frames** from the same scene, asked the same yes/no question. Within a sequence, visual ambiguity peaks at a known **center frame** (`max_ambiguity_index`). All metrics are designed to test whether LLaVA's internals change *smoothly* as ambiguity changes and whether ambiguity is *encoded* somewhere in the network.

---

## Notation

| Symbol | Meaning |
|---|---|
| $z_i$ | mean-pooled **vision encoder** output for frame $i$ (one vector per frame) |
| $p_i$ | mean-pooled **multimodal projector** output for frame $i$ |
| $h_i^{(l)}$ | LM **hidden state** at layer $l$, position = final prompt token, frame $i$ |
| $a_i$ | **attention** distribution over image patches at frame $i$ (softmax probabilities) |
| $c$ | `max_ambiguity_index` — the center frame of the sequence |
| $d_i = \lvert i - c \rvert$ | "ambiguity distance" — how far frame $i$ is from the center |

**Cosine distance** is used throughout:

$$
\text{cos\_dist}(u, v) = 1 - \frac{u \cdot v}{\|u\|\,\|v\| + \varepsilon}
$$

Implementation: `metrics/distance_utils.py`.

---

## 1. Smoothness metrics (vision / projector / hidden)

These check whether adjacent frames produce similar internal representations. Big jumps = unstable. Computed identically for $z$, $p$, and each layer's $h^{(l)}$.

### 1.1 Adjacent jump

$$
\text{jump}_i = \text{cos\_dist}(\text{feat}_i, \text{feat}_{i+1})
$$

One value per consecutive frame pair (9 values per 10-frame sequence). Implementation: `smoothness_metrics.adjacent_jumps`.

### 1.2 Mean adjacent jump

$$
\overline{\text{jump}} = \frac{1}{N-1}\sum_{i=0}^{N-2} \text{jump}_i
$$

**Reported columns:** `mean_vision_adjacent_jump`, `mean_projector_adjacent_jump`, `mean_hidden_adjacent_jump_best_layer`.
**Reading it:** small = smooth representation. Large = unstable / jumpy. Compare across layers to see *where* instability starts.

### 1.3 Max adjacent jump

$$
\max_i \text{jump}_i
$$

Catches single-frame discontinuities that the mean averages away. **Reported columns:** `max_vision_adjacent_jump`, `max_projector_adjacent_jump`.

### 1.4 Path length

$$
\text{path\_length} = \sum_{i=0}^{N-2} \text{jump}_i
$$

Total distance traversed in feature space across the sequence. **Reported columns:** `vision_path_length`, `projector_path_length`. A short path with smooth jumps means the model traces a tidy trajectory through feature space.

### 1.5 Center distances

$$
\text{cd}_i = \text{cos\_dist}(\text{feat}_i, \text{feat}_c)
$$

Distance from each frame to the center frame. If ambiguity is encoded, this should be small at the center and grow as $i$ moves away from $c$.

### 1.6 Center-distance Spearman

$$
\rho = \text{spearman}(\text{cd}_i,\ d_i)
$$

Rank correlation between distance-to-center and ambiguity-distance-to-center. **Reported columns:** `vision_center_distance_spearman`, `projector_center_distance_spearman`, `best_hidden_center_distance_spearman`.

**Reading it:**
- $\rho$ near **+1** → representation smoothly *opens up* away from the center (good: ambiguity is encoded directionally).
- $\rho$ near **0** → no ordering relationship.
- $\rho$ near **−1** → representation moves *toward* center as ambiguity grows (rare; suspicious).

### 1.7 Smoothness drop (projector vs vision)

$$
\Delta = \overline{\text{jump}}_{\text{projector}} - \overline{\text{jump}}_{\text{vision}}
$$

**Column:** `smoothness_drop_projector_minus_vision`. Positive value = the projector amplified instability that wasn't there in the raw vision encoder (localizes failure to the multimodal-projection stage).

### 1.8 Best hidden layer

For every transformer layer, compute `center_distance_spearman`. The **layer with the highest value** is recorded as `best_hidden_layer_by_spearman`. That layer encodes ambiguity most strongly. Implementation: `sequence_metrics.aggregate_sequence`.

---

## 2. Attention metrics

Attention is sliced from the chosen query token (default: final prompt token) to the 576 image-patch keys, then averaged across heads. Each frame produces one **patch-attention vector** $a_i$ of length ~576.

### 2.1 Attention entropy

$$
H(a_i) = -\sum_p \tilde a_i[p]\,\log \tilde a_i[p], \quad \tilde a_i = a_i / \sum_p a_i[p]
$$

High entropy = attention is spread across the image (uncertain grounding). Low entropy = attention is concentrated on a few patches (confident grounding). Implementation: `attention_metrics.attention_entropy`. **Column:** `mean_attention_entropy`.

### 2.2 Attention jump

$$
\text{attn\_jump}_i = 1 - \cos(a_i, a_{i+1})
$$

Cosine-distance between consecutive attention maps. Large jump = attention re-routes suddenly between frames. **Column:** `mean_attention_jump`.

### 2.3 Attention entropy ↔ ambiguity Spearman

$$
\rho = \text{spearman}(H(a_i),\ d_i)
$$

**Column:** `attention_entropy_ambiguity_spearman`.

**Reading it:**
- **Negative $\rho$** = entropy is *highest near the center* (attention diffuses when ambiguity is highest — the model "doesn't know where to look"). This is the *expected* behavior of a well-behaved model.
- $\rho \approx 0$ = no relationship.
- **Positive $\rho$** = entropy is *highest far from center* (unexpected — model is most confident in its grounding at the center).

### 2.4 Attention center-distance Spearman

$$
\rho = \text{spearman}\bigl(1 - \cos(a_i, a_c),\ d_i\bigr)
$$

**Column:** `attention_center_distance_spearman`. Positive $\rho$ = attention maps smoothly diverge from the center map as ambiguity grows.

---

## 3. LM-head / decision metrics

For binary yes/no questions, the final-position logits are reduced to two scalars:

$$
\ell_{\text{yes}} = \text{logsumexp}(\{\text{logit}(t) : t \in \text{yes variants}\}), \quad \ell_{\text{no}} = \text{logsumexp}(\{\text{logit}(t) : t \in \text{no variants}\})
$$

Yes variants: `["yes", "Yes", " yes", " Yes"]`. No variants: analogous. Implementation: `extraction/extract_logits.py`.

### 3.1 Yes/no margin

$$
\text{margin}_i = \ell_{\text{yes}}(i) - \ell_{\text{no}}(i)
$$

**Per-frame column:** `logit_margin_yes_no`. Positive → model says yes. Negative → no. Magnitude → confidence.

### 3.2 Binary entropy

$$
p_{\text{yes}} = \frac{e^{\ell_{\text{yes}}}}{e^{\ell_{\text{yes}}} + e^{\ell_{\text{no}}}}, \quad H = -p_{\text{yes}} \log p_{\text{yes}} - (1-p_{\text{yes}})\log(1-p_{\text{yes}})
$$

**Per-frame column:** `binary_entropy`. Range $[0, \log 2 \approx 0.693]$. High value = the model is genuinely uncertain. Expected to peak at the center frame.

### 3.3 Boundary index

$$
b = \arg\min_i \lvert \text{margin}_i \rvert
$$

The frame where the model is *least confident* — its de facto decision boundary between yes and no. **Column:** `boundary_index`.

### 3.4 Boundary error

$$
\text{boundary\_error} = \lvert b - c \rvert
$$

**Column:** `boundary_error`. **0** = the model's decision boundary aligns exactly with the human-labeled ambiguity center. Larger values = the model becomes uncertain in the wrong place.

### 3.5 Center absolute margin

$$
\lvert \text{margin}_c \rvert
$$

**Column:** `center_abs_margin`. Should be **small** for well-calibrated models — the center frame is supposed to be the hardest. Large = the model is overconfident at the genuinely ambiguous frame.

### 3.6 Mean absolute margin

$$
\frac{1}{N}\sum_i \lvert\text{margin}_i\rvert
$$

**Column:** `mean_abs_margin`. Overall decisiveness of the model across the sequence.

### 3.7 Answer flip rate

$$
\text{flip\_rate} = \frac{\#\{i : \text{ans}_i \ne \text{ans}_{i+1}\}}{N - 1}
$$

**Column:** `answer_flip_rate`. Fraction of consecutive frames where the parsed answer changes. **0** = stable. High value = the model's answer is unstable across visually similar frames.

### 3.8 Accuracy

Fraction of frames where the parsed answer matches `ground_truth`. **Column:** `accuracy`.

---

## 4. Output files

Run-level layout under `outputs/{run_name}/`:

| File | Contents |
|---|---|
| `per_frame_outputs.jsonl` | one line per frame — raw logits, attention entropy per layer, hidden-state norms, paths to saved tensors. Section 11.1 of the instruction file. |
| `per_sequence_summary.jsonl` | one line per sequence — all the metrics above aggregated. Section 11.2. |
| `aggregate_metrics.csv` | flat CSV mirror of the per-sequence JSONL, one row per sequence. Spreadsheet-friendly. |
| `features/{sequence_id}/frame_XX_*.pt` | saved tensors (vision pooled, projector pooled, hidden states by layer, attention maps by layer, raw logits). float16 by default. |
| `figures/` | all plots described below. |

---

## 5. Plots — what each one shows and how to read it

### 5.1 Attention overlays — `figures/attention_overlays/{sequence_id}/frame_XX.png`

One image per frame. Original RGB frame with the per-patch attention probabilities upsampled and overlaid as a **jet colormap** (red = high attention, blue = low). White grid lines mark the 24×24 patch boundaries when enabled.

**Reading it:**
- Bright (red) regions = where the model "looked" before answering.
- Compare across frames in a sequence — does attention drift smoothly toward / away from a target object as ambiguity changes? Or does it leap to unrelated regions at the center frame?
- **Caveat:** attention is correlated with reasoning, not proof of it. Treat as a *grounding diagnostic*.

Title shows: frame index, parsed answer, yes/no margin, binary entropy.

### 5.2 Sequence grid — `figures/sequence_grids/{sequence_id}.png`

3 rows × 10 columns. The whole story of one sequence on one page.

| Row | Contents |
|---|---|
| Row 1 | Original frame thumbnails, sorted by `frame_index` (0 → 9). Center frame highlighted with a **red border**. |
| Row 2 | Attention overlay for each frame (same colormap as 5.1). |
| Row 3 | Text per frame: `ans=yes/no`, `m=<margin>`, `H=<binary_entropy>`. |

**Reading it:**
- Walk left-to-right and ask: does `m` smoothly cross zero near the red-bordered column? Does attention shift gradually or jump? Does `H` peak at the red column?

### 5.3 Trajectory plot — `figures/trajectory_plots/{sequence_id}.png`

2×2 grid for one sequence. Red dotted vertical line marks `max_ambiguity_index`.

| Subplot | Y-axis | What to look for |
|---|---|---|
| Top-left | `logit_margin_yes_no` | Should cross **0** at the red line. Gray dashed horizontal = zero margin. |
| Top-right | `binary_entropy` | Should **peak** at the red line. |
| Bottom-left | First-layer `attention_entropy` (from the per-frame dict) | Higher = more diffuse attention. |
| Bottom-right | $d_i = \lvert i - c \rvert$ | The ground-truth V-shape — reference for what an "ideal" ambiguity-tracking curve looks like. |

### 5.4 Aggregate plots — `figures/aggregate/`

These summarize across **all sequences** in the run.

| File | Plot | Reading it |
|---|---|---|
| `cat_answer_flip_rate.png` | Bar plot: mean `answer_flip_rate` per category (Geometric / Semantic / Compositional). | Tall bars = the model is unstable on that category. |
| `cat_boundary_error.png` | Bar plot: mean `boundary_error` per category. | Tall bars = the model's decision boundary is consistently misplaced for that category. |
| `cat_attention_jump.png` | Bar plot: mean `mean_attention_jump` per category. | Tall bars = attention is jumpy on that category (poor grounding stability). |
| `vision_vs_projector_scatter.png` | Each point = one sequence. X = `mean_vision_adjacent_jump`, Y = `mean_projector_adjacent_jump`. Dashed line = $y = x$. | Points **above** the line: projector adds instability the vision encoder didn't have (localizes failure to the projector). Points **on** the line: projector is faithful. Points **below**: projector is *smoothing* the vision representation. |
| `best_layer_spearman.png` | Each point = one sequence. X = `best_hidden_layer_by_spearman` (which layer best encodes ambiguity). Y = its Spearman value. | Clustering shows whether ambiguity is encoded consistently at early / middle / late layers across the dataset. |

---

## 6. Interpretation cheat sheet

| Observation | Likely localization |
|---|---|
| Large `mean_vision_adjacent_jump` | Perception-level instability (vision encoder). |
| Smooth vision, large `smoothness_drop_projector_minus_vision` | Multimodal projector distorts visual evidence. |
| Smooth vision & projector, jumpy hidden states at some layer | Reasoning-state instability inside the language model. |
| Large `mean_attention_jump` with smooth hidden states | Attention grounding is unstable even though representations are smooth. |
| Smooth hidden states, large `center_abs_margin` | Hidden states encode ambiguity but the LM head is overconfident — motivates an uncertainty head. |
| `boundary_error` consistently > 1 | Model's uncertainty is systematically misaligned with human-labeled ambiguity. |
| `answer_flip_rate` > 0.3 with high `mean_abs_margin` | High-confidence flipping — the worst failure mode. |

---

## 7. Caveats

- **Attention ≠ reasoning.** Report attention findings as *grounding diagnostics*, not causal explanations.
- **Softmax attention is relative.** A high weight on one patch means high *routing weight*, not absolute importance.
- **Spearman on 10 points is noisy.** Treat per-sequence $\rho$ values cautiously; trends across the full dataset are more reliable.
- **Best layer is run-dependent.** "Best" = highest center-distance Spearman within that sequence. Aggregate across sequences to find the population-level best layer.
- Stronger causal claims require **intervention experiments** (patch masking, activation patching) — not part of this pipeline.
