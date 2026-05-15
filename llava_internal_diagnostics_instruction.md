# Instruction File: LLaVA Internal Diagnostics for Continuous Ambiguity VLM Dataset

## 0. Purpose

Implement a **quick internal-diagnostics pipeline** for a continuous visual uncertainty / ambiguity dataset using **LLaVA** first.

The dataset consists of natural visual QA trajectories. Each sequence contains multiple frames from the same scene, usually 10 frames, with the **same question** asked on every frame. The visual ambiguity changes gradually toward or away from a maximally ambiguous frame.

The current paper already evaluates external behavior: accuracy, uncertainty, calibration, and monotonicity. This implementation should inspect **internal VLM behavior** to understand whether LLaVA internally tracks ambiguity smoothly.

The first implementation target is **LLaVA**. Please keep the code modular enough to support more open source VLMs later (like qwen and pixtral).

---

## 1. Main Research Goal

For each ordered image trajectory:

```text
frame_0 → frame_1 → ... → frame_9
same question for every frame
known max_ambiguity_index
```

diagnose whether LLaVA’s internals change smoothly and meaningfully as visual ambiguity changes.

Specifically, assess these six components:

1. VLM input formatting and token indexing
2. Vision encoder features
3. Multimodal projector outputs
4. Transformer hidden states
5. Text-to-image attention maps
6. LM-head logits and answer margins

Also implement an attention visualization idea:

> Extract the softmax attention probabilities after QK dot product inside the transformer. Slice attention from selected text/query tokens to image patch/key tokens. Map those probabilities back to the image patch grid and overlay them on the original image.

---

## 2. Expected Dataset Format

The dataset manifest is a CSV file (`list_questions_revised.csv`). Each row represents one **sequence** (not one frame). The loader must expand each row into per-frame records.

### CSV Schema

```text
video_path,question,frames,center_frames,gt_answer,primary_category
```

| Column | Type | Description |
|---|---|---|
| `video_path` | string | Relative path to the source video, e.g. `images/NExTVideo/0089/3066966990.mp4` |
| `question` | string | Binary yes/no question, same for every frame in the sequence |
| `frames` | list of ints | 10 frame indices (zero-padded, e.g. `[0010, 0015, 0018, ...]`) |
| `center_frames` | list of ints | 1 or 2 frame indices marking the maximally ambiguous moment |
| `gt_answer` | string or list | Ground truth answer — see note below |
| `primary_category` | string | One of `Geometric`, `Semantic`, `Compositional` |

### gt_answer Format

The `gt_answer` column has two possible formats:

1. **Single value** (most common): a plain string `yes` or `no`.

   ```text
   yes
   no
   ```

2. **List with repeated values** (when `center_frames` contains 2 frames): a bracketed list where both elements are identical, e.g. `[yes, yes]` or `[no, no]`.

   ```text
   [yes, yes]
   [no, no]
   ```

   The list form is present in the data but is not used in the current experiments — both entries always agree. For experiments, parse either form to extract a single `yes`/`no` label.

### Image Directory Layout

Extracted frames are stored under:

```text
images/NExTVideo_{folder}_{video_id}/frame_{XXXX}.png
```

The directory name is derived from the `video_path` by replacing path separators with underscores and dropping the `.mp4` suffix:

```text
video_path:  images/NExTVideo/0089/3066966990.mp4
image_dir:   images/NExTVideo_0089_3066966990/
frame files: frame_1840.png, frame_1845.png, ...
```

Frame filenames use the raw integer from the `frames` list, zero-padded to 4 digits:

```text
frame_{frame_index:04d}.png
```

### Derived Fields

The loader should produce these per-frame fields, which are not in the CSV but can be computed:

```text
sequence_id:               derived from video_path, e.g. "NExTVideo_0089_3066966990"
frame_index:               position in the frames list (0–9)
frame_number:              actual frame value from the frames list
image_path:                images/{sequence_id}/frame_{frame_number:04d}.png
max_ambiguity_frame:       center_frames[0] (use first element if list has 2)
max_ambiguity_index:       position of max_ambiguity_frame in frames list
relative_ambiguity_distance: abs(frame_index - max_ambiguity_index)
gt_answer_parsed:          single yes/no string, normalized from either gt_answer format
```

### Loader Behavior

Group rows by `sequence_id` (one row per sequence in the CSV). Within each sequence, iterate over `frames` in order to produce 10 per-frame records sorted by `frame_index`.

---

## 3. Minimum Viable Experiment

Start with a small subset.

```text
Model:
- LLaVA-style HuggingFace model

Data:
- 20 sequences total
- roughly 7 geometric
- roughly 7 semantic
- roughly 6 compositional
- 10 frames per sequence

For each frame:
1. Run LLaVA with the fixed question.
2. Save generated answer.
3. Extract vision encoder features.
4. Extract projected visual tokens.
5. Extract transformer hidden states.
6. Extract attention matrices.
7. Extract yes/no logits if the question is binary.
8. Create attention heatmap overlay from text tokens to image tokens.
9. Compute trajectory-level metrics.
```

---

## 4. LLaVA-Specific Implementation Notes

### 4.1 Model Loading

Use HuggingFace `transformers`.

Recommended model class depends on the specific checkpoint. For newer HF integration, use:

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor

model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager"
)
model.eval()
```

Important:

```python
attn_implementation="eager"
```

is needed because SDPA/FlashAttention often does not return attention weights.

If the model still does not return attention, try:

```python
model.config._attn_implementation = "eager"
```

and avoid FlashAttention kernels.

### 4.2 Disable FlashAttention / SDPA if needed

If attention tensors are missing, explicitly force eager mode.

Possible options:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

or in Python:

```python
model.config._attn_implementation = "eager"
```

If custom loading uses FlashAttention, disable it.

### 4.3 Prompt Format

Use the official processor / chat template when possible.

Example:

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ],
    }
]

prompt = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True
)
```

Fallback LLaVA-style prompt:

```text
USER: <image>
{question}
ASSISTANT:
```

Always use the same prompt format for every frame in a sequence.

---

## 5. Priority 1: Input Formatting and Token Indexing

### Goal

Identify exactly how LLaVA represents:

```text
image tokens
question tokens
final prompt token
generated answer tokens
```

This is required before computing attention maps or hidden-state diagnostics.

### Required Output

Create a function:

```python
def get_token_indices(processor, model, image, question, generated_ids=None):
    """
    Returns token-index metadata needed for internal diagnostics.
    """
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_token_indices": image_token_indices,
        "question_token_indices": question_token_indices,
        "final_prompt_token_index": final_prompt_token_index,
        "first_answer_token_index": first_answer_token_index,
        "image_grid_shape": image_grid_shape,
        "num_image_tokens": num_image_tokens,
        "decoded_prompt_tokens": decoded_prompt_tokens
    }
```

### What to Identify

For LLaVA, image tokens are usually inserted where the special image token appears in the prompt. Depending on implementation, the input may contain a single `<image>` token that is internally expanded into many image features.

You must verify:

```text
1. Does input_ids contain one special image token or expanded image tokens?
2. Does the language model sequence contain expanded visual tokens?
3. What is the actual range of image token positions in the transformer hidden states and attentions?
```

### Expected Token Order

For decoder-only LLaVA-style models, the effective transformer sequence is usually:

```text
[image tokens] + [text/question tokens] + [answer tokens]
```

Due to causal masking:

```text
question tokens can attend to previous image tokens
answer tokens can attend to image and question tokens
image tokens usually cannot attend to future question tokens
```

Therefore, attention maps should focus on:

```text
question token queries → image token keys
final prompt token query → image token keys
first answer token query → image token keys
```

Do **not** assume image tokens attend to question tokens unless the architecture explicitly supports bidirectional/cross-attention.

---

## 6. Priority 2: Vision Encoder Features

### Goal

Check whether the raw vision encoder representation changes smoothly across adjacent frames.

This tells us whether instability starts at the low-level visual perception stage.

### Required Extraction

For each frame:

```python
vision_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
```

or the correct equivalent for the chosen LLaVA implementation.

Save:

```text
vision_last_hidden_state
vision_pooled_feature, if available
patch-level visual features
global mean-pooled visual feature
```

Recommended pooled representation:

```python
z_i = vision_last_hidden_state.mean(dim=1)
```

### Metrics

For each sequence:

#### 1. Adjacent vision jump

```python
vision_adjacent_jump_i = cosine_distance(z_i, z_{i+1})
```

#### 2. Mean adjacent vision jump

```python
mean_vision_adjacent_jump = mean_i vision_adjacent_jump_i
```

#### 3. Max adjacent vision jump

```python
max_vision_adjacent_jump = max_i vision_adjacent_jump_i
```

#### 4. Distance to ambiguity center

```python
vision_center_distance_i = cosine_distance(z_i, z_center)
```

#### 5. Center-distance Spearman

```python
spearman(vision_center_distance_i, abs(frame_index_i - max_ambiguity_index))
```

#### 6. Representation path length

```python
vision_path_length = sum_i cosine_distance(z_i, z_{i+1})
```

### Interpretation

```text
Vision features unstable:
    likely perception-level instability.

Vision features smooth but later states unstable:
    likely projector/alignment/reasoning issue.
```

---

## 7. Priority 3: Multimodal Projector Outputs

### Goal

Check whether smooth visual representations remain smooth after projection into the language-model token space.

This isolates the multimodal alignment module.

### Required Extraction

For each frame, extract:

```python
vision_features_i
projected_visual_tokens_i = model.multi_modal_projector(vision_features_i)
```

or the corresponding LLaVA projector module.

Save:

```text
projected_visual_tokens
mean-pooled projected visual representation
```

Recommended representation:

```python
p_i = projected_visual_tokens_i.mean(dim=1)
```

### Metrics

Compute the same metrics as vision features:

```text
projector_adjacent_jump
mean_projector_adjacent_jump
max_projector_adjacent_jump
projector_center_distance
projector_center_distance_spearman
projector_path_length
```

Also compute:

```python
smoothness_drop = mean_projector_adjacent_jump - mean_vision_adjacent_jump
```

### Interpretation

```text
Vision smooth, projected tokens unstable:
    multimodal projector may distort visual evidence.

Projected tokens smooth, LLM hidden states unstable:
    instability likely occurs inside language/reasoning layers.
```

---

## 8. Priority 4: Transformer Hidden States

### Goal

Analyze whether LLaVA’s language-side hidden states evolve smoothly with ambiguity.

We want to inspect layerwise hidden states and determine where ambiguity information appears.

### Required Extraction

Run model forward with:

```python
outputs = model(
    **inputs,
    output_hidden_states=True,
    output_attentions=True,
    return_dict=True
)
```

Extract hidden states:

```python
hidden_states = outputs.hidden_states
```

For every layer `l`, extract:

```text
1. mean hidden state over image token positions
2. mean hidden state over question token positions
3. hidden state at final prompt token
4. hidden state at first generated answer token, if available
```

Recommended primary representation:

```python
h_i_l = hidden_states[l][batch_idx, final_prompt_token_index, :]
```

Also save:

```python
h_i_l_image_mean = hidden_states[l][batch_idx, image_token_indices, :].mean(dim=0)
h_i_l_question_mean = hidden_states[l][batch_idx, question_token_indices, :].mean(dim=0)
```

### Metrics

For every layer and sequence:

#### 1. Adjacent hidden-state jump

```python
hidden_adjacent_jump_i_l = cosine_distance(h_i_l, h_{i+1}_l)
```

#### 2. Mean adjacent hidden jump

```python
mean_hidden_adjacent_jump_l = mean_i hidden_adjacent_jump_i_l
```

#### 3. Distance to ambiguity center

```python
hidden_center_distance_i_l = cosine_distance(h_i_l, h_center_l)
```

#### 4. Center-distance Spearman

```python
spearman(hidden_center_distance_i_l, abs(frame_index_i - max_ambiguity_index))
```

#### 5. Layerwise ambiguity decodability

Train a simple probe:

```python
probe(h_i_l) -> ambiguity_distance = abs(frame_index_i - max_ambiguity_index)
```

Use a train/test split by sequence, not by frame.

Report:

```text
R^2 for regression
Spearman correlation for predicted vs true ambiguity distance
classification accuracy for ambiguous vs clear, if using bins
```

### Layerwise Plots

Produce:

```text
layer index vs mean hidden adjacent jump
layer index vs hidden center-distance Spearman
layer index vs probe performance
```

### Interpretation

Potential pattern:

```text
early layers: visual similarity
middle layers: ambiguity is encoded
late layers: answer decision dominates
```

Important possible finding:

```text
Hidden states encode ambiguity, but output uncertainty/logits do not.
```

This would motivate adding an uncertainty head or training objective.

---

## 9. Priority 5: Attention Maps

### Goal

Implement text-to-image attention diagnostics.

We want to see whether LLaVA attends to ambiguity-relevant image patches as ambiguity changes.

This implements the main attention idea:

> Use the attention maps after QK dot product and softmax. Slice attention probabilities from text query tokens to image patch key tokens. Map those image-token probabilities back onto the image patch grid and overlay them on the original image using patch boundaries.

### Required Extraction

Run with:

```python
outputs = model(
    **inputs,
    output_attentions=True,
    output_hidden_states=True,
    return_dict=True
)
```

Extract:

```python
attentions = outputs.attentions
```

Typical attention shape:

```python
attentions[l].shape == [batch_size, num_heads, seq_len, seq_len]
```

Attention definition:

```python
A = softmax(QK^T / sqrt(d))
```

Rows are query tokens. Columns are key tokens.

### Attention Slices

Extract:

```python
A_question_to_image = A[:, :, question_token_indices, image_token_indices]
A_final_prompt_to_image = A[:, :, final_prompt_token_index, image_token_indices]
A_answer_to_image = A[:, :, first_answer_token_index, image_token_indices]
```

Recommended default map:

```text
first answer token → image tokens
```

Fallback:

```text
final prompt token → image tokens
```

### Aggregation

Support configurable aggregation over:

```text
layers
heads
query tokens
```

Default settings:

```yaml
attention:
  query_source: "final_prompt" # options: final_prompt, first_answer, question_mean
  layer_selection: "middle_late" # options: all, early, middle, late, specific
  head_aggregation: "mean"
  normalize_heatmap: true
```

Example:

```python
attention_to_image = attentions[layer][0, :, query_index, image_token_indices]
attention_to_image = attention_to_image.mean(dim=0)
```

### Mapping Attention to Image Grid

For LLaVA with CLIP ViT-L/14 at 336 resolution, image grid is often:

```text
24 × 24 = 576 image tokens
```

But do not hard-code this. Infer if possible.

General logic:

```python
num_image_tokens = len(image_token_indices)
H_patch, W_patch = infer_grid_shape(num_image_tokens)
attention_grid = attention_to_image.reshape(H_patch, W_patch)
```

Then:

```python
attention_grid = normalize(attention_grid)
attention_grid_upsampled = resize(attention_grid, original_image_size)
overlay = overlay_heatmap(original_image, attention_grid_upsampled)
```

Save:

```text
outputs/figures/attention_overlays/{model_name}/{sequence_id}/frame_00.png
...
```

Also create sequence-level grid:

```text
10 frames in columns
row 1: original frames
row 2: attention overlays
row 3: generated answer + margin + uncertainty
```

### Attention Metrics

For each frame and sequence:

#### 1. Attention entropy

```python
H(a_i) = -sum_p a_i[p] * log(a_i[p] + eps)
```

where `a_i` is normalized over image patches.

#### 2. Adjacent attention jump

```python
attention_jump_i = 1 - cosine_similarity(a_i, a_{i+1})
```

#### 3. Attention center-distance correlation

```python
spearman(attention_entropy_i, abs(frame_index_i - max_ambiguity_index))
```

and:

```python
spearman(attention_center_distance_i, abs(frame_index_i - max_ambiguity_index))
```

where:

```python
attention_center_distance_i = 1 - cosine_similarity(a_i, a_center)
```

#### 4. Target-region attention mass, if object boxes/masks are available

If bounding boxes or masks exist for relevant objects:

```python
target_mass = sum attention over target patches
background_mass = sum attention over non-target patches
target_attention_ratio = target_mass / (target_mass + background_mass)
```

For question:

```text
Is the child at the right of the woman?
```

target regions:

```text
child + woman + relation area
```

If boxes/masks are unavailable, skip this metric for now.

### Caveats

1. Attention is not proof of reasoning. Phrase it as:

```text
attention-based grounding diagnostic
```

not as a causal explanation.

2. Softmax attention is relative. High weight on one patch means high relative routing weight, not absolute causal importance.

3. Store raw attention values and normalized heatmap values separately.

4. If possible later, strengthen attention results with masking/ablation:

```text
mask attended region → answer/uncertainty should change
mask background → answer/uncertainty should change less
```

---

## 10. Priority 6: LM-Head Logits and Answer Margins

### Goal

Track whether the model’s final answer decision becomes less confident near the maximally ambiguous frame.

This is especially useful for yes/no questions.

### Required Extraction

For each frame, compute first-token logits after the prompt.

```python
with torch.no_grad():
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits
```

For binary yes/no:

```python
yes_token_ids = tokenizer variants for ["yes", "Yes", " yes", " Yes"]
no_token_ids = tokenizer variants for ["no", "No", " no", " No"]
```

Use logsumexp over variants:

```python
logit_yes = logsumexp(logits[first_answer_position, yes_token_ids])
logit_no = logsumexp(logits[first_answer_position, no_token_ids])
margin = logit_yes - logit_no
prob_yes = softmax([logit_yes, logit_no])[0]
binary_entropy = entropy([prob_yes, 1 - prob_yes])
```

### Metrics

For each sequence:

#### 1. Margin curve

```text
margin_i over frame index
```

#### 2. Decision boundary

Frame where margin crosses zero.

If no exact crossing, use frame with minimum absolute margin:

```python
boundary_index = argmin_i abs(margin_i)
```

#### 3. Boundary error

```python
boundary_error = abs(boundary_index - max_ambiguity_index)
```

#### 4. Confidence at ambiguity center

```python
center_confidence = abs(margin_center)
```

#### 5. Answer flip rate

```python
answer_flip_rate = number_of_adjacent_answer_changes / (num_frames - 1)
```

#### 6. High-confidence instability

```python
high_confidence_flip = answer flips where abs(margin_i) and abs(margin_{i+1}) are high
```

### Expected Behavior

For a yes/no spatial question:

```text
clear yes: large positive margin
ambiguous center: margin near zero
clear no: large negative margin
```

Bad behavior:

```text
large confident margin near ambiguity center
sudden margin jump between adjacent frames
decision boundary far from human ambiguity center
```

---

## 11. Required Output Structures

### 11.1 Per-Frame JSONL

Write one line per frame.

File:

```text
outputs/{run_name}/per_frame_outputs.jsonl
```

Schema:

```json
{
  "run_name": "llava_internal_diag_v1",
  "model_name": "llava-hf/llava-1.5-7b-hf",
  "sequence_id": "seq_0001",
  "frame_index": 0,
  "image_path": "/path/to/frame_000.png",
  "question": "Is the child at the right of the woman?",
  "category": "geometric",
  "max_ambiguity_index": 5,
  "relative_ambiguity_distance": 5,
  "ground_truth": "yes",
  "generated_answer": "yes",
  "parsed_answer": "yes",
  "is_correct": true,

  "num_image_tokens": 576,
  "image_grid_shape": [24, 24],
  "final_prompt_token_index": 612,
  "first_answer_token_index": null,

  "logit_yes": 4.21,
  "logit_no": 1.14,
  "logit_margin_yes_no": 3.07,
  "binary_entropy": 0.18,

  "attention_entropy": {
    "layer_16_final_prompt": 4.81,
    "layer_24_final_prompt": 4.2
  },

  "attention_heatmap_path": "outputs/.../frame_00_attention.png",

  "hidden_state_norms": {
    "layer_0_final_prompt": 42.1,
    "layer_16_final_prompt": 51.3,
    "layer_31_final_prompt": 57.7
  },

  "feature_paths": {
    "vision_feature": "outputs/.../features/seq_0001_frame_00_vision.pt",
    "projected_feature": "outputs/.../features/seq_0001_frame_00_projected.pt",
    "hidden_states": "outputs/.../features/seq_0001_frame_00_hidden.pt",
    "attention_map": "outputs/.../features/seq_0001_frame_00_attention.pt"
  }
}
```

### 11.2 Per-Sequence Summary JSONL

Write one line per sequence.

File:

```text
outputs/{run_name}/per_sequence_summary.jsonl
```

Schema:

```json
{
  "run_name": "llava_internal_diag_v1",
  "model_name": "llava-hf/llava-1.5-7b-hf",
  "sequence_id": "seq_0001",
  "category": "geometric",
  "num_frames": 10,
  "max_ambiguity_index": 5,

  "accuracy": 0.8,
  "answer_flip_rate": 0.22,

  "mean_vision_adjacent_jump": 0.04,
  "max_vision_adjacent_jump": 0.09,
  "vision_path_length": 0.33,
  "vision_center_distance_spearman": 0.71,

  "mean_projector_adjacent_jump": 0.08,
  "max_projector_adjacent_jump": 0.15,
  "projector_path_length": 0.61,
  "projector_center_distance_spearman": 0.66,
  "smoothness_drop_projector_minus_vision": 0.04,

  "best_hidden_layer_by_spearman": 18,
  "best_hidden_center_distance_spearman": 0.74,
  "mean_hidden_adjacent_jump_layer_18": 0.11,

  "mean_attention_entropy": 4.55,
  "mean_attention_jump": 0.17,
  "attention_entropy_ambiguity_spearman": -0.42,
  "attention_center_distance_spearman": 0.55,

  "boundary_index": 4,
  "boundary_error": 1,
  "center_abs_margin": 0.32,
  "mean_abs_margin": 2.41,

  "sequence_figure_path": "outputs/.../figures/seq_0001_grid.png"
}
```

### 11.3 Aggregate CSV

File:

```text
outputs/{run_name}/aggregate_metrics.csv
```

One row per sequence.

Columns:

```text
run_name
model_name
sequence_id
category
num_frames
max_ambiguity_index
accuracy
answer_flip_rate
mean_vision_adjacent_jump
max_vision_adjacent_jump
vision_path_length
vision_center_distance_spearman
mean_projector_adjacent_jump
max_projector_adjacent_jump
projector_path_length
projector_center_distance_spearman
smoothness_drop_projector_minus_vision
best_hidden_layer_by_spearman
best_hidden_center_distance_spearman
mean_hidden_adjacent_jump_best_layer
mean_attention_entropy
mean_attention_jump
attention_entropy_ambiguity_spearman
attention_center_distance_spearman
boundary_index
boundary_error
center_abs_margin
mean_abs_margin
```

### 11.4 Saved Tensor Features

Save large tensors separately instead of embedding them in JSON.

Recommended structure:

```text
outputs/{run_name}/features/{sequence_id}/
    frame_00_vision.pt
    frame_00_projected.pt
    frame_00_hidden_selected_layers.pt
    frame_00_attention_selected_layers.pt
    frame_00_logits.pt
```

Do not save all layers/heads for all frames by default if disk usage is too large.

Make this configurable.

---

## 12. Required Visualizations

### 12.1 Attention Overlay per Frame

File:

```text
outputs/{run_name}/figures/attention_overlays/{sequence_id}/frame_00.png
```

Each image should show:

```text
original image
attention heatmap overlay
optional patch grid boundaries
title with frame index, answer, margin, entropy
```

### 12.2 Sequence Grid

File:

```text
outputs/{run_name}/figures/sequence_grids/{sequence_id}.png
```

Recommended layout:

```text
columns = frames 0 to 9

row 1 = original frames
row 2 = attention overlays
row 3 = generated answer + yes/no margin + binary entropy
```

Mark the maximum ambiguity frame with a red border or label.

### 12.3 Trajectory Plots

For each sequence:

```text
outputs/{run_name}/figures/trajectory_plots/{sequence_id}.png
```

Include separate plots:

```text
frame index vs yes/no margin
frame index vs binary entropy
frame index vs attention entropy
frame index vs attention jump
frame index vs hidden-state distance to center
frame index vs vision/projector distance to center
```

### 12.4 Aggregate Plots

Save:

```text
outputs/{run_name}/figures/aggregate/
```

Required plots:

```text
1. layer index vs hidden center-distance Spearman
2. category-wise answer flip rate
3. category-wise boundary error
4. category-wise mean attention jump
5. vision vs projector smoothness scatter
6. hidden-state ambiguity Spearman by layer
```

---

## 13. Config Files

Create configuration files under:

```text
configs/
```

### 13.1 `configs/model_config.yaml`

```yaml
model:
  name: "llava-hf/llava-1.5-7b-hf"
  family: "llava"
  device_map: "auto"
  torch_dtype: "float16"
  attn_implementation: "eager"
  load_in_4bit: false
  load_in_8bit: false

generation:
  max_new_tokens: 32
  do_sample: false
  temperature: 0.0
  top_p: null
  num_beams: 1

prompt:
  use_chat_template: true
  fallback_template: "USER: <image>\n{question}\nASSISTANT:"
```

### 13.2 `configs/data_config.yaml`

```yaml
data:
  manifest_path: "/path/to/manifest.jsonl"
  image_root: "/path/to/images"
  split: "test"
  max_sequences: 20
  frames_per_sequence: 10
  categories:
    - geometric
    - semantic
    - compositional

sequence:
  id_column: "sequence_id"
  frame_index_column: "frame_index"
  image_path_column: "image_path"
  question_column: "question"
  category_column: "category"
  max_ambiguity_column: "max_ambiguity_index"
  ground_truth_column: "ground_truth"
```

### 13.3 `configs/extraction_config.yaml`

```yaml
extraction:
  save_vision_features: true
  save_projector_features: true
  save_hidden_states: true
  save_attentions: true
  save_logits: true

hidden_states:
  token_sources:
    - final_prompt
    - image_mean
    - question_mean
  layers: "all" # options: all, selected
  selected_layers: [0, 8, 16, 24, 31]

attentions:
  enabled: true
  query_source: "final_prompt" # options: final_prompt, first_answer, question_mean
  key_source: "image_tokens"
  layers: [8, 16, 24, 31]
  aggregate_heads: "mean"
  aggregate_layers: false
  save_raw_attention: false
  save_attention_heatmap: true

logits:
  compute_yes_no_margin: true
  yes_variants: ["yes", "Yes", " yes", " Yes"]
  no_variants: ["no", "No", " no", " No"]

storage:
  save_large_tensors: true
  tensor_dtype: "float16"
```

### 13.4 `configs/visualization_config.yaml`

```yaml
visualization:
  create_attention_overlays: true
  create_sequence_grids: true
  create_trajectory_plots: true
  create_aggregate_plots: true

attention_overlay:
  colormap: "jet"
  alpha: 0.45
  draw_patch_boundaries: true
  normalize_per_frame: true
  normalize_per_sequence: false

sequence_grid:
  mark_ambiguity_center: true
  show_generated_answer: true
  show_margin: true
  show_binary_entropy: true
```

### 13.5 `configs/run_config.yaml`

```yaml
run:
  run_name: "llava_internal_diag_v1"
  output_dir: "outputs/llava_internal_diag_v1"
  seed: 42
  overwrite: false
  num_workers: 4
  batch_size: 1
  device: "cuda"

debug:
  enabled: false
  limit_sequences: 3
  limit_frames_per_sequence: 10
```

---

## 14. Expected Repository / File Structure

Create this structure:

```text
vlm_internal_diagnostics/
    README.md

    configs/
        model_config.yaml
        data_config.yaml
        extraction_config.yaml
        visualization_config.yaml
        run_config.yaml

    data/
        dataset_loader.py
        manifest_utils.py

    models/
        llava_loader.py
        token_index_utils.py
        hook_utils.py

    extraction/
        extract_all.py
        extract_vision_features.py
        extract_projector_features.py
        extract_hidden_states.py
        extract_attentions.py
        extract_logits.py

    metrics/
        distance_utils.py
        smoothness_metrics.py
        attention_metrics.py
        logit_metrics.py
        hidden_probe_metrics.py
        sequence_metrics.py

    visualization/
        attention_overlay.py
        trajectory_plots.py
        qualitative_grid.py
        aggregate_plots.py

    scripts/
        run_extract.py
        run_metrics.py
        run_visualize.py
        run_all.py

    outputs/
        llava_internal_diag_v1/
            per_frame_outputs.jsonl
            per_sequence_summary.jsonl
            aggregate_metrics.csv
            features/
            figures/
```

---

## 15. Script Interfaces

### 15.1 Run everything

```bash
python scripts/run_all.py \
  --model_config configs/model_config.yaml \
  --data_config configs/data_config.yaml \
  --extraction_config configs/extraction_config.yaml \
  --visualization_config configs/visualization_config.yaml \
  --run_config configs/run_config.yaml
```

### 15.2 Extraction only

```bash
python scripts/run_extract.py \
  --model_config configs/model_config.yaml \
  --data_config configs/data_config.yaml \
  --extraction_config configs/extraction_config.yaml \
  --run_config configs/run_config.yaml
```

### 15.3 Metrics only

```bash
python scripts/run_metrics.py \
  --run_dir outputs/llava_internal_diag_v1
```

### 15.4 Visualizations only

```bash
python scripts/run_visualize.py \
  --run_dir outputs/llava_internal_diag_v1 \
  --visualization_config configs/visualization_config.yaml
```

---

## 16. Core Metrics to Implement

### 16.1 Cosine Distance

```python
def cosine_distance(a, b, eps=1e-8):
    return 1.0 - cosine_similarity(a, b, eps=eps)
```

### 16.2 Adjacent Jump

```python
def adjacent_jumps(features):
    return [
        cosine_distance(features[i], features[i+1])
        for i in range(len(features)-1)
    ]
```

### 16.3 Path Length

```python
def path_length(features):
    return sum(adjacent_jumps(features))
```

### 16.4 Center Distances

```python
def center_distances(features, center_index):
    center = features[center_index]
    return [cosine_distance(f, center) for f in features]
```

### 16.5 Spearman with Ambiguity Distance

```python
def ambiguity_spearman(values, frame_indices, center_index):
    ambiguity_distances = [abs(i - center_index) for i in frame_indices]
    return spearmanr(values, ambiguity_distances).correlation
```

### 16.6 Attention Entropy

```python
def attention_entropy(attn, eps=1e-12):
    attn = attn / (attn.sum() + eps)
    return -float((attn * (attn + eps).log()).sum())
```

### 16.7 Yes/No Margin

```python
margin = logit_yes - logit_no
```

### 16.8 Binary Entropy

```python
p_yes = softmax([logit_yes, logit_no])[0]
entropy = -p_yes * log(p_yes) - (1 - p_yes) * log(1 - p_yes)
```

### 16.9 Boundary Error

```python
boundary_index = argmin(abs(margins))
boundary_error = abs(boundary_index - max_ambiguity_index)
```

### 16.10 Answer Flip Rate

```python
answer_flip_rate = number_of_adjacent_answer_changes / (num_frames - 1)
```

---

## 17. Implementation Checks and Sanity Tests

Before running full experiments, perform these checks.

### 17.1 Token Index Sanity

For one frame, print:

```text
decoded prompt tokens
image token range
question token range
final prompt token index
number of image tokens
image grid shape
```

Verify:

```text
num_image_tokens == H_patch * W_patch
```

if using patch-grid attention.

### 17.2 Attention Shape Sanity

Print:

```python
len(outputs.attentions)
outputs.attentions[0].shape
```

Expected:

```text
num_layers tensors
each tensor: [batch, heads, seq_len, seq_len]
```

### 17.3 Heatmap Sanity

For one frame:

```text
attention_to_image.shape == [num_image_tokens]
attention_grid.shape == [H_patch, W_patch]
```

Overlay should roughly align with image patches.

### 17.4 Logit Sanity

Print token IDs for yes/no variants:

```text
yes variants and token IDs
no variants and token IDs
```

Check whether tokenization creates multiple tokens. If so, handle carefully.

### 17.5 Generation Sanity

For a few frames, print:

```text
question
generated answer
ground truth
yes/no margin
binary entropy
```

---

## 18. Final Deliverables

The final implementation should produce:

### A. Codebase

A modular Python project with the folder structure described above.

### B. Config files

All required YAML configs.

### C. Raw outputs

```text
per_frame_outputs.jsonl
per_sequence_summary.jsonl
aggregate_metrics.csv
saved tensor features
```

### D. Visual outputs

```text
attention overlays per frame
sequence grids
trajectory plots
aggregate plots
```

### E. README

The README should explain:

```text
1. how to install dependencies
2. how to prepare the manifest
3. how to run extraction
4. how to run metrics
5. how to run visualization
6. known LLaVA-specific caveats
```

### F. Brief analysis report

Create:

```text
outputs/{run_name}/analysis_report.md
```

with:

```text
1. number of sequences processed
2. model used
3. average answer flip rate
4. average boundary error
5. average attention jump
6. best hidden layer for ambiguity tracking
7. qualitative examples
8. major observed failure cases
```

---

## 19. Main Questions the Experiment Should Answer

The implementation should help answer:

1. Does LLaVA’s vision encoder change smoothly across natural adjacent frames?
2. Does the multimodal projector preserve or distort this smoothness?
3. Which transformer layers best encode ambiguity ordering?
4. Do text-to-image attention maps focus on relevant image regions?
5. Does attention shift smoothly or abruptly around the ambiguity center?
6. Do yes/no logits become less confident near the maximally ambiguous frame?
7. Are sudden answer flips accompanied by sudden attention or hidden-state jumps?
8. Are there cases where hidden states encode ambiguity but output logits remain overconfident?
9. Can the source of uncertainty failure be localized to:
   - vision encoder,
   - projector,
   - transformer hidden states,
   - attention grounding,
   - LM head / decoding?

---

## 20. Important Caveats to Preserve in Reporting

When writing conclusions, do not overclaim.

Use:

```text
attention-based grounding diagnostic
```

not:

```text
attention proves reasoning
```

Use:

```text
internal representation smoothness
```

not:

```text
the model understands ambiguity
```

Use:

```text
evidence of possible localization of failure
```

not:

```text
definitive causal source of failure
```

Stronger causal claims require later intervention experiments such as masking, patch ablation, or activation patching.

---

## 21. Recommended First Debug Run

Use:

```yaml
debug:
  enabled: true
  limit_sequences: 3
  limit_frames_per_sequence: 10
```

Run:

```bash
python scripts/run_all.py \
  --model_config configs/model_config.yaml \
  --data_config configs/data_config.yaml \
  --extraction_config configs/extraction_config.yaml \
  --visualization_config configs/visualization_config.yaml \
  --run_config configs/run_config.yaml
```

Expected quick outputs:

```text
3 sequence grids
30 attention overlays
per_frame_outputs.jsonl
per_sequence_summary.jsonl
aggregate_metrics.csv
```

Only after this works, scale to 20 sequences and then to the full dataset.

---

## 22. Summary of the Scientific Framing

This implementation uses natural ambiguity trajectories as controlled near-counterfactuals. Since the question and scene remain mostly fixed while visual ambiguity varies, we can inspect whether internal VLM features, attention maps, hidden states, and logits evolve smoothly with ambiguity.

The key novelty is that isolated VQA samples cannot support this analysis. Same-scene ordered trajectories allow us to distinguish:

```text
perception-level instability
vs
vision-language projection instability
vs
reasoning-state instability
vs
attention-grounding instability
vs
LM-head overconfidence
```

This can become a mechanistic extension of the continuous uncertainty evaluation benchmark.
