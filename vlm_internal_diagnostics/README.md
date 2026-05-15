# VLM Internal Diagnostics

Internal-state diagnostics pipeline for continuous-ambiguity VQA trajectories.
First-target model: `llava-hf/llava-1.5-7b-hf`.

## 1. Install

```bash
pip install torch torchvision transformers==4.51.* accelerate pillow pyyaml \
            numpy scipy scikit-learn matplotlib
```

GPU with >= 16 GB VRAM recommended for fp16 inference of LLaVA-1.5-7B.

## 2. Manifest

The pipeline reads `list_questions_revised.csv` directly (no preprocessing).
Each CSV row is one 10-frame sequence; the loader expands rows into per-frame
records (see `data/manifest_utils.py`). Image files live under:

```
{image_root}/{sequence_id}/frame_{XXXX}.png
```

where `sequence_id` derives from `video_path` by `/`→`_` and stripping `.mp4`.

Defaults in `configs/data_config.yaml`:
- manifest_path: `/home/kalashkala/acl_rebuttal_form/list_questions_revised.csv`
- image_root:    `/home/kalashkala/acl_rebuttal_form/images`

## 3. Run extraction

```bash
cd vlm_internal_diagnostics/scripts
python run_extract.py \
  --model_config ../configs/model_config.yaml \
  --data_config ../configs/data_config.yaml \
  --extraction_config ../configs/extraction_config.yaml \
  --run_config ../configs/run_config.yaml
```

Writes `outputs/{run_name}/per_frame_outputs.jsonl` and per-frame tensors under
`outputs/{run_name}/features/`.

## 4. Run metrics

```bash
python run_metrics.py --run_dir ../outputs/llava_internal_diag_v1
```

Writes `per_sequence_summary.jsonl` and `aggregate_metrics.csv`.

## 5. Run visualizations

```bash
python run_visualize.py \
  --run_dir ../outputs/llava_internal_diag_v1 \
  --visualization_config ../configs/visualization_config.yaml
```

Writes attention overlays, sequence grids, trajectory plots, aggregate plots
under `outputs/{run_name}/figures/`.

## 6. Run everything

```bash
python run_all.py \
  --model_config ../configs/model_config.yaml \
  --data_config ../configs/data_config.yaml \
  --extraction_config ../configs/extraction_config.yaml \
  --visualization_config ../configs/visualization_config.yaml \
  --run_config ../configs/run_config.yaml
```

## 7. Debug mode

Set `debug.enabled: true` in `run_config.yaml`. The loader respects
`limit_sequences` and `limit_frames_per_sequence`.

## 8. LLaVA-specific caveats

- We force `attn_implementation="eager"` (also overwriting
  `model.config._attn_implementation`) because SDPA/FlashAttention does not
  return attention weights.
- LLaVA-1.5 (HF) stores **one** `<image>` sentinel in `input_ids` and expands
  it internally into 576 visual tokens (24x24 CLIP ViT-L/14 at 336). The token
  indexer in `models/token_index_utils.py` infers the expansion from
  `vision_config` (image_size, patch_size, feature_select_strategy) and
  computes expanded LM-sequence positions accordingly.
- The "first answer token" position is not available without a separate
  `model.generate` call; by default we use the **final prompt token** position
  for attention slicing and yes/no margin computation. This matches the
  config default `query_source: final_prompt`.
- Attention is reported as a grounding diagnostic, not a causal explanation.
- Yes/No margin uses logsumexp over first-subword variants of {yes, Yes,
   yes,  Yes} and {no, No,  no,  No}. Multi-subword tokenizations are
  approximated by their leading subword.

## 9. Output layout

```
outputs/{run_name}/
  per_frame_outputs.jsonl
  per_sequence_summary.jsonl
  aggregate_metrics.csv
  features/{sequence_id}/frame_XX_{vision,projected,hidden,attention,logits}.pt
  figures/
    attention_overlays/{sequence_id}/frame_XX.png
    sequence_grids/{sequence_id}.png
    trajectory_plots/{sequence_id}.png
    aggregate/*.png
```
