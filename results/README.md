# Committed results

The exact outputs behind every number in the top-level README, so you can check
the claims — and verify your own re-run — without first reproducing 7 GB of
extraction.

These are **outputs, not inputs.** Nothing in the pipeline reads this directory;
the phase scripts read from and write to `<outputs_root>` (see
`../vlm_internal_diagnostics/configs/paths.yaml`). This is a snapshot for
review and replication checking.

## What is here

| Path | Contents |
|---|---|
| `aggregate_metrics/{llava,phi4,pixtral,qwen2_5_vl}.csv` | per-sequence metrics from extraction, 92 rows each — the input to every analysis phase |
| `phase_1_2/phase1_phase2_results.csv` | per-sequence × per-stage results for all five hypothesis tests |
| `phase_1_2/phase1_phase2_summary_stats.txt` | pass rates by stage, model and category |
| `phase_1_2/validation_report.txt` | the verdict ("STRUCTURE IS WEAK OR ABSENT") — see the warning below |
| `phase_1_2/figures/*.png` | pass-rate bars/heatmap, and histograms of PC1 alignment, LIS ratio, orthogonal Spearman, curvature |
| `phase_3/g3_g4_results.csv` | raw G3 / G4 / hybrid boundary indices and errors per stage |
| `phase_3/g3_g4_with_baselines.csv` | the same, joined with the four baselines + logit |
| `phase_3/g3_g4_summary.txt` | comparison tables + Wilcoxon signed-rank vs chord |
| `phase_4/uncertainty_auroc.csv` | 144 rows: grouping × stage × method AUROC with 90% bootstrap CIs |
| `phase_4/uncertainty_auroc_summary.txt` | the formatted AUROC tables |

**Deliberately excluded:**

- `uncertainty_auroc_frames.csv` (2.9 MB) — the per-frame scores behind Phase 4.
  Regenerate it by running Phase 4, or rsync it from the lab server.
- `features/` (~1.7 GB per model) and `per_frame_outputs.jsonl` — the saved
  tensors the phases actually read. These must come from the lab server; see
  "Getting the data" in the top-level README.

> ⚠️ **Read `validation_report.txt` carefully.** Its "KEY FINDINGS FOR PAPER"
> section states Findings 1, 2 and 4 in positive language for tests that the
> same file marks ✗ FAIL. Those lines are the mean effect sizes of *failing*
> tests, not findings. The top-level README §3.1 has the correct reading.

## Replicating

Phases 3 and 4 are deterministic and have been verified to reproduce these
files **byte-for-byte** from the extraction outputs as shipped. So the
replication check is an exact diff, not an eyeball comparison.

Once `outputs_root` is configured and populated (top-level README §6):

```bash
cd vlm_internal_diagnostics
conda run -n continuous-uncertainty python scripts/phase_3_g3_g4.py
conda run -n continuous-uncertainty python scripts/phase_4_uncertainty_auroc.py

# compare your re-run against what is committed here
diff -r results/phase_3 "$CU_OUTPUTS_ROOT/phase_3"
diff  results/phase_4/uncertainty_auroc.csv "$CU_OUTPUTS_ROOT/phase_4/uncertainty_auroc.csv"
```

Phase 1&2 writes figures too, so diff only its text outputs:

```bash
conda run -n continuous-uncertainty python scripts/phase_1_2_analysis.py
diff results/phase_1_2/phase1_phase2_results.csv       "$CU_OUTPUTS_ROOT/phase_1_2/phase1_phase2_results.csv"
diff results/phase_1_2/phase1_phase2_summary_stats.txt "$CU_OUTPUTS_ROOT/phase_1_2/phase1_phase2_summary_stats.txt"
```

Any difference means something real changed — a different extraction, a
different `outputs_root`, or a library version that moved under you (Isomap in
particular). Investigate rather than assuming noise: none of these phases
sample, and every seed is fixed (`BOOTSTRAP_SEED = 0` in Phase 4).

### Checksums

SHA-256, first 16 hex chars, of the committed text outputs:

```
d53b720b15199611  aggregate_metrics/llava.csv
e0120f41e3109ae1  aggregate_metrics/phi4.csv
82fee18cdbc1b162  aggregate_metrics/pixtral.csv
319449a5aca3132c  aggregate_metrics/qwen2_5_vl.csv
eb5a729790aaf6d8  phase_1_2/phase1_phase2_results.csv
13d45ecc46b2d104  phase_1_2/phase1_phase2_summary_stats.txt
ba3e76c03b14b7a8  phase_1_2/validation_report.txt
0adbee6700b565f8  phase_3/g3_g4_results.csv
7910b811644013d8  phase_3/g3_g4_summary.txt
58ca9725183f399e  phase_3/g3_g4_with_baselines.csv
d1a6057fd586a837  phase_4/uncertainty_auroc.csv
d3997338784e0441  phase_4/uncertainty_auroc_summary.txt
```

Verify with:

```bash
cd results && find . -name '*.csv' -o -name '*.txt' | sort | xargs sha256sum
```

## Provenance

Every file here comes from the run directories
`{llava,phi4,pixtral,qwen2_5_vl}_internal_diag_v1_question_mean_new` on the lab
server. Older run directories exist alongside them (`llava_internal_diag_v1`,
`llava_..._question_mean`, `phi4_multimodal_...`) and are **superseded** — no
number in the top-level README comes from those.
