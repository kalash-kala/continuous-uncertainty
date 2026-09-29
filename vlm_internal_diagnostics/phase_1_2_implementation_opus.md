PHASE 1 & 2: REPRESENTATION GEOMETRY VALIDATION & CHARACTERIZATION
==================================================================

OBJECTIVE
─────────
Validate that internal representations of four VLMs (LLaVA, Phi-4, Qwen, Pixtral) 
encode visual ambiguity through orthogonal structure (decision axis separate from 
uncertainty manifold), and characterize the trajectory geometry.

DATASET OVERVIEW
────────────────
- 92 sequences (visual ambiguity sequences from NaturalVisualQA)
- 10 frames per sequence (indexed 0-9)
- 3 representation stages per model:
  * vision: pooled image encoder output
  * projector: multimodal projection layer output
  * hidden: best hidden layer of transformer (varies by model)
- Ground truth: max_ambiguity_index (center frame, typically 3-7)
- Representations shape: (10, d) where d ∈ {512, 768, 1024, ...}

INPUT DATA
──────────
Four CSV files with metrics (already computed):
  /mnt/project/aggregate_metrics_llava_internal_boundary.csv
  /mnt/project/aggregate_metrics_phi_internal_boundary.csv
  /mnt/project/aggregate_metrics_qwen_internal_boundary.csv
  /mnt/project/aggregate_metrics_pixtral_internal_boundary.csv

Columns you need:
  - sequence_id: unique identifier
  - max_ambiguity_index: ground truth center frame (0-9)
  - category: "Geometric", "Semantic", or "Compositional"
  - (other existing columns ignored for now)

Feature tensor storage (if available):
  Location: /mnt/project/outputs/{model_name}/features/{sequence_id}/
  Files: 
    - frame_XX_vision_pooled.pt (shape: 1, d or d,)
    - frame_XX_projector_pooled.pt (shape: 1, d or d,)
    - frame_XX_hidden_best_layer.pt (shape: 1, d or d,)
  
  If tensors not available, you can load from pickles/HDF5 or use the 
  metrics data (which was computed from these representations).

═══════════════════════════════════════════════════════════════════════════════
PHASE 1: VALIDATION OF ORTHOGONAL STRUCTURE HYPOTHESIS
═══════════════════════════════════════════════════════════════════════════════

HYPOTHESIS
──────────
Representations encode decisions and uncertainty in orthogonal subspaces:
  - Decision axis (dim 1): direction from "clear no" to "clear yes"
  - Orthogonal subspace (dims 2+): where uncertainty manifests (peaks at center)

TESTS TO RUN
────────────

TEST 1: PCA ALIGNMENT WITH DECISION AXIS
═════════════════════════════════════════

For each sequence and each stage (vision, projector, hidden):

Step 1a: Load representations
  - Load 10 frames of (d-dimensional) representations for the sequence
  - Shape: (10, d)

Step 1b: Center the data
  reps_centered = reps - reps.mean(axis=0)

Step 1c: Run PCA
  - SVD: U, S, Vt = np.linalg.svd(reps_centered, full_matrices=False)
  - PC scores: pcs = U @ np.diag(S)  # shape: (10, min(10, d))
  - PC loadings: Vt (shape: min(10, d), d)
  - Explained variance ratio: var_ratio = S**2 / (S**2).sum()

Step 1d: Compute decision axis
  v_chord = reps[9] - reps[0]  # Direction from frame 0 to frame 9
  v_chord_norm = v_chord / np.linalg.norm(v_chord)

Step 1e: Check alignment
  pc1_unnormalized = pcs[:, 0]  # PC scores for first principal component
  
  Option A (simple dot product):
    cos_angle = abs(np.dot(pc1_unnormalized, v_chord_norm)) / np.linalg.norm(pc1_unnormalized)
  
  Option B (better: use loadings):
    pc1_loading = Vt[0, :]  # First principal direction (in original space)
    cos_angle = abs(np.dot(pc1_loading, v_chord_norm))
  
  Use Option B (normalized PC loading vs normalized decision axis)

Step 1f: Check monotonicity of PC1
  pc1_scores = pcs[:, 0]
  # Count how many times it goes backwards
  monotonicity_violations = sum(1 for i in range(9) if pc1_scores[i] > pc1_scores[i+1])
  
  Should be close to 0 (monotonically increasing or decreasing)

Step 1g: Explained variance
  pc1_variance_ratio = var_ratio[0]
  
  Should be > 0.5 (PC1 explains at least 50% of variance)

Output per sequence:
{
  'sequence_id': seq_id,
  'stage': 'vision'/'projector'/'hidden',
  'pc1_alignment_with_chord': cos_angle,  # Range [0, 1]; want > 0.7
  'pc1_monotonicity_violations': num_violations,  # Want = 0 or 1
  'pc1_variance_ratio': var_ratio[0],  # Want > 0.5
  'decision_axis_magnitude': np.linalg.norm(v_chord),
  'total_samples': 10
}

SUCCESS CRITERIA for TEST 1:
  - pc1_alignment > 0.7: counts as PASS
  - monotonicity_violations ≤ 1: counts as PASS
  - pc1_variance_ratio > 0.5: counts as PASS
  
  Sequence passes Test 1 if ALL THREE pass.
  Goal: >80% of sequences pass Test 1 across all stages.

───────────────────────────────────────────────────────────────────────────────

TEST 2: ORTHOGONAL VARIANCE PEAKS AT CENTER
═════════════════════════════════════════════

For each sequence and stage:

Step 2a: Load representations and center
  reps_centered = reps - reps.mean(axis=0)

Step 2b: Compute decision axis
  v_chord = reps[9] - reps[0]
  v_chord_norm = v_chord / np.linalg.norm(v_chord)

Step 2c: Project frames onto decision axis
  lambda_i = reps @ v_chord_norm  # Scalar projection for each frame
  # Shape: (10,)
  
  # Verify monotonicity
  is_monotonic = all(lambda_i[i] <= lambda_i[i+1] for i in range(9)) or \
                 all(lambda_i[i] >= lambda_i[i+1] for i in range(9))

Step 2d: Extract orthogonal component
  # For each frame, subtract its projection onto decision axis
  reps_proj_on_chord = np.outer(lambda_i, v_chord_norm)
  reps_orthogonal = reps_centered - reps_proj_on_chord
  # Shape: (10, d)

Step 2e: Compute orthogonal variance per frame
  ortho_variance = np.linalg.norm(reps_orthogonal, axis=1)**2
  # Shape: (10,)
  # This is the squared norm of the orthogonal component for each frame
  # High at center = good, low at edges = good

Step 2f: Check if orthogonal variance peaks at center
  center_idx = max_ambiguity_index
  distance_from_center = np.abs(np.arange(10) - center_idx)
  
  # Compute Spearman rank correlation
  # If orthogonal variance peaks at center, it should correlate NEGATIVELY
  # with distance from center (negative correlation = inverse relationship)
  from scipy.stats import spearmanr
  
  spearman_rho, p_value = spearmanr(ortho_variance, distance_from_center)

Step 2g: Optional - visualize the pattern
  print(f"Frame:              {list(range(10))}")
  print(f"Orth variance:      {ortho_variance.round(3)}")
  print(f"Distance from cntr: {distance_from_center}")
  print(f"Spearman ρ:         {spearman_rho:.3f}")
  # Should see U-shape: low at 0 and 9, high at center

Output per sequence:
{
  'sequence_id': seq_id,
  'stage': 'vision'/'projector'/'hidden',
  'decision_axis_monotonic': is_monotonic,
  'decision_axis_variance': np.var(lambda_i),
  'orthogonal_variance_mean': ortho_variance.mean(),
  'orthogonal_variance_peak_frame': np.argmax(ortho_variance),
  'orthogonal_variance_center_spearman': spearman_rho,
  'orthogonal_variance_center_p_value': p_value,
  'orthogonal_variance_shows_correct_peak': (
      np.argmax(ortho_variance) == max_ambiguity_index or
      np.argmax(ortho_variance) in [max_ambiguity_index - 1, max_ambiguity_index + 1]
  ),
}

SUCCESS CRITERIA for TEST 2:
  - decision_axis_monotonic = True: PASS
  - orthogonal_variance_center_spearman < -0.5: PASS (negative = peaks at center)
  - orthogonal_variance_shows_correct_peak = True: PASS
  
  Sequence passes Test 2 if ≥2/3 pass.
  Goal: >80% of sequences pass Test 2 across all stages.

───────────────────────────────────────────────────────────────────────────────

TEST 3: FRAME ORDER RECOVERY FROM REPRESENTATION DISTANCES
═══════════════════════════════════════════════════════════

For each sequence and stage, test if you can recover the correct frame order
using only pairwise distances in representation space.

Step 3a: Compute pairwise distances
  from scipy.spatial.distance import pdist, squareform
  
  distances = squareform(pdist(reps, metric='cosine'))
  # Shape: (10, 10)

Step 3b: Nearest neighbor chain
  Start at frame 0, always go to nearest unvisited frame.
  
  nn_order = [0]
  visited = {0}
  current = 0
  
  for _ in range(9):
      # Find nearest unvisited neighbor
      neighbors_dist = distances[current, :]
      neighbors_dist[list(visited)] = np.inf
      next_frame = np.argmin(neighbors_dist)
      nn_order.append(next_frame)
      visited.add(next_frame)
      current = next_frame
  
  nn_order_array = np.array(nn_order)

Step 3c: Score the recovered order
  true_order = np.arange(10)
  
  # Metric 1: Perfect match?
  perfect_match = np.array_equal(nn_order_array, true_order)
  
  # Metric 2: Longest increasing subsequence
  # (how much of the order is correct when read left-to-right?)
  def longest_increasing_subseq(arr):
      n = len(arr)
      dp = [1] * n
      for i in range(1, n):
          for j in range(i):
              if arr[j] < arr[i]:
                  dp[i] = max(dp[i], dp[j] + 1)
      return max(dp)
  
  lis_length = longest_increasing_subseq(nn_order_array)
  lis_ratio = lis_length / 10.0

Step 3d: Optional - Traveling Salesman Problem (TSP)
  (More complex; skip if time-constrained)
  
  Use greedy TSP: repeatedly add nearest unvisited frame
  (same as nearest neighbor chain above, but try multiple starting points)
  
  Or use scipy: from scipy.optimize import linear_sum_assignment
  
  See if TSP solution = true order for >50% sequences

Output per sequence:
{
  'sequence_id': seq_id,
  'stage': 'vision'/'projector'/'hidden',
  'nn_order_perfect_match': perfect_match,
  'nn_order_lis_ratio': lis_ratio,  # Ratio of correct subsequence length
  'nn_order': nn_order_array.tolist(),
  'true_order': true_order.tolist(),
}

SUCCESS CRITERIA for TEST 3:
  - nn_order_perfect_match = True: EXCELLENT
  - nn_order_lis_ratio > 0.7: GOOD (70% of order reconstructed)
  - nn_order_lis_ratio > 0.5: ACCEPTABLE
  
  Sequence passes if lis_ratio > 0.5.
  Goal: >50% of sequences pass Test 3 (this is harder than Tests 1-2).

═══════════════════════════════════════════════════════════════════════════════
PHASE 2: CHARACTERIZATION OF TRAJECTORY GEOMETRY
═════════════════════════════════════════════════════════════════════════════════

TESTS TO RUN
────────────

TEST 4: CURVATURE AND SMOOTHNESS OF TRAJECTORY
════════════════════════════════════════════════

For each sequence and stage:

Step 4a: Compute frame-to-frame velocity vectors
  velocities = np.diff(reps, axis=0)  # Shape: (9, d)
  # velocity[i] = reps[i+1] - reps[i]

Step 4b: Compute discrete curvature (angle between consecutive velocities)
  curvatures = []
  
  for i in range(len(velocities) - 1):
      v1 = velocities[i]
      v2 = velocities[i+1]
      
      # Cosine of angle
      cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
      cos_angle = np.clip(cos_angle, -1, 1)
      
      # Angle in radians
      angle = np.arccos(cos_angle)
      curvatures.append(angle)
  
  curvatures = np.array(curvatures)  # Shape: (8,) — edges between consecutive frames
  # Index i represents the angle at frame i+0.5 (the edge)

Step 4c: Compute acceleration vectors
  accelerations = np.diff(velocities, axis=0)  # Shape: (8, d)

Step 4d: Compute smoothness (magnitude of acceleration)
  smoothness = np.linalg.norm(accelerations, axis=1)  # Shape: (8,)

Step 4e: Check if curvature peaks near center
  center_idx = max_ambiguity_index
  
  # Map edges (8 values) to frame indices
  # Edge between frames i and i+1 is at position i+0.5
  edge_positions = np.arange(8) + 0.5
  distance_from_center = np.abs(edge_positions - center_idx)
  
  from scipy.stats import spearmanr
  spearman_curv, p_curv = spearmanr(curvatures, distance_from_center)
  
  # Should be negative: curvature peaks (high value) when distance is low
  # So correlation should be negative (high curvature ↔ low distance)

Step 4f: Find where maximum curvature occurs
  max_curv_idx = np.argmax(curvatures)  # Index of edge with max curvature
  max_curv_frame_position = max_curv_idx + 0.5  # Frame position
  distance_to_center = abs(max_curv_frame_position - center_idx)

Output per sequence:
{
  'sequence_id': seq_id,
  'stage': 'vision'/'projector'/'hidden',
  'mean_curvature_radians': curvatures.mean(),
  'max_curvature_radians': curvatures.max(),
  'std_curvature_radians': curvatures.std(),
  'curvature_peak_distance_to_center': distance_to_center,
  'curvature_center_spearman': spearman_curv,
  'curvature_center_p_value': p_curv,
  'curvature_peaks_near_center': distance_to_center <= 1.5,  # Within 1.5 frames
  'mean_smoothness': smoothness.mean(),
  'max_smoothness': smoothness.max(),
}

SUCCESS CRITERIA for TEST 4:
  - mean_curvature_radians > 0.15: PASS (trajectory is curved, not straight)
  - curvature_peaks_near_center = True: PASS (peak within 1.5 frames of center)
  - curvature_center_spearman < -0.3: PASS (negative correlation with distance)
  
  Sequence passes if ≥2/3 pass.
  Goal: >70% of sequences pass Test 4.

───────────────────────────────────────────────────────────────────────────────

TEST 5: EFFECTIVE DIMENSIONALITY (MANIFOLD DIMENSIONALITY)
═══════════════════════════════════════════════════════════

For each sequence and stage:

Step 5a: Compute effective dimensionality (how many PCs needed for 95% variance?)
  U, S, Vt = np.linalg.svd(reps - reps.mean(axis=0), full_matrices=False)
  
  cumsum_var = np.cumsum(S**2) / (S**2).sum()
  eff_dim_95pct = np.argmax(cumsum_var >= 0.95)  # Index of 1st PC with cumsum >= 95%
  
  original_dim = reps.shape[1]  # d

Step 5b: Also compute 90% and 99% for reference
  eff_dim_90pct = np.argmax(cumsum_var >= 0.90)
  eff_dim_99pct = np.argmax(cumsum_var >= 0.99)

Step 5c: Compute compression ratio
  compression_ratio_95 = original_dim / eff_dim_95pct

Step 5d: Visualize variance explained
  # For reporting: top 10 PCs should explain >90% of variance typically
  top_10_var = cumsum_var[min(9, len(cumsum_var)-1)]

Output per sequence:
{
  'sequence_id': seq_id,
  'stage': 'vision'/'projector'/'hidden',
  'original_dimension': original_dim,
  'effective_dim_90pct': eff_dim_90pct,
  'effective_dim_95pct': eff_dim_95pct,
  'effective_dim_99pct': eff_dim_99pct,
  'compression_ratio_95pct': compression_ratio_95,
  'variance_by_top_10_pcs': top_10_var,
  'is_low_dimensional': eff_dim_95pct < original_dim / 10,  # Compression > 10x
}

SUCCESS CRITERIA for TEST 5:
  - eff_dim_95pct << original_dim (by factor of 10x or more): PASS
  - is_low_dimensional = True: PASS
  
  Goal: >80% of sequences show 10x+ compression.

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT & AGGREGATION
═════════════════════════════════════════════════════════════════════════════════

For each test, generate:

1. PER-SEQUENCE RESULTS
   - CSV file: results_{stage}_{test_number}.csv
   - One row per sequence with all metrics
   - Columns: sequence_id, category, pass_fail, [all individual metrics]

2. AGGREGATE STATISTICS
   - For each stage (vision, projector, hidden):
     * Overall pass rate per test (% sequences passing)
     * Mean values of key metrics
     * Std dev
     * Min, max, median
   
   Format:
   {
     'stage': 'vision',
     'test_1': {
       'pass_rate': 0.87,  # 87% of sequences passed
       'mean_pc1_alignment': 0.76,
       'std_pc1_alignment': 0.12,
       'mean_monotonicity_violations': 0.3,
       ...
     },
     'test_2': {...},
     ...
   }

3. BREAKDOWN BY CATEGORY
   - For each category (Geometric, Semantic, Compositional):
     * Pass rates
     * Key metrics
   
   Answer: Does structure differ by category?

4. BREAKDOWN BY MODEL
   - LLaVA vs Phi-4 vs Qwen vs Pixtral
   - Do all models show similar structure?
   
   Answer: Is this fundamental across architectures or model-specific?

5. SUMMARY TABLE
   
   ┌────────────────────┬─────────────┬──────────┬──────────────────┐
   │ Test               │ Vision      │ Projector│ Hidden (best)    │
   │                    │ Pass Rate   │ Pass Rate│ Pass Rate        │
   ├────────────────────┼─────────────┼──────────┼──────────────────┤
   │ Test 1: PC1        │ 85% (✓)     │ 78% (✓)  │ 92% (✓)          │
   │ Test 2: Ortho      │ 82% (✓)     │ 75% (✓)  │ 88% (✓)          │
   │ Test 3: Order      │ 65% (?)     │ 62% (?)  │ 70% (✓)          │
   │ Test 4: Curvature  │ 78% (✓)     │ 71% (?)  │ 84% (✓)          │
   │ Test 5: Manifold   │ 89% (✓)     │ 92% (✓)  │ 88% (✓)          │
   └────────────────────┴─────────────┴──────────┴──────────────────┘
   
   Legend: ✓ = Pass (>70%), ? = Marginal (50-70%), ✗ = Fail (<50%)

═══════════════════════════════════════════════════════════════════════════════
VALIDATION DECISION TREE
═════════════════════════════════════════════════════════════════════════════════

After running all tests:

IF (Test 1 AND Test 2 AND Test 4) ALL PASS (>70% each):
  → ORTHOGONAL STRUCTURE HYPOTHESIS IS CONFIRMED
  → Decision: Proceed to Phase 3 (new methods)
  → Paper section: "Representations encode ambiguity orthogonally"

IF (Test 1 AND Test 2) PASS but Test 4 is WEAK (<60%):
  → Structure exists but trajectory is less curved than expected
  → Decision: Skip manifold methods (G3), focus on simpler methods (G1, G2, G4)
  → Paper section: "Geometric structure is evident but subtle"

IF (Test 3) FAILS (frame order recovery <50%):
  → Manifold structure not strong enough for Isomap
  → Decision: Skip method G3 (geodesic distance)
  → Paper section: "Representations encode structure but not as a clean manifold"

IF (Test 5) shows EFFECTIVE DIM << 5:
  → Very low-dimensional manifold discovered
  → Decision: HIGHLY encourage method G3 (Isomap)
  → Paper section: "Surprising finding: 5D manifold discovered in Xd space"

═══════════════════════════════════════════════════════════════════════════════
ERROR HANDLING & EDGE CASES
═════════════════════════════════════════════════════════════════════════════════

1. NaN or Inf values in representations
   → Filter out or report as problematic sequences
   → Log which sequences have issues

2. All frames have identical representations
   → Would cause division by zero in cosine distance
   → Mark as "degenerate" and skip

3. Center frame at boundary (idx=0 or idx=9)
   → Rare but possible if sequence is mislabeled
   → Handle gracefully; report count

4. Dimension d is very small (<10)
   → PCA can use at most min(10, d) components
   → Adjust effective dimensionality tests

5. Sequence has duplicate or near-identical frames
   → Normal; don't filter out
   → Just report

═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL NOTES
═════════════════════════════════════════════════════════════════════════════════

Expected runtime:
  - Load data: ~1 minute
  - Test 1 (PCA): ~5 minutes
  - Test 2 (variance): ~2 minutes
  - Test 3 (NN order): ~3 minutes
  - Test 4 (curvature): ~3 minutes
  - Test 5 (manifold): ~5 minutes
  - Total: ~20 minutes on CPU
  
  If loading from feature tensors (vs CSV): add 5-10 minutes

Memory:
  - 92 sequences × 10 frames × 768 dims = ~7 MB per stage
  - Pairwise distances (92 × 10×10 matrices) = ~7 MB
  - PCA results: negligible
  - Total: ~50 MB (easily fits in memory)

═══════════════════════════════════════════════════════════════════════════════
SUCCESS CRITERIA SUMMARY
═════════════════════════════════════════════════════════════════════════════════

OVERALL VALIDATION SUCCESS:
  ✓ ≥3 of 5 tests pass with >70% pass rate across stages
  ✓ Structure is consistent across LLaVA, Phi-4, Qwen, Pixtral
  ✓ No major differences between categories (Geometric, Semantic, Compositional)

If this is satisfied:
  → Hypothesis STRONGLY VALIDATED
  → Proceed to Phase 3 with confidence
  → Write paper with "orthogonal encoding" as centerpiece

═══════════════════════════════════════════════════════════════════════════════
DELIVERABLES
═════════════════════════════════════════════════════════════════════════════════

Create and save:

1. results_vision_test1.csv, results_vision_test2.csv, etc.
2. results_projector_test*.csv
3. results_hidden_test*.csv

4. summary_statistics.json (aggregate results)

5. validation_report.txt
   - Key findings
   - Pass/fail summary
   - Decision for Phase 3
   - Visualization URLs (if generating plots)

6. Visualizations (highly recommended):
   - Per test: histogram of pass rates across sequences
   - Per model: heatmap of pass rates (tests × stages)
   - Example sequence: show PC1-PC2 plot, orthogonal variance curve
   - Curvature distribution across all sequences

═══════════════════════════════════════════════════════════════════════════════
STARTER CODE STRUCTURE (pseudocode)
═════════════════════════════════════════════════════════════════════════════════

```python
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

# Models and stages
MODELS = ['llava', 'phi', 'qwen', 'pixtral']
STAGES = ['vision', 'projector', 'hidden']

results_all = []

for model in MODELS:
    # Load CSV
    df = pd.read_csv(f'/mnt/project/aggregate_metrics_{model}_internal_boundary.csv')
    
    for idx, row in df.iterrows():
        seq_id = row['sequence_id']
        center_idx = row['max_ambiguity_index']
        category = row['category']
        
        for stage in STAGES:
            # Load representations (function to be implemented)
            reps = load_representations(model, seq_id, stage)
            
            if reps is None:
                continue
            
            # TEST 1: PCA
            test1_result = test_pca_alignment(reps, center_idx)
            
            # TEST 2: Orthogonal variance
            test2_result = test_orthogonal_variance(reps, center_idx)
            
            # TEST 3: Frame order recovery
            test3_result = test_frame_order_recovery(reps)
            
            # TEST 4: Curvature
            test4_result = test_curvature(reps, center_idx)
            
            # TEST 5: Effective dimensionality
            test5_result = test_effective_dimensionality(reps)
            
            # Combine results
            combined = {
                'model': model,
                'stage': stage,
                'sequence_id': seq_id,
                'category': category,
                'center_idx': center_idx,
                **test1_result,
                **test2_result,
                **test3_result,
                **test4_result,
                **test5_result,
            }
            
            results_all.append(combined)

# Convert to DataFrame and save
results_df = pd.DataFrame(results_all)
results_df.to_csv('/mnt/user-data/outputs/phase1_phase2_results.csv', index=False)

# Generate aggregate statistics
print_aggregate_statistics(results_df)
```

═══════════════════════════════════════════════════════════════════════════════
NEXT STEPS FOR OPUS
═════════════════════════════════════════════════════════════════════════════════

1. Implement the 5 test functions:
   - test_pca_alignment(reps, center_idx) → dict of metrics
   - test_orthogonal_variance(reps, center_idx) → dict of metrics
   - test_frame_order_recovery(reps) → dict of metrics
   - test_curvature(reps, center_idx) → dict of metrics
   - test_effective_dimensionality(reps) → dict of metrics

2. Implement load_representations(model, seq_id, stage)
   - Either load from .pt files or use stored metrics
   - Return shape (10, d) array

3. Run full pipeline on all 4 models × 3 stages × 92 sequences

4. Generate summary table and statistics

5. Create optional visualizations

6. Generate validation report with decision for Phase 3
