"""
INTERNAL REPRESENTATION GEOMETRY ANALYSIS FRAMEWORK
====================================================

Goal: Validate orthogonal structure hypothesis and discover new geometric boundary methods.

Key insight: We have 92 sequences × 10 frames, each with d-dimensional pooled representations.
We can analyze the geometry at multiple scales:
  1. Per-sequence geometry (how does each sequence's 10-frame trajectory behave?)
  2. Cross-sequence geometry (is the structure consistent across all sequences?)
  3. Variance decomposition (where does ambiguity variance manifest?)
  4. Trajectory properties (curvature, smoothness, dimensionality)

========================================
PHASE 1: VALIDATE ORTHOGONAL STRUCTURE
========================================

For each sequence, we can decompose the representation space as:
  f_i = f_anchor_space + f_orthogonal_space
  
Where:
  f_anchor_space = component pointing along v_chord = (f_9 - f_0) direction
  f_orthogonal_space = component perpendicular to v_chord

Hypothesis: 
  - f_anchor_space increases monotonically from frame 0 to 9 (decision unfolds)
  - f_orthogonal_space peaks at the center frame (maximum uncertainty)

Test 1: PCA decomposition per sequence
───────────────────────────────────────
For each of 92 sequences:
  1. Stack 10 frame representations into 10×d matrix
  2. Do PCA to get principal components
  3. PC1 should align with v_chord (decision axis)
  4. PC2, PC3, ... should show maximum variance at center frame (U-shape)
  5. Compute alignment: cos(angle between PC1 and v_chord)
  6. For orthogonal PCs, compute variance at each frame position

Expected result:
  - PC1 explains ~60-80% of variance (strong decision axis)
  - Alignment with v_chord: cos(angle) > 0.7 for most sequences
  - PC2+ variance: clear U-shape with peak at frame c

Code sketch:
```python
import numpy as np
from scipy.stats import spearmanr

results = []
for seq_id in range(92):
    reps = load_representations(seq_id, stage='vision')  # Shape: 10 × d
    center_frame = max_ambiguity_index[seq_id]
    
    # PCA
    U, S, Vt = np.linalg.svd(reps - reps.mean(axis=0))
    pcs = U @ np.diag(S)  # PC scores
    
    # PC1 should align with decision axis
    decision_axis = reps[9] - reps[0]
    pc1_alignment = abs(np.dot(pcs[:, 0], decision_axis)) / (np.linalg.norm(pcs[:, 0]) * np.linalg.norm(decision_axis))
    
    # PC2 should show U-shape (peak at center)
    pc2_distance_from_center = np.abs(np.arange(10) - center_frame)
    pc2_variance_at_frame = np.abs(pcs[:, 1])
    spearman_u_shape, p = spearmanr(pc2_variance_at_frame, pc2_distance_from_center)
    
    results.append({
        'seq_id': seq_id,
        'pc1_alignment': pc1_alignment,
        'pc1_variance_ratio': S[0]**2 / (S**2).sum(),
        'pc2_u_shape_spearman': spearman_u_shape,  # Should be NEGATIVE (inverted U)
        'pc3_u_shape_spearman': spearmanr(np.abs(pcs[:, 2]), pc2_distance_from_center)[0],
    })
```

Interpretation:
  - High pc1_alignment + high pc1_variance_ratio → Strong decision axis exists
  - Negative pc2_u_shape_spearman → PC2 peaks at center (uncertainty orthogonal to decision)
  - If >80% of sequences show this pattern → Hypothesis VALIDATED

---

Test 2: Decision axis vs orthogonal subspace variance
─────────────────────────────────────────────────────
For each sequence:
  1. Project all frames onto v_chord (decision axis) → 1D coordinates λ_i
  2. Project all frames onto orthogonal subspace → (d-1) dimensional
  3. Measure variance in each subspace
  4. Check if λ_i is monotonically increasing (clean decision progression)
  5. Check if orthogonal variance peaks at center

Code sketch:
```python
for seq_id in range(92):
    reps = load_representations(seq_id, stage='vision')  # 10 × d
    center_frame = max_ambiguity_index[seq_id]
    
    # Decision axis
    v_chord = reps[9] - reps[0]
    v_chord = v_chord / np.linalg.norm(v_chord)
    
    # Projections
    lambda_i = reps @ v_chord  # 1D coordinates on decision axis
    
    # Orthogonal component
    reps_proj_on_chord = np.outer(lambda_i, v_chord)
    reps_orthogonal = reps - reps_proj_on_chord  # (d-1) dimensional
    
    # Check monotonicity of λ_i
    mono_violations = sum(lambda_i[i] > lambda_i[i+1] for i in range(9))
    
    # Check if orthogonal variance peaks at center
    ortho_variance_per_frame = np.linalg.norm(reps_orthogonal, axis=1)**2
    distance_from_center = np.abs(np.arange(10) - center_frame)
    spearman_peak_at_center, _ = spearmanr(ortho_variance_per_frame, distance_from_center)
    
    results.append({
        'seq_id': seq_id,
        'monotonicity_violations': mono_violations,
        'decision_axis_variance': np.var(lambda_i),
        'orthogonal_variance': np.var(reps_orthogonal),
        'orthogonal_variance_centers_correctly': spearman_peak_at_center < -0.5,  # Negative = peaks at center
    })
```

Interpretation:
  - mono_violations ≈ 0 → Clean decision progression
  - orthogonal_variance >> decision_axis_variance? Probably not; decision axis should dominate.
  - But orthogonal variance should still show structure (U-shape)

---

Test 3: Representability of frame ordering from representation space alone
──────────────────────────────────────────────────────────────────────────
Can we recover the correct frame order just from pairwise distances in representation space?

Code sketch:
```python
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

for seq_id in range(92):
    reps = load_representations(seq_id, stage='vision')  # 10 × d
    
    # Compute all pairwise distances
    distances = squareform(pdist(reps, metric='cosine'))
    
    # Nearest neighbor graph: connect each frame to its nearest neighbor
    nn_order = []
    current = 0
    visited = {0}
    for _ in range(9):
        neighbors = distances[current, :]
        neighbors[list(visited)] = np.inf
        next_frame = np.argmin(neighbors)
        nn_order.append(next_frame)
        visited.add(next_frame)
        current = next_frame
    
    # Did we recover the sequence order?
    true_order = np.arange(10)
    recovered_order_length = longest_increasing_subsequence(nn_order)
    
    # Alternative: Use TSP-like approach (traveling salesman on frames)
    # This finds the shortest path visiting all 10 frames
    # If shortest path = correct order, representation encodes sequence geometry
```

Interpretation:
  - If nearest-neighbor graph recovers >80% of sequences in order → Strong structure
  - If TSP solution = correct frame order for >50% of sequences → Manifold is well-organized

This tests whether the "trajectory" is actually a smooth path you can follow in the representation space.

========================================
PHASE 2: CHARACTERIZE TRAJECTORY GEOMETRY
========================================

Given that the representation DOES encode structure, let's measure its properties.

Test 4: Curvature and smoothness of the trajectory
────────────────────────────────────────────────────
For a sequence of frames f_0, f_1, ..., f_9 in representation space:

Frame-to-frame velocity: v_i = f_{i+1} - f_i
Frame-to-frame acceleration: a_i = v_{i+1} - v_i (change in direction/speed)

Metrics:
  1. Smoothness: mean(||a_i||) - how much does direction/speed change?
     - Low smoothness = sharp turns (unexpected)
     - High smoothness = gradual changes (expected)
  
  2. Discrete curvature: κ_i = angle between v_i and v_{i+1}
     - κ_i ≈ 0 = straight trajectory
     - κ_i > 0 = curved trajectory
  
  3. Average curvature: mean(κ_i)
     - Should be high (the path is NOT straight)
     - Should be asymmetric: higher curvature around center frame
  
  4. Curvature concentration: Is most curvature concentrated at center?

Code sketch:
```python
def compute_trajectory_properties(reps):
    """reps: 10 × d array of frame representations"""
    
    velocities = np.diff(reps, axis=0)  # 9 × d
    accelerations = np.diff(velocities, axis=0)  # 8 × d
    
    # Smoothness
    smoothness = np.linalg.norm(accelerations, axis=1)  # 8 values
    mean_smoothness = np.mean(smoothness)
    
    # Curvature (angle between consecutive velocity vectors)
    cosines = []
    for i in range(len(velocities)-1):
        v1 = velocities[i]
        v2 = velocities[i+1]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        curvatures.append(angle)  # Curvature in radians
    
    return {
        'mean_smoothness': mean_smoothness,
        'mean_curvature': np.mean(curvatures),
        'max_curvature': np.max(curvatures),
        'curvature_std': np.std(curvatures),
    }

results = []
for seq_id in range(92):
    reps = load_representations(seq_id, stage='vision')
    center_frame = max_ambiguity_index[seq_id]
    props = compute_trajectory_properties(reps)
    
    # Is curvature concentrated at center?
    curvatures_array = np.array(props['curvatures'])  # 8 values (frames 0-1 to 8-9)
    distance_from_center = np.abs(np.arange(8) + 0.5 - center_frame)  # Centers of edges
    spearman_curve_at_center, _ = spearmanr(curvatures_array, distance_from_center)
    
    results.append({
        'seq_id': seq_id,
        'mean_curvature': props['mean_curvature'],
        'max_curvature': props['max_curvature'],
        'curvature_peaks_at_center': spearman_curve_at_center < -0.3,
    })
```

Expected result:
  - mean_curvature > 0.2 radians (~12°) — trajectory is NOT straight
  - max_curvature peaks near center frame — curvature concentrates where ambiguity is
  - This validates that the path deviates from chord, especially at center

---

Test 5: Dimensionality of the effective manifold
──────────────────────────────────────────────────
For each sequence, what's the effective dimensionality? Do we really need all d dimensions?

Code sketch:
```python
def effective_dimensionality(reps, threshold=0.95):
    """
    Compute effective dimensionality: how many PCA components to explain
    'threshold' fraction of variance?
    """
    U, S, _ = np.linalg.svd(reps - reps.mean(axis=0))
    cumsum_var = np.cumsum(S**2) / (S**2).sum()
    eff_dim = np.argmax(cumsum_var >= threshold)
    return eff_dim

results = []
for seq_id in range(92):
    reps = load_representations(seq_id, stage='vision')  # shape 10 × d
    eff_dim = effective_dimensionality(reps, threshold=0.95)
    
    results.append({
        'seq_id': seq_id,
        'original_dim': reps.shape[1],
        'effective_dim_95pct': eff_dim,
        'effective_dim_fraction': eff_dim / reps.shape[1],
    })
```

Expected result:
  - effective_dim << original_dim — most of the d dimensions are noise
  - typical effective_dim = 3-10 even if d = 256 or 768
  - This suggests the sequence traces a low-dimensional manifold

Insight: If most sequences lie on a 3-5D manifold, we can use manifold learning!

========================================
PHASE 3: NEW GEOMETRIC METHODS
========================================

Based on what we learn from Phase 1-2, design better methods:

METHOD G1: Curvature-weighted chord distance
─────────────────────────────────────────────
Instead of simple perpendicular distance from chord, weight by curvature:

boundary = argmax_i [d_perp(i) * curvature_weight(i)]

where curvature_weight(i) is high if the frame is at a high-curvature region.

Rationale: High curvature = large deviation from straight path = likely ambiguity

---

METHOD G2: Variance along principal axes
─────────────────────────────────────────
Decompose variance into decision axis vs orthogonal:

boundary = argmax_i [var_orthogonal(i) + w * var_orthogonal_rate(i)]

where:
  - var_orthogonal(i) = magnitude of frame i's component orthogonal to v_chord
  - var_orthogonal_rate(i) = rate of change in orthogonal direction

Rationale: Center frame should have large orthogonal component AND be at a turning point

Code sketch:
```python
def method_g2_variance_decomposition(reps, center_gt=None):
    """
    Boundary detection via orthogonal variance decomposition.
    """
    v_chord = reps[9] - reps[0]
    v_chord = v_chord / np.linalg.norm(v_chord)
    
    # Project onto decision axis
    lambda_i = reps @ v_chord
    
    # Orthogonal component
    reps_ortho = reps - np.outer(lambda_i, v_chord)
    ortho_variance = np.linalg.norm(reps_ortho, axis=1)**2
    
    # Rate of change in orthogonal direction
    ortho_velocity = np.diff(reps_ortho, axis=0)
    ortho_velocity_norm = np.linalg.norm(ortho_velocity, axis=1)
    
    # Combine: orthogonal variance at each frame + orthogonal velocity before/after
    score = np.zeros(10)
    score[0] = ortho_variance[0]
    for i in range(1, 9):
        score[i] = ortho_variance[i] + 0.3 * (ortho_velocity_norm[i-1] + ortho_velocity_norm[i])
    score[9] = ortho_variance[9]
    
    boundary_index = np.argmax(score)
    return boundary_index, score
```

---

METHOD G3: Geodesic distance instead of Euclidean
───────────────────────────────────────────────────
The shortest path between frames in representation space might not be straight line!

Use Isomap or diffusion maps to compute geodesic distances along the manifold.

Code sketch:
```python
from sklearn.manifold import Isomap
from scipy.spatial.distance import pdist, squareform

def method_g3_geodesic_distance(reps):
    """
    Compute boundary using geodesic distance along manifold.
    """
    # Fit Isomap (learns local manifold structure)
    iso = Isomap(n_neighbors=min(5, len(reps)-1), n_components=3)
    reps_iso = iso.fit_transform(reps)  # Low-dim embedding preserving geodesics
    
    # Now compute distances in the learned manifold space
    distances_iso = squareform(pdist(reps_iso, metric='euclidean'))
    
    # Chord and perpendicular distance in the manifold space
    chord_dist_iso = distances_iso[0, 9]
    perp_distances_iso = []
    for i in range(10):
        # Simple approximation: distance from frame i to line connecting 0-9
        # (Proper geodesic perpendicular is complex, so use approximation)
        perp = distances_iso[i, 0]**2 + chord_dist_iso**2 - distances_iso[i, 9]**2
        perp = max(0, perp / (2 * chord_dist_iso))  # Project formula
        perp_distances_iso.append(perp)
    
    boundary_index = np.argmax(perp_distances_iso)
    return boundary_index, np.array(perp_distances_iso)
```

Rationale: If the representation truly lies on a low-dim manifold, geodesic distance is more accurate than Euclidean

---

METHOD G4: Local density + distance combination
────────────────────────────────────────────────
Frames near decision boundaries have different local density properties.

boundary = argmax_i [d_perp(i) + w * local_density_anomaly(i)]

where local_density_anomaly(i) measures how different frame i's local neighborhood is.

Code sketch:
```python
from sklearn.neighbors import NearestNeighbors

def method_g4_density_anomaly(reps):
    """
    Boundary detection via density + geometry.
    """
    # Local density via k-NN
    nbrs = NearestNeighbors(n_neighbors=min(4, len(reps)-1)).fit(reps)
    distances_nn, indices_nn = nbrs.kneighbors(reps)
    local_density = 1.0 / (np.mean(distances_nn, axis=1) + 1e-6)
    
    # Also use chord method
    v_chord = reps[9] - reps[0]
    v_chord = v_chord / np.linalg.norm(v_chord)
    lambda_i = reps @ v_chord
    reps_ortho = reps - np.outer(lambda_i, v_chord)
    perp_dist = np.linalg.norm(reps_ortho, axis=1)
    
    # Normalize and combine
    perp_dist_norm = perp_dist / (np.max(perp_dist) + 1e-6)
    local_density_norm = local_density / (np.max(local_density) + 1e-6)
    
    combined_score = 0.6 * perp_dist_norm + 0.4 * local_density_norm
    boundary_index = np.argmax(combined_score)
    
    return boundary_index, combined_score
```

Rationale: Ambiguous frames are "between" decision regions, so they have lower local density (fewer close neighbors in representation space)

---

METHOD G5: Symmetry detection
──────────────────────────────
If the sequence has temporal symmetry (yes→no has same geometry as no→yes), exploit it.

For ground truth patterns like:
  - no→yes: frames 0-5 → 5-10
  - yes→no: frames 0-5 → 5-10 (reversed)

The center frame should be exactly in the middle AND have maximal "symmetry breaking".

Code sketch:
```python
def method_g5_symmetry_center(reps):
    """
    Boundary detection via symmetry properties.
    """
    # Compute distance from frame i to its temporal mirror frame (9-i)
    mirror_distances = []
    for i in range(10):
        mirror_i = 9 - i
        dist = np.linalg.norm(reps[i] - reps[mirror_i])
        mirror_distances.append(dist)
    
    mirror_distances = np.array(mirror_distances)
    
    # Center frame should have MAXIMUM distance to its mirror
    # (least symmetric, most ambiguous)
    boundary_index = np.argmax(mirror_distances)
    
    return boundary_index, mirror_distances
```

Rationale: For symmetric sequences, the center is the only temporally unique point

---

========================================
PHASE 4: COMPARATIVE TESTING
========================================

For each of 92 sequences and each representation stage (vision, projector, hidden):

1. Compute boundary using Methods A, B, E, F (existing) + G1-G5 (new)
2. Compute boundary error and exact match for each
3. Aggregate across:
   - All sequences
   - Per category (Geometric, Semantic, Compositional)
   - Per ground truth pattern (no→yes, yes→no, yes→yes, no→no)
   - Per model (LLaVA, Phi, Qwen, Pixtral)

Output:
  - Comparison table: Mean BE and exact match for all 11 methods
  - Statistical significance tests (is the improvement significant?)
  - Which method works best for which category?
  - Does any method consistently outperform Chord?

========================================
PHASE 5: THEORETICAL ANALYSIS
========================================

If Phase 1-3 succeed, you can make strong claims:

1. **Orthogonal encoding is real:**
   - PC1 aligns with decision axis (Test 1)
   - Orthogonal variance peaks at center (Test 2)
   - Curvature concentrates at center (Test 4)

2. **The manifold is low-dimensional:**
   - Effective dimensionality << original (Test 5)
   - Can exploit manifold learning (Methods G3)

3. **New methods discovered that beat Chord:**
   - Document which geometric properties matter most
   - Propose a theoretical model of how the model encodes ambiguity
   - Explain why some methods work better on some categories

4. **Generalization to other models/tasks:**
   - If the same geometric structure appears in all 4 models
   - Then this structure is fundamental to how these models work
   - Not just an artifact of one architecture

========================================
SUMMARY: What to compute and check
========================================

Validation Checklist:
  ☐ Test 1: PC1 aligns with decision axis (cos > 0.7 for >80% sequences)
  ☐ Test 2: Orthogonal variance U-shaped at center (Spearman < -0.5 for >80%)
  ☐ Test 3: Can recover frame order from representation distances (>50% perfect)
  ☐ Test 4: Curvature peaks near center frame (negative Spearman for >70%)
  ☐ Test 5: Effective dimensionality << original (typically 3-10 vs 256-1024)

New Methods to Test:
  ☐ G1: Curvature-weighted chord distance
  ☐ G2: Variance along principal axes
  ☐ G3: Geodesic distance via Isomap
  ☐ G4: Local density anomaly + geometry
  ☐ G5: Symmetry-breaking center detection

Success Criteria:
  - At least 3 of 5 validation tests pass strongly
  - At least 1 new method (G1-G5) outperforms Chord method
  - Improvements are consistent across stages and models
  - Improvements are statistically significant

This framework will tell you:
  1. Whether the orthogonal hypothesis is TRUE
  2. What geometric properties matter
  3. How to design even better boundary detection methods
"""