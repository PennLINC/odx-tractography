//! Tiny 3D voxel-grid + mesh-graph primitives used to clean up the
//! pseudo-surface pipeline:
//!
//! - [`mean_filter_3x3x3`] — iterative 27-tap unweighted box filter on a flat
//!   C-order voxel buffer. Mirrors DSI-Studio's `tipl::filter::mean`
//!   *behaviourally* (zero-extrapolation at boundaries, divide by 27). Used as
//!   the pre-marching-cubes smoother in [`crate::qa_otsu::wm_field_otsu`].
//! - [`voxel_cc_label_6conn`] / [`drop_small_components`] — 6-connected
//!   connected-component labelling on a binary mask; the latter is a
//!   convenience for dropping noise floaters.
//! - [`erode_6conn`] — in-place 6-connected morphological erosion. Used on a
//!   *labelling-only copy* of the WM mask to thin the corpus callosum until
//!   left and right hemispheres separate into distinct components.
//! - [`propagate_labels_bfs`] — multi-source BFS from labelled voxels outward
//!   into an un-labelled mask. Used to propagate hemisphere labels from the
//!   eroded core back over the un-eroded mask (so the corpus callosum gets
//!   split roughly down the middle by the wavefront).
//! - [`mesh_cc_label`] — vertex-graph BFS over a triangle index list.
//! - [`taubin_smooth`] — alternating λ/μ Taubin smoothing of triangle-mesh
//!   vertex positions. Non-shrinking by construction (the negative μ step
//!   compensates for the positive λ shrinkage), unlike a plain Laplacian
//!   smoother.
//!
//! C-order convention (matches `qa_otsu.rs` and `pseudo_surfaces.rs`):
//! `flat = i*ny*nz + j*nz + k`, with `k` (z) running fastest. dims are
//! `[nx, ny, nz]`.

#![allow(clippy::needless_range_loop)]

#[inline]
fn flat_idx(i: usize, j: usize, k: usize, dims: [usize; 3]) -> usize {
    i * dims[1] * dims[2] + j * dims[2] + k
}

// ---------------------------------------------------------------------------
// Mean filter (DSI-Studio-style, on a continuous f32 field)
// ---------------------------------------------------------------------------

/// In-place iterative 27-tap unweighted mean (3×3×3 box) on a flat C-order
/// `f32` buffer. Each iteration replaces every voxel with the average of itself
/// and its 26 neighbours. Boundary voxels use zero-extrapolation (samples
/// outside the grid contribute 0 to the sum but still divide by 27, matching
/// DSI-Studio's default).
///
/// Costs one extra `Vec<f32>` allocation per call (the ping-pong buffer),
/// reused across iterations.
pub fn mean_filter_3x3x3(buf: &mut [f32], dims: [usize; 3], iters: u32) {
    let [nx, ny, nz] = dims;
    if iters == 0 || buf.len() != nx * ny * nz {
        return;
    }
    let mut tmp = vec![0.0_f32; buf.len()];
    let inv27 = 1.0_f32 / 27.0;
    for _ in 0..iters {
        for i in 0..nx {
            let i_lo = i.saturating_sub(1);
            let i_hi = (i + 1).min(nx - 1);
            for j in 0..ny {
                let j_lo = j.saturating_sub(1);
                let j_hi = (j + 1).min(ny - 1);
                for k in 0..nz {
                    let k_lo = k.saturating_sub(1);
                    let k_hi = (k + 1).min(nz - 1);
                    let mut sum = 0.0_f32;
                    for ii in i_lo..=i_hi {
                        for jj in j_lo..=j_hi {
                            for kk in k_lo..=k_hi {
                                sum += buf[flat_idx(ii, jj, kk, dims)];
                            }
                        }
                    }
                    // Note: dividing by 27 even when fewer than 27 cells were
                    // actually summed (boundary voxels) is the
                    // zero-extrapolation choice — same as DSI-Studio's
                    // default add-zero-weight behaviour.
                    tmp[flat_idx(i, j, k, dims)] = sum * inv27;
                }
            }
        }
        buf.copy_from_slice(&tmp);
    }
}

// ---------------------------------------------------------------------------
// 6-connected voxel connected components
// ---------------------------------------------------------------------------

/// 6-connected BFS labelling on a binary mask (any non-zero byte = inside).
/// Background voxels keep label 0; foreground components are numbered 1, 2,
/// …. Returns `(labels, counts)` where `counts[id]` is the voxel count of
/// component `id` (count[0] = number of background voxels).
pub fn voxel_cc_label_6conn(mask: &[u8], dims: [usize; 3]) -> (Vec<u32>, Vec<usize>) {
    let total = dims[0] * dims[1] * dims[2];
    assert_eq!(mask.len(), total, "mask length doesn't match dims");
    let mut labels = vec![0_u32; total];
    let mut counts = vec![0_usize; 1];
    counts[0] = mask.iter().filter(|&&b| b == 0).count();
    let mut next_id: u32 = 1;
    let mut queue: Vec<usize> = Vec::new();
    for seed in 0..total {
        if mask[seed] == 0 || labels[seed] != 0 {
            continue;
        }
        // BFS.
        labels[seed] = next_id;
        let mut size: usize = 0;
        queue.clear();
        queue.push(seed);
        while let Some(p) = queue.pop() {
            size += 1;
            let i = p / (dims[1] * dims[2]);
            let rem = p % (dims[1] * dims[2]);
            let j = rem / dims[2];
            let k = rem % dims[2];
            for (di, dj, dk) in [
                (-1, 0, 0), (1, 0, 0),
                (0, -1, 0), (0, 1, 0),
                (0, 0, -1), (0, 0, 1),
            ] {
                let ni = i as isize + di;
                let nj = j as isize + dj;
                let nk = k as isize + dk;
                if ni < 0 || nj < 0 || nk < 0 {
                    continue;
                }
                let (ni, nj, nk) = (ni as usize, nj as usize, nk as usize);
                if ni >= dims[0] || nj >= dims[1] || nk >= dims[2] {
                    continue;
                }
                let q = flat_idx(ni, nj, nk, dims);
                if mask[q] != 0 && labels[q] == 0 {
                    labels[q] = next_id;
                    queue.push(q);
                }
            }
        }
        counts.push(size);
        next_id += 1;
    }
    (labels, counts)
}

/// Drop components smaller than `min_voxels` from a binary mask in place.
pub fn drop_small_components(mask: &mut [u8], dims: [usize; 3], min_voxels: usize) {
    if min_voxels <= 1 {
        return;
    }
    let (labels, counts) = voxel_cc_label_6conn(mask, dims);
    for v in 0..mask.len() {
        if mask[v] == 0 {
            continue;
        }
        let id = labels[v] as usize;
        if id == 0 || counts[id] < min_voxels {
            mask[v] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// 6-connected morphological erosion
// ---------------------------------------------------------------------------

/// In-place 6-connected morphological dilation. After `iters` applications,
/// a voxel is set iff at least one of its `iters`-step 6-neighbours was
/// originally inside the mask. Used to recover from smoothing shrinkage in
/// the morphological cleanup pipeline.
pub fn dilate_6conn(mask: &mut [u8], dims: [usize; 3], iters: u32) {
    let [nx, ny, nz] = dims;
    if iters == 0 || mask.len() != nx * ny * nz {
        return;
    }
    let mut tmp = vec![0_u8; mask.len()];
    for _ in 0..iters {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let p = flat_idx(i, j, k, dims);
                    if mask[p] != 0 {
                        tmp[p] = 1;
                        continue;
                    }
                    // Check 6-neighbours.
                    let mut hit = false;
                    if i > 0 && mask[flat_idx(i - 1, j, k, dims)] != 0 { hit = true; }
                    if !hit && i + 1 < nx && mask[flat_idx(i + 1, j, k, dims)] != 0 { hit = true; }
                    if !hit && j > 0 && mask[flat_idx(i, j - 1, k, dims)] != 0 { hit = true; }
                    if !hit && j + 1 < ny && mask[flat_idx(i, j + 1, k, dims)] != 0 { hit = true; }
                    if !hit && k > 0 && mask[flat_idx(i, j, k - 1, dims)] != 0 { hit = true; }
                    if !hit && k + 1 < nz && mask[flat_idx(i, j, k + 1, dims)] != 0 { hit = true; }
                    tmp[p] = if hit { 1 } else { 0 };
                }
            }
        }
        mask.copy_from_slice(&tmp);
    }
}

/// In-place binary majority-vote smoothing in a 3×3×3 neighbourhood
/// (including the centre voxel). For each voxel, count the number of set
/// neighbours including itself (max 27); set the voxel iff that count > 13.
/// Equivalent to DSI-Studio's `tipl::morphology::smoothing` for 3D images.
/// Used in the morphological cleanup pipeline to round binary mask edges
/// before the soft-attenuation + Gaussian step.
pub fn binary_smoothing_27nbr(mask: &mut [u8], dims: [usize; 3]) {
    let [nx, ny, nz] = dims;
    if mask.len() != nx * ny * nz {
        return;
    }
    let mut tmp = vec![0_u8; mask.len()];
    for i in 0..nx {
        let i_lo = i.saturating_sub(1);
        let i_hi = (i + 1).min(nx - 1);
        for j in 0..ny {
            let j_lo = j.saturating_sub(1);
            let j_hi = (j + 1).min(ny - 1);
            for k in 0..nz {
                let k_lo = k.saturating_sub(1);
                let k_hi = (k + 1).min(nz - 1);
                let mut count: u32 = 0;
                for ii in i_lo..=i_hi {
                    for jj in j_lo..=j_hi {
                        for kk in k_lo..=k_hi {
                            if mask[flat_idx(ii, jj, kk, dims)] != 0 {
                                count += 1;
                            }
                        }
                    }
                }
                tmp[flat_idx(i, j, k, dims)] = if count > 13 { 1 } else { 0 };
            }
        }
    }
    mask.copy_from_slice(&tmp);
}

/// In-place "keep only the largest 6-connected component" — DSI-Studio's
/// `tipl::morphology::defragment`. Drops every voxel that isn't part of the
/// biggest connected component. Used in the cleanup pipeline as both
/// "remove all floating islands" (run on the foreground) and, sandwiched
/// between two `negate` calls, as "fill all internal holes".
pub fn keep_largest_component_6conn(mask: &mut [u8], dims: [usize; 3]) {
    let (labels, counts) = voxel_cc_label_6conn(mask, dims);
    if counts.len() <= 1 {
        return;
    }
    // counts[0] is background; foreground IDs start at 1.
    let mut largest_id: u32 = 1;
    let mut largest_size: usize = 0;
    for (id, &c) in counts.iter().enumerate().skip(1) {
        if c > largest_size {
            largest_size = c;
            largest_id = id as u32;
        }
    }
    for v in 0..mask.len() {
        if mask[v] != 0 && labels[v] != largest_id {
            mask[v] = 0;
        }
    }
}

/// In-place binary negation: every set voxel becomes 0, every zero voxel
/// becomes 1. Trivial helper used in the morphological hole-fill sandwich.
#[inline]
pub fn negate_binary(mask: &mut [u8]) {
    for v in mask.iter_mut() {
        *v = if *v == 0 { 1 } else { 0 };
    }
}

/// In-place 6-connected morphological erosion. After `iters` applications, a
/// voxel survives iff all its `iters`-step 6-neighbours were originally inside
/// the mask. Used to thin the corpus callosum until the two hemispheres
/// separate into distinct connected components.
pub fn erode_6conn(mask: &mut [u8], dims: [usize; 3], iters: u32) {
    let [nx, ny, nz] = dims;
    if iters == 0 || mask.len() != nx * ny * nz {
        return;
    }
    let mut tmp = vec![0_u8; mask.len()];
    for _ in 0..iters {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let p = flat_idx(i, j, k, dims);
                    if mask[p] == 0 {
                        tmp[p] = 0;
                        continue;
                    }
                    // Boundary voxels are treated as outside (eroded away),
                    // which is the standard behaviour and matches the
                    // zero-extrapolation we use elsewhere.
                    let inside = i > 0
                        && j > 0
                        && k > 0
                        && i + 1 < nx
                        && j + 1 < ny
                        && k + 1 < nz
                        && mask[flat_idx(i - 1, j, k, dims)] != 0
                        && mask[flat_idx(i + 1, j, k, dims)] != 0
                        && mask[flat_idx(i, j - 1, k, dims)] != 0
                        && mask[flat_idx(i, j + 1, k, dims)] != 0
                        && mask[flat_idx(i, j, k - 1, dims)] != 0
                        && mask[flat_idx(i, j, k + 1, dims)] != 0;
                    tmp[p] = if inside { 1 } else { 0 };
                }
            }
        }
        mask.copy_from_slice(&tmp);
    }
}

// ---------------------------------------------------------------------------
// Multi-source label propagation
// ---------------------------------------------------------------------------

/// Multi-source BFS from already-labelled voxels (label > 0) outward into the
/// un-labelled portion of `target_mask`. Each unlabelled voxel inside
/// `target_mask` receives the label of its nearest labelled voxel (in 6-conn
/// graph distance). Voxels outside `target_mask` keep label 0.
///
/// Used to take a sparse hemisphere labelling (from the eroded mask's
/// connected components) and grow it back over the full un-eroded WM mask:
/// the wavefront from each hemisphere meets roughly at the corpus-callosum
/// midline, splitting the bridge cleanly.
pub fn propagate_labels_bfs(
    seed_labels: &[u32],
    target_mask: &[u8],
    dims: [usize; 3],
) -> Vec<u32> {
    let total = dims[0] * dims[1] * dims[2];
    assert_eq!(seed_labels.len(), total, "seed_labels length doesn't match dims");
    assert_eq!(target_mask.len(), total, "target_mask length doesn't match dims");
    let mut out = vec![0_u32; total];
    let mut frontier: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for (idx, &lbl) in seed_labels.iter().enumerate() {
        if lbl != 0 && target_mask[idx] != 0 {
            out[idx] = lbl;
            frontier.push_back(idx);
        }
    }
    while let Some(p) = frontier.pop_front() {
        let i = p / (dims[1] * dims[2]);
        let rem = p % (dims[1] * dims[2]);
        let j = rem / dims[2];
        let k = rem % dims[2];
        let label = out[p];
        for (di, dj, dk) in [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ] {
            let ni = i as isize + di;
            let nj = j as isize + dj;
            let nk = k as isize + dk;
            if ni < 0 || nj < 0 || nk < 0 {
                continue;
            }
            let (ni, nj, nk) = (ni as usize, nj as usize, nk as usize);
            if ni >= dims[0] || nj >= dims[1] || nk >= dims[2] {
                continue;
            }
            let q = flat_idx(ni, nj, nk, dims);
            if target_mask[q] != 0 && out[q] == 0 {
                out[q] = label;
                frontier.push_back(q);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Mesh-graph (vertex 1-ring) connected components
// ---------------------------------------------------------------------------

/// Connected components of the *vertex graph* induced by the triangle list:
/// two vertices are neighbours iff they share at least one triangle. Returns
/// `(per_vertex_label, component_vertex_counts)`. Vertices that don't appear
/// in any triangle keep label 0; component IDs are numbered from 1.
pub fn mesh_cc_label(
    triangles: &[[u32; 3]],
    n_vertices: usize,
) -> (Vec<u32>, Vec<usize>) {
    // Build adjacency as a CSR-ish Vec<Vec<u32>> for simplicity. For typical
    // brain meshes this is ~200k vertices × ~6 neighbours, well under 10MB.
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n_vertices];
    for t in triangles {
        let [a, b, c] = *t;
        let (a, b, c) = (a as usize, b as usize, c as usize);
        if a >= n_vertices || b >= n_vertices || c >= n_vertices {
            continue;
        }
        adj[a].push(b as u32);
        adj[a].push(c as u32);
        adj[b].push(a as u32);
        adj[b].push(c as u32);
        adj[c].push(a as u32);
        adj[c].push(b as u32);
    }
    let mut labels = vec![0_u32; n_vertices];
    let mut counts: Vec<usize> = vec![0]; // counts[0] reserved for "no component"
    let mut next_id: u32 = 1;
    for seed in 0..n_vertices {
        if labels[seed] != 0 || adj[seed].is_empty() {
            continue;
        }
        labels[seed] = next_id;
        let mut size = 0_usize;
        let mut stack: Vec<usize> = vec![seed];
        while let Some(v) = stack.pop() {
            size += 1;
            for &n in &adj[v] {
                let n = n as usize;
                if labels[n] == 0 {
                    labels[n] = next_id;
                    stack.push(n);
                }
            }
        }
        counts.push(size);
        next_id += 1;
    }
    (labels, counts)
}

// ---------------------------------------------------------------------------
// Taubin smoothing
// ---------------------------------------------------------------------------

/// Build the 1-ring vertex adjacency from a triangle list. Each entry is a
/// deduplicated list of neighbour vertex indices.
fn build_one_ring(triangles: &[[u32; 3]], n_vertices: usize) -> Vec<Vec<u32>> {
    let mut sets: Vec<std::collections::HashSet<u32>> =
        vec![std::collections::HashSet::new(); n_vertices];
    for t in triangles {
        let [a, b, c] = *t;
        let (au, bu, cu) = (a as usize, b as usize, c as usize);
        if au >= n_vertices || bu >= n_vertices || cu >= n_vertices {
            continue;
        }
        sets[au].insert(b);
        sets[au].insert(c);
        sets[bu].insert(a);
        sets[bu].insert(c);
        sets[cu].insert(a);
        sets[cu].insert(b);
    }
    sets.into_iter()
        .map(|s| s.into_iter().collect::<Vec<u32>>())
        .collect()
}

/// One Laplacian-style smoothing pass: replace each vertex with its 1-ring
/// centroid, weighted by `factor`. `out[v] = pos[v] + factor × (mean_neighbour - pos[v])`.
fn laplacian_pass(
    vertices: &mut [[f32; 3]],
    one_ring: &[Vec<u32>],
    factor: f32,
) {
    let n = vertices.len();
    let mut next = vec![[0.0_f32; 3]; n];
    for v in 0..n {
        let nb = &one_ring[v];
        if nb.is_empty() {
            next[v] = vertices[v];
            continue;
        }
        let mut acc = [0.0_f32; 3];
        for &w in nb {
            let p = vertices[w as usize];
            acc[0] += p[0];
            acc[1] += p[1];
            acc[2] += p[2];
        }
        let inv = 1.0 / nb.len() as f32;
        let mean = [acc[0] * inv, acc[1] * inv, acc[2] * inv];
        let p = vertices[v];
        next[v] = [
            p[0] + factor * (mean[0] - p[0]),
            p[1] + factor * (mean[1] - p[1]),
            p[2] + factor * (mean[2] - p[2]),
        ];
    }
    vertices.copy_from_slice(&next);
}

/// In-place Taubin (λ/μ) mesh smoothing. Runs `iters` *pairs* of passes:
/// alternating a positive λ shrinking pass and a negative μ inflating pass.
/// With μ < −λ < 0 the steady-state displacement on a flat region is exactly
/// zero, so the mesh is smoothed without volume loss — the canonical fix for
/// Laplacian shrinkage.
///
/// Defaults λ = 0.5, μ = −0.53 (Taubin 1995). The user-facing iteration count
/// is the number of *λ/μ pairs* — total work ≈ 2 × `iters` per-vertex passes.
pub fn taubin_smooth(
    vertices: &mut [[f32; 3]],
    triangles: &[[u32; 3]],
    lambda: f32,
    mu: f32,
    iters: u32,
) {
    if iters == 0 || vertices.is_empty() || triangles.is_empty() {
        return;
    }
    let one_ring = build_one_ring(triangles, vertices.len());
    for _ in 0..iters {
        laplacian_pass(vertices, &one_ring, lambda);
        laplacian_pass(vertices, &one_ring, mu);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_filter_smooths_step() {
        // 5×1×1 step: 0 0 0 1 1 → after one mean iter, the boundary blurs.
        let mut v = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        mean_filter_3x3x3(&mut v, [5, 1, 1], 1);
        // Voxel 2 was 0, has neighbour 3=1 in its 27-tap; should be > 0.
        assert!(v[2] > 0.0 && v[2] < 0.5, "blurred voxel = {}", v[2]);
        // The mid-step voxel that was 1 should drop below 1.
        assert!(v[3] > 0.0 && v[3] < 1.0, "blurred voxel = {}", v[3]);
    }

    #[test]
    fn mean_filter_zero_iters_is_noop() {
        let mut v = vec![1.0_f32, 2.0, 3.0];
        let cp = v.clone();
        mean_filter_3x3x3(&mut v, [3, 1, 1], 0);
        assert_eq!(v, cp);
    }

    #[test]
    fn voxel_cc_finds_two_components() {
        // Two disjoint 1×1×1 cubes in a 5×1×1 grid.
        let mask = vec![1_u8, 0, 1, 0, 1];
        let (labels, counts) = voxel_cc_label_6conn(&mask, [5, 1, 1]);
        let n_components = counts.len() - 1; // counts[0] = bg
        assert_eq!(n_components, 3); // three isolated voxels
        // Different positions get different labels.
        assert_ne!(labels[0], labels[2]);
        assert_ne!(labels[2], labels[4]);
    }

    #[test]
    fn drop_small_components_keeps_only_large() {
        let mut mask = vec![1_u8, 1, 0, 1, 0];
        // Component at idx 0..=1 is size 2; idx 3 is size 1.
        drop_small_components(&mut mask, [5, 1, 1], 2);
        assert_eq!(mask, vec![1, 1, 0, 0, 0]);
    }

    #[test]
    fn erosion_breaks_thin_bridge() {
        // 7×3×3 cube: two 2×3×3 "hemispheres" connected by a 3×1×1 bridge.
        // After one erosion the bridge should disappear (it's only 1 wide).
        let dims = [7, 3, 3];
        let mut mask = vec![0_u8; 7 * 3 * 3];
        // Left hemi: i in 0..=1, j&k full.
        // Right hemi: i in 5..=6, j&k full.
        // Bridge: i in 2..=4, j=1, k=1.
        for i in 0..=1 {
            for j in 0..3 {
                for k in 0..3 {
                    mask[flat_idx(i, j, k, dims)] = 1;
                }
            }
        }
        for i in 5..=6 {
            for j in 0..3 {
                for k in 0..3 {
                    mask[flat_idx(i, j, k, dims)] = 1;
                }
            }
        }
        for i in 2..=4 {
            mask[flat_idx(i, 1, 1, dims)] = 1;
        }
        // Pre-erosion: one connected component.
        let (_, counts) = voxel_cc_label_6conn(&mask, dims);
        assert_eq!(counts.len() - 1, 1, "should be one component before erosion");
        // Post-erosion: the bridge is 1-wide so 1 erosion kills it; the
        // hemispheres are 2-wide each so they shrink but survive.
        let mut eroded = mask.clone();
        erode_6conn(&mut eroded, dims, 1);
        let (_, counts) = voxel_cc_label_6conn(&eroded, dims);
        assert!(
            counts.len() - 1 >= 2,
            "erosion should split into ≥2 components, got {}",
            counts.len() - 1
        );
    }

    #[test]
    fn label_propagation_covers_target() {
        // 5×1×1 mask. Seeds at idx 0 (label 1) and idx 4 (label 2). Target
        // mask is full. Propagation should split the middle: 1 1 1|2 2 or
        // similar — every cell labelled, midline assigned by which seed
        // arrived first along the FIFO BFS frontier.
        let dims = [5, 1, 1];
        let mut seeds = vec![0_u32; 5];
        seeds[0] = 1;
        seeds[4] = 2;
        let target = vec![1_u8; 5];
        let out = propagate_labels_bfs(&seeds, &target, dims);
        assert_eq!(out[0], 1);
        assert_eq!(out[4], 2);
        assert!(out.iter().all(|&l| l != 0), "all voxels labelled");
        // The two labels split the line.
        let n_label_1 = out.iter().filter(|&&l| l == 1).count();
        let n_label_2 = out.iter().filter(|&&l| l == 2).count();
        assert!(n_label_1 >= 2 && n_label_2 >= 2);
    }

    #[test]
    fn mesh_cc_finds_two_components() {
        // Two disjoint triangles.
        let triangles = vec![[0_u32, 1, 2], [3_u32, 4, 5]];
        let (labels, counts) = mesh_cc_label(&triangles, 6);
        assert_eq!(counts.len() - 1, 2);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn taubin_does_not_shrink_a_plane() {
        // Five coplanar vertices on a regular grid, one wiggled out of the
        // plane; Taubin smoothing should pull the wiggle back without moving
        // the plane bulk inward.
        let mut verts: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.5], // wiggle
        ];
        // Triangulate as a fan around vertex 0; vertex 5 sits "above" 0.
        let triangles: Vec<[u32; 3]> = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [0, 1, 5], // attach the wiggle
        ];
        let original_z_avg: f32 =
            verts.iter().map(|v| v[2]).sum::<f32>() / verts.len() as f32;
        taubin_smooth(&mut verts, &triangles, 0.5, -0.53, 20);
        let new_z_avg: f32 =
            verts.iter().map(|v| v[2]).sum::<f32>() / verts.len() as f32;
        // Mean z should stay close to the original (Taubin is non-shrinking).
        // A pure Laplacian smoother would drag the whole thing toward the
        // wiggle's centroid; Taubin shouldn't.
        assert!(
            (new_z_avg - original_z_avg).abs() < 0.1,
            "Taubin shrunk z-mean: {} → {}",
            original_z_avg,
            new_z_avg
        );
    }
}
