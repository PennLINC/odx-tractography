//! Build a clean pseudo-surface (and an associated synthetic pial point set
//! for visualisation) from either a binary WM mask or a smoothed continuous
//! WM field.
//!
//! There are two entry points:
//!
//! - [`pseudo_surfaces_from_field`] (preferred) — runs marching cubes on the
//!   *smoothed continuous* field at `field.iso`. Produces a single
//!   [`PseudoSurfaceMesh`] (vertices, triangles, per-vertex inward unit
//!   normals) plus optional voxel-level and mesh-level cleanup (drop small
//!   components, Taubin smoothing). This is what makes the surface look
//!   substantially cleaner than the older binary-mask path: sub-voxel boundary
//!   precision from MC on a smoothed field, plus non-shrinking Taubin
//!   smoothing of the mesh.
//! - [`pseudo_surfaces_from_mask`] (legacy) — runs MC on a binary mask at
//!   isovalue 0.5 with no smoothing or cleanup. Returns the older
//!   [`PseudoSurfacePair`] shape (paired wm/pial vertex sets, where the pial
//!   is constructed by offsetting wm along its outward normal). Kept for
//!   backward compatibility; the offset-based "pial" is mathematically just
//!   the outward normal multiplied by `pial_offset_mm`, so the inward normal
//!   `wm − pial` carries no more information than the surface's own outward
//!   normal.
//!
//! Hemisphere splitting: [`split_by_eroded_voxel_cc`] is the new preferred
//! routine. It tackles the corpus-callosum problem — the WM mask is one big
//! connected component because the CC bridges the hemispheres — by eroding a
//! *labelling-only copy* of the above-iso mask until LH/RH separate, then
//! propagating the labels back over the un-eroded mask via BFS. The mesh
//! itself is never eroded; only the labelling helper. [`split_by_x_sign`]
//! remains as a fallback for non-AC-PC-aligned data and as a sanity check.

use lin_alg::f32::Vec3 as LVec3;
use mcubes::{MarchingCubes, MeshSide};

use crate::mean_3d::{
    drop_small_components, erode_6conn, keep_largest_component_6conn, mean_filter_3x3x3,
    mesh_cc_label, negate_binary, propagate_labels_bfs, taubin_smooth, voxel_cc_label_6conn,
};
use crate::qa_otsu::WmField;

#[derive(Debug, Clone, Copy)]
pub struct PseudoSurfaceParams {
    /// Outward-direction offset (mm) from each wm vertex to its synthetic
    /// pial vertex in the legacy [`PseudoSurfacePair`] / [`pseudo_surfaces_from_mask`]
    /// path. Defaults to 2 mm — a couple of voxels at typical 1–2 mm dMRI.
    /// Also used by the new path when callers ask for a synthetic pial cloud
    /// from a [`PseudoSurfaceMesh`] for visualisation.
    pub pial_offset_mm: f32,
    /// Skip mesh extraction if the mask has fewer than this many in-mask
    /// voxels (avoids running marching cubes on essentially-empty data).
    pub min_mask_voxels: usize,
    /// **New-path only.** Drop voxel-level connected components smaller than
    /// this from the above-iso binary mask before MC. 0 disables. Default 200
    /// voxels — enough to remove Otsu specks without touching anatomy.
    pub min_component_voxels: usize,
    /// **New-path only.** Drop mesh-level connected components smaller than
    /// this many *vertices* after MC. 0 disables. Default 500.
    pub min_mesh_vertices: usize,
    /// **New-path only.** Number of Taubin λ/μ pairs applied to the MC vertex
    /// positions. 0 disables. Default 2 (≈ 4 per-vertex passes), enough to
    /// round MC's worst grid-aligned facets without washing out gyral
    /// texture. Higher values (5+) progressively smooth out the cortical
    /// folding detail; on QA data where you need every bit of structural
    /// signal, leaving mesh smoothing very light is usually right.
    pub mesh_smooth_iters: u32,
    /// **New-path only.** Taubin λ (positive smoothing factor). Default 0.5.
    pub taubin_lambda: f32,
    /// **New-path only.** Taubin μ (negative inflating factor). Must satisfy
    /// `μ < −λ` for non-shrinkage. Default −0.53.
    pub taubin_mu: f32,
    /// **New-path only.** Apply DSI-Studio-style morphological cleanup before
    /// MC: threshold to binary, keep largest connected component, fill
    /// internal holes (negate-defragment-negate), 3×3×3 majority-vote
    /// smoothing, 6-conn dilation, then attenuate scalar voxels outside the
    /// cleaned mask by 0.2× and Gaussian-smooth the soft-attenuated field.
    /// MC then sees a sharp inside / soft outside continuous field and
    /// produces a clean iso surface following the cleaned mask boundary.
    /// This is the key to gyri-following surfaces on noisy QA where raw
    /// thresholding produces a fragmented mask. Default `true`.
    pub morph_cleanup: bool,
}

impl Default for PseudoSurfaceParams {
    fn default() -> Self {
        Self {
            pial_offset_mm: 2.0,
            min_mask_voxels: 100,
            min_component_voxels: 200,
            min_mesh_vertices: 500,
            mesh_smooth_iters: 2,
            morph_cleanup: true,
            taubin_lambda: 0.5,
            taubin_mu: -0.53,
        }
    }
}

/// New: a single MC-extracted mesh in RAS+ mm with per-vertex inward unit
/// normals. There is no synthetic "pial" geometry — downstream code that
/// wants a paired cloud for visualisation can compute it from
/// `vertex − thickness × inward_normal` on the fly.
#[derive(Debug, Clone)]
pub struct PseudoSurfaceMesh {
    /// Vertex positions in RAS+ mm.
    pub vertices: Vec<[f32; 3]>,
    /// Triangle indices.
    pub triangles: Vec<[u32; 3]>,
    /// Per-vertex inward unit normal (points into the brain, i.e. opposite
    /// to the marching-cubes outward face normal). Length == `vertices.len()`.
    pub inward_normals: Vec<[f32; 3]>,
}

impl PseudoSurfaceMesh {
    /// Synthesise a paired (wm, pial) point cloud + shared topology from this
    /// single mesh, by offsetting along the inward normals. **Display-only**:
    /// the pial cloud has no independent topology and shouldn't be used as a
    /// stand-alone surface.
    pub fn to_visualisation_pair(&self, thickness_mm: f32) -> PseudoSurfacePair {
        // The mesh stores *inward* normals; the pial-side offset goes outward
        // (away from the brain), so subtract.
        let pial_vertices: Vec<[f32; 3]> = self
            .vertices
            .iter()
            .zip(self.inward_normals.iter())
            .map(|(p, n)| {
                [
                    p[0] - thickness_mm * n[0],
                    p[1] - thickness_mm * n[1],
                    p[2] - thickness_mm * n[2],
                ]
            })
            .collect();
        PseudoSurfacePair {
            wm_vertices: self.vertices.clone(),
            pial_vertices,
            triangles: self.triangles.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PseudoSurfacePair {
    /// Vertex positions on the WM-side surface in RAS+ mm.
    pub wm_vertices: Vec<[f32; 3]>,
    /// Paired pial vertex set, same length and ordering as `wm_vertices`.
    /// `pial[i] = wm[i] + pial_offset_mm × outward_normal[i]`, so
    /// `wm[i] - pial[i]` points inward (deeper WM).
    pub pial_vertices: Vec<[f32; 3]>,
    /// Triangle indices into both vertex arrays.
    pub triangles: Vec<[u32; 3]>,
}

#[derive(Debug, Clone)]
pub struct PseudoSurfaceHemispheres {
    pub lh: PseudoSurfacePair,
    pub rh: PseudoSurfacePair,
}

#[derive(Debug, Clone)]
pub struct PseudoSurfaceMeshHemispheres {
    pub lh: PseudoSurfaceMesh,
    pub rh: PseudoSurfaceMesh,
}

// ===========================================================================
// New entry point: continuous-field marching cubes
// ===========================================================================

/// Run marching cubes on a smoothed continuous WM field, then optionally
/// clean up and Taubin-smooth the resulting mesh. Returns a single
/// [`PseudoSurfaceMesh`] with per-vertex inward unit normals.
///
/// Returns `None` if the above-iso voxel set is too small or marching cubes
/// produces no triangles.
pub fn pseudo_surfaces_from_field(
    field: &WmField,
    params: &PseudoSurfaceParams,
) -> Option<PseudoSurfaceMesh> {
    let [nx, ny, nz] = field.dims;
    if nx < 2 || ny < 2 || nz < 2 || field.field.len() != nx * ny * nz {
        return None;
    }

    // Build the binary above-iso mask (C-order).
    let mut above_iso_mask = vec![0_u8; nx * ny * nz];
    for (m, &v) in above_iso_mask.iter_mut().zip(field.field.iter()) {
        if v > field.iso {
            *m = 1;
        }
    }

    // Voxel-level CC cleanup of the above-iso mask before MC. Drops Otsu
    // floaters at the source so we don't produce tiny mesh islands.
    // (Skipped when `morph_cleanup` is on, since that path does its own
    // largest-component-only defragment.)
    if !params.morph_cleanup && params.min_component_voxels > 1 {
        drop_small_components(&mut above_iso_mask, field.dims, params.min_component_voxels);
    }

    let nonzero = above_iso_mask.iter().filter(|&&b| b != 0).count();
    if nonzero < params.min_mask_voxels {
        return None;
    }

    // Build the C-order scalar that will be transposed into MC's F-order
    // input. Two paths:
    //
    // - Without `morph_cleanup`: copy raw QA values for above-iso voxels,
    //   zero elsewhere. MC threshold = field.iso. Sub-voxel-precise
    //   boundary, but susceptible to QA noise (small holes inside, small
    //   bright islands outside).
    //
    // - With `morph_cleanup` (default): clean up the binary mask via the
    //   DSI-Studio mode-2 recipe (largest-component-only → fill holes via
    //   negate-defrag-negate → 3×3×3 majority smoothing → 6-conn dilation),
    //   build a soft-attenuated continuous field where outside-mask voxels
    //   are scaled to 0.2× their original QA, then mean-smooth. MC at the
    //   original threshold finds a clean iso contour at the cleaned-mask
    //   boundary. This is what makes the QA-derived surface follow gyri
    //   rather than just outline the brain.
    let scalar_c_order: Vec<f32> = if params.morph_cleanup {
        // **Light cleanup that preserves sulcal indentations**.
        //
        // The full DSI-Studio mode-2 recipe (largest-CC-only + hole-fill +
        // binary majority smoothing + 6-conn dilation) destroys gyrification
        // on noisy dMRI QA: the dilation closes the 1-2 voxel-wide sulci,
        // the binary smoothing rounds away thin gyral stems, and the
        // largest-only defragment kills any gyrus whose WM has disconnected
        // from the main mass under noise. The result is a smooth balloon.
        //
        // Instead we keep just the steps that clean noise without closing
        // anatomy:
        //   1. drop_small_components — removes pure noise islands but keeps
        //      multiple large components (gyri can stay disconnected)
        //   2. negate-largest-CC-negate — fills internal holes (CSF specks
        //      inside WM, no risk of closing sulci because background
        //      sulcal voxels are connected to the outside-brain background)
        //   3. soft-attenuate outside the cleaned mask × 0.2
        //   4. one mean-filter pass for sub-voxel MC precision
        // Steps 3-4 give MC a smooth iso gradient at the mask boundary
        // without dilating the mask itself.
        let mut mask = above_iso_mask.clone();
        // 1. Drop noise islands but keep all decent-sized components.
        drop_small_components(&mut mask, field.dims, params.min_component_voxels.max(50));
        // 2. Fill internal holes only — sulcal background is connected to
        //    the outside-brain background, so the largest BG component is
        //    "outside + sulci" together; only small isolated background
        //    pockets (= internal CSF specks) get filled.
        negate_binary(&mut mask);
        keep_largest_component_6conn(&mut mask, field.dims);
        negate_binary(&mut mask);
        // 3. Soft-attenuate outside the cleaned mask.
        let mut soft: Vec<f32> = vec![0.0; nx * ny * nz];
        for ((sv, &m), &v) in soft.iter_mut()
            .zip(mask.iter())
            .zip(field.field.iter())
        {
            *sv = if m == 0 { v * 0.2 } else { v };
        }
        // 4. Single mean-filter pass for sub-voxel precision.
        mean_filter_3x3x3(&mut soft, field.dims, 1);
        soft
    } else {
        let mut soft: Vec<f32> = vec![0.0; nx * ny * nz];
        for (sv, (&m, &v)) in soft.iter_mut().zip(above_iso_mask.iter().zip(field.field.iter())) {
            if m != 0 {
                *sv = v;
            }
        }
        soft
    };

    // Transpose C-order → F-order for mcubes.
    let mut density: Vec<f32> = vec![0.0; nx * ny * nz];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let cidx = i * ny * nz + j * nz + k;
                let mc_flat = i + j * nx + k * nx * ny;
                density[mc_flat] = scalar_c_order[cidx];
            }
        }
    }

    let mc = MarchingCubes::new(
        (nx, ny, nz),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        LVec3::new(0.0, 0.0, 0.0),
        density,
        field.iso,
    )
    .ok()?;
    let mesh = mc.generate(MeshSide::OutsideOnly);
    if mesh.indices.is_empty() || mesh.vertices.is_empty() {
        return None;
    }

    // mcubes returns a vertex *soup* — each triangle gets its own three
    // vertices even when neighbouring triangles share an edge crossing. Weld
    // coincident vertices into a manifold mesh first; this is essential so
    // that the 1-ring vertex graph has the actual surface neighbourhood
    // (otherwise each vertex's 1-ring is just its triangle-mates and Taubin
    // smoothing collapses each triangle to its centroid).
    let raw_vertices: Vec<[f32; 3]> = mesh
        .vertices
        .iter()
        .map(|v| [v.posit.x, v.posit.y, v.posit.z])
        .collect();
    let raw_triangles: Vec<[u32; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
        .collect();
    let (welded_voxel_vertices, mut triangles) = weld_vertices(&raw_vertices, &raw_triangles);

    // Voxel-index → RAS+ mm via the supplied affine. Apply a deterministic
    // ~0.01 mm jitter per vertex so downstream KD-trees can bucket-split.
    let mut vertices: Vec<[f32; 3]> = welded_voxel_vertices
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut p = ras_from_voxel(
                [v[0] as f64, v[1] as f64, v[2] as f64],
                &field.voxel_to_ras,
            );
            let j = jitter3(i);
            p[0] += j[0];
            p[1] += j[1];
            p[2] += j[2];
            p
        })
        .collect();

    // Mesh-level CC cleanup: drop small connected components (also catches
    // anything the voxel-CC pass let slip through MC's surface extraction).
    if params.min_mesh_vertices > 1 {
        let (labels, counts) = mesh_cc_label(&triangles, vertices.len());
        // Keep components >= threshold.
        let mut keep_label = vec![false; counts.len()];
        for (id, &cnt) in counts.iter().enumerate() {
            if id != 0 && cnt >= params.min_mesh_vertices {
                keep_label[id] = true;
            }
        }
        let keep: Vec<bool> = labels
            .iter()
            .map(|&l| (l as usize) < keep_label.len() && keep_label[l as usize])
            .collect();
        // Remap vertices and filter triangles.
        let mut new_idx = vec![u32::MAX; vertices.len()];
        let mut new_vertices: Vec<[f32; 3]> = Vec::new();
        for (i, v) in vertices.iter().enumerate() {
            if keep[i] {
                new_idx[i] = new_vertices.len() as u32;
                new_vertices.push(*v);
            }
        }
        let new_triangles: Vec<[u32; 3]> = triangles
            .iter()
            .filter_map(|t| {
                let a = new_idx[t[0] as usize];
                let b = new_idx[t[1] as usize];
                let c = new_idx[t[2] as usize];
                if a == u32::MAX || b == u32::MAX || c == u32::MAX {
                    None
                } else {
                    Some([a, b, c])
                }
            })
            .collect();
        if new_vertices.is_empty() || new_triangles.is_empty() {
            return None;
        }
        vertices = new_vertices;
        triangles = new_triangles;
    }

    // Taubin smoothing of the mesh — non-shrinking by construction.
    if params.mesh_smooth_iters > 0 {
        taubin_smooth(
            &mut vertices,
            &triangles,
            params.taubin_lambda,
            params.taubin_mu,
            params.mesh_smooth_iters,
        );
    }

    // Compute per-vertex inward normals via the area-weighted face-normal
    // average. mcubes' `MeshSide::OutsideOnly` winds triangles such that the
    // raw cross-product `cross(b−a, c−a)` already points *into* the mask
    // (toward the brain interior) — so we use the result directly without
    // flipping. (Easy to confuse: the cross-product direction depends on
    // mcubes' winding choice, which we verified empirically with a sphere
    // phantom — see `tests/synthetic_surfaces.rs`.)
    let inward_normals = compute_vertex_normals(&vertices, &triangles);

    Some(PseudoSurfaceMesh {
        vertices,
        triangles,
        inward_normals,
    })
}

/// Run marching cubes on the supplied binary mask, then offset each surface
/// vertex along the cross-product face-normal direction to produce a paired
/// pial vertex set.
///
/// **Legacy entry point** — kept for backward compatibility with callers
/// that consume [`PseudoSurfacePair`]. Prefer [`pseudo_surfaces_from_field`]
/// for new code.
///
/// Note on normal direction: mcubes' `MeshSide::OutsideOnly` winds triangles
/// such that `cross(b−a, c−a)` points *into* the mask rather than outward.
/// Historically this routine called the result an "outward normal" and used
/// it to construct the pial offset, which means the constructed pial vertex
/// actually sits *inside* the WM mask. Downstream consumers that compute
/// `wm[i] − pial[i]` to recover an inward direction therefore get the *true
/// outward* direction. New code should use [`pseudo_surfaces_from_field`],
/// which computes proper inward normals and stores them on the resulting
/// [`PseudoSurfaceMesh`].
///
/// Returns `None` if the mask is too small or marching cubes returns no
/// triangles.
pub fn pseudo_surfaces_from_mask(
    mask: &[u8],
    dims: [usize; 3],
    voxel_to_ras: [[f64; 4]; 4],
    params: &PseudoSurfaceParams,
) -> Option<PseudoSurfacePair> {
    let [nx, ny, nz] = dims;
    if nx < 2 || ny < 2 || nz < 2 || mask.len() != nx * ny * nz {
        return None;
    }
    let nonzero = mask.iter().filter(|&&b| b != 0).count();
    if nonzero < params.min_mask_voxels {
        return None;
    }

    // Marching cubes input: 0.0 / 1.0 density grid, isovalue 0.5.
    // **Important**: mcubes uses F-order (`flat = x + y·nx + z·nx·ny`, x
    // fastest), but the ODX mask convention is C-order (`flat = i·ny·nz +
    // j·nz + k`, k fastest). Passing the ODX-layout flat array directly
    // would reinterpret ODX (i,j,k) as mcubes (z,y,x), producing oblique-
    // sheet garbage. Transpose to F-order so mcubes' (x,y,z) cell ==
    // ODX (i,j,k); the voxel→RAS+ affine can then be applied straight.
    let mut density: Vec<f32> = vec![0.0; nx * ny * nz];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let odx_flat = i * ny * nz + j * nz + k;
                let mc_flat = i + j * nx + k * nx * ny;
                if mask[odx_flat] != 0 {
                    density[mc_flat] = 1.0;
                }
            }
        }
    }

    let mc = MarchingCubes::new(
        (nx, ny, nz),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        LVec3::new(0.0, 0.0, 0.0),
        density,
        0.5,
    )
    .ok()?;
    let mesh = mc.generate(MeshSide::OutsideOnly);
    if mesh.indices.is_empty() || mesh.vertices.is_empty() {
        return None;
    }

    // Voxel-index → RAS+ mm via the supplied affine. Apply a deterministic
    // ~0.01 mm jitter per vertex so downstream KD-trees can bucket-split:
    // marching cubes lands many vertices on integer-edge planes, which
    // confuses kiddo's bucket strategy without jitter.
    let wm_vertices: Vec<[f32; 3]> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut p = ras_from_voxel(
                [v.posit.x as f64, v.posit.y as f64, v.posit.z as f64],
                &voxel_to_ras,
            );
            let j = jitter3(i);
            p[0] += j[0];
            p[1] += j[1];
            p[2] += j[2];
            p
        })
        .collect();

    // mcubes emits triangles as a flat Vec<usize>. Group + cast to u32.
    let triangles: Vec<[u32; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
        .collect();

    // Per-vertex outward normal = averaged adjacent face normals, normalised.
    let normals = compute_vertex_normals(&wm_vertices, &triangles);

    let off = params.pial_offset_mm;
    let pial_vertices: Vec<[f32; 3]> = wm_vertices
        .iter()
        .zip(normals.iter())
        .map(|(p, n)| [p[0] + off * n[0], p[1] + off * n[1], p[2] + off * n[2]])
        .collect();

    Some(PseudoSurfacePair {
        wm_vertices,
        pial_vertices,
        triangles,
    })
}

// ===========================================================================
// Hemisphere splitting
// ===========================================================================

/// **New preferred splitter**: split a single MC mesh into LH/RH by labelling
/// hemispheres on an *eroded* copy of the above-iso mask, then propagating
/// labels back over the un-eroded mask and onto the mesh vertices.
///
/// This handles the corpus-callosum problem: the WM mask is normally one big
/// connected component because the CC bridges the hemispheres. Erosion by
/// `erode_iters` 6-conn steps thins the CC until it severs while leaving the
/// hemisphere bulks largely intact. The eroded mask's two largest components
/// become the LH/RH cores. Multi-source BFS from those cores then re-labels
/// every above-iso voxel; vertices inherit hemisphere assignment by
/// nearest-voxel lookup.
///
/// Optional belt-and-braces: if `midline_slab_mm > 0`, voxels in the
/// labelling-only copy whose RAS x-coordinate is within that distance of the
/// AC-PC midline (`x = 0`) are also zeroed before erosion.
///
/// Falls back to [`split_by_x_sign`] semantics (per-triangle x-sign) if fewer
/// than two adequate components survive — likely indicates non-AC-PC-aligned
/// data or already-disconnected hemispheres.
pub fn split_by_eroded_voxel_cc(
    mesh: &PseudoSurfaceMesh,
    field: &WmField,
    erode_iters: u32,
    midline_slab_mm: f32,
) -> PseudoSurfaceMeshHemispheres {
    let dims = field.dims;
    let total = dims[0] * dims[1] * dims[2];

    // 1. Above-iso mask (full).
    let mut above_iso = vec![0_u8; total];
    for (m, &v) in above_iso.iter_mut().zip(field.field.iter()) {
        if v > field.iso {
            *m = 1;
        }
    }

    // 2. Labelling-only copy.
    let mut labelling = above_iso.clone();

    // Optional midline slab.
    if midline_slab_mm > 0.0 {
        let inv = invert_affine(&field.voxel_to_ras);
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                for k in 0..dims[2] {
                    let flat = i * dims[1] * dims[2] + j * dims[2] + k;
                    if labelling[flat] == 0 {
                        continue;
                    }
                    // Voxel centre in RAS+.
                    let voxel = [i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5];
                    let _ = inv; // not actually needed — we have voxel_to_ras directly
                    let x_ras = field.voxel_to_ras[0][0] * voxel[0]
                        + field.voxel_to_ras[0][1] * voxel[1]
                        + field.voxel_to_ras[0][2] * voxel[2]
                        + field.voxel_to_ras[0][3];
                    if x_ras.abs() < midline_slab_mm as f64 {
                        labelling[flat] = 0;
                    }
                }
            }
        }
    }

    // 3. Erode the labelling copy.
    erode_6conn(&mut labelling, dims, erode_iters);

    // 4. Voxel CC on the eroded mask.
    let (eroded_labels, eroded_counts) = voxel_cc_label_6conn(&labelling, dims);
    // `eroded_counts[0]` is background; foreground IDs run 1..N.
    // Sort foreground IDs by size descending, take top 2.
    let mut foreground: Vec<(u32, usize)> = (1..eroded_counts.len() as u32)
        .map(|id| (id, eroded_counts[id as usize]))
        .collect();
    foreground.sort_by(|a, b| b.1.cmp(&a.1));

    // Validation: need at least 2 components, and the 2nd-largest must be at
    // least 5% of the largest (otherwise it's probably noise, not a real
    // hemisphere). Falls back to x-sign in those cases.
    let use_cc = foreground.len() >= 2
        && foreground[1].1 as f64 >= 0.05 * foreground[0].1 as f64;

    let (lh_label, rh_label, vertex_hemi): (u32, u32, Vec<u8>) = if use_cc {
        let id_a = foreground[0].0;
        let id_b = foreground[1].0;
        // Build seeds: id_a → 1, id_b → 2, others → 0.
        let mut seeds = vec![0_u32; total];
        for (idx, &lbl) in eroded_labels.iter().enumerate() {
            if lbl == id_a {
                seeds[idx] = 1;
            } else if lbl == id_b {
                seeds[idx] = 2;
            }
        }
        // 5. Propagate labels back over the un-eroded mask.
        let propagated = propagate_labels_bfs(&seeds, &above_iso, dims);

        // 6. Decide which is LH (mean x_RAS < 0) vs RH by mean x of each
        //    component's voxels.
        let mut sum_x: [f64; 3] = [0.0; 3]; // index 0 unused
        let mut count: [usize; 3] = [0; 3];
        for (flat, &lbl) in propagated.iter().enumerate() {
            if lbl == 0 {
                continue;
            }
            let i = flat / (dims[1] * dims[2]);
            let rem = flat % (dims[1] * dims[2]);
            let j = rem / dims[2];
            let k = rem % dims[2];
            let x_ras = field.voxel_to_ras[0][0] * (i as f64 + 0.5)
                + field.voxel_to_ras[0][1] * (j as f64 + 0.5)
                + field.voxel_to_ras[0][2] * (k as f64 + 0.5)
                + field.voxel_to_ras[0][3];
            let li = lbl as usize;
            sum_x[li] += x_ras;
            count[li] += 1;
        }
        let mean_x_1 = if count[1] > 0 {
            sum_x[1] / count[1] as f64
        } else {
            0.0
        };
        let mean_x_2 = if count[2] > 0 {
            sum_x[2] / count[2] as f64
        } else {
            0.0
        };
        let (lh_label, rh_label) = if mean_x_1 < mean_x_2 { (1, 2) } else { (2, 1) };

        // 7. Per-vertex hemisphere assignment via nearest voxel.
        let inv = invert_affine(&field.voxel_to_ras);
        let mut vertex_hemi = vec![0_u8; mesh.vertices.len()];
        for (vi, v) in mesh.vertices.iter().enumerate() {
            let voxel = ras_to_voxel(*v, &inv);
            let i = (voxel[0].round() as isize).clamp(0, dims[0] as isize - 1) as usize;
            let j = (voxel[1].round() as isize).clamp(0, dims[1] as isize - 1) as usize;
            let k = (voxel[2].round() as isize).clamp(0, dims[2] as isize - 1) as usize;
            let flat = i * dims[1] * dims[2] + j * dims[2] + k;
            let mut lbl = propagated[flat];
            if lbl == 0 {
                // Vertex's voxel isn't in the propagated mask — fall back to
                // a small 3x3x3 search.
                'outer: for di in -1..=1 {
                    for dj in -1..=1 {
                        for dk in -1..=1 {
                            let ni = i as isize + di;
                            let nj = j as isize + dj;
                            let nk = k as isize + dk;
                            if ni < 0
                                || nj < 0
                                || nk < 0
                                || ni >= dims[0] as isize
                                || nj >= dims[1] as isize
                                || nk >= dims[2] as isize
                            {
                                continue;
                            }
                            let nf = (ni as usize) * dims[1] * dims[2]
                                + (nj as usize) * dims[2]
                                + (nk as usize);
                            if propagated[nf] != 0 {
                                lbl = propagated[nf];
                                break 'outer;
                            }
                        }
                    }
                }
            }
            // Final fallback: assign by sign of x_RAS.
            if lbl == 0 {
                lbl = if v[0] < 0.0 { lh_label } else { rh_label };
            }
            vertex_hemi[vi] = if lbl == lh_label {
                1
            } else if lbl == rh_label {
                2
            } else {
                0
            };
        }

        (lh_label, rh_label, vertex_hemi)
    } else {
        // Fallback: per-vertex x-sign.
        let vertex_hemi: Vec<u8> = mesh
            .vertices
            .iter()
            .map(|v| if v[0] < 0.0 { 1 } else { 2 })
            .collect();
        (1, 2, vertex_hemi)
    };

    let _ = (lh_label, rh_label); // already encoded in vertex_hemi as 1/2

    // 8. Build per-hemisphere submeshes. Triangles assigned by majority of
    //    their 3 vertex labels. No triangles dropped.
    let lh = subset_mesh(mesh, &vertex_hemi, 1);
    let rh = subset_mesh(mesh, &vertex_hemi, 2);
    PseudoSurfaceMeshHemispheres { lh, rh }
}

/// Split a pseudo-surface pair into hemispheres by `sign(x_RAS)`. Triangles
/// that span the midline (one vertex < 0 and another > 0) are dropped from
/// both hemispheres; their vertices remain in the global pool. This is
/// approximate — fine for ACPC-aligned data, may need adjustment for
/// non-aligned scans.
pub fn split_by_x_sign(pair: &PseudoSurfacePair) -> PseudoSurfaceHemispheres {
    let n = pair.wm_vertices.len();
    let mut keep_lh = vec![false; n];
    let mut keep_rh = vec![false; n];
    let mut tri_lh: Vec<[u32; 3]> = Vec::new();
    let mut tri_rh: Vec<[u32; 3]> = Vec::new();
    for &t in &pair.triangles {
        let [a, b, c] = t;
        let xa = pair.wm_vertices[a as usize][0];
        let xb = pair.wm_vertices[b as usize][0];
        let xc = pair.wm_vertices[c as usize][0];
        if xa < 0.0 && xb < 0.0 && xc < 0.0 {
            keep_lh[a as usize] = true;
            keep_lh[b as usize] = true;
            keep_lh[c as usize] = true;
            tri_lh.push(t);
        } else if xa >= 0.0 && xb >= 0.0 && xc >= 0.0 {
            keep_rh[a as usize] = true;
            keep_rh[b as usize] = true;
            keep_rh[c as usize] = true;
            tri_rh.push(t);
        }
        // Mixed-sign triangles dropped.
    }
    PseudoSurfaceHemispheres {
        lh: subset(pair, &keep_lh, &tri_lh),
        rh: subset(pair, &keep_rh, &tri_rh),
    }
}

// ---------------------------------------------------------------------------

fn subset(
    full: &PseudoSurfacePair,
    keep: &[bool],
    tris: &[[u32; 3]],
) -> PseudoSurfacePair {
    // old vertex id -> new id (or u32::MAX if dropped).
    let mut remap = vec![u32::MAX; keep.len()];
    let mut wm = Vec::new();
    let mut pial = Vec::new();
    for i in 0..keep.len() {
        if keep[i] {
            remap[i] = wm.len() as u32;
            wm.push(full.wm_vertices[i]);
            pial.push(full.pial_vertices[i]);
        }
    }
    let triangles: Vec<[u32; 3]> = tris
        .iter()
        .map(|t| [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]])
        .collect();
    PseudoSurfacePair {
        wm_vertices: wm,
        pial_vertices: pial,
        triangles,
    }
}

/// Build a sub-mesh by keeping only vertices labelled `target` plus any
/// triangles whose majority of vertices share that label. Mixed-label
/// triangles are assigned to whichever side holds 2 of 3 vertices.
fn subset_mesh(full: &PseudoSurfaceMesh, vertex_hemi: &[u8], target: u8) -> PseudoSurfaceMesh {
    let n = full.vertices.len();
    // First pass: figure out which triangles belong to this hemisphere
    // (majority of vertices have label == target). Track which vertices that
    // pulls in.
    let mut tris_hemi: Vec<[u32; 3]> = Vec::new();
    let mut keep = vec![false; n];
    for &t in &full.triangles {
        let la = vertex_hemi[t[0] as usize];
        let lb = vertex_hemi[t[1] as usize];
        let lc = vertex_hemi[t[2] as usize];
        let count = (la == target) as u8 + (lb == target) as u8 + (lc == target) as u8;
        if count >= 2 {
            tris_hemi.push(t);
            keep[t[0] as usize] = true;
            keep[t[1] as usize] = true;
            keep[t[2] as usize] = true;
        }
    }
    // Compact vertices.
    let mut remap = vec![u32::MAX; n];
    let mut new_verts: Vec<[f32; 3]> = Vec::new();
    let mut new_norms: Vec<[f32; 3]> = Vec::new();
    for i in 0..n {
        if keep[i] {
            remap[i] = new_verts.len() as u32;
            new_verts.push(full.vertices[i]);
            new_norms.push(full.inward_normals[i]);
        }
    }
    let new_tris: Vec<[u32; 3]> = tris_hemi
        .iter()
        .map(|t| [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]])
        .collect();
    PseudoSurfaceMesh {
        vertices: new_verts,
        triangles: new_tris,
        inward_normals: new_norms,
    }
}

/// Weld vertices with bit-identical f32 coordinates. mcubes returns a
/// "vertex soup" — each triangle has its own three vertices even when
/// neighbouring triangles share an edge crossing — and naïvely treating that
/// soup as a triangle mesh gives each vertex a 1-ring of just its
/// triangle-mates, which collapses Taubin smoothing onto the triangle
/// centroid. Welding by bit pattern (f32 → u32 via `to_bits`) is exact for
/// mcubes' output because the same edge crossing always produces identical
/// f32 bits.
fn weld_vertices(
    vertices: &[[f32; 3]],
    triangles: &[[u32; 3]],
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    use std::collections::HashMap;
    let mut canonical: HashMap<[u32; 3], u32> = HashMap::new();
    let mut remap: Vec<u32> = Vec::with_capacity(vertices.len());
    let mut new_vertices: Vec<[f32; 3]> = Vec::new();
    for v in vertices {
        let key = [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
        let id = match canonical.get(&key) {
            Some(&id) => id,
            None => {
                let id = new_vertices.len() as u32;
                canonical.insert(key, id);
                new_vertices.push(*v);
                id
            }
        };
        remap.push(id);
    }
    let new_triangles: Vec<[u32; 3]> = triangles
        .iter()
        .filter_map(|t| {
            let a = remap[t[0] as usize];
            let b = remap[t[1] as usize];
            let c = remap[t[2] as usize];
            // Drop any triangle that became degenerate after welding (two of
            // its vertices coalesced to the same canonical id).
            if a == b || b == c || a == c {
                None
            } else {
                Some([a, b, c])
            }
        })
        .collect();
    (new_vertices, new_triangles)
}

/// Deterministic ~0.01 mm jitter from a vertex index. Stable across runs so
/// the same input gives the same surface every time.
#[inline]
fn jitter3(i: usize) -> [f32; 3] {
    // Three independent splittable hashes; then map to ±0.01 mm.
    let h = i as u64;
    let a = ((h.wrapping_mul(2654435761) >> 17) & 0xFFFF) as f32 / 65536.0 - 0.5;
    let b = ((h.wrapping_mul(40503) >> 13) & 0xFFFF) as f32 / 65536.0 - 0.5;
    let c = ((h.wrapping_mul(2246822519) >> 11) & 0xFFFF) as f32 / 65536.0 - 0.5;
    [a * 0.02, b * 0.02, c * 0.02]
}

fn ras_from_voxel(p: [f64; 3], aff: &[[f64; 4]; 4]) -> [f32; 3] {
    [
        (aff[0][0] * p[0] + aff[0][1] * p[1] + aff[0][2] * p[2] + aff[0][3]) as f32,
        (aff[1][0] * p[0] + aff[1][1] * p[1] + aff[1][2] * p[2] + aff[1][3]) as f32,
        (aff[2][0] * p[0] + aff[2][1] * p[1] + aff[2][2] * p[2] + aff[2][3]) as f32,
    ]
}

/// Apply `aff_inv` (a 4×4 RAS→voxel matrix) to a RAS+ point, returning the
/// fractional voxel coordinate. Used for nearest-voxel lookup of mesh
/// vertices during hemisphere assignment.
fn ras_to_voxel(p: [f32; 3], aff_inv: &[[f64; 4]; 4]) -> [f64; 3] {
    let pf = [p[0] as f64, p[1] as f64, p[2] as f64];
    [
        aff_inv[0][0] * pf[0] + aff_inv[0][1] * pf[1] + aff_inv[0][2] * pf[2] + aff_inv[0][3],
        aff_inv[1][0] * pf[0] + aff_inv[1][1] * pf[1] + aff_inv[1][2] * pf[2] + aff_inv[1][3],
        aff_inv[2][0] * pf[0] + aff_inv[2][1] * pf[1] + aff_inv[2][2] * pf[2] + aff_inv[2][3],
    ]
}

/// Invert a 4×4 affine. Built specifically for voxel→RAS+ matrices, which
/// are diag(scale)·rotation + translation; nalgebra is overkill so we use a
/// minimal in-line implementation.
fn invert_affine(aff: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let m = nalgebra::Matrix4::new(
        aff[0][0], aff[0][1], aff[0][2], aff[0][3],
        aff[1][0], aff[1][1], aff[1][2], aff[1][3],
        aff[2][0], aff[2][1], aff[2][2], aff[2][3],
        aff[3][0], aff[3][1], aff[3][2], aff[3][3],
    );
    let inv = m.try_inverse().unwrap_or_else(nalgebra::Matrix4::identity);
    [
        [inv[(0, 0)], inv[(0, 1)], inv[(0, 2)], inv[(0, 3)]],
        [inv[(1, 0)], inv[(1, 1)], inv[(1, 2)], inv[(1, 3)]],
        [inv[(2, 0)], inv[(2, 1)], inv[(2, 2)], inv[(2, 3)]],
        [inv[(3, 0)], inv[(3, 1)], inv[(3, 2)], inv[(3, 3)]],
    ]
}

/// Per-vertex outward unit normal: average adjacent face normals, then
/// normalise. Returns a [0, 0, 0] for any vertex with no incident triangle.
fn compute_vertex_normals(
    vertices: &[[f32; 3]],
    triangles: &[[u32; 3]],
) -> Vec<[f32; 3]> {
    let mut acc = vec![[0.0_f32; 3]; vertices.len()];
    for t in triangles {
        let a = vertices[t[0] as usize];
        let b = vertices[t[1] as usize];
        let c = vertices[t[2] as usize];
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        // Cross product = unnormalised face normal (weight ∝ 2 × triangle area;
        // larger faces contribute more, which is the standard area-weighted
        // smoothing behaviour we want).
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for &vi in t.iter() {
            acc[vi as usize][0] += n[0];
            acc[vi as usize][1] += n[1];
            acc[vi as usize][2] += n[2];
        }
    }
    for v in acc.iter_mut() {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 1.0e-9 {
            v[0] /= len;
            v[1] /= len;
            v[2] /= len;
        }
    }
    acc
}

// ===========================================================================
// Shrink-wrap (deformable surface) — start from a smooth balloon and pull
// every vertex inward along its normal until it crosses the target QA iso
// level. Modeled on the active-surface / level-set tradition (FreeSurfer's
// mri_make_surfaces does a more sophisticated version on T1 intensity).
// ===========================================================================

#[derive(Debug, Clone, Copy)]
pub struct ShrinkWrapParams {
    /// Target QA isovalue: vertices stop moving once they reach a voxel
    /// where the smoothed field value crosses this threshold from below.
    /// Typically `factor × Otsu(QA)` with a higher factor than the balloon
    /// initialization (e.g., balloon at 0.5×Otsu, target at 1.25×Otsu).
    pub target_iso: f32,
    /// Maximum number of shrink iterations. Each iteration moves every
    /// vertex inward by up to `step_mm`, then optionally smooths.
    pub n_iters: u32,
    /// Per-iteration inward step (mm). Smaller = stable but more iterations
    /// needed; larger = faster but can overshoot or self-intersect.
    pub step_mm: f32,
    /// Maximum search distance per iteration (mm). The vertex samples the
    /// QA field along its inward normal in steps of `step_mm` up to this
    /// distance, looking for the iso crossing.
    pub max_search_mm: f32,
    /// Apply a Taubin smoothing pair every N iterations to keep the mesh
    /// manifold and prevent vertex bunching. Default 5 (smooth every 5
    /// iters). Set to `n_iters + 1` to disable.
    pub smooth_every: u32,
    /// Taubin λ for the periodic smoothing. Default 0.5.
    pub taubin_lambda: f32,
    /// Taubin μ for the periodic smoothing. Default −0.53.
    pub taubin_mu: f32,
    /// Stop iterating when no vertex has moved by more than this (mm) in
    /// the current iteration. Default 0.05 mm.
    pub convergence_eps_mm: f32,
}

impl Default for ShrinkWrapParams {
    fn default() -> Self {
        Self {
            target_iso: 0.0, // caller should set
            n_iters: 50,
            step_mm: 0.5,
            max_search_mm: 8.0,
            smooth_every: 5,
            taubin_lambda: 0.5,
            taubin_mu: -0.53,
            convergence_eps_mm: 0.05,
        }
    }
}

/// Trilinear interpolation of a flat C-order voxel field at a continuous
/// voxel-index coordinate. Out-of-bounds samples return 0. The 8-corner
/// weights are computed from the fractional part of `voxel`.
fn sample_field_trilinear(field: &[f32], dims: [usize; 3], voxel: [f64; 3]) -> f32 {
    let [nx, ny, nz] = dims;
    let i0 = voxel[0].floor() as i64;
    let j0 = voxel[1].floor() as i64;
    let k0 = voxel[2].floor() as i64;
    let fx = (voxel[0] - i0 as f64) as f32;
    let fy = (voxel[1] - j0 as f64) as f32;
    let fz = (voxel[2] - k0 as f64) as f32;
    let mut acc = 0.0_f32;
    let mut total_w = 0.0_f32;
    for di in 0..2_i64 {
        for dj in 0..2_i64 {
            for dk in 0..2_i64 {
                let ii = i0 + di;
                let jj = j0 + dj;
                let kk = k0 + dk;
                if ii < 0 || jj < 0 || kk < 0 {
                    continue;
                }
                if (ii as usize) >= nx || (jj as usize) >= ny || (kk as usize) >= nz {
                    continue;
                }
                let wx = if di == 0 { 1.0 - fx } else { fx };
                let wy = if dj == 0 { 1.0 - fy } else { fy };
                let wz = if dk == 0 { 1.0 - fz } else { fz };
                let w = wx * wy * wz;
                let flat = (ii as usize) * ny * nz + (jj as usize) * nz + (kk as usize);
                acc += w * field[flat];
                total_w += w;
            }
        }
    }
    if total_w > 1e-9 {
        acc / total_w
    } else {
        0.0
    }
}

/// Iteratively deform a starting surface inward along per-vertex normals
/// until vertices cross the target QA iso level. The starting mesh should
/// be a smooth balloon-like outer surface — the brain-mask outline from a
/// low threshold (e.g., `factor=0.5 × Otsu`) works well. The result inherits
/// the topology of the starting mesh (no holes, no fragments) but follows
/// the QA structure underneath, giving sulcal indentations even where the
/// raw thresholded mask would have produced disconnected pieces.
///
/// Algorithm per iteration:
/// 1. Compute per-vertex inward unit normals from the current mesh winding.
/// 2. For each vertex, sample the QA field at that vertex; if QA ≥ target
///    iso, the vertex is "inside" — don't move (or pull back slightly).
///    Otherwise, march inward in `step_mm` increments up to `max_search_mm`
///    looking for the iso crossing. If found at distance d, move the vertex
///    halfway to that crossing (capped at one full step). If no crossing is
///    found within the search range, take a single step inward anyway —
///    QA-quiet regions get gradually pulled in until the iteration cap.
/// 3. Every `smooth_every` iterations, apply a single Taubin λ/μ pair to
///    keep the mesh manifold and prevent vertex bunching at high-curvature
///    regions.
/// 4. Stop early if the largest vertex displacement in the iteration falls
///    below `convergence_eps_mm`.
pub fn shrink_wrap(
    initial: &PseudoSurfaceMesh,
    field: &WmField,
    params: &ShrinkWrapParams,
) -> PseudoSurfaceMesh {
    let mut vertices = initial.vertices.clone();
    let triangles = initial.triangles.clone();
    let world_to_voxel = invert_affine(&field.voxel_to_ras);
    let n_search_steps = (params.max_search_mm / params.step_mm).ceil() as i32;

    eprintln!(
        "  shrink_wrap: target_iso={:.4}, n_iters={}, step={}mm, max_search={}mm, smooth_every={}",
        params.target_iso,
        params.n_iters,
        params.step_mm,
        params.max_search_mm,
        params.smooth_every,
    );

    for iter in 0..params.n_iters {
        // mcubes' OutsideOnly winding has cross(b-a, c-a) pointing INWARD,
        // so `compute_vertex_normals` already gives inward unit vectors —
        // no negation needed.
        let inward_normals = compute_vertex_normals(&vertices, &triangles);

        let mut max_disp: f32 = 0.0;
        let mut moved_count: usize = 0;
        let new_positions: Vec<[f32; 3]> = vertices
            .iter()
            .zip(inward_normals.iter())
            .map(|(v, n)| {
                // Skip if normal is degenerate (zero-area vertex).
                let nlen2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
                if nlen2 < 1e-6 {
                    return *v;
                }

                // Sample current QA at vertex.
                let voxel0 = ras_to_voxel(*v, &world_to_voxel);
                let cur_qa = sample_field_trilinear(&field.field, field.dims, voxel0);

                if cur_qa >= params.target_iso {
                    // Already inside. Hold position (could pull back if we
                    // wanted to track the iso surface from the inside, but
                    // simple stop is more stable).
                    return *v;
                }

                // March inward looking for the crossing.
                for s in 1..=n_search_steps {
                    let t = s as f32 * params.step_mm;
                    let p = [v[0] + t * n[0], v[1] + t * n[1], v[2] + t * n[2]];
                    let voxel = ras_to_voxel(p, &world_to_voxel);
                    let qa = sample_field_trilinear(&field.field, field.dims, voxel);
                    if qa >= params.target_iso {
                        // Move halfway toward the crossing, capped at one
                        // full step (don't jump too far in a single iter).
                        let move_dist = (t * 0.5).min(params.step_mm).max(params.step_mm * 0.25);
                        return [
                            v[0] + move_dist * n[0],
                            v[1] + move_dist * n[1],
                            v[2] + move_dist * n[2],
                        ];
                    }
                }

                // No crossing within search range. **Don't move** — the
                // vertex is in a region with no nearby WM (e.g., a true
                // gyrus crown that already sits at the WM/GM boundary, or
                // a region of the balloon that overhangs CSF/air with no
                // WM beneath). Marching inward anyway would tear the
                // surface and produce the spike artifacts we want to
                // avoid. The Taubin smoothing pass will gently relax this
                // vertex toward its neighbours' positions.
                *v
            })
            .collect();

        for (a, b) in vertices.iter().zip(new_positions.iter()) {
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let dz = b[2] - a[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > max_disp {
                max_disp = d;
            }
            if d > 1e-6 {
                moved_count += 1;
            }
        }

        vertices = new_positions;

        if (iter + 1) % params.smooth_every == 0 {
            taubin_smooth(
                &mut vertices,
                &triangles,
                params.taubin_lambda,
                params.taubin_mu,
                1,
            );
        }

        if (iter + 1) % 10 == 0 || iter + 1 == params.n_iters {
            eprintln!(
                "    iter {:>3}/{}: {} verts moved, max_disp = {:.3} mm",
                iter + 1,
                params.n_iters,
                moved_count,
                max_disp
            );
        }

        if max_disp < params.convergence_eps_mm {
            eprintln!("    converged at iter {}", iter + 1);
            break;
        }
    }

    let inward_normals = compute_vertex_normals(&vertices, &triangles);
    PseudoSurfaceMesh {
        vertices,
        triangles,
        inward_normals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mask_returns_none() {
        let dims = [4, 4, 4];
        let mask = vec![0_u8; 64];
        let params = PseudoSurfaceParams::default();
        assert!(pseudo_surfaces_from_mask(&mask, dims, identity_affine(), &params).is_none());
    }

    #[test]
    fn solid_cube_yields_a_mesh() {
        let dims = [10, 10, 10];
        let mut mask = vec![0_u8; 1000];
        for x in 2..8 {
            for y in 2..8 {
                for z in 2..8 {
                    mask[x * 100 + y * 10 + z] = 1;
                }
            }
        }
        let params = PseudoSurfaceParams {
            min_mask_voxels: 10,
            ..PseudoSurfaceParams::default()
        };
        let pair = pseudo_surfaces_from_mask(&mask, dims, identity_affine(), &params)
            .expect("solid cube should produce a mesh");
        assert!(!pair.triangles.is_empty());
        assert_eq!(pair.wm_vertices.len(), pair.pial_vertices.len());
        // Pial should be offset outward — for at least one vertex, pial != wm.
        assert!(pair
            .wm_vertices
            .iter()
            .zip(pair.pial_vertices.iter())
            .any(|(w, p)| w != p));
    }

    fn identity_affine() -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}
