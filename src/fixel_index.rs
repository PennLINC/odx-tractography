//! Spatial index over the fixels in an [`OdxDataset`].
//!
//! Each peak in the ODX (one entry in `directions[]`) is one fixel here. The
//! fixel's world position is the parent voxel's RAS+ mm center; its
//! direction is the unit vector from the ODX directions array. We keep a
//! KD-tree over positions for `nearest_within` queries plus a flat
//! per-voxel bucket so callers can iterate fixels-of-a-voxel without
//! re-querying the tree.

use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use odx_rs::OdxDataset;

/// Sub-voxel offset (mm) applied along each fixel's direction so coincident
/// peaks within a voxel become distinct points in the spatial index. Chosen
/// well below typical voxel resolution (~1.5 mm) so radius queries at
/// typical scales are unaffected.
const POSITION_DIR_OFFSET: f32 = 0.01;

/// Maximum magnitude (mm) of the per-fixel deterministic jitter. The ODX
/// voxel grid shares axis-aligned coordinates across many voxels, which
/// causes the kd-tree to fail to split bucket leaves. A microscopic jitter
/// per fixel breaks the ties without affecting radius queries at any
/// reasonable scale.
const JITTER_MAX_MM: f32 = 0.005;

/// Deterministic micro-offset from a fixel id. Returns three independent
/// jitter components in `[-JITTER_MAX_MM, +JITTER_MAX_MM]`. Pure function
/// of the id, so the same ODX yields the same spatial layout across runs.
fn jitter_for(id: u32) -> [f32; 3] {
    // Three different multiplicative hashes mod a prime, then scale into a
    // signed range. Using distinct constants per axis avoids correlated
    // jitter that would still leave plane-aligned coincidences.
    let h = |seed: u32| -> f32 {
        let mut x = id.wrapping_mul(seed).wrapping_add(0x9E37_79B9);
        x ^= x >> 16;
        x = x.wrapping_mul(0x85EB_CA6B);
        x ^= x >> 13;
        // Map u32 -> (-1.0, +1.0) approximately uniformly.
        ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
    };
    [
        h(0x1F1F_1F1F) * JITTER_MAX_MM,
        h(0x2A2A_2A2A) * JITTER_MAX_MM,
        h(0x3B3B_3B3B) * JITTER_MAX_MM,
    ]
}

/// Index of a fixel within the global `directions[]` array.
pub type FixelId = u32;

/// All the data a caller typically needs about a fixel: its global id,
/// world-space position, unit direction, and the compact (masked) voxel
/// index it belongs to.
#[derive(Debug, Clone, Copy)]
pub struct FixelHandle {
    pub id: FixelId,
    pub world_pos: [f32; 3],
    pub dir: [f32; 3],
    pub voxel_idx: u32,
    /// QA/amplitude (= peak − min_odf, per ODX SPECIFICATION). Higher
    /// = more confident the fixel really represents a fiber. Used by
    /// PTT data support to weight contributions: low-QA fixels in
    /// noisy regions contribute less to arc likelihood, so PTT
    /// trajectories don't drift into low-confidence areas (e.g.,
    /// boundary voxels between bundles where one might otherwise
    /// pass through phantom low-FA fixels).
    /// `f32::NAN` if `dpf/amplitude` was not available at build time.
    pub amplitude: f32,
}

/// KD-tree over fixel world positions, plus per-fixel metadata.
///
/// Fixels can be filtered at build time (e.g. Otsu on amplitude). Kept
/// fixels live in `handles` in insertion order; `id_to_local` maps the
/// global ODX peak index to that position. A `u32::MAX` entry means the
/// fixel was dropped by the filter and is not in the tree.
pub struct FixelIndex {
    handles: Vec<FixelHandle>,
    id_to_local: Vec<u32>,
    /// Bucketed kd-tree. The bucket size is generous because noisy voxels
    /// can have many fixels (5+ in random directions); after sub-voxel
    /// jitter they still cluster tightly, so a small bucket leaves
    /// kiddo unable to split.
    tree: KdTree<f32, FixelId, 3, KDTREE_BUCKET, u32>,
    nb_peaks: u32,
}

const KDTREE_BUCKET: usize = 256;

/// Divide each handle's `amplitude` by the median of all finite
/// amplitudes, so the average kept fixel gets weight ~1.0. Handles
/// with NaN amplitude are left as NaN (PTT consumers fall back to
/// weight 1.0). If median is 0 or NaN (no amplitude data), this is
/// a no-op.
fn normalize_amplitudes(handles: &mut [FixelHandle]) {
    let mut amps: Vec<f32> = handles
        .iter()
        .filter_map(|h| if h.amplitude.is_finite() { Some(h.amplitude) } else { None })
        .collect();
    if amps.is_empty() {
        return;
    }
    amps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = amps[amps.len() / 2];
    if !median.is_finite() || median <= 0.0 {
        return;
    }
    for h in handles.iter_mut() {
        if h.amplitude.is_finite() {
            h.amplitude /= median;
        }
    }
}

impl FixelIndex {
    /// Build the index from an [`OdxDataset`]. Includes every peak.
    ///
    /// The world position used for spatial queries is
    /// `voxel_center + POSITION_DIR_OFFSET * dir`, i.e. the "head" of the
    /// peak rather than the voxel center. This disambiguates fixels that
    /// share a voxel (without it the KD-tree refuses coincident points)
    /// while remaining well below a voxel's resolution, so radius queries
    /// at typical scales (≥0.5 mm) behave identically.
    pub fn build(odx: &OdxDataset) -> Self {
        Self::build_filtered(odx, |_id, _amp| true)
    }

    /// Build the index keeping only fixels for which `keep(id, amplitude)`
    /// returns true. Use this to drop noise via an amplitude or QC-class
    /// threshold (e.g. Otsu on `dpf/amplitude`). The `FixelId` returned
    /// is still the global peak index in the underlying ODX, so any DPF
    /// written downstream uses `nb_peaks`-sized arrays with 0 for the
    /// dropped slots.
    ///
    /// `keep` receives the global peak index and the value of
    /// `dpf/amplitude` (or `f32::NAN` if amplitude isn't present).
    pub fn build_filtered<F>(odx: &OdxDataset, mut keep: F) -> Self
    where
        F: FnMut(u32, f32) -> bool,
    {
        let voxel_centers = odx.mask_voxel_centers_ras();
        let directions = odx.directions();
        let offsets = odx.offsets();
        let nb_voxels = odx.nb_voxels();
        let nb_peaks = directions.len();
        let amplitudes = odx
            .scalar_dpf_f32("amplitude")
            .ok()
            .filter(|v| v.len() == nb_peaks);

        let mut handles: Vec<FixelHandle> = Vec::with_capacity(nb_peaks);
        let mut id_to_local: Vec<u32> = vec![u32::MAX; nb_peaks];
        let mut tree: KdTree<f32, FixelId, 3, KDTREE_BUCKET, u32> =
            KdTree::with_capacity(nb_peaks);

        for v in 0..nb_voxels {
            let center = voxel_centers[v];
            let start = offsets[v] as usize;
            let end = offsets[v + 1] as usize;
            for global_idx in start..end {
                let id = global_idx as FixelId;
                let amp = amplitudes
                    .as_ref()
                    .map(|a| a[global_idx])
                    .unwrap_or(f32::NAN);
                if !keep(id, amp) {
                    continue;
                }
                let dir = directions[global_idx];
                let j = jitter_for(id);
                let pos = [
                    center[0] + POSITION_DIR_OFFSET * dir[0] + j[0],
                    center[1] + POSITION_DIR_OFFSET * dir[1] + j[1],
                    center[2] + POSITION_DIR_OFFSET * dir[2] + j[2],
                ];
                let local = handles.len() as u32;
                id_to_local[global_idx] = local;
                handles.push(FixelHandle {
                    id,
                    world_pos: pos,
                    dir,
                    voxel_idx: v as u32,
                    amplitude: amp,
                });
                tree.add(&pos, id);
            }
        }

        // Normalize amplitude by its median so the average kept fixel
        // gets weight 1.0. Keeps PTT data-support thresholds stable
        // across datasets with different amplitude scales while still
        // down-weighting low-FA fixels (and up-weighting high-QA ones).
        normalize_amplitudes(&mut handles);

        Self {
            handles,
            id_to_local,
            tree,
            nb_peaks: nb_peaks as u32,
        }
    }

    /// Build the index directly from a list of fixel positions and
    /// unit directions, bypassing any [`OdxDataset`]. Used for
    /// **synthetic** fixel fields — e.g. one fixel per u-fiber
    /// parabola apex, direction set to the along-fundus axis — so the
    /// same PTT engine that traces streamlines can be propagated over
    /// a derived field.
    ///
    /// `id` is the array index; every entry is its own synthetic
    /// voxel. A deterministic sub-resolution [`jitter_for`] offset is
    /// added to each position so coincident inputs don't collapse the
    /// KD-tree. `amplitudes` (optional, parallel to `positions`)
    /// becomes the per-fixel PTT weight (median-normalised, like the
    /// ODX path); pass `None` for uniform weight. Directions are
    /// normalised here; a zero-length direction is kept as-is.
    pub fn from_handles(
        positions: &[[f32; 3]],
        directions: &[[f32; 3]],
        amplitudes: Option<&[f32]>,
    ) -> Self {
        assert_eq!(
            positions.len(),
            directions.len(),
            "from_handles: positions/directions length mismatch"
        );
        if let Some(a) = amplitudes {
            assert_eq!(
                a.len(),
                positions.len(),
                "from_handles: amplitudes/positions length mismatch"
            );
        }
        let nb_peaks = positions.len();
        let mut handles: Vec<FixelHandle> = Vec::with_capacity(nb_peaks);
        let id_to_local: Vec<u32> = (0..nb_peaks as u32).collect();
        let mut tree: KdTree<f32, FixelId, 3, KDTREE_BUCKET, u32> =
            KdTree::with_capacity(nb_peaks);

        for i in 0..nb_peaks {
            let id = i as FixelId;
            let d = directions[i];
            let dn = {
                let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if n > 1e-9 {
                    [d[0] / n, d[1] / n, d[2] / n]
                } else {
                    d
                }
            };
            let j = jitter_for(id);
            let pos = [
                positions[i][0] + j[0],
                positions[i][1] + j[1],
                positions[i][2] + j[2],
            ];
            let amp = amplitudes.map(|a| a[i]).unwrap_or(f32::NAN);
            handles.push(FixelHandle {
                id,
                world_pos: pos,
                dir: dn,
                voxel_idx: id,
                amplitude: amp,
            });
            tree.add(&pos, id);
        }
        normalize_amplitudes(&mut handles);
        Self {
            handles,
            id_to_local,
            tree,
            nb_peaks: nb_peaks as u32,
        }
    }

    /// Convenience: build with an Otsu threshold on `dpf/amplitude`,
    /// optionally scaled by `scale` (e.g. 0.5 for a less aggressive
    /// cut). Returns the index plus the threshold value used. If
    /// amplitude isn't present in the ODX, falls back to no filtering.
    pub fn build_otsu(odx: &OdxDataset, scale: f32) -> (Self, Option<f32>) {
        let threshold = odx_rs::qc::compute_fixel_otsu(
            odx,
            Some("amplitude"),
            odx_rs::qc::OtsuScope::AllFixels,
        )
        .ok()
        .map(|o| o.threshold * scale);
        let idx = match threshold {
            Some(t) => Self::build_filtered(odx, |_id, amp| amp.is_finite() && amp >= t),
            None => Self::build(odx),
        };
        (idx, threshold)
    }

    /// Convenience: build with a hard amplitude threshold (any unit
    /// matching `dpf/amplitude`). Drops fixels below `threshold`. If
    /// amplitude isn't present, falls back to no filtering.
    pub fn build_threshold(odx: &OdxDataset, threshold: f32) -> Self {
        Self::build_filtered(odx, |_id, amp| !amp.is_finite() || amp >= threshold)
    }

    /// Build the index keeping only fixels whose parent voxel passes
    /// `keep_voxel(voxel_idx)`. Used for per-voxel scalar filters
    /// (QA/FA/GFA threshold) — the DSI-Studio convention is to apply
    /// the threshold per-voxel, dropping ALL fixels in low-anisotropy
    /// voxels rather than per-fixel filtering on amplitude.
    pub fn build_voxel_filtered<F>(odx: &OdxDataset, mut keep_voxel: F) -> Self
    where
        F: FnMut(u32) -> bool,
    {
        let voxel_centers = odx.mask_voxel_centers_ras();
        let directions = odx.directions();
        let offsets = odx.offsets();
        let nb_voxels = odx.nb_voxels();
        let nb_peaks = directions.len();
        let amplitudes = odx
            .scalar_dpf_f32("amplitude")
            .ok()
            .filter(|v| v.len() == nb_peaks);

        let mut handles: Vec<FixelHandle> = Vec::with_capacity(nb_peaks);
        let mut id_to_local: Vec<u32> = vec![u32::MAX; nb_peaks];
        let mut tree: KdTree<f32, FixelId, 3, KDTREE_BUCKET, u32> =
            KdTree::with_capacity(nb_peaks);

        for v in 0..nb_voxels {
            if !keep_voxel(v as u32) {
                continue;
            }
            let center = voxel_centers[v];
            let start = offsets[v] as usize;
            let end = offsets[v + 1] as usize;
            for global_idx in start..end {
                let id = global_idx as FixelId;
                let dir = directions[global_idx];
                let amp = amplitudes
                    .as_ref()
                    .map(|a| a[global_idx])
                    .unwrap_or(f32::NAN);
                let j = jitter_for(id);
                let pos = [
                    center[0] + POSITION_DIR_OFFSET * dir[0] + j[0],
                    center[1] + POSITION_DIR_OFFSET * dir[1] + j[1],
                    center[2] + POSITION_DIR_OFFSET * dir[2] + j[2],
                ];
                let local = handles.len() as u32;
                id_to_local[global_idx] = local;
                handles.push(FixelHandle {
                    id,
                    world_pos: pos,
                    dir,
                    voxel_idx: v as u32,
                    amplitude: amp,
                });
                tree.add(&pos, id);
            }
        }
        normalize_amplitudes(&mut handles);
        Self {
            handles,
            id_to_local,
            tree,
            nb_peaks: nb_peaks as u32,
        }
    }

    /// Build with the DSI-Studio QA Otsu filter: drop voxels whose
    /// primary-peak QA (= `dpf/amplitude[offsets[v]]`) is below
    /// `scale × otsu(primary_peak_qa)`.
    ///
    /// This is the exact analog of DSI-Studio tracking's `fa_threshold`
    /// — see [`fib_data.cpp:338`](DSI-Studio/libs/tracking/fib_data.cpp)
    /// (Otsu on `fa[0]`) and [`tracking_thread.cpp:217-218`](DSI-Studio/libs/tracking/tracking_thread.cpp)
    /// (`fa_threshold = (default_otsu ± 0.1) × fa_otsu`, with
    /// `default_otsu = 0.6` so the typical tracking range is
    /// `[0.5×otsu, 0.7×otsu]`). odx-rs stores QA values under the
    /// (slightly misleading) name `dpf/amplitude` — they're already
    /// `peak − min_odf`, normalized so max primary QA = 1.0
    /// ([SPECIFICATION.md:261](odx-rs/SPECIFICATION.md):
    /// "equiv. to DSI Studio fa0..faN").
    ///
    /// Returns `(index, Some(threshold))` on success, or
    /// `(index, None)` if `dpf/amplitude` is unavailable.
    pub fn build_qa_otsu(odx: &OdxDataset, scale: f32) -> (Self, Option<f32>) {
        let info = match odx_rs::qc::compute_fixel_otsu(
            odx,
            Some("amplitude"),
            odx_rs::qc::OtsuScope::PrimaryPeak,
        ) {
            Ok(info) => info,
            Err(_) => return (Self::build(odx), None),
        };
        let threshold = info.threshold * scale;
        // For per-voxel filtering we need the primary peak's QA per
        // masked voxel. Pull from offsets directly — primary peak is
        // amplitude at offsets[v] for each voxel that has any peaks.
        let amplitudes = match odx.scalar_dpf_f32("amplitude") {
            Ok(v) => v,
            Err(_) => return (Self::build(odx), None),
        };
        let offsets = odx.offsets().to_vec();
        let idx = Self::build_voxel_filtered(odx, |voxel_idx| {
            let v = voxel_idx as usize;
            let start = offsets[v] as usize;
            let end = offsets[v + 1] as usize;
            if start >= end {
                return false;
            }
            let qa = amplitudes[start];
            qa.is_finite() && qa >= threshold
        });
        (idx, Some(threshold))
    }

    /// Build with a hard absolute threshold on the primary-peak QA
    /// (= `dpf/amplitude[offsets[v]]`). Drops entire voxels whose
    /// primary-peak QA is below `threshold`.
    pub fn build_qa_threshold(odx: &OdxDataset, threshold: f32) -> Self {
        let amplitudes = match odx.scalar_dpf_f32("amplitude") {
            Ok(v) => v,
            Err(_) => return Self::build(odx),
        };
        let offsets = odx.offsets().to_vec();
        Self::build_voxel_filtered(odx, |voxel_idx| {
            let v = voxel_idx as usize;
            let start = offsets[v] as usize;
            let end = offsets[v + 1] as usize;
            if start >= end {
                return false;
            }
            let qa = amplitudes[start];
            qa.is_finite() && qa >= threshold
        })
    }

    /// Number of fixels in the index (== `odx.nb_peaks()`).
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Look up a fixel handle by global ODX peak id. Panics if the id
    /// was filtered out at build time. Use [`Self::try_handle`] for a
    /// fallible version.
    pub fn handle(&self, id: FixelId) -> &FixelHandle {
        let local = self.id_to_local[id as usize];
        debug_assert!(local != u32::MAX, "fixel {} was filtered out", id);
        &self.handles[local as usize]
    }

    /// Like [`Self::handle`] but returns `None` for filtered-out fixels.
    pub fn try_handle(&self, id: FixelId) -> Option<&FixelHandle> {
        let local = *self.id_to_local.get(id as usize)?;
        if local == u32::MAX {
            None
        } else {
            Some(&self.handles[local as usize])
        }
    }

    /// True if the global peak id survived the build-time filter.
    pub fn contains(&self, id: FixelId) -> bool {
        self.id_to_local
            .get(id as usize)
            .copied()
            .is_some_and(|l| l != u32::MAX)
    }

    /// Slice of all fixel handles in insertion order. Filtered-out
    /// fixels are not present.
    pub fn handles(&self) -> &[FixelHandle] {
        &self.handles
    }

    /// Total number of peaks in the underlying ODX (including those
    /// filtered out at build time). Use this for sizing per-peak arrays.
    pub fn nb_peaks(&self) -> usize {
        self.nb_peaks as usize
    }

    /// Find all fixels whose world position is within `radius` mm of `pos`.
    /// Returns `(FixelId, squared_distance)` pairs.
    pub fn nearest_within(&self, pos: [f32; 3], radius: f32) -> Vec<(FixelId, f32)> {
        let r2 = radius * radius;
        self.tree
            .within_unsorted::<SquaredEuclidean>(&pos, r2)
            .into_iter()
            .map(|n| (n.item, n.distance))
            .collect()
    }

    /// Median of the nearest-fixel-in-a-different-voxel distance. Skips
    /// within-voxel peaks (which sit ~0.01 mm apart by construction). The
    /// returned value is the typical inter-voxel spacing, which is what you
    /// want for tuning growth radii. O(n log n); only call from diagnostics.
    pub fn median_inter_voxel_nn_distance(&self) -> f32 {
        if self.handles.len() < 2 {
            return 0.0;
        }
        let mut dists: Vec<f32> = Vec::with_capacity(self.handles.len());
        for h in &self.handles {
            // Ask for some neighbors and take the closest one in a
            // different voxel. 16 covers most local crossings + the
            // fixel's own voxel.
            let near = self.tree.nearest_n::<SquaredEuclidean>(&h.world_pos, 16);
            for n in &near {
                let other = self.handle(n.item);
                if other.voxel_idx != h.voxel_idx {
                    dists.push(n.distance.sqrt());
                    break;
                }
            }
        }
        if dists.is_empty() {
            return 0.0;
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists[dists.len() / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_handles_normalises_dir_and_queries_spatially() {
        let pos = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let dir = vec![[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let idx = FixelIndex::from_handles(&pos, &dir, None);

        assert_eq!(idx.len(), 3);
        assert_eq!(idx.nb_peaks(), 3);

        // Direction is unit-normalised.
        let h1 = idx.handle(1);
        assert!((h1.dir[1] - 1.0).abs() < 1e-5, "dir not normalised: {:?}", h1.dir);
        // Sub-resolution jitter only (< JITTER_MAX_MM on each axis).
        let h0 = idx.handle(0);
        for a in 0..3 {
            assert!(h0.world_pos[a].abs() <= JITTER_MAX_MM + 1e-6);
        }

        // Within 3 mm of the origin: ids 0 and 1 (2 mm away), not id 2.
        let near: std::collections::HashSet<u32> = idx
            .nearest_within([0.0, 0.0, 0.0], 3.0)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(near.contains(&0));
        assert!(near.contains(&1));
        assert!(!near.contains(&2));
    }

    #[test]
    fn from_handles_amplitude_is_median_normalised() {
        let pos = vec![[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let dir = vec![[1.0, 0.0, 0.0]; 3];
        let amp = [2.0_f32, 4.0, 8.0]; // median 4 → normalised to 0.5/1/2
        let idx = FixelIndex::from_handles(&pos, &dir, Some(&amp));
        assert!((idx.handle(0).amplitude - 0.5).abs() < 1e-5);
        assert!((idx.handle(1).amplitude - 1.0).abs() < 1e-5);
        assert!((idx.handle(2).amplitude - 2.0).abs() < 1e-5);
    }
}
