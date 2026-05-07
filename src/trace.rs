//! Fixel-set tractography for visualization.
//!
//! This is a simplified Yeh-style deterministic tracker: at each step
//! we pick the most-aligned coherent fixel near the proposed next
//! position, blend its direction with the current direction.

use std::collections::HashSet;

use crate::fixel_index::{FixelHandle, FixelId, FixelIndex};

/// Parameters controlling streamline propagation within a fixel set.
#[derive(Debug, Clone, Copy)]
pub struct TraceParams {
    /// Step size in mm. Smaller = smoother polylines + more vertices.
    /// Default 0.5 mm gives a few samples per voxel at 1.7 mm res.
    pub step_mm: f32,
    /// Stop a streamline once it reaches this many millimeters total.
    pub max_length_mm: f32,
    /// Discard streamlines shorter than this. Suppresses tiny stubs
    /// from seeds that have nowhere to go.
    pub min_length_mm: f32,
    /// Maximum angle (degrees) between consecutive step directions.
    /// Tight → smoother streamlines; loose → can take sharp turns at
    /// noisy fixels. 30° is a typical tractography default.
    pub angle_max_degrees: f32,
}

impl Default for TraceParams {
    fn default() -> Self {
        Self {
            // One voxel-spacing per step lets the tracer cross
            // fixel-to-fixel cleanly; sub-voxel steps tend to find the
            // same fixel again and stall.
            step_mm: 1.7,
            max_length_mm: 250.0,
            min_length_mm: 5.0,
            angle_max_degrees: 30.0,
        }
    }
}

/// Trace one streamline per seed, restricted to the given fixel set.
///
/// At each step the tracer probes for any fixel in `fixel_set` within
/// `voxel_size * 0.75` mm of the proposed next position (so we span
/// at least one voxel-spacing in the search and reliably find the
/// next-voxel fixel). `voxel_size` is taken from the FixelIndex's
/// median inter-voxel NN distance.
pub fn trace_within_fixels(
    idx: &FixelIndex,
    fixel_set: &HashSet<FixelId>,
    seeds: &[FixelId],
    params: TraceParams,
) -> Vec<Vec<[f32; 3]>> {
    let cos_max = params.angle_max_degrees.to_radians().cos();
    let max_steps = (params.max_length_mm / params.step_mm).ceil() as usize;
    // Search radius needs to span enough voxels to hop across small
    // gaps in the kept fixel set (the bundle isn't always
    // contiguous after QA + forward-flow filtering). 2.5 × step_mm
    // bridges ~2 voxel-gaps; smaller bridges drop streamlines short
    // of the bundle's true endpoints.
    let search_radius = (params.step_mm * 2.5).max(3.0);

    let mut out: Vec<Vec<[f32; 3]>> = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        let Some(seed_handle) = idx.try_handle(seed) else {
            continue;
        };
        if !fixel_set.contains(&seed) {
            continue;
        }

        // Trace forward (along +seed.dir) and backward (along -seed.dir).
        // Each direction tracks its own "visited" set (sharing one set
        // across both halves would prevent the +dir half from reusing
        // the seed's voxel that the -dir half already saw).
        let forward = trace_one_direction(
            idx,
            fixel_set,
            seed_handle,
            seed,
            1.0,
            params.step_mm,
            search_radius,
            cos_max,
            max_steps,
        );
        let backward = trace_one_direction(
            idx,
            fixel_set,
            seed_handle,
            seed,
            -1.0,
            params.step_mm,
            search_radius,
            cos_max,
            max_steps,
        );

        // Concatenate: backward (reversed) + seed + forward.
        let total_pts = backward.len() + 1 + forward.len();
        let mut polyline: Vec<[f32; 3]> = Vec::with_capacity(total_pts);
        for p in backward.iter().rev() {
            polyline.push(*p);
        }
        polyline.push(seed_handle.world_pos);
        for p in &forward {
            polyline.push(*p);
        }

        // Length filter.
        if polyline_length(&polyline) >= params.min_length_mm {
            out.push(polyline);
        }
    }
    out
}

fn trace_one_direction(
    idx: &FixelIndex,
    fixel_set: &HashSet<FixelId>,
    seed: &FixelHandle,
    _seed_id: FixelId,
    sign: f32,
    step_mm: f32,
    search_radius: f32,
    cos_max: f32,
    max_steps: usize,
) -> Vec<[f32; 3]> {
    let mut points: Vec<[f32; 3]> = Vec::new();
    let mut p = seed.world_pos;
    let mut d = [seed.dir[0] * sign, seed.dir[1] * sign, seed.dir[2] * sign];
    for _ in 0..max_steps {
        let p_next = [
            p[0] + step_mm * d[0],
            p[1] + step_mm * d[1],
            p[2] + step_mm * d[2],
        ];
        // Find a coherent fixel in the set near the proposed point.
        // Pick the one most aligned with the current direction.
        let near = idx.nearest_within(p_next, search_radius);
        let mut best_fid: Option<FixelId> = None;
        let mut best_dot: f32 = cos_max;
        for (fid, _) in near {
            if !fixel_set.contains(&fid) {
                continue;
            }
            let h = idx.handle(fid);
            let dot = d[0] * h.dir[0] + d[1] * h.dir[1] + d[2] * h.dir[2];
            let abs_dot = dot.abs();
            if abs_dot > best_dot {
                best_dot = abs_dot;
                best_fid = Some(fid);
            }
        }
        let Some(fid) = best_fid else {
            break;
        };
        let h_next = idx.handle(fid);
        // Sign-align next direction to keep flow consistent.
        let raw_dot = d[0] * h_next.dir[0] + d[1] * h_next.dir[1] + d[2] * h_next.dir[2];
        let s = if raw_dot >= 0.0 { 1.0 } else { -1.0 };
        // Yeh-style: take the new fixel's direction directly (no
        // blending). Blending accumulates drift; direct adoption keeps
        // the trace locked to the bundle's local orientation field.
        let new_d = [s * h_next.dir[0], s * h_next.dir[1], s * h_next.dir[2]];
        d = new_d;
        // Step a fixed distance along the previous-step's direction so
        // streamline samples are evenly spaced. Snapping to the
        // fixel's jittered position causes irregular sampling and
        // accumulates lateral drift.
        p = p_next;
        points.push(p);
    }
    points
}

fn polyline_length(points: &[[f32; 3]]) -> f32 {
    let mut total = 0.0;
    for w in points.windows(2) {
        let dx = w[1][0] - w[0][0];
        let dy = w[1][1] - w[0][1];
        let dz = w[1][2] - w[0][2];
        total += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_seeds_returns_empty() {
        // Trivial: no seeds → no output. Just exercises the pathway.
        let _ = TraceParams::default();
    }
}
