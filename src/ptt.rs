//! Reference-free Parallel Transport Tractography (PTT) primitives on fixels.
//!
//! Adapts the PTT framework (Aydogan & Shi 2021; nibrary from continuous ODFs
//! to discrete fixels. This module hosts
//! propagator math, parallel-transport frames, and pure-fixel arc likelihood.
//!
//! ## What stays the same vs nibrary PTT
//!
//! - Parallel-transport frame `F = (T, N1, N2)`: an orthonormal triad
//!   carried along the trajectory; `T` is the local tangent.
//! - Two-curvature parametrization `(k1, k2)`: a candidate next step is a
//!   tiny helical arc whose shape is fully determined by these curvatures.
//! - **Probe arc**: rather than scoring a step by the *single* local fixel
//!   pattern at the current position, walk `probe_length` forward in
//!   `probe_quality` substeps along the candidate arc and score the *whole
//!   arc* by summing local data support at each substep.
//! - Analytic propagator [`prep_propagator`]: the 3×3 matrix that advances
//!   `(p, F)` along an arc of length `t` with curvatures `(k1, k2)`. Direct
//!   port of nibrary's formula.
//!
//! ## What's different
//!
//! `data_support(p, dir)` swaps from "interpolate SH coeffs at `p`, evaluate
//! amplitude in direction `dir`" to "find fixels within `support_radius` of
//! `p` and sum `amplitude_fixel × |d_fixel · dir|^angular_power` across
//! them". Same role — a scalar likelihood for the local
//! `(position, direction)` configuration — but driven by the discrete fixel
//! set instead of a continuous FOD.

use crate::fixel_index::{FixelHandle, FixelId, FixelIndex};

/// Configuration for PTT-style coherence calculations.
#[derive(Debug, Clone, Copy)]
pub struct PttParams {
    /// Probe arc length (mm). The arc extends forward from the
    /// frame's current position by this much.
    pub probe_length_mm: f32,
    /// Number of arc samples (including start). With `probe_length=5`
    /// and `probe_quality=5`, samples are spaced 1.25 mm apart.
    pub probe_quality: usize,
    /// Spatial neighborhood radius (mm) for the data-support kernel
    /// at each arc sample.
    pub support_radius_mm: f32,
    /// Power applied to `|d_fixel · dir|` in the angular kernel.
    /// Higher = more selective (a 30° misalignment costs more).
    /// 4 is a reasonable starting default.
    pub angular_power: u32,
    /// Maximum |k1| and |k2| (1/mm) sampled in the candidate grid.
    /// `k_max = 0.2` corresponds to a minimum radius of curvature
    /// of 5 mm.
    pub k_max: f32,
    /// Number of curvature candidates per axis. Total candidates per
    /// fixel = `n_k_samples * n_k_samples`. At 5 the grid is
    /// {-k_max, -k_max/2, 0, +k_max/2, +k_max} on each axis (25
    /// total).
    pub n_k_samples: usize,
}

impl PttParams {
    /// Reasonable defaults for ~1.7 mm voxel data and bundles like CST.
    pub fn defaults() -> Self {
        Self {
            probe_length_mm: 5.0,
            probe_quality: 5,
            support_radius_mm: 2.5,
            angular_power: 4,
            k_max: 0.2,
            n_k_samples: 5,
        }
    }
}

/// Parallel-transport frame: position + orthonormal triad.
#[derive(Debug, Clone, Copy)]
pub struct PtfFrame {
    /// Current position in RAS+ mm.
    pub p: [f32; 3],
    /// Frame: `f[0]` = tangent T, `f[1]` = normal N1, `f[2]` = normal N2.
    pub f: [[f32; 3]; 3],
}

const EPS: f32 = 1e-4;

/// Build the 9-element propagator `[a00..a22]` that advances
/// `(p, F)` along an arc of length `t` with curvatures `(k1, k2)`.
/// Direct port of nibrary's analytic formula.
///
/// Layout:
/// - `[0..3]` are the position-update coefficients (in `(T, N1, N2)`
///   coordinates).
/// - `[3..6]` are the new-tangent coefficients.
/// - `[6..9]` are the new-N2 coefficients.
pub fn prep_propagator(k1: f32, k2: f32, t: f32) -> [f32; 9] {
    if k1.abs() < EPS && k2.abs() < EPS {
        // Straight-line arc.
        return [t, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    }
    let k1 = if k1.abs() < EPS { EPS } else { k1 };
    let k2 = if k2.abs() < EPS { EPS } else { k2 };
    let k = (k1 * k1 + k2 * k2).sqrt();
    let sinkt = (k * t).sin();
    let coskt = (k * t).cos();
    let kk = 1.0 / (k * k);
    [
        sinkt / k,
        k1 * (1.0 - coskt) * kk,
        k2 * (1.0 - coskt) * kk,
        coskt,
        k1 * sinkt / k,
        k2 * sinkt / k,
        -k2 * sinkt / k,
        k1 * k2 * (coskt - 1.0) * kk,
        (k1 * k1 + k2 * k2 * coskt) * kk,
    ]
}

#[inline]
pub(crate) fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub(crate) fn normalize(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-9 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        v
    }
}

/// Apply the propagator to advance the frame by step `t` with the
/// candidate curvatures `(k1, k2)`. Updates `frame` in-place. The
/// frame is reorthogonalized after the update.
pub fn walk(frame: &mut PtfFrame, k1: f32, k2: f32, step_mm: f32) {
    let pp = prep_propagator(k1, k2, step_mm);
    let t_old = frame.f[0];
    let n1_old = frame.f[1];
    let n2_old = frame.f[2];

    for i in 0..3 {
        frame.p[i] += pp[0] * t_old[i] + pp[1] * n1_old[i] + pp[2] * n2_old[i];
    }
    let t_new = normalize([
        pp[3] * t_old[0] + pp[4] * n1_old[0] + pp[5] * n2_old[0],
        pp[3] * t_old[1] + pp[4] * n1_old[1] + pp[5] * n2_old[1],
        pp[3] * t_old[2] + pp[4] * n1_old[2] + pp[5] * n2_old[2],
    ]);
    let n2_raw = [
        pp[6] * t_old[0] + pp[7] * n1_old[0] + pp[8] * n2_old[0],
        pp[6] * t_old[1] + pp[7] * n1_old[1] + pp[8] * n2_old[1],
        pp[6] * t_old[2] + pp[7] * n1_old[2] + pp[8] * n2_old[2],
    ];
    // Re-orthogonalize: N1 = N2 × T, then N2 = T × N1.
    let n1_new = normalize(cross(n2_raw, t_new));
    let n2_new = cross(t_new, n1_new);

    frame.f = [t_new, n1_new, n2_new];
}

/// Initialize a frame at a fixel: position = world_pos, T = dir,
/// N1, N2 = arbitrary orthonormal complement.
pub fn frame_at_fixel_handle(handle: &FixelHandle) -> PtfFrame {
    let t = handle.dir;
    // Pick an axis least aligned with T as the seed for N1.
    let seed = if t[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let n1 = normalize(cross(seed, t));
    let n2 = cross(t, n1);
    PtfFrame {
        p: handle.world_pos,
        f: [t, n1, n2],
    }
}

/// Pure-fixel local data support at `(p, dir)`: sum over fixels
/// within `support_radius_mm` of
/// `amplitude × |d_fixel · dir|^angular_power`.
/// Antipodal-symmetric (uses `|·|`).
///
/// Each fixel's contribution is weighted by its **amplitude (= QA)**:
/// a low-FA fixel sitting in a phantom or noisy region contributes
/// proportionally less to arc likelihood than a high-QA fixel on
/// the bundle's main path. Fixels with NaN amplitude (when
/// `dpf/amplitude` is unavailable at index-build time) fall back to
/// weight 1.0.
pub fn data_support(p: [f32; 3], dir: [f32; 3], idx: &FixelIndex, params: &PttParams) -> f32 {
    let mut total = 0.0_f32;
    for (fid, _d2) in idx.nearest_within(p, params.support_radius_mm) {
        let h = idx.handle(fid);
        let dot = h.dir[0] * dir[0] + h.dir[1] * dir[1] + h.dir[2] * dir[2];
        let aligned = dot.abs();
        let mut a = aligned;
        for _ in 1..params.angular_power {
            a *= aligned;
        }
        let amp = if h.amplitude.is_finite() { h.amplitude } else { 1.0 };
        total += a * amp;
    }
    total
}

/// Walk a probe arc and sum a per-sample support score. The
/// `support_fn` decides what each sample's contribution is — pass
/// [`data_support`] for tractography (fixel-only), or a closure that
/// adds reference gating for bundle validation (in `odx-bundles`).
///
/// Does NOT mutate `frame` — walks a local copy.
pub fn arc_likelihood_with<F>(
    frame: &PtfFrame,
    k1: f32,
    k2: f32,
    params: &PttParams,
    mut support_fn: F,
) -> f32
where
    F: FnMut([f32; 3], [f32; 3]) -> f32,
{
    let q = params.probe_quality.max(1);
    let probe_step = if q > 1 {
        params.probe_length_mm / (q as f32 - 1.0)
    } else {
        0.0
    };
    let mut p = frame.p;
    let mut f = frame.f;

    let mut total = support_fn(p, f[0]);

    for _ in 0..q.saturating_sub(1) {
        let pp = prep_propagator(k1, k2, probe_step);
        for i in 0..3 {
            p[i] += pp[0] * f[0][i] + pp[1] * f[1][i] + pp[2] * f[2][i];
        }
        let t_new = normalize([
            pp[3] * f[0][0] + pp[4] * f[1][0] + pp[5] * f[2][0],
            pp[3] * f[0][1] + pp[4] * f[1][1] + pp[5] * f[2][1],
            pp[3] * f[0][2] + pp[4] * f[1][2] + pp[5] * f[2][2],
        ]);
        let n2_raw = [
            pp[6] * f[0][0] + pp[7] * f[1][0] + pp[8] * f[2][0],
            pp[6] * f[0][1] + pp[7] * f[1][1] + pp[8] * f[2][1],
            pp[6] * f[0][2] + pp[7] * f[1][2] + pp[8] * f[2][2],
        ];
        let n1_new = normalize(cross(n2_raw, t_new));
        let n2_new = cross(t_new, n1_new);
        f = [t_new, n1_new, n2_new];
        total += support_fn(p, f[0]);
    }
    total
}

/// Pure-fixel arc likelihood (sum of [`data_support`] along the arc).
pub fn arc_likelihood(
    frame: &PtfFrame,
    k1: f32,
    k2: f32,
    idx: &FixelIndex,
    params: &PttParams,
) -> f32 {
    arc_likelihood_with(frame, k1, k2, params, |p, dir| {
        data_support(p, dir, idx, params)
    })
}

/// Best arc likelihood through a single fixel (pure-fixel; for
/// tractography). Sweeps `n_k_samples^2` candidates and returns max.
pub fn best_arc_likelihood(fid: FixelId, idx: &FixelIndex, params: &PttParams) -> f32 {
    let frame = frame_at_fixel_handle(idx.handle(fid));
    let n_k = params.n_k_samples.max(1);
    if n_k == 1 {
        return arc_likelihood(&frame, 0.0, 0.0, idx, params);
    }
    let dk = 2.0 * params.k_max / (n_k as f32 - 1.0);
    let mut best = 0.0_f32;
    for i in 0..n_k {
        let k1 = -params.k_max + dk * i as f32;
        for j in 0..n_k {
            let k2 = -params.k_max + dk * j as f32;
            let lik = arc_likelihood(&frame, k1, k2, idx, params);
            if lik > best {
                best = lik;
            }
        }
    }
    best
}

/// One PTT trajectory's vertices, tangents, and per-step likelihoods.
#[derive(Debug, Clone)]
pub struct PttTrajectory {
    pub points: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 3]>,
    /// Likelihood of each step *after* the seed point. `likelihoods[i]`
    /// is the score for advancing from `points[i]` to `points[i+1]`.
    pub likelihoods: Vec<f32>,
}

/// Visit every fixel within `capture_radius_mm` of any trajectory
/// point whose direction is aligned (`|d · t_traj| ≥ cos_theta`)
/// with the local trajectory tangent. Returns the visited fixel ids
/// (deduped, sorted).
pub fn capture_visited_fixels(
    traj: &PttTrajectory,
    idx: &FixelIndex,
    capture_radius_mm: f32,
    cos_theta_capture: f32,
) -> Vec<FixelId> {
    use std::collections::HashSet;
    let mut visited: HashSet<FixelId> = HashSet::new();
    for (p, t) in traj.points.iter().zip(traj.tangents.iter()) {
        for (fid, _d2) in idx.nearest_within(*p, capture_radius_mm) {
            let h = idx.handle(fid);
            let dot = h.dir[0] * t[0] + h.dir[1] * t[1] + h.dir[2] * t[2];
            if dot.abs() >= cos_theta_capture {
                visited.insert(fid);
            }
        }
    }
    let mut v: Vec<FixelId> = visited.into_iter().collect();
    v.sort_unstable();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_propagator_is_identity_on_t() {
        let pp = prep_propagator(0.0, 0.0, 2.5);
        // Straight arc: position advances by t along T; T unchanged; N2 unchanged.
        assert!((pp[0] - 2.5).abs() < 1e-6);
        assert!(pp[1].abs() < 1e-6);
        assert!(pp[2].abs() < 1e-6);
        assert!((pp[3] - 1.0).abs() < 1e-6);
        assert!(pp[4].abs() < 1e-6);
        assert!((pp[8] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn frame_at_fixel_orthonormal() {
        let h = FixelHandle {
            id: 0,
            world_pos: [0.0, 0.0, 0.0],
            dir: [0.0, 0.0, 1.0],
            voxel_idx: 0,
            amplitude: 1.0,
        };
        let frame = frame_at_fixel_handle(&h);
        // T should equal dir.
        assert!((frame.f[0][2] - 1.0).abs() < 1e-6);
        // N1, N2 should be unit-length and orthogonal to T and each other.
        for i in 0..3 {
            let v = frame.f[i];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "vector {} not unit ({})", i, len);
        }
        let dot01 = frame.f[0][0] * frame.f[1][0]
            + frame.f[0][1] * frame.f[1][1]
            + frame.f[0][2] * frame.f[1][2];
        let dot02 = frame.f[0][0] * frame.f[2][0]
            + frame.f[0][1] * frame.f[2][1]
            + frame.f[0][2] * frame.f[2][2];
        let dot12 = frame.f[1][0] * frame.f[2][0]
            + frame.f[1][1] * frame.f[2][1]
            + frame.f[1][2] * frame.f[2][2];
        assert!(dot01.abs() < 1e-5);
        assert!(dot02.abs() < 1e-5);
        assert!(dot12.abs() < 1e-5);
    }
}
