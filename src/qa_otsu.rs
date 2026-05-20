//! Otsu-derived quantities on per-fixel QA (`dpf/amplitude`) — the same
//! "primary peak" / Otsu-threshold combination that DSI-Studio's autotrack
//! uses to define a coarse WM mask for tracking.
//!
//! - [`primary_peak_qa`] reduces the per-fixel `dpf/amplitude` array to one
//!   value per masked voxel: the largest amplitude across that voxel's fixels.
//! - [`compute_otsu_threshold`] runs Otsu's method (256-bin histogram) on a
//!   positive-valued sample.
//! - [`wm_mask_otsu`] combines them and flags voxels with
//!   `primary_peak_qa > factor × Otsu` as WM. Returns a flat C-order voxel
//!   mask plus the dataset's dims and affine, ready to feed into
//!   [`crate::pseudo_surfaces`].
//! - [`wm_field_otsu`] is the *cleaner* entry point used by the new
//!   pseudo-surface pipeline: it returns the smoothed continuous QA field on
//!   the full grid plus an isovalue chosen so marching cubes lands on the same
//!   boundary as `wm_mask_otsu` would, only with sub-voxel precision and a
//!   pre-rounded level set. The smoother and isovalue together mirror
//!   DSI-Studio's region-render pipeline (iterative 3×3×3 mean + adaptive
//!   threshold) without copying its code.

use anyhow::{anyhow, Result};
use odx_rs::OdxDataset;

use crate::mean_3d::mean_filter_3x3x3;

/// How to reduce a per-fixel scalar (DPF — variable number of fixels per
/// voxel) to one value per voxel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpfReduction {
    /// Largest fixel value in the voxel. Equivalent to "primary-peak"
    /// readouts (DSI-Studio's QA, AFD of the dominant fibre, etc.).
    /// Default for amplitude / AFD-style scalars.
    Max,
    /// Sum of all fixel values. Useful when each fixel contributes
    /// independently — total fibre density per voxel for AFD.
    Sum,
    /// Arithmetic mean of all fixels. Avoids the bias of Sum toward
    /// crossing-fibre voxels.
    Mean,
}

/// Reduce a per-fixel scalar (variable-length per voxel) to a per-voxel
/// scalar via the chosen reduction. `dpf_name` is the array name inside the
/// ODX (e.g., `"amplitude"`, `"afd"`, `"qa"`). Empty voxels (no fixels)
/// return 0.
pub fn reduce_dpf_to_voxel(
    dataset: &OdxDataset,
    dpf_name: &str,
    reduction: DpfReduction,
) -> Result<Vec<f32>> {
    let values = dataset
        .scalar_dpf_f32(dpf_name)
        .map_err(|e| anyhow!("ODX has no scalar DPF `{dpf_name}`: {e}"))?;
    let offsets = dataset.offsets();
    let nb_voxels = dataset.nb_voxels();
    let expected = offsets.last().copied().unwrap_or(0) as usize;
    if values.len() != expected {
        return Err(anyhow!(
            "dpf/{dpf_name} length {} != offsets sentinel {}",
            values.len(),
            expected
        ));
    }
    let mut out = Vec::with_capacity(nb_voxels);
    for v in 0..nb_voxels {
        let s = offsets[v] as usize;
        let e = offsets[v + 1] as usize;
        let slice = &values[s..e];
        let reduced = match reduction {
            DpfReduction::Max => slice.iter().copied().fold(0.0_f32, f32::max),
            DpfReduction::Sum => slice.iter().copied().sum(),
            DpfReduction::Mean => {
                if slice.is_empty() {
                    0.0
                } else {
                    slice.iter().copied().sum::<f32>() / slice.len() as f32
                }
            }
        };
        out.push(reduced);
    }
    Ok(out)
}

/// For each masked voxel in C-order, the maximum `dpf/amplitude` across that
/// voxel's fixels (= primary-peak QA per DSI-Studio convention). Length =
/// `dataset.nb_voxels()`. Returns an error if `dpf/amplitude` is missing or
/// is the wrong size. Convenience wrapper over [`reduce_dpf_to_voxel`].
pub fn primary_peak_qa(dataset: &OdxDataset) -> Result<Vec<f32>> {
    reduce_dpf_to_voxel(dataset, "amplitude", DpfReduction::Max)
}

/// Otsu's method on a positive-valued sample, 256 bins. Returns the upper
/// bin-edge of the bin maximizing inter-class variance, or 0 for empty /
/// non-positive input.
pub fn compute_otsu_threshold(values: &[f32]) -> f32 {
    let pos: Vec<f32> = values.iter().copied().filter(|&v| v > 0.0).collect();
    if pos.is_empty() {
        return 0.0;
    }
    let max = pos.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max <= 0.0 {
        return 0.0;
    }

    const NBINS: usize = 256;
    let bin_w = max / NBINS as f32;
    let mut hist = [0_u64; NBINS];
    for &v in &pos {
        let mut b = (v / bin_w) as usize;
        if b >= NBINS {
            b = NBINS - 1;
        }
        hist[b] += 1;
    }

    let total: u64 = pos.len() as u64;
    let sum: f64 = (0..NBINS)
        .map(|i| (i as f64 + 0.5) * bin_w as f64 * hist[i] as f64)
        .sum();

    let mut w_b: u64 = 0;
    let mut sum_b: f64 = 0.0;
    let mut max_var: f64 = 0.0;
    let mut best_t: f32 = 0.0;
    for i in 0..NBINS {
        w_b += hist[i];
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += (i as f64 + 0.5) * bin_w as f64 * hist[i] as f64;
        let mu_b = sum_b / w_b as f64;
        let mu_f = (sum - sum_b) / w_f as f64;
        let var = (w_b as f64) * (w_f as f64) * (mu_b - mu_f).powi(2);
        if var > max_var {
            max_var = var;
            best_t = (i as f32 + 1.0) * bin_w;
        }
    }
    best_t
}

/// Threshold an ODX's primary-peak QA at `factor × Otsu(QA)` and project the
/// result onto the dataset's full voxel grid (C-order). Returns the flat WM
/// mask, the grid dimensions, and the voxel→RAS+ mm affine.
///
/// `factor` follows DSI-Studio's convention: 0.5–0.7 is the typical tracking
/// range; lower values flag more WM, higher values restrict to the densest
/// bundles only.
pub fn wm_mask_otsu(
    dataset: &OdxDataset,
    factor: f32,
) -> Result<(Vec<u8>, [usize; 3], [[f64; 4]; 4])> {
    let primary_qa = primary_peak_qa(dataset)?;
    let otsu = compute_otsu_threshold(&primary_qa);
    let threshold = otsu * factor;

    let h = dataset.header();
    let dims = [
        h.dimensions[0] as usize,
        h.dimensions[1] as usize,
        h.dimensions[2] as usize,
    ];
    let total = dims[0] * dims[1] * dims[2];
    let mut wm = vec![0_u8; total];
    let ijks = dataset.compact_to_ijk();
    for (compact, ijk) in ijks.iter().enumerate() {
        if primary_qa[compact] > threshold {
            let flat = (ijk[0] as usize) * dims[1] * dims[2]
                + (ijk[1] as usize) * dims[2]
                + (ijk[2] as usize);
            wm[flat] = 1;
        }
    }
    Ok((wm, dims, h.voxel_to_rasmm))
}

/// The smoothed primary-peak-QA field on the full voxel grid plus the
/// marching-cubes isovalue derived from it. Replaces the binary-mask MC input
/// produced by [`wm_mask_otsu`] with a continuous-field MC input — the
/// fundamental change behind the cleaner pseudo-surface output.
#[derive(Debug, Clone)]
pub struct WmField {
    /// Smoothed primary-peak QA, flat C-order (`flat = i*ny*nz + j*nz + k`),
    /// size `dims[0] * dims[1] * dims[2]`. Zero outside the dataset's compact
    /// voxel set, which gives the implicit brain-mask boundary at the field
    /// edges.
    pub field: Vec<f32>,
    /// Voxel-grid dimensions `[nx, ny, nz]`.
    pub dims: [usize; 3],
    /// Voxel→RAS+ mm affine (4×4 row-major).
    pub voxel_to_ras: [[f64; 4]; 4],
    /// Marching-cubes isovalue. Iso = `factor × Otsu(smoothed_field)`. Note
    /// that Otsu is computed *on the smoothed field*, which is the
    /// volume-preservation trick: the same voxel fraction sits above iso
    /// before and after smoothing, so the surface anchors at the same
    /// anatomical boundary as the unsmoothed binary mask.
    pub iso: f32,
}

/// Build a smoothed continuous WM field for marching cubes.
///
/// 1. Project `primary_peak_qa` onto the full voxel grid (zero outside the
///    compact voxel set).
/// 2. **Compute Otsu on the *raw* (unsmoothed) projected field.** The
///    threshold is anchored to the actual WM-vs-GM bimodal distribution of
///    the primary-peak QA, exactly as DSI-Studio's autotrack uses it. Then
///    `iso = factor × otsu`.
/// 3. Apply `mean_filter_3x3x3` `smooth_iters` times to the *field* (but not
///    to the threshold). Smoothing rounds the iso surface for sub-voxel-
///    precise marching cubes without changing where the WM/GM boundary
///    *should* sit.
///
/// Earlier versions of this function computed Otsu on the *smoothed* field
/// to chase volume preservation, but that backfires: smoothing creates a
/// halo of small positive values around the original WM mask, the
/// positive-value histogram seen by Otsu becomes dominated by that halo,
/// and the threshold collapses — the surface inflates from a WM/GM contour
/// to a brain-mask outline. Anchoring Otsu on the raw field keeps the
/// surface at the genuine WM boundary.
pub fn wm_field_otsu(
    dataset: &OdxDataset,
    factor: f32,
    smooth_iters: u32,
) -> Result<WmField> {
    let primary_qa = primary_peak_qa(dataset)?;

    let h = dataset.header();
    let dims = [
        h.dimensions[0] as usize,
        h.dimensions[1] as usize,
        h.dimensions[2] as usize,
    ];
    let total = dims[0] * dims[1] * dims[2];

    // Project compact-voxel primary-peak QA values onto the full grid.
    let mut field = vec![0.0_f32; total];
    let ijks = dataset.compact_to_ijk();
    for (compact, ijk) in ijks.iter().enumerate() {
        let flat = (ijk[0] as usize) * dims[1] * dims[2]
            + (ijk[1] as usize) * dims[2]
            + (ijk[2] as usize);
        field[flat] = primary_qa[compact];
    }

    // Otsu on the RAW projected QA — anchored to the genuine WM/GM bimodal
    // distribution. Computing it before smoothing prevents the brain-mask
    // inflation we'd otherwise get from the smoothing halo.
    let otsu_raw = compute_otsu_threshold(&field);
    let iso = otsu_raw * factor;

    // Diagnostic — surface quality is highly sensitive to the iso level, so
    // log enough to debug "the surface is a brain mask" / "doesn't follow
    // gyri" complaints. Computed once per ODX, not per voxel; cheap.
    let n_pos = field.iter().filter(|&&v| v > 0.0).count();
    let max_qa = field.iter().cloned().fold(0.0_f32, f32::max);
    let mean_pos: f32 = if n_pos > 0 {
        field.iter().filter(|&&v| v > 0.0).sum::<f32>() / n_pos as f32
    } else {
        0.0
    };
    let n_above_iso = field.iter().filter(|&&v| v > iso).count();
    eprintln!(
        "  qa_field_otsu: positive_voxels={}, max={:.4}, mean(positive)={:.4}, otsu={:.4}, iso=factor*otsu={:.4}, voxels_above_iso={} ({:.1}% of positive)",
        n_pos,
        max_qa,
        mean_pos,
        otsu_raw,
        iso,
        n_above_iso,
        100.0 * n_above_iso as f32 / n_pos.max(1) as f32,
    );

    // Now smooth the field for sub-voxel-precise MC. The threshold is fixed
    // already so smoothing only rounds the level set; it doesn't shift the
    // surface inward or outward in any systematic way.
    mean_filter_3x3x3(&mut field, dims, smooth_iters);

    Ok(WmField {
        field,
        dims,
        voxel_to_ras: h.voxel_to_rasmm,
        iso,
    })
}

/// Build a [`WmField`] by reducing a per-fixel DPF (variable number of
/// fixels per voxel) to a per-voxel scalar via the chosen reduction, then
/// thresholding at `factor × Otsu`. **Use this when you have an SH-
/// deconvolution-style ODX (CSD, SS3T, MSMT-CSD)** where GFA isn't
/// well-defined: the DPF typically holds AFD ("apparent fibre density")
/// per fixel, and `Max`-reducing it gives a per-voxel scalar with similar
/// WM/GM contrast to GFA. For `Max` on `dpf/amplitude` this is identical
/// to [`wm_field_otsu`] (which is hardcoded to that combination).
pub fn wm_field_from_dpf(
    dataset: &OdxDataset,
    dpf_name: &str,
    reduction: DpfReduction,
    factor: f32,
    smooth_iters: u32,
) -> Result<WmField> {
    let scalar = reduce_dpf_to_voxel(dataset, dpf_name, reduction)?;

    let h = dataset.header();
    let dims = [
        h.dimensions[0] as usize,
        h.dimensions[1] as usize,
        h.dimensions[2] as usize,
    ];
    let total = dims[0] * dims[1] * dims[2];

    let mut field = vec![0.0_f32; total];
    let ijks = dataset.compact_to_ijk();
    for (compact, ijk) in ijks.iter().enumerate() {
        let flat = (ijk[0] as usize) * dims[1] * dims[2]
            + (ijk[1] as usize) * dims[2]
            + (ijk[2] as usize);
        field[flat] = scalar[compact];
    }

    let otsu = compute_otsu_threshold(&field);
    let iso = otsu * factor;
    mean_filter_3x3x3(&mut field, dims, smooth_iters);

    Ok(WmField {
        field,
        dims,
        voxel_to_ras: h.voxel_to_rasmm,
        iso,
    })
}

/// Build a [`WmField`] from an arbitrary scalar DPV (Data Per Voxel) in the
/// ODX rather than from the per-peak QA. Useful when the pipeline that
/// wrote the ODX also stored a smoother, less noisy per-voxel scalar —
/// `gfa` (Generalized Fractional Anisotropy) is the prime example: it's
/// computed across the entire ODF instead of just the peaks, so it has
/// much cleaner WM/GM contrast than `dpf/amplitude`, and shrink-wrap on
/// GFA gives more anatomically faithful gyrification.
///
/// The DPV is one f32 per voxel in compact-mask order; this projects it
/// onto the full grid (zero outside the compact set), computes Otsu on the
/// raw projected values, then mean-smooths the field for sub-voxel-precise
/// marching cubes.
pub fn wm_field_from_dpv(
    dataset: &OdxDataset,
    dpv_name: &str,
    factor: f32,
    smooth_iters: u32,
) -> Result<WmField> {
    let scalar = dataset
        .scalar_dpv_f32(dpv_name)
        .map_err(|e| anyhow!("ODX has no scalar DPV `{dpv_name}`: {e}"))?;

    let h = dataset.header();
    let dims = [
        h.dimensions[0] as usize,
        h.dimensions[1] as usize,
        h.dimensions[2] as usize,
    ];
    let total = dims[0] * dims[1] * dims[2];
    if scalar.len() != dataset.nb_voxels() {
        return Err(anyhow!(
            "DPV `{dpv_name}` has {} entries, expected {} compact voxels",
            scalar.len(),
            dataset.nb_voxels()
        ));
    }

    // Project compact-voxel values onto the full grid (zero outside).
    let mut field = vec![0.0_f32; total];
    let ijks = dataset.compact_to_ijk();
    for (compact, ijk) in ijks.iter().enumerate() {
        let flat = (ijk[0] as usize) * dims[1] * dims[2]
            + (ijk[1] as usize) * dims[2]
            + (ijk[2] as usize);
        field[flat] = scalar[compact];
    }

    let otsu = compute_otsu_threshold(&field);
    let iso = otsu * factor;
    mean_filter_3x3x3(&mut field, dims, smooth_iters);

    Ok(WmField {
        field,
        dims,
        voxel_to_ras: h.voxel_to_rasmm,
        iso,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otsu_separates_bimodal() {
        let mut v = Vec::new();
        for _ in 0..100 {
            v.push(0.1);
        }
        for _ in 0..100 {
            v.push(1.0);
        }
        let t = compute_otsu_threshold(&v);
        assert!(t > 0.1 && t < 1.0, "expected threshold between modes, got {t}");
    }

    #[test]
    fn otsu_handles_empty() {
        assert_eq!(compute_otsu_threshold(&[]), 0.0);
    }
}
