//! Export per-voxel scalars to NIfTI-1 volumes.
//!
//! ODX DPV arrays live in compact (mask-only) order; this module projects them
//! onto the dataset's full voxel grid and writes a NIfTI with the dataset's
//! voxel→RAS+ mm affine baked into both `sform` and `qform` slots. The
//! data goes into both, but the **codes** are asymmetric on purpose:
//!
//! * `qform_code = 1 (ScannerAnat)` — authoritative slot; what every
//!   reader should consult.
//! * `sform_code = 0 (Unknown)` — `srow_*` is filled with the affine but
//!   flagged inactive, so software that prefers `sform` falls back to the
//!   `qform` instead. This eliminates any chance of sform/qform divergence
//!   confusing downstream tools.
//!
//! This duplicates a similar header-building pattern inside `odx-rs`
//! (`compare.rs`, `formats/mrtrix.rs`) on purpose: a sibling crate that
//! talks to NIfTI files but isn't tied to an ODX-rs release version. The
//! `set_qform`/`set_sform` pair from the `nifti` crate handles the
//! quaternion math for us — ODX affines are rigid+zoom so the quaternion
//! round-trip is exact.

use std::path::Path;

use anyhow::{Context, Result};
use nalgebra::Matrix4;
use ndarray::Array3;
use nifti::writer::WriterOptions;
use nifti::{NiftiHeader, NiftiType, XForm};

/// Compact-voxel `(i, j, k)` indices in C-order (i-slowest, k-fastest), one
/// entry per compact voxel — matches what
/// [`odx_rs::OdxDataset::compact_to_ijk`] returns.
pub type CompactIjk<'a> = &'a [[u32; 3]];

/// Project a `u16` per-compact-voxel scalar onto the dataset's full
/// `dims[0] × dims[1] × dims[2]` grid and write as a NIfTI-1 volume at
/// `path`. Voxels outside the compact (mask) set get 0.
///
/// `voxel_to_ras` is the 4×4 voxel→RAS+ mm affine from the ODX header. The
/// `qform` is the authoritative slot (`qform_code = ScannerAnat`); `sform`
/// data is also written but flagged `sform_code = Unknown` so every reader
/// converges on the qform geometry.
///
/// The output extension determines compression: `.nii` is uncompressed,
/// `.nii.gz` is gz-compressed (the writer figures this out from the path).
pub fn write_voxel_u16_nifti(
    path: &Path,
    values_compact: &[u16],
    compact_to_ijk: CompactIjk<'_>,
    dims: [usize; 3],
    voxel_to_ras: [[f64; 4]; 4],
) -> Result<()> {
    assert_eq!(values_compact.len(), compact_to_ijk.len());

    // 1. Project compact-order values onto the (nx, ny, nz) grid.
    let mut vol = Array3::<u16>::zeros((dims[0], dims[1], dims[2]));
    for (compact, ijk) in compact_to_ijk.iter().enumerate() {
        let v = values_compact[compact];
        if v == 0 {
            continue;
        }
        vol[[ijk[0] as usize, ijk[1] as usize, ijk[2] as usize]] = v;
    }

    let hdr = build_header(dims, voxel_to_ras, NiftiType::Uint16);

    WriterOptions::new(path)
        .reference_header(&hdr)
        .write_nifti_with_type(&vol, NiftiType::Uint16)
        .with_context(|| format!("writing NIfTI to {}", path.display()))?;
    Ok(())
}

/// Build a `NiftiHeader` whose spatial metadata exactly matches a 3-D ODX
/// grid (`dims`, `voxel_to_ras`). Both `sform` and `qform` are set with
/// code `ScannerAnat`; `xyzt_units` is flagged as mm.
fn build_header(
    dims: [usize; 3],
    voxel_to_ras: [[f64; 4]; 4],
    datatype: NiftiType,
) -> NiftiHeader {
    let mut hdr = NiftiHeader::default();

    hdr.datatype = datatype as i16;
    hdr.bitpix = (datatype.size_of() * 8) as i16;
    hdr.dim = [3, dims[0] as u16, dims[1] as u16, dims[2] as u16, 1, 1, 1, 1];

    // Voxel-size magnitudes along each spatial axis.
    let dx = (voxel_to_ras[0][0].powi(2)
        + voxel_to_ras[1][0].powi(2)
        + voxel_to_ras[2][0].powi(2))
    .sqrt() as f32;
    let dy = (voxel_to_ras[0][1].powi(2)
        + voxel_to_ras[1][1].powi(2)
        + voxel_to_ras[2][1].powi(2))
    .sqrt() as f32;
    let dz = (voxel_to_ras[0][2].powi(2)
        + voxel_to_ras[1][2].powi(2)
        + voxel_to_ras[2][2].powi(2))
    .sqrt() as f32;
    hdr.pixdim = [1.0, dx, dy, dz, 0.0, 0.0, 0.0, 0.0];

    // 2 = NIFTI_UNITS_MM (spatial units in millimetres, no temporal flag).
    hdr.xyzt_units = 2;

    let affine_mat = Matrix4::<f64>::new(
        voxel_to_ras[0][0], voxel_to_ras[0][1], voxel_to_ras[0][2], voxel_to_ras[0][3],
        voxel_to_ras[1][0], voxel_to_ras[1][1], voxel_to_ras[1][2], voxel_to_ras[1][3],
        voxel_to_ras[2][0], voxel_to_ras[2][1], voxel_to_ras[2][2], voxel_to_ras[2][3],
        voxel_to_ras[3][0], voxel_to_ras[3][1], voxel_to_ras[3][2], voxel_to_ras[3][3],
    );

    // Write both slots. `set_qform` derives a quaternion + offsets and
    // overwrites pixdim[0..3]; `set_sform` populates `srow_*`. The qform
    // is the authoritative slot; the sform fields are written for
    // bookkeeping but `sform_code = Unknown` keeps readers off them so we
    // can't ship a file with sform/qform that disagree.
    hdr.set_qform(&affine_mat, XForm::ScannerAnat);
    hdr.set_sform(&affine_mat, XForm::Unknown);

    hdr
}
