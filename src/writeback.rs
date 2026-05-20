//! Writing per-fixel scalars and per-bundle voxel groups back into an ODX
//! archive. Uses the in-place archive-edit machinery exposed by
//! [`odx_rs::io::zip`] so we don't rewrite the whole 80+ MB file each time.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use odx_rs::data_array::DataArray;
use odx_rs::dtype::DType;
use odx_rs::io::directory::{
    append_dpf_to_directory, append_dpv_to_directory, append_groups_to_directory,
};
use odx_rs::io::zip::{append_dpf_to_zip, append_dpv_to_zip, append_groups_to_zip};
use zip::CompressionMethod;

/// Append a single per-fixel scalar (DPF) to the ODX at `path`. Routes to
/// the directory or zip variant based on `path`. Overwrite is on so
/// re-runs of the pipeline replace prior values rather than erroring.
pub fn write_dpf_u8(path: &Path, name: &str, values: &[u8]) -> Result<()> {
    let arr = DataArray::owned_bytes(values.to_vec(), 1, DType::UInt8);
    let mut map = HashMap::new();
    map.insert(name.to_string(), arr);
    append_dpf_dispatch(path, &map)
}

pub fn write_dpf_u16(path: &Path, name: &str, values: &[u16]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let arr = DataArray::owned_bytes(bytes, 1, DType::UInt16);
    let mut map = HashMap::new();
    map.insert(name.to_string(), arr);
    append_dpf_dispatch(path, &map)
}

pub fn write_dpf_u32(path: &Path, name: &str, values: &[u32]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let arr = DataArray::owned_bytes(bytes, 1, DType::UInt32);
    let mut map = HashMap::new();
    map.insert(name.to_string(), arr);
    append_dpf_dispatch(path, &map)
}

pub fn write_dpf_f32(path: &Path, name: &str, values: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let arr = DataArray::owned_bytes(bytes, 1, DType::Float32);
    let mut map = HashMap::new();
    map.insert(name.to_string(), arr);
    append_dpf_dispatch(path, &map)
}

/// Append a single per-voxel scalar (DPV) to the ODX at `path`. `values`
/// must have length `dataset.nb_voxels()` in compact (mask-only) order.
pub fn write_dpv_u8(path: &Path, name: &str, values: &[u8]) -> Result<()> {
    let arr = DataArray::owned_bytes(values.to_vec(), 1, DType::UInt8);
    let mut map = HashMap::new();
    map.insert(name.to_string(), arr);
    append_dpv_dispatch(path, &map)
}

pub fn write_dpv_u16(path: &Path, name: &str, values: &[u16]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let arr = DataArray::owned_bytes(bytes, 1, DType::UInt16);
    let mut map = HashMap::new();
    map.insert(name.to_string(), arr);
    append_dpv_dispatch(path, &map)
}

pub fn write_dpv_u32(path: &Path, name: &str, values: &[u32]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let arr = DataArray::owned_bytes(bytes, 1, DType::UInt32);
    let mut map = HashMap::new();
    map.insert(name.to_string(), arr);
    append_dpv_dispatch(path, &map)
}

pub fn write_dpv_f32(path: &Path, name: &str, values: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let arr = DataArray::owned_bytes(bytes, 1, DType::Float32);
    let mut map = HashMap::new();
    map.insert(name.to_string(), arr);
    append_dpv_dispatch(path, &map)
}

/// Append a voxel group (a list of compact voxel indices) to the ODX. Used
/// for `groups/bundles/<name>.uint32` in Phase 4+.
pub fn write_voxel_group(path: &Path, name: &str, voxel_indices: &[u32]) -> Result<()> {
    let bytes: Vec<u8> = voxel_indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    let arr = DataArray::owned_bytes(bytes, 1, DType::UInt32);
    let mut map = HashMap::new();
    map.insert(name.to_string(), arr);
    append_groups_dispatch(path, &map)
}

fn append_dpf_dispatch(path: &Path, dpf: &HashMap<String, DataArray>) -> Result<()> {
    if path.is_dir() {
        append_dpf_to_directory(path, dpf, /*overwrite=*/ true)
            .with_context(|| format!("appending dpf to directory {}", path.display()))?;
    } else if is_odx_archive(path) {
        append_dpf_to_zip(path, dpf, CompressionMethod::Deflated, /*overwrite=*/ true)
            .with_context(|| format!("appending dpf to archive {}", path.display()))?;
    } else {
        return Err(anyhow!(
            "expected ODX directory or .odx archive: {}",
            path.display()
        ));
    }
    Ok(())
}

fn append_dpv_dispatch(path: &Path, dpv: &HashMap<String, DataArray>) -> Result<()> {
    if path.is_dir() {
        append_dpv_to_directory(path, dpv, /*overwrite=*/ true)
            .with_context(|| format!("appending dpv to directory {}", path.display()))?;
    } else if is_odx_archive(path) {
        append_dpv_to_zip(path, dpv, CompressionMethod::Deflated, /*overwrite=*/ true)
            .with_context(|| format!("appending dpv to archive {}", path.display()))?;
    } else {
        return Err(anyhow!(
            "expected ODX directory or .odx archive: {}",
            path.display()
        ));
    }
    Ok(())
}

fn append_groups_dispatch(path: &Path, groups: &HashMap<String, DataArray>) -> Result<()> {
    if path.is_dir() {
        append_groups_to_directory(path, groups, /*overwrite=*/ true)
            .with_context(|| format!("appending groups to directory {}", path.display()))?;
    } else if is_odx_archive(path) {
        append_groups_to_zip(path, groups, CompressionMethod::Deflated, /*overwrite=*/ true)
            .with_context(|| format!("appending groups to archive {}", path.display()))?;
    } else {
        return Err(anyhow!(
            "expected ODX directory or .odx archive: {}",
            path.display()
        ));
    }
    Ok(())
}

fn is_odx_archive(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e == "odx")
}
