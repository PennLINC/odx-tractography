# odx-tractography

Fixel tractography primitives on [ODX](https://github.com/PennLINC/odx-rs)
datasets, in pure Rust.

A spatial fixel index, an analytic Parallel Transport Tractography (PTT)
propagator adapted to discrete fixels, a Yeh-style fixel-set tracker,
QA / Otsu helpers, mid-cortical pseudo-surface bootstrapping (no GIFTI
required), a small in-place writeback layer for appending per-fixel
scalars or per-bundle voxel groups to an existing ODX archive, and a NIfTI
voxel I/O helper.

## Modules

- **[`fixel_index`]** — `FixelIndex`, a [kiddo](https://crates.io/crates/kiddo)
  KD-tree over per-fixel world positions, with sub-voxel jitter to handle
  coincident peaks and per-fixel amplitude (QA) metadata loaded from
  `dpf/amplitude` when present. Supports build-time filtering by amplitude
  Otsu, threshold, or arbitrary per-voxel predicates, plus
  **`FixelIndex::from_handles(positions, directions, amplitudes)`** — a
  constructor for *synthetic* fixel fields (e.g. one fixel per u-fiber
  parabola apex, direction set to the along-fundus axis), so the same PTT
  engine that traces streamlines can be propagated over a derived field.

- **[`ptt`]** — Parallel Transport Tractography on fixels. Includes the
  analytic propagator (`prep_propagator`, `walk`), parallel-transport frames
  (`PtfFrame`, `frame_at_fixel_handle`), pure-fixel data support
  (`data_support`), and arc-likelihood scoring with a closure-based variant
  (`arc_likelihood_with`) so callers can plug in their own per-sample
  scoring. Adapts the PTT framework (Aydoğan & Shi 2021) from continuous
  ODFs to the discrete fixel set in an ODX. `PttParams` exposes the nibrary
  **fanned probe** (`probe_radius_mm`, `probe_count`) — when both are set,
  arc support at each probe sample is averaged over `probe_count` points
  offset by `probe_radius_mm` around the tangent in the N1/N2 plane,
  smoothing the spiky discrete-fixel posterior. Defaults `0.0 / 1` are
  byte-identical to the single-arc historical behaviour.

- **[`trace`]** — Yeh-style deterministic fixel-set tractography for
  visualisation. Given a set of fixel ids, `trace_within_fixels` produces
  one polyline per seed that stays inside the set.

- **[`qa_otsu`]** — QA / Otsu helpers shared by ufixels' Phase 1 and
  pseudo-surface bootstrap: `primary_peak_qa`, `compute_otsu_threshold`,
  `reduce_dpf_to_voxel` (`DpfReduction::{Max, Sum, Mean}`), `wm_field_otsu`
  (smoothed continuous field + Otsu isovalue for clean marching-cubes
  surfaces), `wm_mask_otsu`, `wm_field_from_dpf`, `wm_field_from_dpv`.

- **[`pseudo_surfaces`]** — mid-cortical mesh bootstrap from the ODX itself
  when GIFTI surfaces aren't available. `pseudo_surfaces_from_mask` /
  `pseudo_surfaces_from_field` run a balloon + shrink-wrap pipeline against
  the WM mask or Otsu-thresholded QA field; hemisphere split via
  `split_by_eroded_voxel_cc` or `split_by_x_sign`; outputs paired
  white/pial-like meshes with vertex correspondence. Parameters in
  `PseudoSurfaceParams` and `ShrinkWrapParams`.

- **[`mean_3d`]** — `mean_filter_3x3x3` (and friends) for smoothing the
  continuous WM field before marching cubes.

- **[`voxel_nifti`]** — small NIfTI-1 writer for voxel scalar fields (per-voxel
  sheet count, Otsu threshold maps, etc.). Pairs with the ODX
  `voxel_to_rasmm` / `dimensions` for round-trippable headers.

- **[`writeback`]** — append `dpf/<name>.<ncols>.<dtype>` arrays or
  `groups/<name>.uint32` voxel-id lists to an existing ODX directory or
  zip archive in place. Avoids rewriting the multi-hundred-MB ODF/SH
  payload when all you want to add is a few KB of per-fixel scalars.

## TRX writing

This crate re-exports [`trx_rs::Tractogram`](https://docs.rs/trx-rs) so
consumers can write TRX files without an extra dependency:

```rust
use odx_tractography::{Tractogram, TrxHeader, TrxDType};

let header = TrxHeader {
    voxel_to_rasmm: dataset.header().voxel_to_rasmm,
    dimensions: dataset.header().dimensions,
    nb_streamlines: 0,
    nb_vertices: 0,
    extra: Default::default(),
};
let mut tractogram = Tractogram::with_header(header);
for poly in &polylines {
    tractogram.push_streamline(poly)?;
}
tractogram.insert_group("my_bundle".to_string(), vec![0, 1, 2]);
tractogram.to_trx(TrxDType::Float32)?.save(out_path)?;
```

## PTT example

Score the best curvature-grid arc through every fixel in an ODX:

```rust
use odx_tractography::{FixelIndex, PttParams, best_arc_likelihood};
use odx_rs::OdxDataset;

let dataset = OdxDataset::open("subject.odx".as_ref())?;
let idx = FixelIndex::build(&dataset);
let params = PttParams::defaults();

for fid in 0..idx.len() as u32 {
    let lik = best_arc_likelihood(fid, &idx, &params);
    if lik > 1.0 {
        // ... high-coherence fixel ...
    }
}
```

For a custom scoring kernel (e.g. amplitude × ROI mask), use
`arc_likelihood_with` and pass any closure satisfying
`FnMut([f32; 3], [f32; 3]) -> f32`.

## Synthetic-field PTT (`FixelIndex::from_handles`)

Build a fixel field from raw positions + directions, bypassing any
`OdxDataset`. Useful for *derived* fields such as one fixel per u-fiber
parabola apex:

```rust
use odx_tractography::FixelIndex;

// `apex_pos[i]` = apex of streamline i; `apex_dir[i]` = triangle plane normal
let idx = FixelIndex::from_handles(&apex_pos, &apex_dir, Some(&apex_amplitude));

// Now PTT primitives (`data_support`, `arc_likelihood`, ...) work over this
// synthetic field — used by ufixels' `trace_apex_fundi` to trace the sulcal-
// fundus spine through each sheet's apex cloud with the real PTT engine.
```

## In-place DPF append

```rust
use odx_tractography::writeback;

writeback::write_dpf_u8(&path, "my_selection", &selected_bytes)?;
writeback::write_dpf_f32(&path, "my_score", &per_fixel_scores)?;
writeback::write_voxel_group(&path, "my_bundle", &voxel_ids)?;
```

These mutate the ODX at `path` directly — copy first if you want to keep
the source pristine.

## Tests + examples

Unit tests in [`tests/`](tests/) (round-trip synthetic surfaces +
`from_handles` + fanned probe consistency); usage examples in
[`examples/`](examples/).

## License

Dual-licensed under MIT or Apache-2.0 at your option.
