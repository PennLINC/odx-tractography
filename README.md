# odx-tractography

Fixel tractography primitives on [ODX](https://github.com/PennLINC/odx-rs)
datasets, in pure Rust.

A spatial fixel index, an analytic Parallel Transport Tractography (PTT)
propagator adapted to discrete fixels, a Yeh-style fixel-set tracker, and
a small in-place writeback layer for appending per-fixel scalars or
per-bundle voxel groups to an existing ODX archive without rewriting the
rest of the file.

## Modules

- **[`fixel_index`]** — `FixelIndex`, a [kiddo](https://crates.io/crates/kiddo)
  KD-tree over per-fixel world positions, with sub-voxel jitter to handle
  coincident peaks and per-fixel amplitude (QA) metadata loaded from
  `dpf/amplitude` when present. Supports build-time filtering by amplitude
  Otsu, threshold, or arbitrary per-voxel predicates.

- **[`ptt`]** — Parallel Transport Tractography on fixels. Includes the
  analytic propagator (`prep_propagator`, `walk`), parallel-transport frames
  (`PtfFrame`, `frame_at_fixel_handle`), pure-fixel data support
  (`data_support`), and arc-likelihood scoring with a closure-based variant
  (`arc_likelihood_with`) so callers can plug in their own per-sample scoring.
  Adapts the PTT framework (Aydogan & Shi 2021) from continuous ODFs to the
  discrete fixel set already present in an ODX.

- **[`trace`]** — Yeh-style deterministic fixel-set tractography for
  visualisation. Given a set of fixel ids, `trace_within_fixels` produces
  one polyline per seed that stays inside the set.

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

## In-place DPF append

```rust
use odx_tractography::writeback;

writeback::write_dpf_u8(&path, "my_selection", &selected_bytes)?;
writeback::write_dpf_f32(&path, "my_score", &per_fixel_scores)?;
writeback::write_voxel_group(&path, "my_bundle", &voxel_ids)?;
```

These mutate the ODX at `path` directly — copy first if you want to keep
the source pristine.

## License

Dual-licensed under MIT or Apache-2.0 at your option.
