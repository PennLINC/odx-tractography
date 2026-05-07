//! Reference-free fixel tractography on ODX datasets.
//!
//! This crate hosts the algorithm-only tractography primitives that don't
//! require a reference template:
//!
//! - [`fixel_index::FixelIndex`] — KD-tree over per-fixel positions with
//!   amplitude metadata, the spatial primitive shared by everything else.
//! - [`ptt`] — reference-free Parallel Transport Tractography: propagator
//!   math, parallel-transport frames, pure-fixel arc likelihood, and
//!   trajectory utilities.
//! - [`trace`] — Yeh-style fixel-set deterministic tractography for
//!   visualisation.
//! - [`writeback`] — append per-fixel scalars and per-bundle voxel groups
//!   into an existing ODX archive in place (no full rewrite).
//!
//! For TRX writing, use [`trx_rs::Tractogram`] directly — its
//! [`push_streamline`](https://docs.rs/trx-rs/latest/trx_rs/tractogram/struct.Tractogram.html#method.push_streamline)
//! / [`insert_group`](https://docs.rs/trx-rs/latest/trx_rs/tractogram/struct.Tractogram.html#method.insert_group)
//! / [`save`](https://docs.rs/trx-rs/latest/trx_rs/tractogram/struct.Tractogram.html#method.save)
//! API supersedes the hand-rolled writer this crate previously exposed.
//! Re-exported here for convenience.
//!
//! Reference-template features (ROI gating, Reeb-graph topology, bundle
//! validation, region-growing segmentation) live in the consuming
//! `odx-bundles` crate.

#![warn(rust_2018_idioms)]

pub mod fixel_index;
pub mod ptt;
pub mod trace;
pub mod writeback;

pub use fixel_index::{FixelHandle, FixelId, FixelIndex};
pub use ptt::{
    arc_likelihood, arc_likelihood_with, best_arc_likelihood, capture_visited_fixels, data_support,
    frame_at_fixel_handle, prep_propagator, walk as ptt_walk, PtfFrame, PttParams, PttTrajectory,
};
pub use trace::{trace_within_fixels, TraceParams};

// Re-export trx-rs's Tractogram + Header + DType so consumers don't need
// to depend on trx-rs separately.
pub use trx_rs::{DType as TrxDType, Header as TrxHeader, Tractogram};
