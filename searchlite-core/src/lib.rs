//! searchlite-core: embedded search engine core.
//! v0.1 focuses on a simple single-writer/multi-reader index with BM25.

pub mod analysis;
pub mod api;
mod index;
pub mod query;
pub mod storage;
pub mod util;

#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "vectors")]
pub mod vectors;

/// Document identifier within a segment.
pub type DocId = u32;

pub use index::wal;
pub use index::{manifest::Schema, Index};
