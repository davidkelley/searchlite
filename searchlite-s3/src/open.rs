//! Stage 10c: top-level `Index` constructor for S3-backed read-only
//! deployments.
//!
//! See [`open_index_read_only`] for the contract. The shape mirrors
//! the bake-locally-then-serve-from-cloud pattern: callers run a
//! mutable Index locally to commit + compact + bake content hashes,
//! upload via [`crate::sync_to_s3`], then serve via
//! [`open_index_read_only`] against the same `S3Config`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use searchlite_core::api::types::IndexOptions;
use searchlite_core::storage::blob::BlobStore;
use searchlite_core::storage::{
  BlobStoreAdapter, CachedBlobStore, Storage, DEFAULT_CACHE_CAPACITY_BYTES,
};
use searchlite_core::Index;

use crate::config::S3Config;
use crate::store::S3BlobStore;

/// Open an index read-only against an S3-compatible backend.
///
/// ## S3 namespace mapping
///
/// The S3 namespace lives in [`S3Config::prefix`]. The index helpers
/// operate against a **logical-empty root** so that:
///
/// 1. `Manifest::manifest_path(opts.path)` produces the bare key
///    `MANIFEST.json` (relative).
/// 2. `BlobStoreAdapter::resolve` joins it against the adapter's
///    own root (also empty) → still `MANIFEST.json`.
/// 3. `S3BlobStore::resolve_key` is the **only** place the prefix
///    is added: `MANIFEST.json` → `prefix/MANIFEST.json`.
///
/// Without this discipline, an `IndexOptions.path = "idx"` plus
/// `BlobStoreAdapter::root = "idx"` would either double-prefix
/// (`idx/idx/MANIFEST.json`) or pass an absolute path to
/// `S3BlobStore::resolve_key`, which rejects absolute keys.
///
/// ## Cache layer
///
/// `CachedBlobStore<S3BlobStore>` with the default RAM capacity
/// (64 MiB byte-weighted LRU). Both the BlobStore-side argument and
/// the BlobStoreAdapter wrap the same `Arc<CachedBlobStore<...>>`,
/// so the Stage 8a [`Storage::as_blob_store`] hint avoids
/// double-wrapping in `default_blob_store`.
///
/// ## Read-only enforcement
///
/// `IndexOptions.read_only = true` is set unconditionally:
/// `Index::writer`, `compact`, and `merge_segments` will all error
/// (Stage 10a). Pending-manifest recovery is also fail-closed —
/// reopen the source index mutably to reconcile, then re-sync.
pub async fn open_index_read_only(s3_config: S3Config) -> Result<Index> {
  let s3 = Arc::new(S3BlobStore::new(s3_config).await?);
  let cached: Arc<dyn BlobStore> = Arc::new(CachedBlobStore::with_capacity(
    s3,
    DEFAULT_CACHE_CAPACITY_BYTES,
  ));
  // Logical-empty root: every key the index emits is relative
  // (Stage 9a v2 invariant) and gets prefix-joined inside
  // `S3BlobStore::resolve_key` — never here.
  let logical_root = PathBuf::new();
  let adapter: Arc<dyn Storage> =
    Arc::new(BlobStoreAdapter::new(cached.clone(), logical_root.clone()));
  let opts = IndexOptions {
    path: logical_root,
    create_if_missing: false,
    read_only: true,
    ..Default::default()
  };
  Index::open_with_storage_and_blob_store(opts, adapter, cached)
}
