//! Top-level `Index` constructors for S3-backed read-only deployments.
//!
//! See [`open_index_read_only`] for the contract. The shape mirrors
//! the bake-locally-then-serve-from-cloud pattern: callers run a
//! mutable Index locally to commit + compact + bake content hashes,
//! upload via [`crate::sync_to_s3`], then serve via
//! [`open_index_read_only`] against the same `S3Config`.
//!
//! [`open_index_read_only_with_options`] lets the caller customize
//! [`IndexOptions`] when the defaults aren't a fit — for example,
//! switching `checksum_policy` to
//! [`ChecksumPolicy::TrustManifest`](searchlite_core::api::types::ChecksumPolicy::TrustManifest)
//! to skip whole-file SHA-256 verification on every fresh
//! `Index::reader()`.

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

/// Open an index read-only against an S3-compatible backend, using
/// default [`IndexOptions`] for everything except `path`,
/// `create_if_missing`, and `read_only`. For full control over options
/// (including `checksum_policy`), use
/// [`open_index_read_only_with_options`].
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
/// so the [`Storage::as_blob_store`] hint avoids double-wrapping in
/// `default_blob_store`.
///
/// ## Read-only enforcement
///
/// `IndexOptions.read_only = true` is set unconditionally:
/// `Index::writer`, `compact`, and `merge_segments` will all error.
/// Pending-manifest recovery is also fail-closed — reopen the source
/// index mutably to reconcile, then re-sync.
pub async fn open_index_read_only(s3_config: S3Config) -> Result<Index> {
  open_index_read_only_with_options(s3_config, IndexOptions::default()).await
}

/// Like [`open_index_read_only`] but lets the caller customize
/// [`IndexOptions`].
///
/// Use this when you need to override a default that
/// [`open_index_read_only`] doesn't expose — most commonly
/// `checksum_policy`, to switch from the default
/// [`ChecksumPolicy::Strict`](searchlite_core::api::types::ChecksumPolicy::Strict)
/// to
/// [`ChecksumPolicy::TrustManifest`](searchlite_core::api::types::ChecksumPolicy::TrustManifest)
/// or
/// [`ChecksumPolicy::Audit`](searchlite_core::api::types::ChecksumPolicy::Audit).
///
/// Three fields on the supplied [`IndexOptions`] are forced regardless
/// of what the caller passes:
///
/// * `path` — overridden to the logical-empty root (see the namespace
///   mapping notes on [`open_index_read_only`]).
/// * `create_if_missing` — forced to `false`; S3-backed readers never
///   create indexes.
/// * `read_only` — forced to `true`; mutators on the resulting `Index`
///   will all error.
///
/// Every other field on the supplied options is preserved.
///
/// # Example: skip whole-file checksum verification
///
/// ```no_run
/// use searchlite_core::api::types::{ChecksumPolicy, IndexOptions};
/// use searchlite_s3::{open_index_read_only_with_options, S3Config, S3Credentials};
/// # async fn example() -> anyhow::Result<()> {
/// let s3_config = S3Config {
///     region: "us-east-1".into(),
///     bucket: "my-search-indexes".into(),
///     prefix: Some("products/v1".into()),
///     credentials: S3Credentials::LoadFromEnv,
///     ..S3Config::aws_default()
/// };
/// let opts = IndexOptions {
///     checksum_policy: ChecksumPolicy::TrustManifest,
///     ..Default::default()
/// };
/// let index = open_index_read_only_with_options(s3_config, opts).await?;
/// # let _ = index;
/// # Ok(())
/// # }
/// ```
pub async fn open_index_read_only_with_options(
  s3_config: S3Config,
  mut opts: IndexOptions,
) -> Result<Index> {
  let s3 = Arc::new(S3BlobStore::new(s3_config).await?);
  let cached: Arc<dyn BlobStore> = Arc::new(CachedBlobStore::with_capacity(
    s3,
    DEFAULT_CACHE_CAPACITY_BYTES,
  ));
  // Logical-empty root: every key the index emits is relative and
  // gets prefix-joined inside `S3BlobStore::resolve_key` — never here.
  let logical_root = PathBuf::new();
  let adapter: Arc<dyn Storage> =
    Arc::new(BlobStoreAdapter::new(cached.clone(), logical_root.clone()));
  opts.path = logical_root;
  opts.create_if_missing = false;
  opts.read_only = true;
  Index::open_with_storage_and_blob_store(opts, adapter, cached)
}
