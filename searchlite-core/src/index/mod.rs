use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};

use crate::api::errors::WriteKeyError;
use crate::api::types::{Document, IndexOptions, StorageType};
use crate::index::directory::ensure_root;
use crate::index::manifest::{Manifest, Schema};
use crate::index::segment::SegmentWriter;
use crate::index::wal::Wal;
use crate::storage::{BlobStore, FsStorage, InMemoryStorage, Storage, StorageAsBlobStore};
#[cfg(feature = "write-key")]
use crate::util::write_key::derive_write_key_meta;

pub mod codec;
pub mod directory;
pub mod docstore;
pub mod fastfields;
pub mod highlight;
pub mod json_schema;
pub mod manifest;
pub mod merge;
pub mod postings;
pub mod segment;
pub mod terms;
pub mod wal;

pub struct Index {
  pub(crate) inner: Arc<InnerIndex>,
}

pub(crate) struct InnerIndex {
  pub path: PathBuf,
  pub options: IndexOptions,
  pub manifest: RwLock<Manifest>,
  pub writer_lock: Mutex<()>,
  pub storage: Arc<dyn Storage>,
  /// `BlobStore`-shaped view over `storage`. Stage 8a wires this in so
  /// segment readers can open `Object` handles for postings and (in 8b)
  /// docstore via bounded `Object::read_range` calls. Default is
  /// `StorageAsBlobStore::new(storage.clone())` — a transitional bridge
  /// that serves raw file bytes without the LocalBlobStore header
  /// format; existing segment files (written via `Storage::open_write`)
  /// are routable as-is. Stage 9+ replaces this with a real BlobStore
  /// (LocalBlobStore for local FS, S3BlobStore for cloud).
  pub blob_store: Arc<dyn BlobStore>,
  /// Cache of immutable `SegmentCore`s keyed by `(segment_id, fingerprint)`.
  /// `IndexReader::open` consults this cache when materializing per-manifest
  /// `SegmentReader` views, so the expensive segment-open work (checksum
  /// verification, term-dict load, fast-field parse) runs at most once per
  /// `(id, fingerprint)` per process — not once per reader open.
  pub(crate) segment_cache: crate::index::segment::SegmentCache,
  /// Monotonic count of successful `Index::reader()` calls. Exposed via
  /// `Index::reader_open_count` so pooling/caching layers can be regression-
  /// tested without relying on wall-clock heuristics.
  pub(crate) reader_opens: AtomicUsize,
}

impl Index {
  pub fn create(path: &Path, schema: Schema, opts: IndexOptions) -> Result<Self> {
    Self::create_with_write_key(path, schema, opts, None)
  }

  pub fn create_with_write_key(
    path: &Path,
    schema: Schema,
    opts: IndexOptions,
    write_key: Option<&str>,
  ) -> Result<Self> {
    let storage = storage_from_options(&opts);
    Self::create_with_storage_and_key(path, schema, opts, storage, write_key)
  }

  pub fn create_with_storage(
    path: &Path,
    schema: Schema,
    opts: IndexOptions,
    storage: Arc<dyn Storage>,
  ) -> Result<Self> {
    Self::create_with_storage_and_key(path, schema, opts, storage, None)
  }

  pub fn create_with_storage_and_key(
    path: &Path,
    schema: Schema,
    opts: IndexOptions,
    storage: Arc<dyn Storage>,
    write_key: Option<&str>,
  ) -> Result<Self> {
    let blob_store = default_blob_store(&storage);
    Self::create_with_storage_blob_store_and_key(path, schema, opts, storage, blob_store, write_key)
  }

  /// Stage 8a: explicit blob_store constructor. Production callers use
  /// the storage-only variants which build a `StorageAsBlobStore`
  /// bridge; tests inject a custom `BlobStore` (e.g. `RecordingBlobStore`)
  /// via this entry point to assert range-read counts. Stage 9+ will
  /// expose this for cloud-backed deployments where the native
  /// BlobStore differs from the local Storage.
  pub fn create_with_storage_and_blob_store(
    path: &Path,
    schema: Schema,
    opts: IndexOptions,
    storage: Arc<dyn Storage>,
    blob_store: Arc<dyn BlobStore>,
  ) -> Result<Self> {
    Self::create_with_storage_blob_store_and_key(path, schema, opts, storage, blob_store, None)
  }

  pub fn create_with_storage_blob_store_and_key(
    path: &Path,
    schema: Schema,
    opts: IndexOptions,
    storage: Arc<dyn Storage>,
    blob_store: Arc<dyn BlobStore>,
    write_key: Option<&str>,
  ) -> Result<Self> {
    // Stage 10a v2 [P2] (Codex review): a read-only index cannot be
    // *created* — creation issues writes (`ensure_root`,
    // `Manifest::store`). Reject before touching storage so a
    // read-only token (S3/R2) can't be tricked into issuing writes
    // through a misconfigured `IndexOptions`.
    if opts.read_only {
      bail!(
        "Index::create*: cannot create an index with IndexOptions.read_only = true; \
         creation requires writes — open an existing index with read_only = true to serve it"
      );
    }
    let mut opts = opts;
    opts.path = path.to_path_buf();
    schema.validate_config()?;
    ensure_root(storage.as_ref(), path)?;
    #[allow(unused_mut)]
    let mut manifest = Manifest::new(schema);
    if let Some(key) = write_key {
      #[cfg(feature = "write-key")]
      {
        manifest.write_key = Some(derive_write_key_meta(key, None)?);
      }
      #[cfg(not(feature = "write-key"))]
      {
        let _ = key;
        crate::util::write_key::require_write_key_feature()?;
      }
    }
    manifest.store(storage.as_ref(), &Manifest::manifest_path(path))?;
    let inner = Arc::new(InnerIndex {
      path: path.to_path_buf(),
      storage,
      blob_store,
      options: opts,
      manifest: RwLock::new(manifest),
      writer_lock: Mutex::new(()),
      segment_cache: crate::index::segment::SegmentCache::new(),
      reader_opens: AtomicUsize::new(0),
    });
    Ok(Self { inner })
  }

  pub fn open(opts: IndexOptions) -> Result<Self> {
    let storage = storage_from_options(&opts);
    Self::open_with_storage(opts, storage)
  }

  pub fn open_with_storage(opts: IndexOptions, storage: Arc<dyn Storage>) -> Result<Self> {
    let blob_store = default_blob_store(&storage);
    Self::open_with_storage_and_blob_store(opts, storage, blob_store)
  }

  /// Stage 8a: explicit blob_store entry point for `Index::open`. See
  /// `create_with_storage_and_blob_store` for the rationale.
  pub fn open_with_storage_and_blob_store(
    opts: IndexOptions,
    storage: Arc<dyn Storage>,
    blob_store: Arc<dyn BlobStore>,
  ) -> Result<Self> {
    // Stage 10a v2 [P2] (Codex review): when `read_only` is set, the
    // open path must NOT issue any writes — that includes
    // `ensure_root` (creates the dir if missing), the auto-create
    // manifest write below, and the recovery promote in
    // `reconcile_pending_manifest`. The intended deployment is
    // serving a baked index from a read-only S3/R2 token; any of
    // those writes would otherwise issue a 403 from the storage
    // layer with a backend-specific message rather than a clear
    // "index is read-only" error.
    let manifest_path = Manifest::manifest_path(&opts.path);
    if opts.read_only {
      // Recovery is a write. Auto-create is a write. Both must
      // be refused before any storage-layer call.
      if !storage.exists(&manifest_path) {
        if opts.create_if_missing {
          bail!(
            "Index::open: cannot auto-create an index with read_only = true; \
             create_if_missing and read_only are mutually exclusive"
          );
        }
        bail!("index does not exist at {manifest_path:?}");
      }
      // Stage 10a v3 [P1] (Codex review): if a `MANIFEST.json.pending`
      // exists, it may carry a durably-committed batch (BUG-018: the
      // WAL has crossed the commit fence but the live manifest
      // publish was interrupted). Loading the live manifest without
      // first promoting the pending file would silently serve a
      // stale state and hide committed docs. Read-only mode cannot
      // promote-or-discard the pending file (both are writes), so
      // fail closed with a clear error and ask the operator to
      // reopen mutably to reconcile.
      let pending_path = Manifest::manifest_pending_path(&opts.path);
      if storage.exists(&pending_path) {
        bail!(
          "Index::open: a `MANIFEST.json.pending` exists at {pending_path:?} \
           and read_only mode cannot perform recovery (promote or discard the \
           pending file). Reopen with read_only = false once to reconcile, \
           then reopen read-only to serve. This protects against silently \
           serving stale state when the pending file carries a durably \
           committed batch."
        );
      }
      // Existing manifest. Skip ensure_root (might create dirs) and
      // skip pending-manifest reconciliation (would atomic_write).
      let manifest = Manifest::load(storage.as_ref(), &manifest_path)?;
      let inner = Arc::new(InnerIndex {
        path: opts.path.clone(),
        storage,
        blob_store,
        options: opts,
        manifest: RwLock::new(manifest),
        writer_lock: Mutex::new(()),
        segment_cache: crate::index::segment::SegmentCache::new(),
        reader_opens: AtomicUsize::new(0),
      });
      return Ok(Self { inner });
    }

    ensure_root(storage.as_ref(), &opts.path)?;
    // BUG-018 recovery: if a previous `Writer::commit` crashed between the
    // WAL commit fence and the live manifest publish, finish promoting the
    // staged manifest now (or discard it if the WAL never crossed the fence).
    reconcile_pending_manifest(storage.as_ref(), &opts.path, &manifest_path)?;
    let manifest = if storage.exists(&manifest_path) {
      Manifest::load(storage.as_ref(), &manifest_path)?
    } else if opts.create_if_missing {
      let schema = Schema::default_text_body();
      let m = Manifest::new(schema);
      m.store(storage.as_ref(), &manifest_path)?;
      m
    } else {
      bail!("index does not exist at {manifest_path:?}");
    };
    let inner = Arc::new(InnerIndex {
      path: opts.path.clone(),
      storage,
      blob_store,
      options: opts,
      manifest: RwLock::new(manifest),
      writer_lock: Mutex::new(()),
      segment_cache: crate::index::segment::SegmentCache::new(),
      reader_opens: AtomicUsize::new(0),
    });
    Ok(Self { inner })
  }

  pub fn writer(&self) -> Result<crate::api::writer::IndexWriter> {
    self.writer_with_key(None)
  }

  pub fn writer_with_key(
    &self,
    write_key: Option<&str>,
  ) -> Result<crate::api::writer::IndexWriter> {
    self.ensure_mutable("writer")?;
    crate::api::writer::IndexWriter::new(self.inner.clone(), write_key)
  }

  /// Stage 10a: gate every mutator entry point on `IndexOptions.read_only`.
  /// Returns a clear error rather than letting the mutation proceed and
  /// fail later at the storage/blob-store layer, where the message would
  /// be backend-specific (e.g. `403 Forbidden` from a read-only S3 token).
  pub(crate) fn ensure_mutable(&self, mutator: &'static str) -> Result<()> {
    if self.inner.options.read_only {
      bail!(
        "Index::{mutator}: index is open read-only \
         (IndexOptions.read_only = true); reopen with read_only = false to mutate"
      );
    }
    Ok(())
  }

  pub fn reader(&self) -> Result<crate::api::reader::IndexReader> {
    let reader = crate::api::reader::IndexReader::open(self.inner.clone())?;
    self.inner.reader_opens.fetch_add(1, Ordering::Relaxed);
    Ok(reader)
  }

  /// Async surface for `reader()`. In Stage 4 the body is the same sync work
  /// behind an `async fn`; the future has no internal `.await` points and
  /// resolves on first poll. Stage 8 replaces this body with real async I/O
  /// against `BlobStore`, at which point callers in async contexts (Workers,
  /// future migrations of `searchlite-http`) get genuine non-blocking opens.
  ///
  /// Sync callers should keep using `reader()` directly — wrapping this
  /// future in `block_on` from inside an active Tokio runtime panics, and
  /// today's body has no async work to begin with.
  pub async fn reader_async(&self) -> Result<crate::api::reader::IndexReader> {
    self.reader()
  }

  /// Number of successful `reader()` calls issued against this `Index`.
  ///
  /// Exposed as an observability hook so reader-reuse behavior (e.g. the
  /// bounded pool in HTTP `multi_search`) can be exercised in regression
  /// tests without relying on timing or log parsing.
  #[doc(hidden)]
  pub fn reader_open_count(&self) -> usize {
    self.inner.reader_opens.load(Ordering::Relaxed)
  }

  /// Number of times a `SegmentCore` was loaded from storage in this index's
  /// lifetime. Cache hits do not increment this counter, so two
  /// `Index::reader()` calls over a stable manifest should leave it
  /// unchanged after the first. Used by Stage 1 regression tests to assert
  /// segment caching is effective.
  #[doc(hidden)]
  pub fn segment_core_loads(&self) -> usize {
    self.inner.segment_cache.loads()
  }

  pub fn compact(&self) -> Result<()> {
    self.compact_with_key(None)
  }

  pub fn compact_with_key(&self, write_key: Option<&str>) -> Result<()> {
    self.ensure_mutable("compact")?;
    let _writer_guard = self.inner.writer_lock.lock();
    let reader = self.reader()?;
    let manifest_snapshot = reader.manifest.clone();
    ensure_compact_safe(&manifest_snapshot.schema)?;
    #[allow(unused_mut)]
    let mut write_binding: Option<Vec<u8>> = None;
    let mut seg_bindings: Vec<Vec<u8>> = manifest_snapshot
      .segments
      .iter()
      .filter_map(|s| s.write_binding_b64.as_deref())
      .map(|b64| {
        BASE64
          .decode(b64)
          .map_err(|e| anyhow!("invalid base64 in segment write_binding_b64: {e}"))
      })
      .collect::<Result<Vec<_>>>()?;
    for seg in manifest_snapshot.segments.iter() {
      let resolved = seg.paths.resolve(self.inner.storage.root());
      let bytes = self
        .inner
        .storage
        .read_to_end(&resolved.meta)
        .map_err(|e| anyhow!("failed to read segment meta {}: {e}", seg.id))?;
      let seg_meta: crate::index::segment::SegmentFileMeta = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow!("failed to parse segment meta {}: {e}", seg.id))?;
      if let Some(b64) = seg_meta.write_binding_b64.as_deref() {
        let decoded = BASE64
          .decode(b64)
          .map_err(|e| anyhow!("invalid base64 in segment metadata write_binding_b64: {e}"))?;
        seg_bindings.push(decoded);
      }
    }
    if manifest_snapshot.write_key.is_some() || !seg_bindings.is_empty() {
      #[cfg(feature = "write-key")]
      {
        let key = write_key.ok_or(WriteKeyError::Required)?;
        if let Some(meta) = manifest_snapshot.write_key.as_ref() {
          crate::util::write_key::verify_write_key(key, meta)?;
        }
        let binding = crate::util::write_key::binding_for_uuid(key, &manifest_snapshot.uuid);
        for seg_binding in seg_bindings.iter() {
          if !crate::util::write_key::verify_binding(seg_binding, &binding) {
            return Err(WriteKeyError::Mismatch("segment binding; index may be tampered").into());
          }
        }
        write_binding = Some(binding);
      }
      #[cfg(not(feature = "write-key"))]
      {
        let _ = write_key;
        return Err(WriteKeyError::FeatureDisabled.into());
      }
    }
    if manifest_snapshot.segments.len() <= 1 {
      return Ok(());
    }
    let old_segments = manifest_snapshot.segments.clone();
    let inner = &self.inner;
    let schema = manifest_snapshot.schema.clone();
    let generation = manifest_snapshot
      .segments
      .iter()
      .map(|s| s.generation)
      .max()
      .unwrap_or(0)
      + 1;
    let docs = reader.segments.iter().flat_map(|seg| {
      (0..seg.meta.doc_count).filter_map(move |doc_id| {
        if seg.is_deleted(doc_id) {
          return None;
        }
        Some(seg.get_doc(doc_id).and_then(|doc_json| {
          let map = doc_json.as_object().ok_or_else(|| {
            anyhow!(
              "document {doc_id} in segment {} is not an object",
              seg.meta.id
            )
          })?;
          let fields = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
          Ok(Document { fields })
        }))
      })
    });
    let mut manifest_guard = inner.manifest.write();
    let writer = SegmentWriter::new(
      &inner.path,
      &schema,
      inner.options.enable_positions,
      cfg!(feature = "zstd"),
      inner.storage.clone(),
      write_binding.clone(),
    );
    let new_seg = writer.write_segment_from_iter(docs, generation)?;
    manifest_guard.segments = vec![new_seg];
    manifest_guard.committed_at = Utc::now().to_rfc3339();
    // Stage 9a v3 [P2] (Codex review): if the in-memory manifest was
    // loaded as legacy v1, upgrade it before publish — `Manifest::store`
    // (via `serialize_for_write`) refuses to write below the latest
    // version. Without this, compact on a legacy index would fail
    // instead of upgrading the manifest to v2.
    manifest_guard
      .upgrade_to_latest(&inner.path)
      .context("upgrading manifest to latest version before compact publish")?;
    manifest_guard.store(
      inner.storage.as_ref(),
      &Manifest::manifest_path(&inner.path),
    )?;
    drop(manifest_guard);
    cleanup_segments(inner.storage.as_ref(), &old_segments)?;
    Ok(())
  }

  /// Merge a specific set of segments into a single new segment.
  ///
  /// Unlike `compact`, this only touches the listed segments and leaves the
  /// rest of the index untouched. Deleted documents within the merged
  /// segments are excluded from the output.
  pub fn merge_segments(&self, segment_ids: &[String], write_key: Option<&str>) -> Result<()> {
    if segment_ids.is_empty() {
      return Ok(());
    }
    self.ensure_mutable("merge_segments")?;
    let _writer_guard = self.inner.writer_lock.lock();
    let manifest_snapshot = self.inner.manifest.read().clone();
    ensure_compact_safe(&manifest_snapshot.schema)?;

    // Identify which segments participate in the merge (dedup input IDs).
    let merge_set: std::collections::HashSet<&str> =
      segment_ids.iter().map(|s| s.as_str()).collect();
    let merge_metas: Vec<&crate::index::manifest::SegmentMeta> = manifest_snapshot
      .segments
      .iter()
      .filter(|s| merge_set.contains(s.id.as_str()))
      .collect();
    // Verify all unique requested segment IDs exist before deciding there is
    // nothing to merge, so typo'd or half-committed IDs surface as errors
    // rather than being silently swallowed by the `< 2` early return below.
    if merge_metas.len() != merge_set.len() {
      let found: std::collections::HashSet<&str> =
        merge_metas.iter().map(|s| s.id.as_str()).collect();
      let mut missing: Vec<&str> = merge_set
        .iter()
        .copied()
        .filter(|id| !found.contains(id))
        .collect();
      missing.sort_unstable();
      bail!(
        "some segment IDs not found in manifest: {:?} (requested {}, found {})",
        missing,
        merge_set.len(),
        merge_metas.len()
      );
    }
    if merge_metas.len() < 2 {
      // Nothing useful to merge.
      return Ok(());
    }

    // Handle write-key bindings.
    #[allow(unused_mut)]
    let mut write_binding: Option<Vec<u8>> = None;
    let mut seg_bindings: Vec<Vec<u8>> = merge_metas
      .iter()
      .filter_map(|s| s.write_binding_b64.as_deref())
      .map(|b64| {
        BASE64
          .decode(b64)
          .map_err(|e| anyhow!("invalid base64 in segment write_binding_b64: {e}"))
      })
      .collect::<Result<Vec<_>>>()?;
    for seg in merge_metas.iter() {
      let resolved = seg.paths.resolve(self.inner.storage.root());
      let bytes = self
        .inner
        .storage
        .read_to_end(&resolved.meta)
        .map_err(|e| anyhow!("failed to read segment meta {}: {e}", seg.id))?;
      let seg_meta: crate::index::segment::SegmentFileMeta = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow!("failed to parse segment meta {}: {e}", seg.id))?;
      if let Some(b64) = seg_meta.write_binding_b64.as_deref() {
        let decoded = BASE64
          .decode(b64)
          .map_err(|e| anyhow!("invalid base64 in segment metadata write_binding_b64: {e}"))?;
        seg_bindings.push(decoded);
      }
    }
    if manifest_snapshot.write_key.is_some() || !seg_bindings.is_empty() {
      #[cfg(feature = "write-key")]
      {
        let key = write_key.ok_or(WriteKeyError::Required)?;
        if let Some(meta) = manifest_snapshot.write_key.as_ref() {
          crate::util::write_key::verify_write_key(key, meta)?;
        }
        let binding = crate::util::write_key::binding_for_uuid(key, &manifest_snapshot.uuid);
        for seg_binding in seg_bindings.iter() {
          if !crate::util::write_key::verify_binding(seg_binding, &binding) {
            return Err(WriteKeyError::Mismatch("segment binding; index may be tampered").into());
          }
        }
        write_binding = Some(binding);
      }
      #[cfg(not(feature = "write-key"))]
      {
        let _ = write_key;
        return Err(WriteKeyError::FeatureDisabled.into());
      }
    }

    // Open segment readers for the merge set only.
    let inner = &self.inner;
    let schema = manifest_snapshot.schema.clone();
    // Compaction reads through the same cache as `IndexReader::open` so a
    // segment that was just opened by a query thread isn't re-loaded from
    // storage here. Cache misses (first time a segment is touched) still go
    // through the full `SegmentCore::load` path.
    let load_ctx = crate::index::segment::SegmentLoadCtx::from_options(&inner.options);
    let readers: Vec<crate::index::segment::SegmentReader> = merge_metas
      .iter()
      .map(|seg| -> Result<_> {
        let core = inner
          .segment_cache
          .get_or_load(seg, &load_ctx, inner.storage.clone())?;
        crate::index::segment::SegmentReader::from_core(
          core,
          (*seg).clone(),
          inner.blob_store.clone(),
          inner.storage.root(),
        )
      })
      .collect::<Result<Vec<_>>>()?;

    let generation = manifest_snapshot
      .segments
      .iter()
      .map(|s| s.generation)
      .max()
      .unwrap_or(0)
      + 1;

    // Build an iterator over live docs from the merge segments.
    let docs = readers.iter().flat_map(|seg| {
      (0..seg.meta.doc_count).filter_map(move |doc_id| {
        if seg.is_deleted(doc_id) {
          return None;
        }
        Some(seg.get_doc(doc_id).and_then(|doc_json| {
          let map = doc_json.as_object().ok_or_else(|| {
            anyhow!(
              "document {doc_id} in segment {} is not an object",
              seg.meta.id
            )
          })?;
          let fields = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
          Ok(crate::api::types::Document { fields })
        }))
      })
    });

    let writer = SegmentWriter::new(
      &inner.path,
      &schema,
      inner.options.enable_positions,
      cfg!(feature = "zstd"),
      inner.storage.clone(),
      write_binding.clone(),
    );
    let new_seg = writer.write_segment_from_iter(docs, generation)?;

    // Build new manifest: keep non-merged segments, append the new merged one.
    let old_merged_segments: Vec<crate::index::manifest::SegmentMeta> = manifest_snapshot
      .segments
      .iter()
      .filter(|s| merge_set.contains(s.id.as_str()))
      .cloned()
      .collect();
    let mut manifest_guard = inner.manifest.write();
    manifest_guard.segments = manifest_snapshot
      .segments
      .iter()
      .filter(|s| !merge_set.contains(s.id.as_str()))
      .cloned()
      .collect();
    manifest_guard.segments.push(new_seg);
    manifest_guard.committed_at = Utc::now().to_rfc3339();
    // Stage 9a v3 [P2] (Codex review): same upgrade-before-store dance
    // as `compact` — without this, `merge_segments` against a legacy
    // v1 index would fail at `store()` since `serialize_for_write`
    // refuses to write a v1 manifest.
    manifest_guard
      .upgrade_to_latest(&inner.path)
      .context("upgrading manifest to latest version before merge_segments publish")?;
    manifest_guard.store(
      inner.storage.as_ref(),
      &Manifest::manifest_path(&inner.path),
    )?;
    drop(manifest_guard);

    // Clean up old segment files.
    cleanup_segments(inner.storage.as_ref(), &old_merged_segments)?;
    Ok(())
  }

  pub fn manifest(&self) -> Manifest {
    self.inner.manifest.read().clone()
  }
}

impl InnerIndex {
  pub(crate) fn wal(&self) -> Result<Wal> {
    let wal_path = directory::wal_path(&self.path);
    Wal::open(self.storage.clone(), &wal_path)
  }

  pub(crate) fn manifest_path(&self) -> PathBuf {
    Manifest::manifest_path(&self.path)
  }
}

fn storage_from_options(opts: &IndexOptions) -> Arc<dyn Storage> {
  match opts.storage {
    StorageType::Filesystem => Arc::new(FsStorage::new(opts.path.clone())),
    StorageType::InMemory => Arc::new(InMemoryStorage::new(opts.path.clone())),
  }
}

/// Build the default `Arc<dyn BlobStore>` for an `Index` from its
/// `Storage`. Stage 8a uses `StorageAsBlobStore` as a transitional
/// bridge: it serves raw file bytes (no LocalBlobStore header) so
/// existing segment files written via `Storage::open_write` can be
/// read through the BlobStore surface without rewriting them. Stage 9+
/// can pass an explicit blob_store via the `*_with_storage_and_blob_store`
/// constructors when a richer backend (e.g. S3) is configured.
///
/// Stage 8a [P1] (Codex review): if `storage` is itself a
/// `BlobStoreAdapter` (or any future Storage wrapping a BlobStore),
/// prefer the inner store via [`Storage::as_blob_store`] instead of
/// wrapping again. Wrapping a `BlobStoreAdapter` in `StorageAsBlobStore`
/// produces a nested `block_on` chain (segment reader →
/// `StorageAsBlobStore::stat` → `BlobStoreAdapter::*` → another
/// `block_on`) that the `LocalPool` executor refuses with `EnterError`.
fn default_blob_store(storage: &Arc<dyn Storage>) -> Arc<dyn BlobStore> {
  storage
    .as_blob_store()
    .unwrap_or_else(|| Arc::new(StorageAsBlobStore::new(storage.clone())))
}

/// Reconcile a `MANIFEST.json.pending` left behind by a crashed `Writer::commit`.
///
/// The commit pipeline writes the staged manifest to `MANIFEST.json.pending`
/// *before* appending the WAL commit record (the durability fence), then
/// promotes the staging file to `MANIFEST.json` once the fence has been crossed.
/// A crash between those steps therefore leaves one of two recoverable states:
///
/// * **WAL contains at least one `Commit` record** — the batch was durably
///   committed but the live manifest publish (or the cleanup that follows
///   it) did not complete. The staged file is the authoritative manifest for
///   that batch and we promote it now so the next reader/writer sees a
///   consistent index. Uncommitted entries appended *after* the durable
///   `Commit` (e.g. an `AddDoc` written before the crash) do not invalidate
///   this — `last_pending_ops` correctly replays only post-commit entries.
///
/// * **No `Commit` record in WAL** — the WAL never crossed the durability
///   fence, so the staged manifest belongs to a batch that was effectively
///   rolled back. The pending entries still in the WAL will replay through
///   the next commit; we discard the staging file.
///
/// In either case the staging file is removed before we return, so subsequent
/// opens see a clean slate. This is the BUG-018 reconciler.
fn reconcile_pending_manifest(
  storage: &dyn Storage,
  root: &Path,
  manifest_path: &Path,
) -> Result<()> {
  let pending_path = Manifest::manifest_pending_path(root);
  if !storage.exists(&pending_path) {
    return Ok(());
  }
  let wal_path = directory::wal_path(root);
  if Wal::contains_commit(storage, &wal_path)? {
    let pending_data = storage
      .read_to_end(&pending_path)
      .with_context(|| format!("reading staged manifest at {pending_path:?}"))?;
    // Stage 9a v3 [P1] (Codex review): legacy v1 `.pending` files
    // produced by pre-Stage-9 builds are still valid recovery state
    // (the WAL fence is durable; the pending bytes describe a
    // committed batch). Upgrade them to v2 before validating, rather
    // than rejecting them outright — a rejection here would leave the
    // pending file in place and block recovery on the next open.
    //
    // The same `serialize_for_write` validator is applied **after**
    // the upgrade so a malformed pending file (e.g. v2 with absolute
    // paths) still gets rejected.
    let mut pending_manifest: Manifest = serde_json::from_slice(&pending_data)
      .with_context(|| format!("parsing staged manifest at {pending_path:?}"))?;
    pending_manifest
      .upgrade_to_latest(root)
      .with_context(|| format!("upgrading staged manifest at {pending_path:?}"))?;
    let validated = pending_manifest
      .serialize_for_write()
      .with_context(|| format!("validating staged manifest at {pending_path:?}"))?;
    storage
      .atomic_write(manifest_path, &validated)
      .with_context(|| format!("promoting staged manifest to {manifest_path:?}"))?;
  }
  // Best-effort cleanup of the staging file in either branch.
  let _ = storage.remove(&pending_path);
  Ok(())
}

pub(crate) fn cleanup_segments(
  storage: &dyn Storage,
  segments: &[crate::index::manifest::SegmentMeta],
) -> Result<()> {
  for seg in segments {
    let resolved = seg.paths.resolve(storage.root());
    for path in [
      &resolved.terms,
      &resolved.postings,
      &resolved.docstore,
      &resolved.fast,
      &resolved.meta,
    ] {
      let _ = storage.remove(path);
    }
    #[cfg(feature = "vectors")]
    if let Some(dir) = resolved.vector_dir.as_ref() {
      let _ = storage.remove_dir_all(dir);
    }
  }
  Ok(())
}

fn ensure_compact_safe(schema: &Schema) -> Result<()> {
  for field in schema.resolved_fields().into_iter() {
    if (field.indexed || field.fast) && !field.stored {
      bail!(
        "cannot compact index: field `{}` is indexed/fast but not stored; compaction would drop its data",
        field.path
      );
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::api::types::{Document, IndexOptions, StorageType};
  use crate::index::manifest::{default_doc_id_field, TextField};
  use tempfile::tempdir;

  fn opts(path: &Path) -> IndexOptions {
    IndexOptions {
      path: path.to_path_buf(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 1.2,
      bm25_b: 0.75,
      storage: StorageType::Filesystem,
      checksum_policy: Default::default(),
      checksum_audit_failure_hook: None,
      read_only: false,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    }
  }

  #[test]
  fn reconcile_pending_manifest_promotes_when_wal_has_commit() {
    // BUG-018: a `.pending` manifest paired with a trailing WAL `Commit`
    // marker means the prior commit crossed the durability fence but the
    // live `MANIFEST.json` was never published. Recovery must copy the
    // staged content over and clean up the staging file.
    let dir = tempdir().unwrap();
    let storage: Arc<dyn Storage> =
      Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
    ensure_root(storage.as_ref(), dir.path()).unwrap();
    let manifest_path = Manifest::manifest_path(dir.path());
    let pending_path = Manifest::manifest_pending_path(dir.path());

    let live = Manifest::new(Schema::default_text_body());
    live.store(storage.as_ref(), &manifest_path).unwrap();
    let mut staged = live.clone();
    staged.committed_at = "2099-01-01T00:00:00+00:00".into();
    let staged_bytes = serde_json::to_vec_pretty(&staged).unwrap();
    storage.atomic_write(&pending_path, &staged_bytes).unwrap();

    // Simulate "WAL crossed the fence": append a commit marker.
    {
      let mut wal = Wal::open(storage.clone(), &directory::wal_path(dir.path())).unwrap();
      wal.append_commit().unwrap();
    }

    super::reconcile_pending_manifest(storage.as_ref(), dir.path(), &manifest_path).unwrap();

    let promoted = Manifest::load(storage.as_ref(), &manifest_path).unwrap();
    assert_eq!(
      promoted.committed_at, "2099-01-01T00:00:00+00:00",
      "the staged manifest should have been promoted to MANIFEST.json",
    );
    assert!(
      !storage.exists(&pending_path),
      "the staging file must be removed after promotion",
    );
  }

  #[test]
  fn reconcile_pending_manifest_discards_when_wal_has_no_trailing_commit() {
    // BUG-018: a `.pending` manifest with no trailing WAL `Commit`
    // corresponds to a commit that never reached the durability fence —
    // its pending ops will be replayed via the WAL, so the staged file
    // is stale and must be discarded without touching the live manifest.
    let dir = tempdir().unwrap();
    let storage: Arc<dyn Storage> =
      Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
    ensure_root(storage.as_ref(), dir.path()).unwrap();
    let manifest_path = Manifest::manifest_path(dir.path());
    let pending_path = Manifest::manifest_pending_path(dir.path());

    let live = Manifest::new(Schema::default_text_body());
    live.store(storage.as_ref(), &manifest_path).unwrap();
    let original_committed_at = live.committed_at.clone();
    let mut staged = live.clone();
    staged.committed_at = "2099-01-01T00:00:00+00:00".into();
    let staged_bytes = serde_json::to_vec_pretty(&staged).unwrap();
    storage.atomic_write(&pending_path, &staged_bytes).unwrap();

    // No WAL written → no trailing commit.
    super::reconcile_pending_manifest(storage.as_ref(), dir.path(), &manifest_path).unwrap();

    let live_after = Manifest::load(storage.as_ref(), &manifest_path).unwrap();
    assert_eq!(
      live_after.committed_at, original_committed_at,
      "the live manifest must be untouched when the WAL did not cross the fence",
    );
    assert!(
      !storage.exists(&pending_path),
      "the orphan staging file must be removed even when not promoted",
    );
  }

  #[test]
  fn compact_rejects_non_stored_indexed_fields() {
    let dir = tempdir().unwrap();
    let schema = Schema {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: vec![TextField {
        name: "body".into(),
        analyzer: "default".into(),
        search_analyzer: None,
        stored: false,
        indexed: true,
        nullable: false,
        search_as_you_type: None,
      }],
      keyword_fields: Vec::new(),
      numeric_fields: Vec::new(),
      nested_fields: Vec::new(),
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("hello")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("2")),
            ("body".into(), serde_json::json!("world")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    let err = idx.compact().unwrap_err();
    assert!(
      err.to_string().contains("indexed/fast but not stored"),
      "unexpected error: {err}"
    );
  }

  #[test]
  #[cfg(feature = "write-key")]
  fn compaction_requires_write_key_even_if_manifest_tampered() {
    use std::fs;

    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let key = "super-secret";
    let idx = Index::create_with_write_key(dir.path(), schema.clone(), opts(dir.path()), Some(key))
      .unwrap();
    {
      let mut writer = idx.writer_with_key(Some(key)).unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("hello")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    // Ensure segment metadata contains a binding written at commit time.
    let manifest = idx.manifest();
    let seg = &manifest.segments[0];
    let seg_meta_bytes = std::fs::read(dir.path().join(&seg.paths.meta)).unwrap();
    let seg_meta: crate::index::segment::SegmentFileMeta =
      serde_json::from_slice(&seg_meta_bytes).unwrap();
    assert!(
      seg_meta.write_binding_b64.is_some(),
      "segment metadata should contain write binding"
    );

    // Tamper manifest to strip write_key and segment binding, but keep segment meta binding.
    let manifest_path = Manifest::manifest_path(dir.path());
    let mut manifest_json: serde_json::Value =
      serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest_json.as_object_mut().unwrap().remove("write_key");
    if let Some(segments) = manifest_json
      .get_mut("segments")
      .and_then(|v| v.as_array_mut())
    {
      for seg in segments.iter_mut() {
        if let Some(obj) = seg.as_object_mut() {
          obj.remove("write_binding_b64");
        }
      }
    }
    fs::write(
      &manifest_path,
      serde_json::to_vec_pretty(&manifest_json).unwrap(),
    )
    .unwrap();

    let mut tampered_opts = opts(dir.path());
    tampered_opts.create_if_missing = false;
    let idx_tampered = Index::open(tampered_opts).unwrap();

    // Recompute bindings as compaction would.
    let manifest_snapshot = idx_tampered.manifest();
    let mut seg_bindings = Vec::new();
    for seg in manifest_snapshot.segments.iter() {
      let bytes = std::fs::read(dir.path().join(&seg.paths.meta))
        .expect("segment meta readable for tampered manifest");
      let seg_meta: crate::index::segment::SegmentFileMeta =
        serde_json::from_slice(&bytes).expect("parse segment meta");
      if let Some(b64) = seg_meta.write_binding_b64.as_deref() {
        seg_bindings.push(b64.to_string());
      }
    }
    assert!(
      !seg_bindings.is_empty(),
      "expected binding present in segment metadata after tamper"
    );

    // Without key, compaction should fail because segment metadata still carries the binding.
    assert!(idx_tampered.compact_with_key(None).is_err());

    // With correct key, compaction should succeed.
    idx_tampered.compact_with_key(Some(key)).unwrap();
  }

  #[test]
  fn tiered_merge_selects_small_segments() {
    use crate::index::manifest::{SegmentMeta, SegmentPaths};
    use crate::index::merge::TieredMergePolicy;
    use std::collections::HashMap;

    let make_seg = |id: &str, doc_count: u32| -> SegmentMeta {
      SegmentMeta {
        id: id.to_string(),
        generation: 1,
        paths: SegmentPaths {
          terms: String::new(),
          postings: String::new(),
          docstore: String::new(),
          fast: String::new(),
          meta: String::new(),
          #[cfg(feature = "vectors")]
          vector_dir: None,
        },
        doc_count,
        max_doc_id: doc_count,
        blockmax: false,
        deleted_docs: Vec::new(),
        avg_field_lengths: HashMap::new(),
        content_hashes: std::collections::BTreeMap::new(),
        write_binding_b64: None,
      }
    };

    let policy = TieredMergePolicy {
      segments_per_tier: 3,
      max_merge_at_once: 5,
      floor_segment_docs: 1_000,
      max_merged_segment_docs: 5_000_000,
    };

    // 6 small segments in the floor tier, threshold is 3.
    let segments: Vec<_> = (0..6)
      .map(|i| make_seg(&format!("s{i}"), 100 + i * 10))
      .collect();
    let merges = policy.find_merges(&segments);
    assert_eq!(merges.len(), 1, "expected exactly one merge group");
    assert_eq!(merges[0].len(), 5, "expected max_merge_at_once segments");
    // The selected segments should be the 5 smallest.
    assert_eq!(merges[0][0], "s0");
    assert_eq!(merges[0][4], "s4");

    // Fewer than segments_per_tier => no merge.
    let small: Vec<_> = (0..3).map(|i| make_seg(&format!("s{i}"), 50)).collect();
    assert!(policy.find_merges(&small).is_empty());
  }

  #[test]
  fn merge_segments_preserves_search() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

    // Create 5 segments, each with one document.
    let words = ["alpha", "bravo", "charlie", "delta", "echo"];
    for (i, word) in words.iter().enumerate() {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(format!("doc{i}"))),
            ("body".into(), serde_json::json!(word)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    assert_eq!(idx.manifest().segments.len(), 5);

    // Pick the first 3 segments to merge.
    let seg_ids: Vec<String> = idx
      .manifest()
      .segments
      .iter()
      .take(3)
      .map(|s| s.id.clone())
      .collect();
    idx.merge_segments(&seg_ids, None).unwrap();

    // Should now have 3 segments: 1 merged + 2 untouched.
    let manifest = idx.manifest();
    assert_eq!(manifest.segments.len(), 3);

    // All 5 documents should still be searchable.
    let reader = idx.reader().unwrap();
    for word in &words {
      let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
        "query": word.to_string(),
        "limit": 10,
      }))
      .unwrap();
      let result = reader.search(&req).unwrap();
      assert_eq!(
        result.total_hits_estimate, 1,
        "expected 1 hit for '{word}', got {}",
        result.total_hits_estimate
      );
    }
  }

  #[test]
  fn merge_segments_errors_when_any_id_missing() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

    // Create two real segments so the manifest has something to look up.
    for (i, word) in ["alpha", "bravo"].iter().enumerate() {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(format!("doc{i}"))),
            ("body".into(), serde_json::json!(word)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    assert_eq!(idx.manifest().segments.len(), 2);
    let real_id = idx.manifest().segments[0].id.clone();

    // Requesting a merge of one real segment and one typo'd ID must fail
    // rather than silently returning Ok. Before BUG-028 was fixed, the
    // `merge_metas.len() < 2` early return masked the missing ID.
    let err = idx
      .merge_segments(&[real_id.clone(), "does-not-exist".into()], None)
      .expect_err("expected merge_segments to error on unknown segment id");
    let msg = err.to_string();
    assert!(
      msg.contains("does-not-exist"),
      "error should name the missing segment: {msg}"
    );

    // A request that names only missing IDs should also error (previously
    // silent-Ok because the filtered set had fewer than two entries).
    let err = idx
      .merge_segments(&["missing-a".into(), "missing-b".into()], None)
      .expect_err("expected merge_segments to error when all ids missing");
    let msg = err.to_string();
    assert!(
      msg.contains("missing-a") && msg.contains("missing-b"),
      "error should name both missing segments: {msg}"
    );

    // Sanity check: the no-op paths we intentionally keep still succeed.
    idx.merge_segments(&[], None).unwrap(); // empty input
    idx
      .merge_segments(std::slice::from_ref(&real_id), None)
      .unwrap(); // single real id
                 // The manifest must be unchanged by the no-op / error cases.
    assert_eq!(idx.manifest().segments.len(), 2);
  }

  #[test]
  fn commit_with_merge_reduces_segments() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

    // Insert enough segments to trigger a merge with a low threshold.
    // Default segments_per_tier is 10, floor_segment_docs is 1000.
    // We'll create 12 segments with small doc counts (< 1000 each).
    for i in 0..12 {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(format!("doc{i}"))),
            ("body".into(), serde_json::json!(format!("word{i}"))),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    assert_eq!(idx.manifest().segments.len(), 12);

    // Now do one more commit_with_merge(true) that should trigger auto merge.
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!("doc_final")),
          ("body".into(), serde_json::json!("final")),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    writer.commit_with_merge(true).unwrap();

    // After auto merge, segment count should have decreased.
    let final_count = idx.manifest().segments.len();
    assert!(
      final_count < 13,
      "expected fewer than 13 segments after auto merge, got {final_count}"
    );

    // All 13 documents should still be searchable via match_all.
    let reader = idx.reader().unwrap();
    let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
      "query": { "type": "match_all" },
      "limit": 20,
      "track_total_hits": true,
    }))
    .unwrap();
    let result = reader.search(&req).unwrap();
    assert_eq!(
      result.total_hits_estimate, 13,
      "expected all 13 docs after merge, got {}",
      result.total_hits_estimate
    );
  }

  /// Stage 1, P1 invariant: the *typical* sequential reader pattern — open
  /// reader, serve a query, drop the reader, open another reader for the next
  /// request — must hit the cache. The previous Weak-only cache failed this
  /// because dropping the only reader released the last strong ref to each
  /// `SegmentCore`, so the next open reloaded everything. The bounded-LRU
  /// strong-ref cache fixes that.
  #[test]
  fn segment_cache_hits_for_sequential_non_overlapping_readers() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

    // Three independent commits → three segments.
    for (i, word) in ["alpha", "bravo", "charlie"].iter().enumerate() {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(format!("doc{i}"))),
            ("body".into(), serde_json::json!(word)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    assert_eq!(idx.manifest().segments.len(), 3);

    let baseline = idx.segment_core_loads();
    // First reader: loads all three segments. Drop it before opening any
    // others so the only path that keeps cores alive is the cache itself.
    {
      let _r1 = idx.reader().unwrap();
    }
    let after_first = idx.segment_core_loads();
    assert_eq!(
      after_first - baseline,
      3,
      "first reader open must load all three segments"
    );

    // Five subsequent reader opens, each dropped immediately. With strong
    // refs in the cache these must all hit; with the prior Weak design they
    // would each reload all three segments (3 + 3*5 = 18 total loads).
    for _ in 0..5 {
      let _r = idx.reader().unwrap();
    }
    assert_eq!(
      idx.segment_core_loads(),
      after_first,
      "sequential non-overlapping reader opens must hit the cache; \
       observed {} extra loads",
      idx.segment_core_loads() - after_first
    );

    // Sanity: queries against a fresh reader still produce correct results.
    let r = idx.reader().unwrap();
    let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
      "query": { "type": "match_all" },
      "limit": 10,
      "track_total_hits": true,
    }))
    .unwrap();
    let result = r.search(&req).unwrap();
    assert_eq!(result.total_hits_estimate, 3);
  }

  /// Stage 1: a tombstone-only commit (upsert / delete) does not change any
  /// segment's file checksums, so the cache fingerprint stays the same and a
  /// post-commit reader reuses the original `SegmentCore`. The new
  /// `SegmentReader` view must still pick up the new manifest's
  /// `deleted_docs`. This test asserts both: that the core is reused (no
  /// new load) AND that the tombstoned doc is no longer findable.
  #[test]
  fn segment_cache_reuses_core_on_tombstone_only_commit() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

    // Use non-overlapping single-word bodies so a string query unambiguously
    // matches exactly one doc.
    {
      let mut writer = idx.writer().unwrap();
      for (id, body) in [("1", "alpha"), ("2", "stable")] {
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(id)),
              ("body".into(), serde_json::json!(body)),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
      }
      writer.commit().unwrap();
    }
    assert_eq!(idx.manifest().segments.len(), 1);

    // Materialize the original segment core in the cache.
    let r0 = idx.reader().unwrap();
    let search = |reader: &crate::api::reader::IndexReader, q: &str| {
      let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
        "query": q.to_string(),
        "limit": 10,
        "track_total_hits": true,
      }))
      .unwrap();
      reader.search(&req).unwrap().total_hits_estimate
    };
    assert_eq!(search(&r0, "alpha"), 1);
    let loads_after_first_open = idx.segment_core_loads();
    assert_eq!(loads_after_first_open, 1);
    drop(r0);

    // Upsert: re-add doc "1" with a new body. The original segment's
    // `deleted_docs` gains the old doc-id (manifest-only change, no segment
    // file mutation). A new segment is written for the new doc.
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("bravo")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    assert_eq!(idx.manifest().segments.len(), 2);

    // Open a new reader. The original segment core must be REUSED from the
    // cache (its file checksums are unchanged → fingerprint unchanged → key
    // unchanged → cache hit). Only the brand-new segment should incur a
    // fresh load. Total loads delta = 1.
    let r1 = idx.reader().unwrap();
    let loads_after_upsert = idx.segment_core_loads();
    assert_eq!(
      loads_after_upsert - loads_after_first_open,
      1,
      "tombstone-only commit must reuse the original SegmentCore; \
       observed {} new loads (expected 1, for the brand-new segment only)",
      loads_after_upsert - loads_after_first_open
    );

    // The reused core must NOT surface stale `deleted_docs`: the new view
    // built via `SegmentReader::from_core(reused_core, new_meta, storage)`
    // derives `deleted` from `new_meta.deleted_docs`, which contains the
    // tombstone for the old doc.
    assert_eq!(
      search(&r1, "alpha"),
      0,
      "stale deleted_docs leak: tombstoned doc resurrected from reused core"
    );
    assert_eq!(
      search(&r1, "bravo"),
      1,
      "new version must be visible after upsert"
    );
  }

  /// Stage 1: an in-flight reader must keep its snapshotted `SegmentCore`s
  /// alive even when a concurrent `compact` removes those segments from the
  /// manifest. With strong refs in the cache, compaction-orphaned cores stay
  /// in the cache until the LRU evicts them; in-flight reader `Arc`s also
  /// keep the cores live. Either way the in-flight reader's queries must
  /// continue to return correct results from its snapshot.
  #[test]
  fn segment_cache_survives_compaction_for_in_flight_readers() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

    let words = ["alpha", "bravo", "charlie"];
    for (i, word) in words.iter().enumerate() {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(format!("doc{i}"))),
            ("body".into(), serde_json::json!(word)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    assert_eq!(idx.manifest().segments.len(), 3);

    // Open a reader holding all three segments. We deliberately do not
    // consume `pre_compact_reader` until after the compaction completes.
    let pre_compact_reader = idx.reader().unwrap();

    // Compact all three segments into one. This rewrites the manifest and
    // calls `cleanup_segments` on the originals — but the kernel keeps the
    // file handles `pre_compact_reader` already opened valid (Unix unlink
    // semantics), and the in-flight reader's `Arc<SegmentCore>` keeps each
    // core alive even if a concurrent cache miss caused an LRU eviction.
    let seg_ids: Vec<String> = idx
      .manifest()
      .segments
      .iter()
      .map(|s| s.id.clone())
      .collect();
    idx.merge_segments(&seg_ids, None).unwrap();
    assert_eq!(idx.manifest().segments.len(), 1);

    // The pre-compact reader must still serve queries correctly against its
    // own snapshot of three segments — the merge doesn't yank cores out from
    // under it.
    let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
      "query": { "type": "match_all" },
      "limit": 10,
      "track_total_hits": true,
    }))
    .unwrap();
    let result = pre_compact_reader.search(&req).unwrap();
    assert_eq!(
      result.total_hits_estimate, 3,
      "in-flight reader must see its snapshotted segments after compaction"
    );

    // A new reader opened *after* the compaction sees the merged segment.
    drop(pre_compact_reader);
    let post = idx.reader().unwrap();
    let result = post.search(&req).unwrap();
    assert_eq!(result.total_hits_estimate, 3);
    assert_eq!(post.manifest.segments.len(), 1);
  }

  /// Stage 1: bounded LRU eviction. With a small-capacity cache and more
  /// distinct segments than the cache can hold, the oldest entry must be
  /// evicted on each new insert. An `Arc<SegmentCore>` cloned out of the
  /// cache before eviction must remain usable: only the cache's own strong
  /// ref goes away on eviction; in-flight readers' refs keep the core alive.
  #[test]
  fn segment_cache_lru_evicts_oldest_and_keeps_in_flight_arcs_alive() {
    use crate::index::segment::{SegmentCache, SegmentCacheKey};

    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

    // Build three independent segments. Use the index only to produce real
    // SegmentMetas with valid checksums and on-disk files; the eviction test
    // itself runs against a fresh small-capacity cache so we don't have to
    // make the production cache size configurable for this stage.
    for (i, word) in ["alpha", "bravo", "charlie"].iter().enumerate() {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(format!("doc{i}"))),
            ("body".into(), serde_json::json!(word)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    let metas = idx.manifest().segments.clone();
    assert_eq!(metas.len(), 3);

    let cache = SegmentCache::with_capacity(2);
    let storage = Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
    let ctx = crate::index::segment::SegmentLoadCtx::from_options(&idx.inner.options);

    // Load segment 0 and clone an Arc out of the cache so we can verify the
    // core stays usable after it's evicted from the cache itself.
    let core_0 = cache.get_or_load(&metas[0], &ctx, storage.clone()).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.loads(), 1);

    // Load segment 1 and segment 2. After the third insert the cache is over
    // capacity (cap=2) and segment 0 — the LRU — must have been evicted.
    let _core_1 = cache.get_or_load(&metas[1], &ctx, storage.clone()).unwrap();
    let _core_2 = cache.get_or_load(&metas[2], &ctx, storage.clone()).unwrap();
    assert_eq!(cache.len(), 2, "cache must respect its capacity bound");
    assert_eq!(
      cache.loads(),
      3,
      "each distinct meta must have caused one load"
    );
    let key_0 = SegmentCacheKey::from_meta(&metas[0]);
    assert!(
      !cache.contains_key(&key_0),
      "the oldest (LRU) entry must have been evicted"
    );

    // The Arc we cloned out earlier still works: its terms / fast fields /
    // doc table are owned by the SegmentCore, which lives until the last
    // strong ref drops. Eviction only removed the cache's own ref.
    assert_eq!(
      Arc::strong_count(&core_0),
      1,
      "after eviction, the only remaining Arc to segment 0 is `core_0`"
    );

    // Re-requesting segment 0 now causes a fresh load (cache miss) — proving
    // it really was evicted, not just hidden by a hash collision.
    let loads_before = cache.loads();
    let _core_0_again = cache.get_or_load(&metas[0], &ctx, storage.clone()).unwrap();
    assert_eq!(
      cache.loads() - loads_before,
      1,
      "evicted entry must be reloaded on next request"
    );
  }

  /// Build an index with a single committed doc and return the on-disk path
  /// of its (only) segment's postings file. Used by the Stage 3 checksum
  /// tests to corrupt a segment AFTER the manifest has been written.
  fn build_index_and_postings_path(dir: &std::path::Path) -> (Index, std::path::PathBuf) {
    let schema = Schema::default_text_body();
    let idx = Index::create(dir, schema, opts(dir)).unwrap();
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!("1")),
          ("body".into(), serde_json::json!("alpha")),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    writer.commit().unwrap();
    // Stage 9a: paths in v2 manifests are relative-to-root keys.
    // Resolve against the test dir to recover an absolute path.
    let postings_path = dir.join(&idx.manifest().segments[0].paths.postings);
    (idx, postings_path)
  }

  /// Corrupt a segment file by appending a trailing byte. The reader-side
  /// postings decoder stops at the encoded `doc_freq` count (and similarly
  /// the inner postings-entry counts), so a trailing byte changes the
  /// manifest checksum without affecting any structural read — exactly the
  /// bit-rot signature `verify_checksums` is designed to catch. (Flipping
  /// an in-payload byte would also fail verification but might additionally
  /// change a varint or `f32` value mid-decode, producing wrong query
  /// results that obscure what we're actually trying to test.)
  fn append_trailing_byte(path: &std::path::Path) {
    let mut bytes = std::fs::read(path).unwrap();
    bytes.push(0xAB);
    std::fs::write(path, bytes).unwrap();
  }

  /// Stage 3 P1 fix: under `Strict`, a cache hit must re-verify the
  /// on-disk segment files against the manifest's recorded checksums.
  /// Pre-Stage-1, every `Index::reader()` open re-verified; Stage 1's
  /// caching silently regressed that guarantee. Codex flagged it; this
  /// test pins the restored behavior.
  ///
  /// Scenario: open a reader (populates the cache), mutate `postings.bin`
  /// in-place WITHOUT touching the manifest, open a second reader against
  /// the same `Index`. Under `Strict` the second open must fail with a
  /// checksum mismatch — even though the cache holds a valid parsed core.
  #[test]
  fn checksum_policy_strict_re_verifies_on_cache_hit() {
    let dir = tempdir().unwrap();
    let (idx, postings_path) = build_index_and_postings_path(dir.path());

    // First reader: populates the cache via a fresh load (which also runs
    // synchronous verification under Strict). Capture loads counter.
    let _r1 = idx.reader().unwrap();
    let loads_before = idx.segment_core_loads();
    assert_eq!(loads_before, 1);

    // Mutate the segment file AFTER the cache is populated. Simulates
    // external bit rot or out-of-process tampering between reader opens.
    append_trailing_byte(&postings_path);

    // Second reader open under the (default) Strict policy must re-verify
    // on the cache hit. Without the P1 fix this would silently succeed
    // and serve queries against the corrupted file.
    let err = match idx.reader() {
      Ok(_) => panic!("Strict cache hit must re-verify and reject the externally-mutated segment"),
      Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
      // Stage 9b: Strict path now reports SHA-256 verification by
      // default; legacy manifests with only CRC32 still produce a
      // "failed checksum" error. Either substring is acceptable.
      msg.contains("postings") && (msg.contains("checksum") || msg.contains("SHA-256")),
      "expected postings checksum/SHA-256 failure, got: {msg}"
    );

    // No new SegmentCore was loaded — the re-verify path doesn't bump the
    // loads counter. We re-read bytes for verification only; the cached
    // parsed structures stay intact and reused for the next attempt.
    assert_eq!(
      idx.segment_core_loads(),
      loads_before,
      "Strict re-verify must not trigger a fresh load on cache hit"
    );
  }

  /// Stage 3, default policy: `Strict` is the default for a freshly-built
  /// `IndexOptions`, and a corrupted segment must fail to open with a
  /// checksum error. This is the load-bearing safety guarantee that the
  /// new policy enum must NOT silently regress.
  #[test]
  fn checksum_policy_strict_default_rejects_corrupted_segment() {
    let dir = tempdir().unwrap();
    let (idx, postings_path) = build_index_and_postings_path(dir.path());
    drop(idx); // release the cache so the next open is a fresh load
    append_trailing_byte(&postings_path);

    let mut reopen_opts = opts(dir.path());
    reopen_opts.create_if_missing = false;
    assert_eq!(
      reopen_opts.checksum_policy,
      crate::api::types::ChecksumPolicy::Strict,
      "Strict must be the default policy"
    );
    let reopened = Index::open(reopen_opts).unwrap();
    let err = match reopened.reader() {
      Ok(_) => panic!("corrupted segment under Strict must fail at reader open"),
      Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
      // Stage 9b: Strict path now reports SHA-256 verification by
      // default; legacy manifests with only CRC32 still produce a
      // "failed checksum" error. Either substring is acceptable.
      msg.contains("postings") && (msg.contains("checksum") || msg.contains("SHA-256")),
      "expected postings checksum/SHA-256 failure, got: {msg}"
    );
  }

  /// Stage 3: under `TrustManifest`, a corrupted segment opens without
  /// error because whole-file verification is skipped. Format-level reads
  /// (term-dict CRC, fast-fields self-check) still fire — those protect
  /// structural integrity, not cross-file manifest consistency. Postings
  /// and docstore are NOT touched at load time.
  #[test]
  fn checksum_policy_trust_manifest_skips_whole_file_verification() {
    use crate::api::types::ChecksumPolicy;

    let dir = tempdir().unwrap();
    let (idx, postings_path) = build_index_and_postings_path(dir.path());
    drop(idx);
    append_trailing_byte(&postings_path);

    let mut reopen_opts = opts(dir.path());
    reopen_opts.create_if_missing = false;
    reopen_opts.checksum_policy = ChecksumPolicy::TrustManifest;
    let reopened = Index::open(reopen_opts).unwrap();
    // The reader open succeeds despite the postings-file corruption.
    let reader = reopened
      .reader()
      .expect("TrustManifest must open without verifying postings");
    // And basic queries still produce correct results because the encoded
    // postings list is intact through its declared length; only the
    // trailing byte (irrelevant to read_at) was flipped.
    let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
      "query": "alpha",
      "limit": 10,
      "track_total_hits": true,
    }))
    .unwrap();
    let result = reader.search(&req).unwrap();
    assert_eq!(result.total_hits_estimate, 1);
  }

  /// Stage 3: `Audit` opens segments synchronously (like `TrustManifest`)
  /// but dispatches a background verification. A corrupted segment under
  /// `Audit` must (a) open without error and (b) cause the audit hook to
  /// fire with a failure within a reasonable timeout.
  #[test]
  fn checksum_policy_audit_fires_hook_on_corrupted_segment() {
    use crate::api::types::{ChecksumAuditFailureHook, ChecksumPolicy};
    use std::sync::mpsc;
    use std::time::Duration;

    let dir = tempdir().unwrap();
    let (idx, postings_path) = build_index_and_postings_path(dir.path());
    let expected_segment_id = idx.manifest().segments[0].id.clone();
    drop(idx);
    append_trailing_byte(&postings_path);

    let (tx, rx) = mpsc::sync_channel::<(String, String)>(4);
    let hook = ChecksumAuditFailureHook::new(move |segment_id, err| {
      // `try_send` so the channel saturating doesn't block the rayon worker.
      let _ = tx.try_send((segment_id.to_string(), format!("{err:#}")));
    });

    let mut reopen_opts = opts(dir.path());
    reopen_opts.create_if_missing = false;
    reopen_opts.checksum_policy = ChecksumPolicy::Audit;
    reopen_opts.checksum_audit_failure_hook = Some(hook);
    let reopened = Index::open(reopen_opts).unwrap();
    let _reader = reopened
      .reader()
      .expect("Audit must open without synchronous verification");

    // Wait for the rayon worker to complete the background verification.
    let (got_segment_id, err_msg) = rx
      .recv_timeout(Duration::from_secs(5))
      .expect("audit hook should fire within 5s on a corrupted segment");
    assert_eq!(got_segment_id, expected_segment_id);
    assert!(
      // Stage 9b: Audit path now reports SHA-256 verification by
      // default; legacy manifests still produce a "failed checksum"
      // error.
      err_msg.contains("postings") && (err_msg.contains("checksum") || err_msg.contains("SHA-256")),
      "hook should report a checksum/SHA-256 postings error, got: {err_msg}"
    );
  }

  /// Stage 3: cache-aware audit. Successive `Index::reader()` calls against
  /// the same manifest hit the cache and must NOT re-dispatch the audit —
  /// otherwise opening N readers against M segments produces N*M audit
  /// runs and defeats Stage 1's caching win for the `Audit` policy.
  #[test]
  fn checksum_policy_audit_does_not_refire_on_cache_hits() {
    use crate::api::types::{ChecksumAuditFailureHook, ChecksumPolicy};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    let dir = tempdir().unwrap();
    let (idx, postings_path) = build_index_and_postings_path(dir.path());
    drop(idx);
    append_trailing_byte(&postings_path);

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();
    let (tx, rx) = mpsc::sync_channel::<()>(4);
    let hook = ChecksumAuditFailureHook::new(move |_segment_id, _err| {
      counter_clone.fetch_add(1, Ordering::SeqCst);
      let _ = tx.try_send(());
    });

    let mut reopen_opts = opts(dir.path());
    reopen_opts.create_if_missing = false;
    reopen_opts.checksum_policy = ChecksumPolicy::Audit;
    reopen_opts.checksum_audit_failure_hook = Some(hook);
    let reopened = Index::open(reopen_opts).unwrap();

    // First reader open: triggers an audit dispatch (cache miss).
    let _r1 = reopened.reader().unwrap();
    rx.recv_timeout(Duration::from_secs(5))
      .expect("first audit should fire");

    // Subsequent reader opens hit the cache; no new audit dispatches.
    for _ in 0..5 {
      let _r = reopened.reader().unwrap();
    }
    // Give any (incorrect) extra dispatches a chance to surface.
    std::thread::sleep(Duration::from_millis(200));

    assert_eq!(
      counter.load(Ordering::SeqCst),
      1,
      "audit hook must fire exactly once across N cache-hit reader opens; \
       observed {} firings",
      counter.load(Ordering::SeqCst)
    );
  }

  /// Stage 4 helper: drive a `Future` to completion without depending on any
  /// async runtime. The Stage 4 contract is that `*_async` futures contain
  /// no internal `.await` points and resolve on first poll, so a noop waker
  /// is sufficient. If a future ever returns `Pending` here, that signals
  /// Stage 4's contract has been broken (something inside the async body
  /// started awaiting real I/O before its stage was meant to).
  fn block_on_immediate<F: std::future::Future>(fut: F) -> F::Output {
    use std::pin::pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct Noop;
    impl Wake for Noop {
      fn wake(self: Arc<Self>) {}
    }

    let waker = Waker::from(Arc::new(Noop));
    let mut cx = Context::from_waker(&waker);
    let mut fut = pin!(fut);
    match fut.as_mut().poll(&mut cx) {
      Poll::Ready(out) => out,
      Poll::Pending => panic!(
        "Stage 4 *_async future returned Pending; bodies should be sync work \
         until Stage 8 introduces real async I/O"
      ),
    }
  }

  /// Stage 4: `Index::reader_async` produces a future that resolves to the
  /// same `IndexReader` as `Index::reader`, and bumps `reader_open_count`
  /// the same way (so observability hooks behave identically). On Stage 4
  /// the future is required to resolve on first poll, which `block_on_
  /// immediate` enforces.
  #[test]
  fn reader_async_resolves_immediately_and_matches_sync() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("alpha")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }

    let baseline = idx.reader_open_count();
    let _r_sync = idx.reader().unwrap();
    let after_sync = idx.reader_open_count();
    assert_eq!(after_sync - baseline, 1);

    let _r_async = block_on_immediate(idx.reader_async()).unwrap();
    assert_eq!(
      idx.reader_open_count() - after_sync,
      1,
      "reader_async must bump reader_open_count exactly the same way as reader()"
    );
  }

  /// Stage 4: `IndexReader::search_async` produces results identical to
  /// `IndexReader::search` for the same request. This locks in the API
  /// shape so Stage 8's body change (async BlobStore reads) has a clear
  /// equivalence target.
  #[test]
  fn search_async_returns_identical_results_to_sync() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      for (id, body) in [("1", "alpha"), ("2", "alpha bravo"), ("3", "bravo")] {
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(id)),
              ("body".into(), serde_json::json!(body)),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
      }
      writer.commit().unwrap();
    }
    let reader = idx.reader().unwrap();
    let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
      "query": "alpha",
      "limit": 10,
      "track_total_hits": true,
    }))
    .unwrap();

    let sync_result = reader.search(&req).unwrap();
    let async_result = block_on_immediate(reader.search_async(&req)).unwrap();
    assert_eq!(
      sync_result.total_hits_estimate,
      async_result.total_hits_estimate
    );
    assert_eq!(sync_result.hits.len(), async_result.hits.len());
    for (sync_hit, async_hit) in sync_result.hits.iter().zip(async_result.hits.iter()) {
      assert_eq!(sync_hit.doc_id, async_hit.doc_id);
      assert!(
        (sync_hit.score - async_hit.score).abs() < f32::EPSILON,
        "search_async score must match search score for the same request"
      );
    }
  }

  /// Stage 4: `IndexReader::mget_async` produces results identical to
  /// `IndexReader::mget` for the same id list and `return_stored` flag.
  #[test]
  fn mget_async_returns_identical_results_to_sync() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      for (id, body) in [("1", "alpha"), ("2", "bravo"), ("3", "charlie")] {
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(id)),
              ("body".into(), serde_json::json!(body)),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
      }
      writer.commit().unwrap();
    }
    let reader = idx.reader().unwrap();
    let ids = vec!["1".to_string(), "2".to_string(), "missing".to_string()];

    let sync_result = reader.mget(&ids, true).unwrap();
    let async_result = block_on_immediate(reader.mget_async(&ids, true)).unwrap();
    assert_eq!(sync_result.len(), async_result.len());
    for (s, a) in sync_result.iter().zip(async_result.iter()) {
      assert_eq!(s.doc_id, a.doc_id);
      assert_eq!(s.found, a.found);
      assert_eq!(s._source, a._source);
    }
  }

  /// Stage 6 expressiveness gate: an Index opened via `open_with_storage`
  /// against a `BlobStoreAdapter` wrapping `LocalBlobStore` runs an
  /// end-to-end write + commit + reopen + search workflow indistinguishably
  /// from one running on `FsStorage` directly. This is the load-bearing
  /// proof that the BlobStore trait surface (Stage 5) is expressive
  /// enough to back everything `Storage`-shaped index code does today.
  ///
  /// If this test fails, Stage 5's trait surface or Stage 6's adapter is
  /// missing something the index code depends on.
  ///
  /// Stage 8a [P1] (Codex review): the storage-only constructor path
  /// must keep working when `Storage` is a `BlobStoreAdapter`. Going
  /// through `BlobStoreAdapter` for segment reads would create a nested
  /// `futures::executor::block_on` (BlobStoreAdapter's bridge inside
  /// SegmentReader's bridge), which the LocalPool executor refuses with
  /// `EnterError`. The fix: `default_blob_store` consults
  /// [`Storage::as_blob_store`], and `BlobStoreAdapter` overrides it to
  /// return its inner `Arc<dyn BlobStore>` directly — so this test
  /// exercises that the unwrap actually happens (any regression brings
  /// the `EnterError` back).
  #[test]
  fn index_runs_end_to_end_through_blob_store_adapter() {
    use crate::storage::{BlobStoreAdapter, LocalBlobStore};

    let dir = tempdir().unwrap();

    let blob: Arc<dyn crate::storage::BlobStore> =
      Arc::new(LocalBlobStore::new(dir.path().to_path_buf()));
    let adapter: Arc<dyn Storage> = Arc::new(BlobStoreAdapter::new(blob, dir.path().to_path_buf()));

    let schema = Schema::default_text_body();
    let idx =
      Index::create_with_storage(dir.path(), schema, opts(dir.path()), adapter.clone()).unwrap();

    // Two commits in two separate segments to exercise multi-segment
    // reader open through the adapter as well.
    for (id, body) in [("1", "alpha"), ("2", "bravo charlie"), ("3", "charlie")] {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(id)),
            ("body".into(), serde_json::json!(body)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    assert_eq!(idx.manifest().segments.len(), 3);

    // Search through the adapter-backed index.
    let reader = idx.reader().unwrap();
    let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
      "query": "charlie",
      "limit": 10,
      "track_total_hits": true,
    }))
    .unwrap();
    let result = reader.search(&req).unwrap();
    assert_eq!(
      result.total_hits_estimate, 2,
      "search through BlobStoreAdapter must return correct results"
    );

    // Drop the index and reopen it via the same adapter chain. This
    // exercises manifest read, segment open, and segment-cache rebuild
    // through the BlobStore-only path.
    drop(reader);
    drop(idx);
    let mut reopen_opts = opts(dir.path());
    reopen_opts.create_if_missing = false;
    let reopened = Index::open_with_storage(reopen_opts, adapter).unwrap();
    let reader = reopened.reader().unwrap();
    let result = reader.search(&req).unwrap();
    assert_eq!(
      result.total_hits_estimate, 2,
      "search after reopen-through-adapter must return correct results"
    );
    assert_eq!(reopened.manifest().segments.len(), 3);
  }

  /// Stage 8 (a + b) regression suite. With postings (8a) and docstore
  /// (8b) both routed through `BlobStore`, segment reads must issue
  /// bounded `Object::read_range` calls, not whole-file reads. A
  /// `RecordingBlobStore` wrapper logs every `get_range` and
  /// `Object::read_range` call (path + range) for inspection.
  ///
  /// **Postings (Stage 8a)** — properties are *structural*, not
  /// exact-count, because the search path may issue more than one
  /// bounded postings read per segment (e.g. one for `doc_freq` stats,
  /// one for iteration):
  ///
  /// * `missing_term_performs_zero_postings_range_reads` — query for a
  ///   missing term ⇒ **zero** postings reads (FST gate).
  /// * `hit_term_postings_reads_are_strictly_bounded_vs_whole_file` —
  ///   each postings read is a proper subset of the file (no fallback
  ///   to `0..len`).
  /// * `default_blob_store_does_not_read_postings_to_end_during_open`
  ///   — `StorageAsBlobStore::stat` must not slurp the postings file
  ///   on segment open (Stage 8a v2 P1 regression).
  ///
  /// **Docstore (Stage 8b)** — properties are *exact-count*, because
  /// the offset table makes a single bounded read per fetched doc
  /// achievable (and per Codex's Stage 8 done criterion: a top-K=10
  /// search produces exactly 10 docstore range reads):
  ///
  /// * `mget_yields_exactly_one_docstore_range_read_per_returned_doc`
  ///   — N IDs with `_source: true` ⇒ exactly N docstore reads.
  /// * `mget_with_source_false_issues_zero_docstore_reads` —
  ///   `_source: false` ⇒ zero docstore reads (no consult at all).
  /// * `top_k_search_with_source_yields_one_docstore_range_read_per_hit`
  ///   — a top-10 search with `_source: true` ⇒ exactly 10 docstore
  ///   reads (the original Stage 8 done criterion).
  /// * `docstore_range_reads_match_offset_table` — each recorded
  ///   range exactly equals `offsets[doc_id]..offsets[doc_id+1]`
  ///   (or `..docstore_len` for the last doc).
  /// * `docstore_offset_length_mismatch_is_detected_as_corruption` —
  ///   if the offset table implies a longer record than the embedded
  ///   length actually encodes (but still within the span-guard
  ///   bound), `get_doc` returns an error rather than silently
  ///   ignoring trailing bytes.
  /// * `oversized_offset_derived_range_is_rejected_without_issuing_read`
  ///   — Stage 8b v2 P1 regression. An offset-derived span larger
  ///   than `MAX_DOCSTORE_BYTES + 4` must be rejected **before**
  ///   `read_range` is issued, so a corrupt offset table can't
  ///   trigger a multi-GB object-store GET / `Vec` allocation.
  ///
  /// These tests live at the `Index` integration layer so they
  /// exercise the full `IndexReader::{search,mget}` →
  /// `SegmentReader::{postings,get_doc}` → `Object::read_range` path
  /// end to end.
  mod stage8_postings_and_docstore_range_reads {
    use super::*;
    use crate::storage::blob::{
      BlobStore, Capabilities, Object, ObjectStat, ObjectWriter, PutIfMatchError,
    };
    use crate::storage::StorageAsBlobStore;
    use anyhow::Result;
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::ops::Range;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    /// Records every `get_range` and `Object::read_range` call against
    /// the inner BlobStore, plus the cumulative byte count, for tests
    /// that want to assert exact range-read shapes.
    struct RecordingBlobStore {
      inner: Arc<dyn BlobStore>,
      log: Arc<Mutex<Vec<RangeReadEntry>>>,
    }

    #[derive(Debug, Clone)]
    struct RangeReadEntry {
      key: PathBuf,
      range: Range<u64>,
    }

    impl RecordingBlobStore {
      fn new(inner: Arc<dyn BlobStore>) -> (Arc<Self>, Arc<Mutex<Vec<RangeReadEntry>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let store = Arc::new(Self {
          inner,
          log: log.clone(),
        });
        (store, log)
      }
    }

    #[async_trait]
    impl BlobStore for RecordingBlobStore {
      async fn stat(&self, key: &Path) -> Result<ObjectStat> {
        self.inner.stat(key).await
      }

      async fn open(&self, key: &Path) -> Result<Arc<dyn Object>> {
        let inner_obj = self.inner.open(key).await?;
        Ok(Arc::new(RecordingObject {
          inner: inner_obj,
          key: key.to_path_buf(),
          log: self.log.clone(),
        }))
      }

      async fn get_range(&self, key: &Path, range: Range<u64>) -> Result<Bytes> {
        self.log.lock().unwrap().push(RangeReadEntry {
          key: key.to_path_buf(),
          range: range.clone(),
        });
        self.inner.get_range(key, range).await
      }

      async fn get(&self, key: &Path) -> Result<Bytes> {
        self.inner.get(key).await
      }
      async fn put(&self, key: &Path, body: Bytes) -> Result<ObjectStat> {
        self.inner.put(key, body).await
      }
      async fn put_stream(&self, key: &Path) -> Result<Box<dyn ObjectWriter>> {
        self.inner.put_stream(key).await
      }
      async fn put_if_match(
        &self,
        key: &Path,
        body: Bytes,
        expected: Option<&str>,
      ) -> std::result::Result<ObjectStat, PutIfMatchError> {
        self.inner.put_if_match(key, body, expected).await
      }
      async fn delete(&self, key: &Path) -> Result<()> {
        self.inner.delete(key).await
      }
      fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
      }
    }

    struct RecordingObject {
      inner: Arc<dyn Object>,
      key: PathBuf,
      log: Arc<Mutex<Vec<RangeReadEntry>>>,
    }

    #[async_trait]
    impl Object for RecordingObject {
      fn stat(&self) -> &ObjectStat {
        self.inner.stat()
      }

      async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        self.log.lock().unwrap().push(RangeReadEntry {
          key: self.key.clone(),
          range: range.clone(),
        });
        self.inner.read_range(range).await
      }
    }

    /// Build an `Index` whose `blob_store` is a `RecordingBlobStore`
    /// wrapping the default `StorageAsBlobStore` over `FsStorage`.
    /// Returns the index plus the log for assertions.
    fn make_index_with_recording_blob_store(
      dir: &Path,
    ) -> (Index, Arc<Mutex<Vec<RangeReadEntry>>>) {
      let storage: Arc<dyn Storage> = Arc::new(crate::storage::FsStorage::new(dir.to_path_buf()));
      let inner_blob: Arc<dyn BlobStore> = Arc::new(StorageAsBlobStore::new(storage.clone()));
      let (recording, log) = RecordingBlobStore::new(inner_blob);
      let blob_store: Arc<dyn BlobStore> = recording;
      let schema = Schema::default_text_body();
      let idx =
        Index::create_with_storage_and_blob_store(dir, schema, opts(dir), storage, blob_store)
          .unwrap();
      (idx, log)
    }

    /// Filter recorded reads to those targeting the given segment's
    /// postings file. Used to assert range-read counts against
    /// postings specifically (not docstore, not other segments).
    fn postings_reads_for_segment<'a>(
      log: &'a [RangeReadEntry],
      postings_path: &Path,
    ) -> Vec<&'a RangeReadEntry> {
      log.iter().filter(|e| e.key == postings_path).collect()
    }

    /// Stage 8a regression: a query with a term that does not exist in
    /// any segment performs **zero** postings range reads. Catches any
    /// path that accidentally fetches the postings whole-file or
    /// queries unconditionally.
    #[test]
    fn missing_term_performs_zero_postings_range_reads() {
      let dir = tempdir().unwrap();
      let (idx, log) = make_index_with_recording_blob_store(dir.path());
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("alpha bravo")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();

      let postings_path = dir.path().join(&idx.manifest().segments[0].paths.postings);

      // Snapshot the log AFTER the index is built (writes may have
      // their own range reads) so we measure only the search phase.
      let before = log.lock().unwrap().len();

      let reader = idx.reader().unwrap();
      let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
        "query": "thisterm_does_not_exist_anywhere",
        "limit": 10,
        "track_total_hits": true,
      }))
      .unwrap();
      let result = reader.search(&req).unwrap();
      assert_eq!(result.total_hits_estimate, 0);

      let after = log.lock().unwrap();
      let search_phase = &after[before..];
      let postings_reads = postings_reads_for_segment(search_phase, &postings_path);
      assert_eq!(
        postings_reads.len(),
        0,
        "missing term must not issue any postings range reads; got {} reads: {:?}",
        postings_reads.len(),
        postings_reads
      );
    }

    /// Stage 8a regression: a hit-term query performs **bounded**
    /// postings range reads against the matching segment — every read
    /// is strictly less than the whole postings file size. This is the
    /// load-bearing property: even if the search path issues multiple
    /// reads (e.g. one for `doc_freq` stats and one for iteration),
    /// each one is a `TinyFst::range_for` slice rather than a
    /// fallback-to-`0..len` whole-file fetch.
    ///
    /// Uses a multi-term corpus so each term's range is a proper
    /// subset of the file. With a single-term corpus the term's range
    /// IS the whole file and the boundedness property is unobservable.
    #[test]
    fn hit_term_postings_reads_are_strictly_bounded_vs_whole_file() {
      let dir = tempdir().unwrap();
      let (idx, log) = make_index_with_recording_blob_store(dir.path());

      // Multi-term, multi-doc corpus — ensures the "alpha" payload is
      // a small slice of the postings file rather than the entire
      // contents. Without this, a 1-term segment's range is 0..len
      // and the assertion below trivially passes via inequality on
      // identical numbers, defeating the test's intent.
      let mut writer = idx.writer().unwrap();
      for (id, body) in [
        ("1", "alpha bravo"),
        ("2", "bravo charlie delta"),
        ("3", "echo foxtrot golf hotel"),
        ("4", "india juliet kilo lima"),
        ("5", "mike november oscar"),
      ] {
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(id)),
              ("body".into(), serde_json::json!(body)),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
      }
      writer.commit().unwrap();

      let segment = &idx.manifest().segments[0];
      let postings_path = dir.path().join(&segment.paths.postings);
      let postings_total_len = std::fs::metadata(&postings_path).unwrap().len();
      assert!(
        postings_total_len > 100,
        "test prerequisite: postings file should be substantially larger than any single term's range"
      );

      let before = log.lock().unwrap().len();

      let reader = idx.reader().unwrap();
      let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
        "query": "alpha",
        "limit": 10,
        "track_total_hits": true,
      }))
      .unwrap();
      let result = reader.search(&req).unwrap();
      assert_eq!(result.total_hits_estimate, 1);

      let after = log.lock().unwrap();
      let search_phase = &after[before..];
      let postings_reads = postings_reads_for_segment(search_phase, &postings_path);

      assert!(
        !postings_reads.is_empty(),
        "hit-term search must issue at least one postings range read"
      );

      // The structural property: every postings read is a bounded
      // range, NOT a fallback to whole-file. If a future change
      // accidentally re-introduces a whole-file fetch for the term's
      // payload, this assertion fails.
      for r in &postings_reads {
        assert!(
          r.range.end <= postings_total_len,
          "range {:?} must not exceed postings length {}",
          r.range,
          postings_total_len
        );
        assert!(
          (r.range.end - r.range.start) < postings_total_len,
          "range {:?} must be strictly bounded vs whole-file fetch \
           (postings_total_len = {}); accidental fallback to 0..len?",
          r.range,
          postings_total_len
        );
      }
    }

    /// Stage 8a [P1] regression (Codex review): the default
    /// `StorageAsBlobStore::open` path MUST NOT slurp the entire
    /// postings file via `Storage::read_to_end` to compute its length.
    /// The previous shape (`read_to_end(&path)?.len()` inside `stat`)
    /// defeated the entire point of the Stage 8a migration: every
    /// segment reader open re-read the full postings before any
    /// bounded `read_range` happened.
    ///
    /// We wrap an `FsStorage` with a `RecordingStorage` that tallies
    /// every `Storage::read_to_end` call, then drive the default
    /// blob-store path (via `Index::create_with_storage`, with no
    /// explicit `blob_store` override). After committing a segment we
    /// open a reader and assert no `read_to_end` calls landed on the
    /// postings file during reader open. The fixed shape uses
    /// `open_read + seek(End)` to discover length without touching
    /// any bytes.
    #[test]
    fn default_blob_store_does_not_read_postings_to_end_during_open() {
      use crate::storage::DynFile;

      struct RecordingStorage {
        inner: Arc<dyn Storage>,
        read_to_end_calls: Arc<Mutex<Vec<PathBuf>>>,
      }

      impl Storage for RecordingStorage {
        fn root(&self) -> &Path {
          self.inner.root()
        }
        fn ensure_dir(&self, path: &Path) -> Result<()> {
          self.inner.ensure_dir(path)
        }
        fn exists(&self, path: &Path) -> bool {
          self.inner.exists(path)
        }
        fn open_read(&self, path: &Path) -> Result<DynFile> {
          self.inner.open_read(path)
        }
        fn open_write(&self, path: &Path) -> Result<DynFile> {
          self.inner.open_write(path)
        }
        fn open_append(&self, path: &Path) -> Result<DynFile> {
          self.inner.open_append(path)
        }
        fn read_to_end(&self, path: &Path) -> Result<Vec<u8>> {
          self
            .read_to_end_calls
            .lock()
            .unwrap()
            .push(path.to_path_buf());
          self.inner.read_to_end(path)
        }
        fn write_all(&self, path: &Path, data: &[u8]) -> Result<()> {
          self.inner.write_all(path, data)
        }
        fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()> {
          self.inner.atomic_write(path, data)
        }
        fn remove(&self, path: &Path) -> Result<()> {
          self.inner.remove(path)
        }
        fn remove_dir_all(&self, path: &Path) -> Result<()> {
          self.inner.remove_dir_all(path)
        }
      }

      let dir = tempdir().unwrap();
      let inner: Arc<dyn Storage> =
        Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
      let read_to_end_calls = Arc::new(Mutex::new(Vec::new()));
      let recording: Arc<dyn Storage> = Arc::new(RecordingStorage {
        inner,
        read_to_end_calls: read_to_end_calls.clone(),
      });

      // `ChecksumPolicy::Strict` (the default) reads each segment file
      // end-to-end to verify the manifest checksum on segment open.
      // That's an unrelated whole-file read driven by policy, not by
      // the `StorageAsBlobStore::stat` slurping bug we're guarding
      // against. Switch to `TrustManifest` so the only `read_to_end`
      // path under test is the BlobStore open path.
      let mut idx_opts = opts(dir.path());
      idx_opts.checksum_policy = crate::api::types::ChecksumPolicy::TrustManifest;

      // Use `create_with_storage` (NOT the
      // `*_with_storage_and_blob_store` form) so the default
      // `StorageAsBlobStore` bridge is exactly what's exercised.
      let schema = Schema::default_text_body();
      let idx =
        Index::create_with_storage(dir.path(), schema, idx_opts, recording.clone()).unwrap();

      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            (
              "body".into(),
              serde_json::json!("alpha bravo charlie delta echo"),
            ),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
      let postings_path = dir.path().join(&idx.manifest().segments[0].paths.postings);

      // Snapshot the call count so write-side reads don't pollute the
      // read-side measurement.
      let before = read_to_end_calls.lock().unwrap().len();

      let reader = idx.reader().unwrap();
      drop(reader);

      let calls = read_to_end_calls.lock().unwrap();
      let suspect: Vec<&PathBuf> = calls[before..]
        .iter()
        .filter(|p| **p == postings_path)
        .collect();
      assert!(
        suspect.is_empty(),
        "Stage 8a [P1] regression: StorageAsBlobStore::stat must not \
         call Storage::read_to_end on the postings file during reader \
         open; got {} calls: {:?}",
        suspect.len(),
        suspect
      );
    }

    // ───────────────────────── Stage 8b: docstore ─────────────────────────
    //
    // Stage 8b migrated `SegmentReader::get_doc` to one bounded
    // `Object::read_range` per fetched doc. The offset table makes the
    // exact range derivable up-front, so unlike postings (where the
    // search path may issue multiple reads per term) the docstore path
    // can — and is required to — issue **exactly one** range read per
    // returned doc. The next four tests guard that contract; the fifth
    // covers the strict-validation corruption path.

    /// Filter to docstore reads on a specific segment.
    fn docstore_reads_for_segment<'a>(
      log: &'a [RangeReadEntry],
      docstore_path: &Path,
    ) -> Vec<&'a RangeReadEntry> {
      log.iter().filter(|e| e.key == docstore_path).collect()
    }

    /// Stage 8b: `mget` of N IDs with `_source: true` against a single
    /// segment must issue **exactly N** docstore range reads — one per
    /// returned doc.
    #[test]
    fn mget_yields_exactly_one_docstore_range_read_per_returned_doc() {
      let dir = tempdir().unwrap();
      let (idx, log) = make_index_with_recording_blob_store(dir.path());
      let mut writer = idx.writer().unwrap();
      for (id, body) in [
        ("1", "alpha bravo"),
        ("2", "bravo charlie"),
        ("3", "charlie delta"),
        ("4", "delta echo"),
        ("5", "echo foxtrot"),
      ] {
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(id)),
              ("body".into(), serde_json::json!(body)),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
      }
      writer.commit().unwrap();
      let docstore_path = dir.path().join(&idx.manifest().segments[0].paths.docstore);

      let before = log.lock().unwrap().len();

      let reader = idx.reader().unwrap();
      let ids = vec!["1".to_string(), "3".to_string(), "5".to_string()];
      let results = reader.mget(&ids, true).unwrap();
      assert_eq!(results.len(), 3);
      assert!(results.iter().all(|r| r._source.is_some()));

      let after = log.lock().unwrap();
      let phase = &after[before..];
      let docstore_reads = docstore_reads_for_segment(phase, &docstore_path);
      assert_eq!(
        docstore_reads.len(),
        3,
        "Stage 8b: mget of 3 IDs with _source=true must issue exactly 3 \
         docstore range reads; got {} reads: {:?}",
        docstore_reads.len(),
        docstore_reads
      );
    }

    /// Stage 8b: `mget` with `_source: false` (i.e. just existence
    /// check, no payload fetch) must issue **zero** docstore reads —
    /// the materialization fast-path in `mget` skips `seg.get_doc`
    /// entirely.
    #[test]
    fn mget_with_source_false_issues_zero_docstore_reads() {
      let dir = tempdir().unwrap();
      let (idx, log) = make_index_with_recording_blob_store(dir.path());
      let mut writer = idx.writer().unwrap();
      for (id, body) in [("1", "alpha"), ("2", "bravo"), ("3", "charlie")] {
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(id)),
              ("body".into(), serde_json::json!(body)),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
      }
      writer.commit().unwrap();
      let docstore_path = dir.path().join(&idx.manifest().segments[0].paths.docstore);

      let before = log.lock().unwrap().len();

      let reader = idx.reader().unwrap();
      let ids = vec!["1".to_string(), "2".to_string(), "3".to_string()];
      let results = reader.mget(&ids, false).unwrap();
      assert_eq!(results.len(), 3);
      assert!(
        results.iter().all(|r| r._source.is_none()),
        "_source=false must not populate _source"
      );

      let after = log.lock().unwrap();
      let phase = &after[before..];
      let docstore_reads = docstore_reads_for_segment(phase, &docstore_path);
      assert_eq!(
        docstore_reads.len(),
        0,
        "Stage 8b: mget with _source=false must NOT consult the \
         docstore; got {} reads: {:?}",
        docstore_reads.len(),
        docstore_reads
      );
    }

    /// Stage 8b (the original Stage 8 done criterion): a top-K=10
    /// scoring search with `return_stored: true` must materialize
    /// exactly 10 hits and issue **exactly 10** docstore range reads —
    /// one per hit. Catches any path that fetches sources for
    /// pruned/non-returned candidates, or any path that doesn't fetch
    /// for a returned hit.
    #[test]
    fn top_k_search_with_source_yields_one_docstore_range_read_per_hit() {
      let dir = tempdir().unwrap();
      let (idx, log) = make_index_with_recording_blob_store(dir.path());
      let mut writer = idx.writer().unwrap();
      // Twenty docs — top-K=10 must prune ten of them out of the
      // returned set. If the search path still reads docstore for
      // pruned candidates the count will exceed 10.
      for i in 0..20 {
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(format!("doc{i}"))),
              (
                "body".into(),
                serde_json::json!(format!("alpha pos{i} term{i}")),
              ),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
      }
      writer.commit().unwrap();
      let docstore_path = dir.path().join(&idx.manifest().segments[0].paths.docstore);

      let before = log.lock().unwrap().len();

      let reader = idx.reader().unwrap();
      let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
        "query": "alpha",
        "limit": 10,
        "return_stored": true,
        "track_total_hits": true,
      }))
      .unwrap();
      let result = reader.search(&req).unwrap();
      assert_eq!(result.hits.len(), 10);
      assert!(
        result.hits.iter().all(|h| h.fields.is_some()),
        "all returned hits must carry fields when return_stored=true"
      );

      let after = log.lock().unwrap();
      let phase = &after[before..];
      let docstore_reads = docstore_reads_for_segment(phase, &docstore_path);
      assert_eq!(
        docstore_reads.len(),
        10,
        "Stage 8b: top-10 search with return_stored=true must issue \
         exactly 10 docstore range reads; got {} reads: {:?}",
        docstore_reads.len(),
        docstore_reads
      );
    }

    /// Stage 8b: each recorded docstore range exactly equals
    /// `offsets[doc_id]..offsets[doc_id+1]` (or `..docstore_len` for
    /// the last doc). Catches any drift in the offsets→range
    /// derivation logic in `SegmentReader::get_doc`.
    #[test]
    fn docstore_range_reads_match_offset_table() {
      let dir = tempdir().unwrap();
      let (idx, log) = make_index_with_recording_blob_store(dir.path());
      let mut writer = idx.writer().unwrap();
      let bodies: Vec<&str> = vec![
        "alpha bravo charlie",
        "delta echo",
        "foxtrot golf hotel india juliet",
        "kilo",
      ];
      for (i, body) in bodies.iter().enumerate() {
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(format!("d{i}"))),
              ("body".into(), serde_json::json!(body)),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
      }
      writer.commit().unwrap();
      let segment = &idx.manifest().segments[0];
      let docstore_path = dir.path().join(&segment.paths.docstore);
      let docstore_total_len = std::fs::metadata(&docstore_path).unwrap().len();

      let before = log.lock().unwrap().len();
      let reader = idx.reader().unwrap();
      // Fetch docs in non-monotonic order so the test isn't trivially
      // satisfied by any in-order slicing scheme.
      let ids: Vec<String> = vec!["d2".into(), "d0".into(), "d3".into(), "d1".into()];
      let results = reader.mget(&ids, true).unwrap();
      assert!(results.iter().all(|r| r._source.is_some()));
      drop(results);
      let after = log.lock().unwrap();
      let phase = &after[before..];
      let docstore_reads = docstore_reads_for_segment(phase, &docstore_path);

      // Recover the offset table from the segment's *.meta.json
      // (the same file `SegmentCore::load` parses).
      let seg_meta_bytes = std::fs::read(dir.path().join(&segment.paths.meta)).unwrap();
      let seg_meta: crate::index::segment::SegmentFileMeta =
        serde_json::from_slice(&seg_meta_bytes).unwrap();
      let offsets = seg_meta.doc_offsets;

      assert_eq!(docstore_reads.len(), 4);
      // Build the expected ranges per doc id.
      let expected_for = |doc_idx: usize| -> std::ops::Range<u64> {
        let start = offsets[doc_idx];
        let end = offsets
          .get(doc_idx + 1)
          .copied()
          .unwrap_or(docstore_total_len);
        start..end
      };

      // `mget` iterates a `HashMap`, so the fetch order across doc
      // ids is non-deterministic. Assert the recorded ranges as a
      // sorted multiset against the sorted expected ranges — a
      // bijection: each requested doc produced exactly one range
      // read, and each range matches some doc's offset-derived range.
      let mut got: Vec<std::ops::Range<u64>> =
        docstore_reads.iter().map(|r| r.range.clone()).collect();
      let mut expected: Vec<std::ops::Range<u64>> = vec![
        expected_for(0),
        expected_for(1),
        expected_for(2),
        expected_for(3),
      ];
      got.sort_by_key(|r| (r.start, r.end));
      expected.sort_by_key(|r| (r.start, r.end));
      assert_eq!(
        got, expected,
        "Stage 8b: docstore range reads must equal the offset-derived \
         ranges as a multiset"
      );
    }

    /// Stage 8b v2 [P1] (Codex review): an offset-derived span larger
    /// than `MAX_DOCSTORE_BYTES + 4` MUST be rejected **before**
    /// `read_range` is issued — otherwise a corrupt offset table or a
    /// docstore that's been sparsely extended out-of-band can trigger
    /// a multi-GB object-store GET / `Vec` allocation before
    /// parse-time validation has a chance to reject.
    ///
    /// We trigger an oversized span by sparsely extending the
    /// docstore file via `set_len` to `MAX_DOCSTORE_BYTES + 4096`
    /// after segment publish (sparse so the test stays cheap on disk
    /// and doesn't actually allocate 32 MiB). With a single-doc
    /// segment, `get_doc(0)` derives `end = docstore_len`, so the
    /// span equals the whole inflated file size — well past the
    /// guard threshold.
    ///
    /// The test wires a `RecordingBlobStore` and asserts:
    /// * `get_doc` errors with a message mentioning the span / bundle
    ///   size guard.
    /// * the recording log contains **zero** `read_range` calls
    ///   against the docstore — the guard fired before the read.
    #[test]
    fn oversized_offset_derived_range_is_rejected_without_issuing_read() {
      use crate::index::docstore::MAX_DOCSTORE_BYTES;

      let dir = tempdir().unwrap();

      // Phase 1: build a normal index with a single doc, then drop it.
      {
        let storage: Arc<dyn Storage> =
          Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
        let schema = Schema::default_text_body();
        let idx =
          Index::create_with_storage(dir.path(), schema, opts(dir.path()), storage).unwrap();
        let mut writer = idx.writer().unwrap();
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!("1")),
              ("body".into(), serde_json::json!("alpha bravo")),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
        writer.commit().unwrap();
      }

      // Find the docstore on disk by reading the manifest directly.
      let manifest_bytes =
        std::fs::read(dir.path().join("MANIFEST.json")).expect("manifest must exist");
      let manifest_value: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).expect("manifest must be valid JSON");
      let docstore_str = manifest_value["segments"][0]["paths"]["docstore"]
        .as_str()
        .expect("manifest segment[0].paths.docstore present")
        .to_string();
      // Stage 9a: v2 manifests record relative keys, so resolve
      // against the test dir to get the on-disk path.
      let docstore_path = dir.path().join(&docstore_str);

      // Phase 2: sparsely extend the docstore so `docstore_len` at
      // next open exceeds `MAX_DOCSTORE_BYTES + 4`. `set_len` extends
      // the file logically without writing zeros (sparse) so the test
      // doesn't pay a 32 MiB write cost.
      {
        let f = std::fs::OpenOptions::new()
          .write(true)
          .open(&docstore_path)
          .unwrap();
        let inflated_len = MAX_DOCSTORE_BYTES as u64 + 4096;
        f.set_len(inflated_len).unwrap();
        f.sync_all().unwrap();
      }

      // Phase 3: reopen with a `RecordingBlobStore` so we can assert
      // **zero** `read_range` calls against the docstore. Use
      // `TrustManifest` so the docstore checksum mismatch caused by
      // our sparse extension doesn't reject the segment open before
      // `get_doc` runs.
      let storage: Arc<dyn Storage> =
        Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
      let inner_blob: Arc<dyn BlobStore> = Arc::new(StorageAsBlobStore::new(storage.clone()));
      let (recording, log) = RecordingBlobStore::new(inner_blob);
      let blob_store: Arc<dyn BlobStore> = recording;

      let mut reopen_opts = opts(dir.path());
      reopen_opts.create_if_missing = false;
      reopen_opts.checksum_policy = crate::api::types::ChecksumPolicy::TrustManifest;
      let idx = Index::open_with_storage_and_blob_store(reopen_opts, storage, blob_store).unwrap();
      let reader = idx.reader().unwrap();

      let before = log.lock().unwrap().len();
      let err = reader
        .mget(&["1".to_string()], true)
        .expect_err("oversized offset-derived span must surface as an error");
      let msg = format!("{err:#}");
      assert!(
        msg.contains("exceeds maximum bundle size") || msg.contains("oversized"),
        "expected span-guard error mentioning 'maximum bundle size' or 'oversized'; got: {msg}"
      );

      let after = log.lock().unwrap();
      let phase = &after[before..];
      let docstore_reads = docstore_reads_for_segment(phase, &docstore_path);
      assert_eq!(
        docstore_reads.len(),
        0,
        "Stage 8b [P1]: oversized offset-derived span must be rejected \
         BEFORE issuing read_range; got {} unexpected reads: {:?}",
        docstore_reads.len(),
        docstore_reads
      );
    }

    /// Stage 8b corruption guard: if the offset table implies a longer
    /// record than the embedded length actually encodes (but still
    /// within the span guard's bound), `get_doc` must error rather
    /// than silently ignore the trailing bytes. We trigger this by
    /// appending a small amount of junk to the on-disk docstore file
    /// after segment publish, then re-opening with `TrustManifest`
    /// (so the docstore file's manifest checksum mismatch doesn't
    /// reject the open before `get_doc` runs).
    #[test]
    fn docstore_offset_length_mismatch_is_detected_as_corruption() {
      let dir = tempdir().unwrap();
      let storage: Arc<dyn Storage> =
        Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
      let schema = Schema::default_text_body();
      let idx =
        Index::create_with_storage(dir.path(), schema, opts(dir.path()), storage.clone()).unwrap();
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("alpha bravo")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
      let docstore_path = dir.path().join(&idx.manifest().segments[0].paths.docstore);
      drop(idx);

      // Append junk so docstore_len at next open exceeds the actual
      // last record's `4 + embedded_len` boundary. The single-doc
      // range becomes 0..(real_len + junk_len), and
      // `decode_docstore_record` rejects the mismatch.
      {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
          .append(true)
          .open(&docstore_path)
          .unwrap();
        f.write_all(&[0u8; 64]).unwrap();
        f.sync_all().unwrap();
      }

      // Reopen with TrustManifest so the recorded docstore checksum
      // mismatch caused by our corruption doesn't reject the segment
      // before `get_doc` runs.
      let mut reopen_opts = opts(dir.path());
      reopen_opts.create_if_missing = false;
      reopen_opts.checksum_policy = crate::api::types::ChecksumPolicy::TrustManifest;
      let reopened = Index::open_with_storage(reopen_opts, storage).unwrap();
      let reader = reopened.reader().unwrap();
      let err = reader
        .mget(&["1".to_string()], true)
        .expect_err("offset/length mismatch must surface as an error");
      let msg = format!("{err:#}");
      assert!(
        msg.contains("length mismatch") || msg.contains("offset table") || msg.contains("corrupt"),
        "expected corruption error mentioning length mismatch / offset table / corrupt; got: {msg}"
      );
    }
  }

  /// Stage 9a regression suite — portable manifest format. The
  /// invariant is: v2 manifests record only relative-to-root segment
  /// keys (no absolute paths, no `..`), so an index can be physically
  /// moved to a new root and reopened without rewriting the manifest.
  /// v1 manifests with absolute paths are still accepted in-place
  /// (back-compat for existing on-disk indexes).
  mod stage9a_portable_manifest {
    use super::*;
    use crate::index::manifest::{Manifest, MANIFEST_LATEST_VERSION};

    /// Stage 9a: a freshly-committed manifest must be `version: 2`
    /// and must record relative-to-root segment keys.
    #[test]
    fn fresh_commits_emit_v2_manifest_with_relative_keys() {
      let dir = tempdir().unwrap();
      let schema = Schema::default_text_body();
      let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("alpha")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();

      let manifest = idx.manifest();
      assert_eq!(manifest.version, MANIFEST_LATEST_VERSION);
      assert_eq!(manifest.segments.len(), 1);
      let seg = &manifest.segments[0];
      // Each key is the bare filename, no embedded root.
      assert_eq!(seg.paths.terms, format!("seg_{}.terms", seg.id));
      assert_eq!(seg.paths.postings, format!("seg_{}.post", seg.id));
      assert_eq!(seg.paths.docstore, format!("seg_{}.docs", seg.id));
      assert_eq!(seg.paths.fast, format!("seg_{}.fast", seg.id));
      assert_eq!(seg.paths.meta, format!("seg_{}.meta", seg.id));
      // Validation passes because all keys are relative.
      seg.paths.validate_v2_relative().unwrap();
    }

    /// Stage 9a (the load-bearing portability test): write an index
    /// in dir A, physically move every file to dir B, open against
    /// the new root, and verify every read path still works
    /// end-to-end (search, mget with `_source: true`).
    #[test]
    fn v2_index_can_be_relocated_to_new_root_and_searched() {
      let dir_a = tempdir().unwrap();
      let dir_b = tempdir().unwrap();

      let schema = Schema::default_text_body();
      let idx = Index::create(dir_a.path(), schema, opts(dir_a.path())).unwrap();
      let mut writer = idx.writer().unwrap();
      for (id, body) in [
        ("1", "alpha bravo"),
        ("2", "bravo charlie"),
        ("3", "charlie delta"),
      ] {
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(id)),
              ("body".into(), serde_json::json!(body)),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
      }
      writer.commit().unwrap();
      drop(idx);

      // Move every file from dir_a to dir_b.
      for entry in std::fs::read_dir(dir_a.path()).unwrap() {
        let entry = entry.unwrap();
        let from = entry.path();
        let to = dir_b.path().join(entry.file_name());
        if from.is_dir() {
          // For vectors-feature segment dirs (or any nested dir),
          // walk recursively. Without `vectors` feature there are
          // never any.
          copy_dir_recursive(&from, &to).unwrap();
        } else {
          std::fs::rename(&from, &to).unwrap();
        }
      }

      // Open against the new root. Every path-using read must
      // resolve through the new root (no stale absolute paths).
      let mut reopen_opts = opts(dir_b.path());
      reopen_opts.create_if_missing = false;
      let reopened = Index::open(reopen_opts).unwrap();
      let reader = reopened.reader().unwrap();

      // Search (postings path).
      let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
        "query": "bravo",
        "limit": 10,
        "track_total_hits": true,
      }))
      .unwrap();
      let result = reader.search(&req).unwrap();
      assert_eq!(
        result.total_hits_estimate, 2,
        "search at relocated root must return correct hits"
      );

      // mget with _source (docstore path).
      let mget_results = reader
        .mget(&["1".to_string(), "3".to_string()], true)
        .unwrap();
      assert_eq!(mget_results.len(), 2);
      assert!(mget_results.iter().all(|r| r._source.is_some()));
    }

    fn copy_dir_recursive(from: &Path, to: &Path) -> Result<()> {
      std::fs::create_dir_all(to)?;
      for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let src = entry.path();
        let dst = to.join(entry.file_name());
        if src.is_dir() {
          copy_dir_recursive(&src, &dst)?;
        } else {
          std::fs::rename(&src, &dst)?;
        }
      }
      Ok(())
    }

    /// Stage 9a: a hand-authored v1 manifest with absolute paths
    /// must still open in-place (legacy back-compat). We synthesize
    /// such a manifest by writing a fresh v2 index, then rewriting
    /// the manifest as v1 with absolute paths in the JSON.
    #[test]
    fn v1_absolute_path_manifest_opens_in_place() {
      let dir = tempdir().unwrap();
      let schema = Schema::default_text_body();
      let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("alpha")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
      drop(idx);

      // Surgically rewrite the manifest: bump version 2 → 1 and
      // expand each segment path to its absolute form. This mimics
      // an on-disk manifest produced by an older searchlite build.
      let manifest_path = dir.path().join("MANIFEST.json");
      let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
      value["version"] = serde_json::json!(1);
      let segments = value["segments"].as_array_mut().unwrap();
      for seg in segments.iter_mut() {
        let paths = seg["paths"].as_object_mut().unwrap();
        for key in ["terms", "postings", "docstore", "fast", "meta"] {
          let rel = paths[key].as_str().unwrap().to_string();
          let abs = dir.path().join(&rel).to_string_lossy().into_owned();
          paths[key] = serde_json::json!(abs);
        }
        if let Some(serde_json::Value::String(rel)) = paths.get("vector_dir").cloned() {
          let abs = dir.path().join(&rel).to_string_lossy().into_owned();
          paths.insert("vector_dir".into(), serde_json::json!(abs));
        }
      }
      std::fs::write(&manifest_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

      // Open against the (still-original) root. Absolute paths
      // resolve through `SegmentPaths::resolve` unchanged, so reads
      // succeed.
      let mut reopen_opts = opts(dir.path());
      reopen_opts.create_if_missing = false;
      let reopened = Index::open(reopen_opts).unwrap();
      assert_eq!(reopened.manifest().version, 1);
      let reader = reopened.reader().unwrap();
      let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
        "query": "alpha",
        "limit": 10,
        "track_total_hits": true,
      }))
      .unwrap();
      let result = reader.search(&req).unwrap();
      assert_eq!(result.total_hits_estimate, 1);
    }

    /// Stage 9a: after relocation, a compact run on the relocated
    /// index must succeed without path failures. Compact reads every
    /// segment's meta + checksums and writes new ones; if any path
    /// resolution missed the relocation, this would fail.
    #[test]
    fn relocated_v2_index_supports_compact() {
      let dir_a = tempdir().unwrap();
      let dir_b = tempdir().unwrap();

      let schema = Schema::default_text_body();
      let idx = Index::create(dir_a.path(), schema, opts(dir_a.path())).unwrap();
      // Two commits → two segments → eligible for compact.
      for body in ["alpha", "bravo"] {
        let mut writer = idx.writer().unwrap();
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(body)),
              ("body".into(), serde_json::json!(body)),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
        writer.commit().unwrap();
      }
      drop(idx);

      for entry in std::fs::read_dir(dir_a.path()).unwrap() {
        let entry = entry.unwrap();
        let from = entry.path();
        let to = dir_b.path().join(entry.file_name());
        if from.is_dir() {
          copy_dir_recursive(&from, &to).unwrap();
        } else {
          std::fs::rename(&from, &to).unwrap();
        }
      }

      let mut reopen_opts = opts(dir_b.path());
      reopen_opts.create_if_missing = false;
      let reopened = Index::open(reopen_opts).unwrap();
      assert_eq!(reopened.manifest().segments.len(), 2);
      reopened.compact().unwrap();
      assert_eq!(
        reopened.manifest().segments.len(),
        1,
        "compact must produce a single merged segment after relocation"
      );

      // Re-search to prove the new merged segment is reachable.
      let reader = reopened.reader().unwrap();
      let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
        "query": "alpha",
        "limit": 10,
        "track_total_hits": true,
      }))
      .unwrap();
      let result = reader.search(&req).unwrap();
      assert_eq!(result.total_hits_estimate, 1);
    }

    /// Stage 9a: `Manifest::store` must reject any v2 manifest whose
    /// segment paths are absolute or contain `..` components, even
    /// if hand-constructed in-process. Catches accidental
    /// regressions in the writer that emit non-portable keys.
    #[test]
    fn manifest_store_rejects_v2_absolute_or_dotdot_paths() {
      let dir = tempdir().unwrap();
      let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
      let manifest_path = Manifest::manifest_path(dir.path());

      // Build a minimal valid v2 manifest, then mutate one segment
      // path into an absolute form and verify `store` rejects it.
      let schema = Schema::default_text_body();
      let mut manifest = Manifest::new(schema.clone());
      manifest.segments.push(crate::index::manifest::SegmentMeta {
        id: "abs".into(),
        generation: 1,
        paths: crate::index::manifest::SegmentPaths {
          terms: "/abs/seg_abs.terms".into(),
          postings: "seg_abs.post".into(),
          docstore: "seg_abs.docs".into(),
          fast: "seg_abs.fast".into(),
          meta: "seg_abs.meta".into(),
          #[cfg(feature = "vectors")]
          vector_dir: None,
        },
        doc_count: 0,
        max_doc_id: 0,
        blockmax: true,
        deleted_docs: Vec::new(),
        avg_field_lengths: Default::default(),
        content_hashes: Default::default(),
        write_binding_b64: None,
      });
      let err = manifest
        .store(&storage, &manifest_path)
        .expect_err("v2 manifest with absolute segment path must be rejected by Manifest::store");
      assert!(format!("{err:#}").contains("absolute"));

      // Now `..`.
      manifest.segments[0].paths.terms = "../escape.terms".into();
      let err = manifest
        .store(&storage, &manifest_path)
        .expect_err("v2 manifest with `..` segment path must be rejected by Manifest::store");
      assert!(format!("{err:#}").contains(".."));
    }

    /// Stage 9a: `Manifest::load` must reject v2 manifests on disk
    /// whose paths violate the relative-key invariant. Catches
    /// corruption / hand-edits / older buggy writers.
    #[test]
    fn manifest_load_rejects_v2_absolute_paths() {
      let dir = tempdir().unwrap();
      let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
      let manifest_path = Manifest::manifest_path(dir.path());
      let v2_with_abs = serde_json::json!({
        "version": 2,
        "uuid": uuid::Uuid::new_v4(),
        "segments": [{
          "id": "abs",
          "generation": 1,
          "paths": {
            "terms": "/abs/seg_abs.terms",
            "postings": "seg_abs.post",
            "docstore": "seg_abs.docs",
            "fast": "seg_abs.fast",
            "meta": "seg_abs.meta"
          },
          "doc_count": 0,
          "max_doc_id": 0,
          "blockmax": true,
          "deleted_docs": [],
          "avg_field_lengths": {},
          "checksums": {}
        }],
        "committed_at": "2024-01-01T00:00:00Z",
        "schema": {}
      });
      std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&v2_with_abs).unwrap(),
      )
      .unwrap();

      let err = Manifest::load(&storage, &manifest_path)
        .expect_err("v2 manifest with absolute paths must be rejected on load");
      assert!(format!("{err:#}").contains("absolute"));
    }

    // ───────────────────── Stage 9a v2 P2 regressions ─────────────────────

    /// Stage 9a [P2] (Codex review): a legacy v1 manifest must be
    /// upgraded in-place to v2 on the **first commit** after open.
    /// Without this, the writer's clone-then-mutate flow keeps
    /// `version: 1` and the new manifest mixes absolute legacy paths
    /// with freshly-relative new-segment paths — the index never
    /// becomes portable.
    #[test]
    fn legacy_v1_manifest_is_upgraded_to_v2_on_first_commit() {
      let dir = tempdir().unwrap();
      let schema = Schema::default_text_body();
      let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("seed")),
            ("body".into(), serde_json::json!("alpha")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
      drop(idx);

      // Hand-roll a v1 manifest by rewriting the on-disk file.
      let manifest_path = dir.path().join("MANIFEST.json");
      let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
      value["version"] = serde_json::json!(1);
      let segments = value["segments"].as_array_mut().unwrap();
      for seg in segments.iter_mut() {
        let paths = seg["paths"].as_object_mut().unwrap();
        for key in ["terms", "postings", "docstore", "fast", "meta"] {
          let rel = paths[key].as_str().unwrap().to_string();
          let abs = dir.path().join(&rel).to_string_lossy().into_owned();
          paths[key] = serde_json::json!(abs);
        }
        if let Some(serde_json::Value::String(rel)) = paths.get("vector_dir").cloned() {
          let abs = dir.path().join(&rel).to_string_lossy().into_owned();
          paths.insert("vector_dir".into(), serde_json::json!(abs));
        }
      }
      std::fs::write(&manifest_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

      // Open. The v1 manifest is accepted (legacy back-compat).
      let mut reopen_opts = opts(dir.path());
      reopen_opts.create_if_missing = false;
      let reopened = Index::open(reopen_opts).unwrap();
      assert_eq!(reopened.manifest().version, 1);

      // Commit a second doc → triggers `upgrade_to_latest` before
      // staging.
      let mut writer = reopened.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("post-upgrade")),
            ("body".into(), serde_json::json!("bravo")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();

      // Live manifest must now be v2 with all relative paths.
      let upgraded_bytes = std::fs::read(&manifest_path).unwrap();
      let upgraded: Manifest = serde_json::from_slice(&upgraded_bytes).unwrap();
      assert_eq!(upgraded.version, MANIFEST_LATEST_VERSION);
      assert_eq!(upgraded.segments.len(), 2);
      for seg in &upgraded.segments {
        seg
          .paths
          .validate_v2_relative()
          .expect("every segment path must be relative after upgrade");
      }
    }

    /// Stage 9a [P2] (Codex review): a malformed v1 manifest with
    /// `..` segment paths must be rejected at load time.
    /// `Manifest::load` historically skipped all v1 path validation,
    /// so a hand-edited `../escape.terms` would resolve under the
    /// current root and could drive reads outside the index.
    ///
    /// Stage 9a v4 update: relative paths are now LEGITIMATE for v1
    /// (the old writer used `root.join(filename)` and `root` could be
    /// relative). Only `..`-bearing or empty paths are rejected.
    #[test]
    fn manifest_load_rejects_v1_dotdot_paths() {
      let dir = tempdir().unwrap();
      let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
      let manifest_path = Manifest::manifest_path(dir.path());

      // v1 + `..` path → rejected.
      let v1_with_dotdot = serde_json::json!({
        "version": 1,
        "uuid": uuid::Uuid::new_v4(),
        "segments": [{
          "id": "dotdot",
          "generation": 1,
          "paths": {
            "terms": "/abs/../escape.terms",
            "postings": "/abs/seg.post",
            "docstore": "/abs/seg.docs",
            "fast": "/abs/seg.fast",
            "meta": "/abs/seg.meta"
          },
          "doc_count": 0,
          "max_doc_id": 0,
          "blockmax": true,
          "deleted_docs": [],
          "avg_field_lengths": {},
          "checksums": {}
        }],
        "committed_at": "2024-01-01T00:00:00Z",
        "schema": {}
      });
      std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&v1_with_dotdot).unwrap(),
      )
      .unwrap();
      let err = Manifest::load(&storage, &manifest_path)
        .expect_err("v1 manifest with `..` must be rejected");
      let msg = format!("{err:#}");
      assert!(
        msg.contains(".."),
        "expected error mentioning ..; got: {msg}"
      );
    }

    /// Stage 9a [P2] (Codex review): the recovery promote in
    /// `reconcile_pending_manifest` and the leftover-pending promote
    /// in `Writer::commit` MUST validate the staged bytes before
    /// publishing them. Without this, a `.pending` file produced by
    /// an older buggy writer could carry a v1 manifest with mixed
    /// shape paths and we'd publish it verbatim.
    ///
    /// Drives the recovery path by writing a malformed `.pending`
    /// file (v2 + absolute path) alongside a WAL with a `Commit`
    /// record, then re-opening the index. `reconcile_pending_manifest`
    /// must reject the bad pending file rather than promoting it.
    #[test]
    fn recovery_rejects_invalid_pending_manifest_instead_of_promoting() {
      let dir = tempdir().unwrap();
      let schema = Schema::default_text_body();
      let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("alpha")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
      drop(writer);
      drop(idx);

      // Plant a malformed v2 pending manifest (absolute path) next
      // to the live manifest. The WAL already contains a Commit
      // record from the prior commit, so `reconcile_pending_manifest`
      // would otherwise promote this on next open.
      let pending_path = Manifest::manifest_pending_path(dir.path());
      let bogus_pending = serde_json::json!({
        "version": 2,
        "uuid": uuid::Uuid::new_v4(),
        "segments": [{
          "id": "abs",
          "generation": 99,
          "paths": {
            "terms": "/abs/poisoned.terms",
            "postings": "/abs/poisoned.post",
            "docstore": "/abs/poisoned.docs",
            "fast": "/abs/poisoned.fast",
            "meta": "/abs/poisoned.meta"
          },
          "doc_count": 0,
          "max_doc_id": 0,
          "blockmax": true,
          "deleted_docs": [],
          "avg_field_lengths": {},
          "checksums": {}
        }],
        "committed_at": "2099-01-01T00:00:00Z",
        "schema": {}
      });
      std::fs::write(
        &pending_path,
        serde_json::to_vec_pretty(&bogus_pending).unwrap(),
      )
      .unwrap();

      // Re-open. Reconcile must reject the bogus pending bytes
      // rather than overwrite the live manifest with them.
      let mut reopen_opts = opts(dir.path());
      reopen_opts.create_if_missing = false;
      let result = Index::open(reopen_opts);

      // Either the open errors with a validation failure (preferred)
      // or it succeeds and the live manifest is unchanged. Both
      // outcomes prove the bogus pending was NOT promoted verbatim.
      let live_after =
        std::fs::read(Manifest::manifest_path(dir.path())).expect("live manifest still exists");
      let live: Manifest = serde_json::from_slice(&live_after).expect("live manifest still parses");
      assert!(
        live.segments.iter().all(|s| s.id != "abs"),
        "Stage 9a [P2]: bogus pending manifest with id=abs must NOT be promoted; \
         live manifest segments: {:?}",
        live.segments.iter().map(|s| &s.id).collect::<Vec<_>>()
      );
      // If open succeeded, search must still be intact (the original
      // alpha doc).
      if let Ok(reopened) = result {
        let reader = reopened.reader().unwrap();
        let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
          "query": "alpha",
          "limit": 10,
          "track_total_hits": true,
        }))
        .unwrap();
        let r = reader.search(&req).unwrap();
        assert_eq!(r.total_hits_estimate, 1);
      }
    }

    // ───────────────────── Stage 9a v3 regressions ─────────────────────

    /// Helper: rewrite the live manifest at `dir/MANIFEST.json` into
    /// a v1 absolute-path shape, simulating what a pre-Stage-9 build
    /// would have produced. Returns once the on-disk manifest is
    /// valid v1 (passes `validate_v1_legacy`).
    fn rewrite_manifest_as_v1_absolute(dir: &Path) {
      let manifest_path = dir.join("MANIFEST.json");
      let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
      value["version"] = serde_json::json!(1);
      let segments = value["segments"].as_array_mut().unwrap();
      for seg in segments.iter_mut() {
        let paths = seg["paths"].as_object_mut().unwrap();
        for key in ["terms", "postings", "docstore", "fast", "meta"] {
          let rel = paths[key].as_str().unwrap().to_string();
          let abs = dir.join(&rel).to_string_lossy().into_owned();
          paths[key] = serde_json::json!(abs);
        }
        if let Some(serde_json::Value::String(rel)) = paths.get("vector_dir").cloned() {
          let abs = dir.join(&rel).to_string_lossy().into_owned();
          paths.insert("vector_dir".into(), serde_json::json!(abs));
        }
      }
      std::fs::write(&manifest_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    }

    /// Stage 9a v3 [P2] (Codex review): `compact` against a legacy
    /// v1 index must succeed by upgrading the manifest to v2 in place,
    /// not fail at `Manifest::store` (which now refuses v1).
    #[test]
    fn compact_upgrades_legacy_v1_manifest_to_v2() {
      let dir = tempdir().unwrap();
      let schema = Schema::default_text_body();
      let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
      // Need ≥2 segments for compact to do anything substantial.
      for body in ["alpha", "bravo"] {
        let mut writer = idx.writer().unwrap();
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(body)),
              ("body".into(), serde_json::json!(body)),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
        writer.commit().unwrap();
      }
      drop(idx);

      rewrite_manifest_as_v1_absolute(dir.path());

      let mut reopen_opts = opts(dir.path());
      reopen_opts.create_if_missing = false;
      let reopened = Index::open(reopen_opts).unwrap();
      assert_eq!(reopened.manifest().version, 1);
      reopened.compact().unwrap();

      let upgraded: Manifest =
        serde_json::from_slice(&std::fs::read(dir.path().join("MANIFEST.json")).unwrap()).unwrap();
      assert_eq!(upgraded.version, MANIFEST_LATEST_VERSION);
      assert_eq!(upgraded.segments.len(), 1);
      for seg in &upgraded.segments {
        seg.paths.validate_v2_relative().unwrap();
      }
    }

    /// Stage 9a v3 [P2] (Codex review): `merge_segments` against a
    /// legacy v1 index must succeed by upgrading to v2.
    #[test]
    fn merge_segments_upgrades_legacy_v1_manifest_to_v2() {
      let dir = tempdir().unwrap();
      let schema = Schema::default_text_body();
      let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
      for body in ["alpha", "bravo", "charlie"] {
        let mut writer = idx.writer().unwrap();
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(body)),
              ("body".into(), serde_json::json!(body)),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
        writer.commit().unwrap();
      }
      let v2_segment_ids: Vec<String> = idx
        .manifest()
        .segments
        .iter()
        .map(|s| s.id.clone())
        .collect();
      assert_eq!(v2_segment_ids.len(), 3);
      drop(idx);

      rewrite_manifest_as_v1_absolute(dir.path());

      let mut reopen_opts = opts(dir.path());
      reopen_opts.create_if_missing = false;
      let reopened = Index::open(reopen_opts).unwrap();
      assert_eq!(reopened.manifest().version, 1);

      // Merge two of the three segments. The third stays untouched
      // and must also be relativized by the upgrade (relativize_under
      // covers every segment, not just the merge target).
      let merge_ids = v2_segment_ids[..2].to_vec();
      reopened.merge_segments(&merge_ids, None).unwrap();

      let upgraded: Manifest =
        serde_json::from_slice(&std::fs::read(dir.path().join("MANIFEST.json")).unwrap()).unwrap();
      assert_eq!(upgraded.version, MANIFEST_LATEST_VERSION);
      assert_eq!(upgraded.segments.len(), 2);
      for seg in &upgraded.segments {
        seg.paths.validate_v2_relative().unwrap();
      }
    }

    /// Stage 9a v3 [P1] (Codex review): a valid pre-Stage-9 crash
    /// state has a v1 absolute-path `.pending` file plus a WAL with a
    /// durable Commit record. `Index::open`'s reconciler MUST upgrade
    /// the v1 pending bytes to v2 and promote them, not reject them
    /// — rejecting would leave the pending file in place and block
    /// recovery.
    #[test]
    fn recovery_upgrades_legacy_v1_pending_manifest() {
      let dir = tempdir().unwrap();
      let schema = Schema::default_text_body();
      // Phase 1: build a normal index with one durable commit. This
      // gives us a legitimate WAL with a Commit record we can reuse
      // as the recovery fence below.
      let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("seed")),
            ("body".into(), serde_json::json!("alpha")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
      drop(writer);
      drop(idx);

      // Phase 2: hand-craft a v1 absolute-path pending manifest that
      // mirrors what an old searchlite build would have written. The
      // segment files referenced by it ARE present on disk (we wrote
      // them in Phase 1), and the WAL already has a Commit fence —
      // so this is exactly the post-crash state of a legacy index.
      let live_path = Manifest::manifest_path(dir.path());
      let live: Manifest = serde_json::from_slice(&std::fs::read(&live_path).unwrap()).unwrap();
      let pending_path = Manifest::manifest_pending_path(dir.path());
      let mut pending = live.clone();
      pending.version = 1;
      for seg in pending.segments.iter_mut() {
        for key in [
          &mut seg.paths.terms,
          &mut seg.paths.postings,
          &mut seg.paths.docstore,
          &mut seg.paths.fast,
          &mut seg.paths.meta,
        ] {
          *key = dir.path().join(&*key).to_string_lossy().into_owned();
        }
        #[cfg(feature = "vectors")]
        if let Some(d) = seg.paths.vector_dir.as_mut() {
          *d = dir.path().join(&*d).to_string_lossy().into_owned();
        }
      }
      std::fs::write(&pending_path, serde_json::to_vec_pretty(&pending).unwrap()).unwrap();
      // Bump the live manifest's `committed_at` back so the pending
      // looks "newer" — though reconcile_pending_manifest's promote
      // is unconditional when the WAL has a Commit, so this is just
      // belt-and-suspenders.
      let mut older_live = live.clone();
      older_live.committed_at = "2000-01-01T00:00:00Z".into();
      std::fs::write(&live_path, serde_json::to_vec_pretty(&older_live).unwrap()).unwrap();

      // Phase 3: reopen. The reconciler must upgrade the v1 pending
      // to v2 and promote it. After open, the live manifest must be
      // v2 with relative paths, and the pending file must be gone.
      let mut reopen_opts = opts(dir.path());
      reopen_opts.create_if_missing = false;
      let reopened = Index::open(reopen_opts).unwrap();
      assert!(
        !std::fs::exists(&pending_path).unwrap(),
        "pending file must be cleaned up after successful recovery"
      );
      assert_eq!(reopened.manifest().version, MANIFEST_LATEST_VERSION);
      for seg in &reopened.manifest().segments {
        seg.paths.validate_v2_relative().unwrap();
      }

      // The live manifest on disk is also v2.
      let live_after: Manifest =
        serde_json::from_slice(&std::fs::read(&live_path).unwrap()).unwrap();
      assert_eq!(live_after.version, MANIFEST_LATEST_VERSION);

      // Search still works (post-recovery, post-upgrade).
      let reader = reopened.reader().unwrap();
      let req: crate::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
        "query": "alpha",
        "limit": 10,
        "track_total_hits": true,
      }))
      .unwrap();
      let r = reader.search(&req).unwrap();
      assert_eq!(r.total_hits_estimate, 1);
    }

    // ───────────────────── Stage 9a v4 regressions ─────────────────────

    /// Stage 9a v4 [P2] (Codex review): the pre-Stage-9 writer used
    /// `IndexOptions.path.join(filename)`. When the caller passed a
    /// **relative** path (e.g. `--index idx`), the resulting v1
    /// manifest paths were relative-with-root-prefix
    /// (`idx/seg_X.terms`). Three invariants must hold for that shape:
    ///
    /// 1. `validate_v1_legacy` accepts relative + non-`..` keys.
    /// 2. `resolve_segment_path` detects the root prefix and does
    ///    NOT double-prefix to `idx/idx/seg_X.terms`.
    /// 3. `relativize_under` strips the root prefix from the relative
    ///    key (not just from absolute paths) so the upgrade produces
    ///    bare v2 keys.
    ///
    /// We exercise these at the helper level (where the actual logic
    /// lives) — a full `Index::open` round-trip with a relative root
    /// would also need CWD manipulation, since `FsStorage` resolves
    /// relative paths through the process CWD. The integration shape
    /// is covered indirectly by the existing absolute-path legacy
    /// upgrade tests and by the `Manifest::load` round-trip below.
    #[test]
    fn legacy_v1_relative_root_paths_load_resolve_and_upgrade() {
      use crate::index::manifest::{resolve_segment_path, ResolvedSegmentPaths, SegmentPaths};

      // (1) `validate_v1_legacy` accepts relative + non-`..`.
      let v1_relative_paths = SegmentPaths {
        terms: "idx/a.terms".into(),
        postings: "idx/a.post".into(),
        docstore: "idx/a.docs".into(),
        fast: "idx/a.fast".into(),
        meta: "idx/a.meta".into(),
        #[cfg(feature = "vectors")]
        vector_dir: None,
      };
      v1_relative_paths.validate_v1_legacy().unwrap();

      // (2) `resolve_segment_path` doesn't double-prefix when the
      // relative key already starts with the root.
      let resolved = resolve_segment_path(Path::new("idx"), "idx/seg_xyz.terms");
      assert_eq!(resolved, PathBuf::from("idx/seg_xyz.terms"));
      // And bare-relative keys (v2 shape) are still joined under root.
      let resolved_bare = resolve_segment_path(Path::new("idx"), "seg_xyz.terms");
      assert_eq!(resolved_bare, PathBuf::from("idx/seg_xyz.terms"));
      // Absolute paths under a different root (legacy v1) pass through.
      let resolved_abs = resolve_segment_path(Path::new("/some/root"), "/elsewhere/seg.terms");
      assert_eq!(resolved_abs, PathBuf::from("/elsewhere/seg.terms"));

      // (3) `relativize_under` strips the root-prefix from a relative
      // key as well as from an absolute one.
      let mut rp = v1_relative_paths.clone();
      rp.relativize_under(Path::new("idx")).unwrap();
      assert_eq!(rp.terms, "a.terms");
      assert_eq!(rp.postings, "a.post");
      // Post-upgrade, `validate_v2_relative` passes.
      rp.validate_v2_relative().unwrap();

      // `ResolvedSegmentPaths::resolve` round-trips through the same
      // logic — sanity check the struct-level helper.
      let r: ResolvedSegmentPaths = v1_relative_paths.resolve(Path::new("idx"));
      assert_eq!(r.terms, PathBuf::from("idx/a.terms"));

      // Round-trip through `Manifest::load`: serialize a v1 manifest
      // with relative-root keys to disk, load it, and verify
      // validation passes (it would have errored under the v3-era
      // strict v1-must-be-absolute rule).
      let dir = tempdir().unwrap();
      let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
      let manifest_path = Manifest::manifest_path(dir.path());
      let v1_with_relative_root = serde_json::json!({
        "version": 1,
        "uuid": uuid::Uuid::new_v4(),
        "segments": [{
          "id": "rel",
          "generation": 1,
          "paths": {
            "terms": "idx/seg_rel.terms",
            "postings": "idx/seg_rel.post",
            "docstore": "idx/seg_rel.docs",
            "fast": "idx/seg_rel.fast",
            "meta": "idx/seg_rel.meta"
          },
          "doc_count": 0,
          "max_doc_id": 0,
          "blockmax": true,
          "deleted_docs": [],
          "avg_field_lengths": {},
          "checksums": {}
        }],
        "committed_at": "2024-01-01T00:00:00Z",
        "schema": {}
      });
      std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&v1_with_relative_root).unwrap(),
      )
      .unwrap();
      let loaded =
        Manifest::load(&storage, &manifest_path).expect("v1 with relative-root paths must load");
      assert_eq!(loaded.version, 1);
    }

    /// Stage 9a v4 [P3] (Codex review): a manifest with `version: 0`
    /// and an empty segment list must be rejected. Previously the
    /// per-segment loop owned the unsupported-version check, so an
    /// empty manifest skipped validation entirely.
    #[test]
    fn manifest_load_rejects_unsupported_version_even_when_segments_empty() {
      let dir = tempdir().unwrap();
      let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
      let manifest_path = Manifest::manifest_path(dir.path());

      let bogus = serde_json::json!({
        "version": 0,
        "uuid": uuid::Uuid::new_v4(),
        "segments": [],
        "committed_at": "2024-01-01T00:00:00Z",
        "schema": {}
      });
      std::fs::write(&manifest_path, serde_json::to_vec_pretty(&bogus).unwrap()).unwrap();
      let err = Manifest::load(&storage, &manifest_path)
        .expect_err("manifest with version 0 must be rejected");
      assert!(
        format!("{err:#}").contains("unsupported version 0"),
        "expected error mentioning unsupported version 0; got: {err:#}"
      );

      let too_new = serde_json::json!({
        "version": 99,
        "uuid": uuid::Uuid::new_v4(),
        "segments": [],
        "committed_at": "2024-01-01T00:00:00Z",
        "schema": {}
      });
      std::fs::write(&manifest_path, serde_json::to_vec_pretty(&too_new).unwrap()).unwrap();
      let err = Manifest::load(&storage, &manifest_path)
        .expect_err("manifest with newer-than-supported version must be rejected");
      assert!(
        format!("{err:#}").contains("unsupported version 99"),
        "expected error mentioning unsupported version 99; got: {err:#}"
      );
    }

    // ───────────────────── Stage 9a v5 regressions ─────────────────────

    /// Stage 9a v5 [P2] (Codex review): the recovery and
    /// leftover-pending promote paths parse pending bytes directly
    /// and call `upgrade_to_latest` without going through
    /// `Manifest::load`. Any unsupported `version` (0 or >
    /// `MANIFEST_LATEST_VERSION`) on a pending file would otherwise
    /// be silently mutated to v2 and promoted. `upgrade_to_latest`
    /// must reject those before mutating, mirroring `Manifest::load`.
    #[test]
    fn upgrade_to_latest_rejects_unsupported_versions_before_mutating() {
      let mut bogus = Manifest::new(Schema::default_text_body());
      bogus.version = 0;
      let err = bogus
        .upgrade_to_latest(Path::new("/tmp"))
        .expect_err("v0 must be rejected before mutating");
      assert!(
        format!("{err:#}").contains("unsupported version 0"),
        "expected error mentioning unsupported version 0; got: {err:#}"
      );
      // The manifest is left untouched (still v0).
      assert_eq!(bogus.version, 0);

      let mut too_new = Manifest::new(Schema::default_text_body());
      too_new.version = 99;
      let err = too_new
        .upgrade_to_latest(Path::new("/tmp"))
        .expect_err("v99 must be rejected before mutating");
      assert!(
        format!("{err:#}").contains("unsupported version 99"),
        "expected error mentioning unsupported version 99; got: {err:#}"
      );
      assert_eq!(too_new.version, 99);
    }

    /// Stage 9a v5 [P2] (Codex review): a v1 manifest with absolute
    /// segment paths must upgrade cleanly even when the user opens
    /// the index with a **relative** root. Lexical
    /// `Path::strip_prefix("idx")` against `/cwd/idx/seg_X.terms`
    /// fails because the absolute path's first component is
    /// `RootDir`, not `Normal("idx")`. `relativize_under` must fall
    /// back to comparing against the *absolute* form of the relative
    /// root.
    ///
    /// Stage 9a v6 [P3] update: previously this test mutated the
    /// process CWD to drive the relative-root scenario, which races
    /// against the parallel cargo test harness. Now uses the
    /// `relativize_under_with_cwd` testable variant that takes an
    /// explicit cwd parameter, so the test is hermetic.
    #[test]
    fn relativize_under_handles_absolute_key_with_relative_root() {
      use crate::index::manifest::SegmentPaths;

      let outer = tempdir().unwrap();
      let nested = outer.path().join("idx");
      std::fs::create_dir_all(&nested).unwrap();

      // Create the on-disk segment files so `relativize_under` can
      // canonicalize them when the absolute key uses a different
      // symlink path than the absolute form of the relative root
      // (e.g. macOS `/var/folders/...` vs `/private/var/folders/...`).
      for stem in [
        "seg_X.terms",
        "seg_X.post",
        "seg_X.docs",
        "seg_X.fast",
        "seg_X.meta",
      ] {
        std::fs::write(nested.join(stem), b"").unwrap();
      }
      let absolute_seg = nested.join("seg_X.terms").to_string_lossy().into_owned();
      let absolute_post = nested.join("seg_X.post").to_string_lossy().into_owned();
      let absolute_docs = nested.join("seg_X.docs").to_string_lossy().into_owned();
      let absolute_fast = nested.join("seg_X.fast").to_string_lossy().into_owned();
      let absolute_meta = nested.join("seg_X.meta").to_string_lossy().into_owned();
      let mut paths = SegmentPaths {
        terms: absolute_seg.clone(),
        postings: absolute_post,
        docstore: absolute_docs,
        fast: absolute_fast,
        meta: absolute_meta,
        #[cfg(feature = "vectors")]
        vector_dir: None,
      };

      // The relative root used at "open" time. With an explicit cwd
      // of `outer.path()`, `idx` resolves to `outer.path()/idx`,
      // matching where the segment files live.
      let relative_root = Path::new("idx");

      // Pre-fix: this errored with "not under index root". Post-fix:
      // succeeds because we strip against the absolute form of root
      // (and its symlink-canonicalized form).
      paths
        .relativize_under_with_cwd(relative_root, Some(outer.path()))
        .unwrap();
      assert_eq!(paths.terms, "seg_X.terms");
      assert_eq!(paths.postings, "seg_X.post");
      paths.validate_v2_relative().unwrap();
    }
  }

  /// Stage 9b regression suite — `ContentHash` (SHA-256) per segment
  /// artifact. Codex's plan calls out four invariants that this
  /// suite guards:
  ///
  /// 1. Fresh commits record valid lowercase-hex SHA-256 hashes for
  ///    every segment artifact (and vector files under the feature).
  /// 2. `Strict` policy verifies SHA-256 and rejects corruption with
  ///    a SHA-256-mismatch error.
  /// 3. A `content_hashes` map missing any expected entry is treated
  ///    as corruption (Stage 9c removed the CRC32 fall-through).
  /// 4. `SegmentCacheKey::from_meta` produces stable identity:
  ///    same `(id, content_hashes)` ⇒ same key regardless of paths;
  ///    any one hash change ⇒ different key.
  mod stage9b_content_hashes {
    use super::*;
    use crate::index::manifest::SegmentMeta;
    use crate::index::segment::{SegmentCacheKey, SegmentFingerprint};
    use std::collections::BTreeMap;

    /// Stage 9b: every fresh segment artifact gets a SHA-256 hash
    /// recorded in `content_hashes`, formatted as 64-char lowercase
    /// hex.
    #[test]
    fn fresh_commits_record_sha256_for_every_segment_artifact() {
      let dir = tempdir().unwrap();
      let schema = Schema::default_text_body();
      let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("alpha")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();

      let seg = &idx.manifest().segments[0];
      // Required artifact names for a non-vector segment.
      for name in ["meta", "terms", "postings", "docstore", "fast"] {
        let hash = seg
          .content_hashes
          .get(name)
          .unwrap_or_else(|| panic!("content_hashes missing {name}: {seg:?}"));
        assert_eq!(
          hash.len(),
          64,
          "expected 64-char SHA-256 hex for {name}; got {hash:?}"
        );
        assert!(
          hash
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
          "expected lowercase hex for {name}; got {hash:?}"
        );
      }
    }

    /// Stage 9b: under `Strict` policy, a corrupted postings file
    /// must be rejected at reader open with a SHA-256 mismatch
    /// error (rather than falling through silently).
    #[test]
    fn strict_policy_rejects_sha256_mismatch_on_corrupted_postings() {
      let dir = tempdir().unwrap();
      let schema = Schema::default_text_body();
      let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("alpha")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
      let postings_path = dir.path().join(&idx.manifest().segments[0].paths.postings);
      drop(idx);

      // Corrupt by flipping a byte mid-file.
      let mut bytes = std::fs::read(&postings_path).unwrap();
      assert!(bytes.len() > 4, "postings file must be non-trivial");
      bytes[2] ^= 0xff;
      std::fs::write(&postings_path, &bytes).unwrap();

      let mut reopen_opts = opts(dir.path());
      reopen_opts.create_if_missing = false;
      // Strict is the default; spell it out for clarity.
      reopen_opts.checksum_policy = crate::api::types::ChecksumPolicy::Strict;
      let reopened = Index::open(reopen_opts).unwrap();
      let err = match reopened.reader() {
        Ok(_) => panic!("Strict must reject corrupted postings"),
        Err(e) => e,
      };
      let msg = format!("{err:#}");
      assert!(
        msg.contains("SHA-256") && msg.contains("postings"),
        "expected SHA-256-mismatch error mentioning postings; got: {msg}"
      );
    }

    /// Stage 9b: a non-empty `content_hashes` map missing any
    /// expected artifact is treated as corruption (no fall-through
    /// to CRC32). Codex's #2 invariant.
    #[test]
    fn partial_content_hashes_is_treated_as_corruption_under_strict() {
      let dir = tempdir().unwrap();
      let schema = Schema::default_text_body();
      let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("alpha")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
      drop(idx);

      // Surgically remove the `postings` entry from `content_hashes`
      // in the live manifest. Other entries remain → `content_hashes`
      // is non-empty but incomplete.
      let manifest_path = dir.path().join("MANIFEST.json");
      let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
      let segments = value["segments"].as_array_mut().unwrap();
      for seg in segments.iter_mut() {
        let ch = seg["content_hashes"].as_object_mut().unwrap();
        ch.remove("postings");
      }
      std::fs::write(&manifest_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

      let mut reopen_opts = opts(dir.path());
      reopen_opts.create_if_missing = false;
      let reopened = Index::open(reopen_opts).unwrap();
      let err = match reopened.reader() {
        Ok(_) => panic!("partial content_hashes must be rejected under Strict"),
        Err(e) => e,
      };
      let msg = format!("{err:#}");
      assert!(
        msg.contains("missing artifact") && msg.contains("postings"),
        "expected 'missing artifact \"postings\"' error; got: {msg}"
      );
    }

    /// Stage 9c: a manifest written before Stage 9b carries a CRC32
    /// `checksums` map and an empty `content_hashes`. The current
    /// struct silently drops the legacy `checksums` field on
    /// deserialization, so the manifest reaches `verify_checksums`
    /// with an empty map; without the (now-removed) CRC32 fallback,
    /// `Strict` must reject the open with a clear "rebuild" error
    /// rather than panicking or silently skipping verification.
    #[test]
    fn legacy_manifest_without_content_hashes_is_rejected_under_strict() {
      let dir = tempdir().unwrap();
      let schema = Schema::default_text_body();
      let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("alpha")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
      drop(idx);

      // Strip `content_hashes` entirely from the on-disk manifest to
      // mimic a pre-Stage-9b artifact.
      let manifest_path = dir.path().join("MANIFEST.json");
      let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
      let segments = value["segments"].as_array_mut().unwrap();
      for seg in segments.iter_mut() {
        seg.as_object_mut().unwrap().remove("content_hashes");
      }
      std::fs::write(&manifest_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

      let mut reopen_opts = opts(dir.path());
      reopen_opts.create_if_missing = false;
      let reopened = Index::open(reopen_opts).unwrap();
      assert!(
        reopened.manifest().segments[0].content_hashes.is_empty(),
        "test setup: content_hashes must be empty to exercise the rejection path"
      );
      let err = match reopened.reader() {
        Ok(_) => panic!("Stage 9c removed the CRC32 fallback — empty content_hashes must reject"),
        Err(e) => e,
      };
      let msg = format!("{err:#}");
      assert!(
        msg.contains("no SHA-256 content_hashes") && msg.contains("rebuild the index"),
        "expected pre-Stage-9b rejection with rebuild guidance; got: {msg}"
      );
    }

    /// Stage 9b: cache key identity properties (Codex plan #4 / #5).
    /// Tests are at the helper level — `SegmentCache` is per-Index,
    /// so cross-index dedupe assertions need direct access to the
    /// key constructor rather than the cache itself.
    #[test]
    fn segment_cache_key_identity_properties() {
      use crate::index::manifest::SegmentPaths;

      let make_meta =
        |id: &str, paths: SegmentPaths, hashes: BTreeMap<String, String>| -> SegmentMeta {
          SegmentMeta {
            id: id.to_string(),
            generation: 1,
            paths,
            doc_count: 0,
            max_doc_id: 0,
            blockmax: true,
            deleted_docs: Vec::new(),
            avg_field_lengths: Default::default(),
            content_hashes: hashes,
            write_binding_b64: None,
          }
        };
      let bare_paths = SegmentPaths {
        terms: "seg_a.terms".into(),
        postings: "seg_a.post".into(),
        docstore: "seg_a.docs".into(),
        fast: "seg_a.fast".into(),
        meta: "seg_a.meta".into(),
        #[cfg(feature = "vectors")]
        vector_dir: None,
      };
      let prefixed_paths = SegmentPaths {
        terms: "/abs/seg_a.terms".into(),
        postings: "/abs/seg_a.post".into(),
        docstore: "/abs/seg_a.docs".into(),
        fast: "/abs/seg_a.fast".into(),
        meta: "/abs/seg_a.meta".into(),
        #[cfg(feature = "vectors")]
        vector_dir: None,
      };
      let mut hashes = BTreeMap::new();
      hashes.insert("meta".into(), "a".repeat(64));
      hashes.insert("terms".into(), "b".repeat(64));
      hashes.insert("postings".into(), "c".repeat(64));
      hashes.insert("docstore".into(), "d".repeat(64));
      hashes.insert("fast".into(), "e".repeat(64));

      // Same id + content_hashes ⇒ same key, regardless of paths.
      let key_a = SegmentCacheKey::from_meta(&make_meta("seg_a", bare_paths, hashes.clone()));
      let key_a_relocated =
        SegmentCacheKey::from_meta(&make_meta("seg_a", prefixed_paths, hashes.clone()));
      assert_eq!(
        key_a, key_a_relocated,
        "cross-location dedupe: same content_hashes ⇒ same cache key"
      );
      // Stage 9c: SegmentFingerprint collapsed to a single tuple-struct
      // variant. The 32-byte digest carries the SHA-256 of the canonical
      // content_hashes encoding; here we assert it's not the all-zero
      // sentinel that an empty map would hash through.
      let SegmentFingerprint(digest_a) = &key_a.fingerprint;
      assert_ne!(
        digest_a, &[0u8; 32],
        "populated content_hashes must produce a non-sentinel digest"
      );

      // Changing any one hash ⇒ different key.
      let mut hashes_changed = hashes.clone();
      hashes_changed.insert("postings".into(), "0".repeat(64));
      let bare_paths_2 = SegmentPaths {
        terms: "seg_a.terms".into(),
        postings: "seg_a.post".into(),
        docstore: "seg_a.docs".into(),
        fast: "seg_a.fast".into(),
        meta: "seg_a.meta".into(),
        #[cfg(feature = "vectors")]
        vector_dir: None,
      };
      let key_a_changed =
        SegmentCacheKey::from_meta(&make_meta("seg_a", bare_paths_2, hashes_changed));
      assert_ne!(
        key_a, key_a_changed,
        "any hash change must change the cache key"
      );
    }
  }

  /// Vectors-feature regression: SHA-256 must be recorded for every
  /// vector artifact (`vector_{field}_bin` + `vector_{field}_hnsw`),
  /// and `Strict` verification must reject corruption in either.
  #[cfg(feature = "vectors")]
  mod stage9b_content_hashes_vectors {
    use super::*;

    #[test]
    fn vector_artifacts_recorded_and_verified_via_sha256() {
      use crate::api::types::VectorMetric as ApiVectorMetric;

      let dir = tempdir().unwrap();
      let mut schema = Schema::default_text_body();
      schema
        .vector_fields
        .push(crate::index::manifest::VectorField {
          name: "v".into(),
          dim: 3,
          metric: ApiVectorMetric::Cosine.into(),
          hnsw: None,
        });
      let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("alpha")),
            ("v".into(), serde_json::json!([0.1, 0.2, 0.3])),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();

      let bin_path;
      {
        let manifest = idx.manifest();
        let seg = &manifest.segments[0];
        let bin = seg
          .content_hashes
          .get("vector_v_bin")
          .expect("vector_v_bin hash must be recorded");
        let hnsw = seg
          .content_hashes
          .get("vector_v_hnsw")
          .expect("vector_v_hnsw hash must be recorded");
        for h in [bin, hnsw] {
          assert_eq!(h.len(), 64);
          assert!(h
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
        }
        let vec_dir = dir
          .path()
          .join(seg.paths.vector_dir.as_ref().expect("vector dir set"));
        bin_path = vec_dir.join("v.bin");
      }
      drop(idx);

      // Corrupt the vector bin and reopen under Strict — must fail.
      let mut bin_bytes = std::fs::read(&bin_path).unwrap();
      assert!(!bin_bytes.is_empty());
      bin_bytes[0] ^= 0xff;
      std::fs::write(&bin_path, bin_bytes).unwrap();

      let mut reopen_opts = opts(dir.path());
      reopen_opts.create_if_missing = false;
      let reopened = Index::open(reopen_opts).unwrap();
      let err = match reopened.reader() {
        Ok(_) => panic!("corrupted vector bin must be rejected under Strict"),
        Err(e) => e,
      };
      let msg = format!("{err:#}");
      assert!(
        msg.contains("SHA-256") && msg.contains("vector_v_bin"),
        "expected SHA-256 mismatch on vector_v_bin, got: {msg}"
      );
    }
  }

  /// Stage 10a regression suite — read-only enforcement.
  /// `IndexOptions.read_only = true` must refuse every mutator entry
  /// point (`writer`, `compact`, `merge_segments`) with a clear error
  /// message, rather than letting the mutation proceed and fail later
  /// at the storage/blob-store layer where the message would be
  /// backend-specific.
  mod stage10a_read_only {
    use super::*;

    fn build_committed_index(dir: &Path) -> Index {
      let schema = Schema::default_text_body();
      let idx = Index::create(dir, schema, opts(dir)).unwrap();
      // At least one committed segment so `compact` and
      // `merge_segments` have something to act on.
      for body in ["alpha", "bravo"] {
        let mut writer = idx.writer().unwrap();
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), serde_json::json!(body)),
              ("body".into(), serde_json::json!(body)),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
        writer.commit().unwrap();
      }
      idx
    }

    fn read_only_opts(dir: &Path) -> crate::api::types::IndexOptions {
      let mut o = opts(dir);
      o.create_if_missing = false;
      o.read_only = true;
      o
    }

    /// Stage 10a: `Index::writer` must error when `read_only = true`.
    #[test]
    fn read_only_index_refuses_writer() {
      let dir = tempdir().unwrap();
      let idx = build_committed_index(dir.path());
      drop(idx);

      let reopened = Index::open(read_only_opts(dir.path())).unwrap();
      assert!(reopened.manifest().segments.len() >= 2);
      let err = match reopened.writer() {
        Ok(_) => panic!("read_only index must refuse writer()"),
        Err(e) => e,
      };
      let msg = format!("{err:#}");
      assert!(
        msg.contains("read-only") && msg.contains("writer"),
        "expected error mentioning read-only writer; got: {msg}"
      );
    }

    /// Stage 10a: `Index::compact` must error when `read_only = true`.
    #[test]
    fn read_only_index_refuses_compact() {
      let dir = tempdir().unwrap();
      let idx = build_committed_index(dir.path());
      drop(idx);

      let reopened = Index::open(read_only_opts(dir.path())).unwrap();
      let err = reopened
        .compact()
        .expect_err("read_only index must refuse compact()");
      let msg = format!("{err:#}");
      assert!(
        msg.contains("read-only") && msg.contains("compact"),
        "expected error mentioning read-only compact; got: {msg}"
      );
      // Manifest must be unchanged (still 2 segments, not merged).
      assert!(reopened.manifest().segments.len() >= 2);
    }

    /// Stage 10a: `Index::merge_segments` must error when
    /// `read_only = true`. Empty input still short-circuits to Ok
    /// (the early-return is before the read-only check), which
    /// matches the contract.
    #[test]
    fn read_only_index_refuses_merge_segments() {
      let dir = tempdir().unwrap();
      let idx = build_committed_index(dir.path());
      let segment_ids: Vec<String> = idx
        .manifest()
        .segments
        .iter()
        .map(|s| s.id.clone())
        .collect();
      drop(idx);

      let reopened = Index::open(read_only_opts(dir.path())).unwrap();
      // Empty list is a no-op even on read-only indexes — matches
      // the existing contract that `merge_segments(&[], ...)` is Ok.
      reopened.merge_segments(&[], None).unwrap();

      // Non-empty list must error.
      let err = reopened
        .merge_segments(&segment_ids, None)
        .expect_err("read_only index must refuse merge_segments()");
      let msg = format!("{err:#}");
      assert!(
        msg.contains("read-only") && msg.contains("merge_segments"),
        "expected error mentioning read-only merge_segments; got: {msg}"
      );
      // Manifest unchanged — segments not merged.
      assert_eq!(
        reopened.manifest().segments.len(),
        segment_ids.len(),
        "merge_segments error must not mutate the manifest"
      );
    }

    /// Stage 10a: `read_only = false` (the default) keeps the historical
    /// mutator behavior — no regression.
    #[test]
    fn read_only_false_does_not_block_mutators() {
      let dir = tempdir().unwrap();
      let idx = build_committed_index(dir.path());
      // Read_only defaults to false; a fresh writer still works.
      let _ = idx.writer().unwrap();
      // compact succeeds (≥2 segments → 1).
      idx.compact().unwrap();
      assert_eq!(idx.manifest().segments.len(), 1);
    }

    /// Stage 10a v2 [P2] (Codex review): `Index::create*` must reject
    /// `read_only = true` BEFORE issuing any storage write. Otherwise
    /// a misconfigured deployment with a read-only S3/R2 token would
    /// surface a backend-specific 403 instead of a clear "cannot
    /// create read-only index" error.
    #[test]
    fn read_only_index_refuses_create() {
      let dir = tempdir().unwrap();
      let mut o = opts(dir.path());
      o.read_only = true;
      let schema = Schema::default_text_body();
      let err = match Index::create(dir.path(), schema, o) {
        Ok(_) => panic!("Index::create with read_only = true must error"),
        Err(e) => e,
      };
      let msg = format!("{err:#}");
      assert!(
        msg.contains("read_only") && msg.contains("create"),
        "expected error mentioning create + read_only; got: {msg}"
      );
      // No MANIFEST.json was written. `ensure_root` and
      // `Manifest::store` were both skipped.
      assert!(
        !dir.path().join("MANIFEST.json").exists(),
        "Index::create with read_only must NOT write MANIFEST.json"
      );
    }

    /// Stage 10a v2 [P2] (Codex review): `Index::open` with both
    /// `read_only = true` AND `create_if_missing = true` must error
    /// rather than silently auto-creating a manifest with a write.
    #[test]
    fn read_only_index_refuses_auto_create_on_open() {
      let dir = tempdir().unwrap();
      let mut o = opts(dir.path());
      o.read_only = true;
      o.create_if_missing = true;
      let err = match Index::open(o) {
        Ok(_) => panic!("Index::open with read_only + create_if_missing must error"),
        Err(e) => e,
      };
      let msg = format!("{err:#}");
      assert!(
        msg.contains("auto-create") && msg.contains("read_only"),
        "expected error mentioning auto-create vs read_only; got: {msg}"
      );
      assert!(
        !dir.path().join("MANIFEST.json").exists(),
        "open(read_only, create_if_missing) must NOT write MANIFEST.json"
      );
    }

    /// Stage 10a v2 [P2] (Codex review): `Index::open` with
    /// `read_only = true` against a non-existent index errors
    /// cleanly (without create_if_missing, regardless of read_only).
    #[test]
    fn read_only_index_open_on_missing_errors_without_writing() {
      let dir = tempdir().unwrap();
      let mut o = opts(dir.path());
      o.read_only = true;
      o.create_if_missing = false;
      let err = match Index::open(o) {
        Ok(_) => panic!("open must error when manifest absent"),
        Err(e) => e,
      };
      let msg = format!("{err:#}");
      assert!(
        msg.contains("does not exist"),
        "expected 'does not exist' error; got: {msg}"
      );
      assert!(
        !dir.path().join("MANIFEST.json").exists(),
        "missing-index open must NOT write MANIFEST.json"
      );
    }

    /// Stage 10a v3 [P1] (Codex review): when `MANIFEST.json.pending`
    /// exists and `read_only = true`, the open must FAIL CLOSED with
    /// a clear "pending recovery requires mutable open" error. The
    /// pending file may carry a durably-committed batch (BUG-018:
    /// WAL crossed the commit fence but the live manifest publish
    /// was interrupted); silently loading the live manifest would
    /// hide those committed docs.
    ///
    /// This test was previously phrased as "read_only must skip
    /// recovery writes and leave the pending file untouched", which
    /// is correct on the no-write side but missed the load-bearing
    /// safety property: read_only must not serve stale state. The
    /// inversion here is deliberate.
    #[test]
    fn read_only_index_open_fails_closed_when_pending_manifest_exists() {
      let dir = tempdir().unwrap();
      let idx = build_committed_index(dir.path());
      drop(idx);

      // Plant a `MANIFEST.json.pending` simulating an interrupted
      // commit publish. We don't need it to be a valid manifest —
      // we want to verify open errors before any parse.
      let pending_path = dir.path().join("MANIFEST.json.pending");
      std::fs::write(&pending_path, b"\"placeholder-pending-bytes\"").unwrap();
      let pending_before = std::fs::read(&pending_path).unwrap();
      let live_before = std::fs::read(dir.path().join("MANIFEST.json")).unwrap();

      let err = match Index::open(read_only_opts(dir.path())) {
        Ok(_) => panic!(
          "read_only open with a pending manifest must fail closed; \
           silently loading the live manifest would hide durably committed docs"
        ),
        Err(e) => e,
      };
      let msg = format!("{err:#}");
      assert!(
        msg.contains("MANIFEST.json.pending") && msg.contains("read_only"),
        "expected error mentioning the pending file and read_only mode; got: {msg}"
      );

      // Both files are untouched (no writes issued by the failed
      // open).
      assert_eq!(
        std::fs::read(&pending_path).unwrap(),
        pending_before,
        "failed read_only open must NOT touch MANIFEST.json.pending"
      );
      assert_eq!(
        std::fs::read(dir.path().join("MANIFEST.json")).unwrap(),
        live_before,
        "failed read_only open must NOT touch the live manifest"
      );
    }

    /// Stage 10a v3 [P1] companion: after a normal mutable open
    /// reconciles the pending file (promoting or discarding it), a
    /// read-only reopen succeeds with the recovered state.
    #[test]
    fn read_only_open_succeeds_once_pending_is_reconciled_via_mutable_open() {
      let dir = tempdir().unwrap();
      let idx = build_committed_index(dir.path());
      drop(idx);

      // Plant a pending file with bytes that won't survive
      // validation — `reconcile_pending_manifest` will read the
      // WAL, see that there IS a Commit fence (because the prior
      // commits were durable), and try to promote. The promote
      // step parses + re-validates via `serialize_for_write`, which
      // will reject the placeholder payload — but in either case
      // the pending file is removed.
      let pending_path = dir.path().join("MANIFEST.json.pending");
      std::fs::write(&pending_path, b"\"placeholder-pending-bytes\"").unwrap();

      // Mutable open performs reconciliation. The placeholder
      // payload may surface an error during promote; what matters
      // here is that the pending file is removed (best-effort
      // cleanup runs in either branch of `reconcile_pending_manifest`).
      // We tolerate either outcome (Ok or Err) and verify the
      // pending file is gone afterwards.
      let mut mutable_opts = opts(dir.path());
      mutable_opts.create_if_missing = false;
      let _ = Index::open(mutable_opts);
      assert!(
        !pending_path.exists(),
        "mutable open must clean up MANIFEST.json.pending"
      );

      // Now read-only open succeeds.
      let reopened = Index::open(read_only_opts(dir.path())).unwrap();
      assert!(reopened.manifest().segments.len() >= 2);
    }

    /// Stage 10a: `read_only` survives `IndexOptions` JSON
    /// serialization round-trip with `default = false` /
    /// `skip_serializing_if = false`. Confirms the serde annotations
    /// don't drop the flag.
    #[test]
    fn read_only_serialize_round_trip() {
      let dir = tempdir().unwrap();
      let mut o = opts(dir.path());
      o.read_only = true;
      let json = serde_json::to_string(&o).unwrap();
      let parsed: crate::api::types::IndexOptions = serde_json::from_str(&json).unwrap();
      assert!(parsed.read_only, "read_only=true must survive round-trip");

      let mut default_o = opts(dir.path());
      default_o.read_only = false;
      let default_json = serde_json::to_string(&default_o).unwrap();
      assert!(
        !default_json.contains("read_only"),
        "read_only=false must be skipped during serialization to keep \
         serialized IndexOptions byte-stable; got: {default_json}"
      );
    }
  }
}
