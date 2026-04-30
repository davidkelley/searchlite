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
use crate::storage::{FsStorage, InMemoryStorage, Storage};
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
    ensure_root(storage.as_ref(), &opts.path)?;
    let manifest_path = Manifest::manifest_path(&opts.path);
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
    crate::api::writer::IndexWriter::new(self.inner.clone(), write_key)
  }

  pub fn reader(&self) -> Result<crate::api::reader::IndexReader> {
    let reader = crate::api::reader::IndexReader::open(self.inner.clone())?;
    self.inner.reader_opens.fetch_add(1, Ordering::Relaxed);
    Ok(reader)
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
      let bytes = self
        .inner
        .storage
        .read_to_end(Path::new(&seg.paths.meta))
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
      let bytes = self
        .inner
        .storage
        .read_to_end(Path::new(&seg.paths.meta))
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
    let readers: Vec<crate::index::segment::SegmentReader> = merge_metas
      .iter()
      .map(|seg| -> Result<_> {
        let core = inner.segment_cache.get_or_load(
          seg,
          inner.options.enable_positions,
          inner.storage.clone(),
        )?;
        crate::index::segment::SegmentReader::from_core(
          core,
          (*seg).clone(),
          inner.storage.clone(),
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
    storage
      .atomic_write(manifest_path, &pending_data)
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
    for path in [
      &seg.paths.terms,
      &seg.paths.postings,
      &seg.paths.docstore,
      &seg.paths.fast,
      &seg.paths.meta,
    ] {
      let _ = storage.remove(Path::new(path));
    }
    #[cfg(feature = "vectors")]
    if let Some(dir) = seg.paths.vector_dir.as_ref() {
      let _ = storage.remove_dir_all(Path::new(dir));
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
    let seg_meta_bytes = std::fs::read(&seg.paths.meta).unwrap();
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
      let bytes =
        std::fs::read(&seg.paths.meta).expect("segment meta readable for tampered manifest");
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
        checksums: HashMap::new(),
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

    // Load segment 0 and clone an Arc out of the cache so we can verify the
    // core stays usable after it's evicted from the cache itself.
    let core_0 = cache
      .get_or_load(&metas[0], idx.inner.options.enable_positions, storage.clone())
      .unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.loads(), 1);

    // Load segment 1 and segment 2. After the third insert the cache is over
    // capacity (cap=2) and segment 0 — the LRU — must have been evicted.
    let _core_1 = cache
      .get_or_load(&metas[1], idx.inner.options.enable_positions, storage.clone())
      .unwrap();
    let _core_2 = cache
      .get_or_load(&metas[2], idx.inner.options.enable_positions, storage.clone())
      .unwrap();
    assert_eq!(cache.len(), 2, "cache must respect its capacity bound");
    assert_eq!(cache.loads(), 3, "each distinct meta must have caused one load");
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
    let _core_0_again = cache
      .get_or_load(&metas[0], idx.inner.options.enable_positions, storage.clone())
      .unwrap();
    assert_eq!(
      cache.loads() - loads_before,
      1,
      "evicted entry must be reloaded on next request"
    );
  }
}
