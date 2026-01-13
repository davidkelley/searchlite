use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use chrono::Utc;

use crate::api::types::Document;
use crate::index::manifest::{Manifest, Schema};
use crate::index::segment::{SegmentFileMeta, SegmentWriter};
use crate::index::wal::{Wal, WalEntry};
use crate::index::InnerIndex;
use crate::DocId;

#[derive(Debug, Clone)]
struct DocAddress {
  segment_id: String,
  doc_id: DocId,
}

#[derive(Debug, Clone)]
enum PendingOp {
  Add { doc_id: String, doc: Document },
  Delete { doc_id: String },
}

pub struct IndexWriter {
  inner: Arc<InnerIndex>,
  wal: Wal,
  pending_ops: Vec<PendingOp>,
  schema: Schema,
  live_docs: HashMap<String, DocAddress>,
  live_generation: u32,
}

impl IndexWriter {
  pub(crate) fn new(inner: Arc<InnerIndex>) -> Result<Self> {
    // Hold the writer lock during initialization to avoid racing with a commit.
    let _guard = inner.writer_lock.lock();
    let wal_path = crate::index::directory::wal_path(&inner.path);
    let pending_entries = Wal::last_pending_ops(inner.storage.as_ref(), &wal_path)?;
    let wal = inner.wal()?;
    let manifest = inner.manifest.read().clone();
    let schema = manifest.schema.clone();
    let live_generation = manifest
      .segments
      .iter()
      .map(|s| s.generation)
      .max()
      .unwrap_or(0);
    let live_docs = load_live_docs(inner.as_ref(), &manifest)?;
    let mut pending_ops = Vec::new();
    for entry in pending_entries {
      match entry {
        WalEntry::AddDoc(doc) => {
          let doc_id = doc_id_from_document(&schema, &doc)?;
          pending_ops.push(PendingOp::Add { doc_id, doc });
        }
        WalEntry::DeleteDocId(doc_id) => pending_ops.push(PendingOp::Delete { doc_id }),
        WalEntry::Commit => {}
      }
    }
    drop(_guard);
    Ok(Self {
      inner,
      wal,
      pending_ops,
      schema,
      live_docs,
      live_generation,
    })
  }

  pub fn add_document(&mut self, doc: &Document) -> Result<u32> {
    let _guard = self.inner.writer_lock.lock();
    self.schema.validate_document(doc)?;
    let doc_id = doc_id_from_document(&self.schema, doc)?;
    self.wal.append_add_doc(doc)?;
    self.pending_ops.push(PendingOp::Add {
      doc_id: doc_id.clone(),
      doc: doc.clone(),
    });
    let add_count = self
      .pending_ops
      .iter()
      .filter(|op| matches!(op, PendingOp::Add { .. }))
      .count();
    Ok(add_count as u32 - 1)
  }

  pub fn delete_document(&mut self, doc_id: &str) -> Result<()> {
    self.delete_documents(&[doc_id.to_string()])
  }

  pub fn delete_documents(&mut self, doc_ids: &[String]) -> Result<()> {
    let _guard = self.inner.writer_lock.lock();
    for id in doc_ids {
      self.wal.append_delete_doc_id(id)?;
      self
        .pending_ops
        .push(PendingOp::Delete { doc_id: id.clone() });
    }
    Ok(())
  }

  pub fn commit(&mut self) -> Result<()> {
    let inner = self.inner.clone();
    let _guard = inner.writer_lock.lock();
    if self.pending_ops.is_empty() {
      return Ok(());
    }
    self.wal.sync()?;
    let manifest_snapshot = inner.manifest.read().clone();
    self.schema = manifest_snapshot.schema.clone();
    let manifest_generation = manifest_snapshot
      .segments
      .iter()
      .map(|s| s.generation)
      .max()
      .unwrap_or(0);
    let mut live_docs = if manifest_generation == self.live_generation {
      self.live_docs.clone()
    } else {
      load_live_docs(inner.as_ref(), &manifest_snapshot)?
    };
    let mut pending_new: BTreeMap<String, Document> = BTreeMap::new();
    let mut tombstones: HashMap<String, Vec<DocId>> = HashMap::new();
    for op in self.pending_ops.iter() {
      match op {
        PendingOp::Add { doc_id, doc } => {
          if let Some(addr) = live_docs.remove(doc_id) {
            tombstones
              .entry(addr.segment_id)
              .or_default()
              .push(addr.doc_id);
          }
          pending_new.insert(doc_id.clone(), doc.clone());
        }
        PendingOp::Delete { doc_id } => {
          pending_new.remove(doc_id);
          if let Some(addr) = live_docs.remove(doc_id) {
            tombstones
              .entry(addr.segment_id)
              .or_default()
              .push(addr.doc_id);
          }
        }
      }
    }
    let mut new_manifest = manifest_snapshot.clone();
    for seg in new_manifest.segments.iter_mut() {
      if let Some(deleted) = tombstones.remove(&seg.id) {
        let mut set: HashSet<DocId> = seg.deleted_docs.iter().copied().collect();
        set.extend(deleted.into_iter());
        let mut merged: Vec<DocId> = set.into_iter().collect();
        merged.sort_unstable();
        seg.deleted_docs = merged;
      }
    }
    if !pending_new.is_empty() {
      let generation = new_manifest
        .segments
        .iter()
        .map(|s| s.generation)
        .max()
        .unwrap_or(0)
        + 1;
      let writer = SegmentWriter::new(
        &self.inner.path,
        &self.schema,
        self.inner.options.enable_positions,
        cfg!(feature = "zstd"),
        self.inner.storage.clone(),
      );
      let docs: Vec<Document> = pending_new.values().cloned().collect();
      let segment = writer.write_segment(&docs, generation)?;
      new_manifest.segments.push(segment.clone());
      for (offset, doc_id) in pending_new.keys().enumerate() {
        live_docs.insert(
          doc_id.clone(),
          DocAddress {
            segment_id: segment.id.clone(),
            doc_id: offset as DocId,
          },
        );
      }
    }
    let new_generation = new_manifest
      .segments
      .iter()
      .map(|s| s.generation)
      .max()
      .unwrap_or(0);
    new_manifest.committed_at = Utc::now().to_rfc3339();
    let manifest_path = self.inner.manifest_path();
    let wal_len = self.wal.len()?;
    if let Err(e) = (|| -> Result<()> {
      new_manifest.store(self.inner.storage.as_ref(), &manifest_path)?;
      self.wal.append_commit()?;
      self.wal.sync()?;
      Ok(())
    })() {
      // Roll back manifest to the previous snapshot and restore WAL to its
      // pre-commit length so pending ops can be retried safely.
      if let Err(truncate_err) = self.wal.truncate_to(wal_len) {
        log::error!(
          "WAL rollback failed while handling commit error: \
           unable to truncate WAL back to length {}: {}",
          wal_len,
          truncate_err
        );
      }
      if let Err(manifest_err) =
        manifest_snapshot.store(self.inner.storage.as_ref(), &manifest_path)
      {
        log::error!(
          "Manifest rollback failed while handling commit error: {}. \
           The on-disk manifest and WAL may be inconsistent.",
          manifest_err
        );
      }
      return Err(e);
    }
    {
      let mut manifest_guard = self.inner.manifest.write();
      *manifest_guard = new_manifest;
    }
    self.wal.truncate()?;
    self.pending_ops.clear();
    self.live_docs = live_docs;
    self.live_generation = new_generation;
    Ok(())
  }

  pub fn rollback(&mut self) -> Result<()> {
    let _guard = self.inner.writer_lock.lock();
    self.pending_ops.clear();
    self.wal.truncate()?;
    Ok(())
  }
}

impl Drop for IndexWriter {
  fn drop(&mut self) {
    if !self.pending_ops.is_empty() {
      if let Err(e) = self.wal.sync() {
        eprintln!(
          "IndexWriter: failed to sync WAL on drop ({} pending ops): {e}",
          self.pending_ops.len()
        );
      }
    }
  }
}

fn doc_id_from_document(schema: &Schema, doc: &Document) -> Result<String> {
  let doc_id = doc
    .fields
    .get(schema.doc_id_field())
    .and_then(|v| v.as_str())
    .ok_or_else(|| {
      anyhow!(
        "missing or empty required document id field `{}`",
        schema.doc_id_field()
      )
    })?;
  if doc_id.trim().is_empty() {
    bail!(
      "missing or empty required document id field `{}`",
      schema.doc_id_field()
    );
  }
  Ok(doc_id.to_string())
}

fn load_live_docs(inner: &InnerIndex, manifest: &Manifest) -> Result<HashMap<String, DocAddress>> {
  let mut map = HashMap::new();
  for seg in manifest.segments.iter() {
    let meta_bytes = inner
      .storage
      .read_to_end(Path::new(&seg.paths.meta))
      .map_err(|e| anyhow!("reading segment meta for {}: {}", seg.id, e))?;
    let seg_meta: SegmentFileMeta = serde_json::from_slice(&meta_bytes)?;
    if seg_meta.doc_ids.len() != seg_meta.doc_offsets.len() {
      bail!(
        "segment {} is missing document ids; reindex or compact to repair",
        seg.id
      );
    }
    let deleted: HashSet<u32> = seg.deleted_docs.iter().copied().collect();
    for (idx, doc_id) in seg_meta.doc_ids.iter().enumerate() {
      if deleted.contains(&(idx as u32)) {
        continue;
      }
      map.insert(
        doc_id.clone(),
        DocAddress {
          segment_id: seg.id.clone(),
          doc_id: idx as DocId,
        },
      );
    }
  }
  Ok(map)
}

#[cfg(test)]
mod tests {
  use std::path::PathBuf;
  use std::sync::atomic::{AtomicBool, Ordering};
  use std::sync::Arc;

  use anyhow::anyhow;
  use parking_lot::{Mutex, RwLock};

  use super::PendingOp;
  use crate::api::types::{Document, IndexOptions, Schema, StorageType};
  use crate::index::{directory, manifest::Manifest, wal::Wal, Index, InnerIndex};
  use crate::storage::{InMemoryStorage, Storage};
  use tempfile::tempdir;

  fn opts(path: &std::path::Path) -> IndexOptions {
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

  struct FailingManifestStorage {
    inner: InMemoryStorage,
    fail_manifest: AtomicBool,
  }

  impl FailingManifestStorage {
    fn new(root: PathBuf) -> Self {
      Self {
        inner: InMemoryStorage::new(root),
        fail_manifest: AtomicBool::new(false),
      }
    }

    fn fail_next_manifest_store(&self) {
      self.fail_manifest.store(true, Ordering::SeqCst);
    }

    fn should_fail(&self, path: &std::path::Path) -> bool {
      path.ends_with("MANIFEST.json") && self.fail_manifest.swap(false, Ordering::SeqCst)
    }
  }

  impl Storage for FailingManifestStorage {
    fn root(&self) -> &std::path::Path {
      self.inner.root()
    }

    fn ensure_dir(&self, path: &std::path::Path) -> anyhow::Result<()> {
      self.inner.ensure_dir(path)
    }

    fn exists(&self, path: &std::path::Path) -> bool {
      self.inner.exists(path)
    }

    fn open_read(&self, path: &std::path::Path) -> anyhow::Result<crate::storage::DynFile> {
      self.inner.open_read(path)
    }

    fn open_write(&self, path: &std::path::Path) -> anyhow::Result<crate::storage::DynFile> {
      self.inner.open_write(path)
    }

    fn open_append(&self, path: &std::path::Path) -> anyhow::Result<crate::storage::DynFile> {
      self.inner.open_append(path)
    }

    fn read_to_end(&self, path: &std::path::Path) -> anyhow::Result<Vec<u8>> {
      self.inner.read_to_end(path)
    }

    fn write_all(&self, path: &std::path::Path, data: &[u8]) -> anyhow::Result<()> {
      self.inner.write_all(path, data)
    }

    fn atomic_write(&self, path: &std::path::Path, data: &[u8]) -> anyhow::Result<()> {
      if self.should_fail(path) {
        return Err(anyhow!("manifest write failed"));
      }
      self.inner.atomic_write(path, data)
    }

    fn remove(&self, path: &std::path::Path) -> anyhow::Result<()> {
      self.inner.remove(path)
    }

    fn remove_dir_all(&self, path: &std::path::Path) -> anyhow::Result<()> {
      self.inner.remove_dir_all(path)
    }
  }

  #[test]
  fn wal_retains_pending_when_manifest_store_fails() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let storage = Arc::new(FailingManifestStorage::new(dir.path().to_path_buf()));
    let manifest_path = Manifest::manifest_path(dir.path());
    let manifest = Manifest::new(schema.clone());
    manifest.store(storage.as_ref(), &manifest_path).unwrap();

    let mut opts = opts(dir.path());
    opts.storage = StorageType::InMemory;
    let inner = Arc::new(InnerIndex {
      path: dir.path().to_path_buf(),
      options: opts,
      manifest: RwLock::new(manifest),
      writer_lock: Mutex::new(()),
      storage: storage.clone(),
    });

    storage.fail_next_manifest_store();

    let mut writer = super::IndexWriter::new(inner).unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!("1")),
          ("body".into(), serde_json::json!("commit wal safety")),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    let err = writer.commit();
    assert!(err.is_err());
    assert_eq!(
      writer
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Add { .. }))
        .count(),
      1
    );

    let wal_path = directory::wal_path(dir.path());
    let pending = Wal::last_pending_ops(storage.as_ref(), &wal_path).unwrap();
    assert!(
      !pending.is_empty(),
      "wal should retain pending ops when manifest persistence fails"
    );
  }

  #[test]
  fn replay_pending_from_wal() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("pending doc")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      // Drop without commit so wal retains entry.
    }
    let restored = idx.writer().unwrap();
    assert_eq!(
      restored
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Add { .. }))
        .count(),
      1
    );
  }

  #[test]
  fn replay_pending_delete_from_wal() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("to delete")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    {
      let mut writer = idx.writer().unwrap();
      writer.delete_document("1").unwrap();
      // Drop without commit so wal retains delete entry.
    }
    let mut restored = idx.writer().unwrap();
    assert_eq!(
      restored
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Delete { .. }))
        .count(),
      1
    );
    restored.commit().unwrap();
    let manifest = idx.manifest();
    assert_eq!(manifest.segments.len(), 1);
    assert_eq!(manifest.segments[0].deleted_docs, vec![0]);
  }

  #[test]
  fn replay_add_then_delete_same_id() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("original")),
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
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("updated")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.delete_document("1").unwrap();
      // Drop without commit so wal retains ordered ops.
    }
    let mut restored = idx.writer().unwrap();
    assert_eq!(
      restored
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Add { .. }))
        .count(),
      1
    );
    assert_eq!(
      restored
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Delete { .. }))
        .count(),
      1
    );
    restored.commit().unwrap();
    let manifest = idx.manifest();
    assert_eq!(manifest.segments.len(), 1);
    assert_eq!(manifest.segments[0].deleted_docs, vec![0]);
  }

  #[test]
  fn rollback_clears_pending_and_wal() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!("1")),
          ("body".into(), serde_json::json!("to rollback")),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    assert_eq!(
      writer
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Add { .. }))
        .count(),
      1
    );
    writer.rollback().unwrap();
    assert!(writer.pending_ops.is_empty());
    let wal_path = crate::index::directory::wal_path(&writer.inner.path);
    assert_eq!(std::fs::metadata(wal_path).unwrap().len(), 0);
  }

  #[test]
  fn rollback_discards_pending_delete_ops() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("to keep")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    let mut writer = idx.writer().unwrap();
    writer.delete_document("1").unwrap();
    assert_eq!(
      writer
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Delete { .. }))
        .count(),
      1
    );
    writer.rollback().unwrap();
    assert!(writer.pending_ops.is_empty());
    let wal_path = crate::index::directory::wal_path(&writer.inner.path);
    assert_eq!(std::fs::metadata(wal_path).unwrap().len(), 0);
    let manifest = idx.manifest();
    assert!(manifest.segments[0].deleted_docs.is_empty());
  }

  #[test]
  fn commit_clears_wal_and_pending_entries() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("commit durability")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    let wal_path = directory::wal_path(dir.path());
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let pending = Wal::last_pending_ops(&storage, &wal_path).unwrap();
    assert!(
      pending.is_empty(),
      "pending WAL ops should be cleared on commit"
    );
    let wal_len = std::fs::metadata(&wal_path).unwrap().len();
    assert_eq!(wal_len, 0, "wal should be truncated after commit");
    let manifest = idx.manifest();
    assert_eq!(manifest.segments.len(), 1);
    assert_eq!(manifest.segments[0].doc_count, 1);
  }
}
