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
}

impl IndexWriter {
  pub(crate) fn new(inner: Arc<InnerIndex>) -> Result<Self> {
    let _guard = inner.writer_lock.lock();
    drop(_guard);
    let wal_path = crate::index::directory::wal_path(&inner.path);
    let pending_entries = Wal::last_pending_ops(inner.storage.as_ref(), &wal_path)?;
    let wal = inner.wal()?;
    let manifest = inner.manifest.read().clone();
    let schema = manifest.schema.clone();
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
    Ok(Self {
      inner,
      wal,
      pending_ops,
      schema,
      live_docs,
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
    let _guard = self.inner.writer_lock.lock();
    if self.pending_ops.is_empty() {
      return Ok(());
    }
    let mut live_docs = self.live_docs.clone();
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
    self.wal.append_commit()?;
    let mut manifest = self.inner.manifest.write();
    for seg in manifest.segments.iter_mut() {
      if let Some(deleted) = tombstones.remove(&seg.id) {
        let mut set: HashSet<DocId> = seg.deleted_docs.iter().copied().collect();
        set.extend(deleted.into_iter());
        let mut merged: Vec<DocId> = set.into_iter().collect();
        merged.sort_unstable();
        seg.deleted_docs = merged;
      }
    }
    if !pending_new.is_empty() {
      let generation = manifest
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
      manifest.segments.push(segment.clone());
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
    manifest.committed_at = Utc::now().to_rfc3339();
    manifest.store(self.inner.storage.as_ref(), &self.inner.manifest_path())?;
    self.wal.truncate()?;
    self.pending_ops.clear();
    self.live_docs = live_docs;
    Ok(())
  }

  pub fn rollback(&mut self) -> Result<()> {
    let _guard = self.inner.writer_lock.lock();
    self.pending_ops.clear();
    self.wal.truncate()?;
    Ok(())
  }
}

fn doc_id_from_document(schema: &Schema, doc: &Document) -> Result<String> {
  doc
    .fields
    .get(schema.doc_id_field())
    .and_then(|v| v.as_str())
    .map(|s| s.to_string())
    .ok_or_else(|| {
      anyhow!(
        "missing required document id field `{}`",
        schema.doc_id_field()
      )
    })
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
  use super::PendingOp;
  use crate::api::types::{Document, IndexOptions, Schema, StorageType};
  use crate::index::Index;
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
}
