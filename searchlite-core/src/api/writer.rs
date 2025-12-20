use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;

use crate::api::types::Document;
use crate::index::manifest::Schema;
use crate::index::segment::SegmentWriter;
use crate::index::wal::Wal;
use crate::index::InnerIndex;

pub struct IndexWriter {
  inner: Arc<InnerIndex>,
  wal: Wal,
  pub(crate) pending: Vec<Document>,
  schema: Schema,
}

impl IndexWriter {
  pub(crate) fn new(inner: Arc<InnerIndex>) -> Result<Self> {
    let _guard = inner.writer_lock.lock();
    drop(_guard);
    let wal_path = crate::index::directory::wal_path(&inner.path);
    let pending = Wal::last_pending_documents(&wal_path)?;
    let wal = inner.wal()?;
    let schema = inner.manifest.read().schema.clone();
    Ok(Self {
      inner,
      wal,
      pending,
      schema,
    })
  }

  pub fn add_document(&mut self, doc: &Document) -> Result<u32> {
    let _guard = self.inner.writer_lock.lock();
    self.wal.append_add_doc(doc)?;
    self.pending.push(doc.clone());
    Ok(self.pending.len() as u32 - 1)
  }

  pub fn commit(&mut self) -> Result<()> {
    let _guard = self.inner.writer_lock.lock();
    if self.pending.is_empty() {
      return Ok(());
    }
    self.wal.append_commit()?;
    let mut manifest = self.inner.manifest.write();
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
    );
    let segment = writer.write_segment(&self.pending, generation)?;
    manifest.segments.push(segment);
    manifest.committed_at = Utc::now().to_rfc3339();
    manifest.store(&self.inner.manifest_path())?;
    self.wal.truncate()?;
    self.pending.clear();
    Ok(())
  }

  pub fn rollback(&mut self) -> Result<()> {
    let _guard = self.inner.writer_lock.lock();
    self.pending.clear();
    self.wal.truncate()?;
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use crate::api::types::{Document, IndexOptions, Schema};
  use crate::index::Index;
  use tempfile::tempdir;

  fn opts(path: &std::path::Path) -> IndexOptions {
    IndexOptions {
      path: path.to_path_buf(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 1.2,
      bm25_b: 0.75,
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
          fields: [("body".into(), serde_json::json!("pending doc"))]
            .into_iter()
            .collect(),
        })
        .unwrap();
      // Drop without commit so wal retains entry.
    }
    let restored = idx.writer().unwrap();
    assert_eq!(restored.pending.len(), 1);
  }

  #[test]
  fn rollback_clears_pending_and_wal() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&Document {
        fields: [("body".into(), serde_json::json!("to rollback"))]
          .into_iter()
          .collect(),
      })
      .unwrap();
    assert_eq!(writer.pending.len(), 1);
    writer.rollback().unwrap();
    assert!(writer.pending.is_empty());
    let wal_path = crate::index::directory::wal_path(&writer.inner.path);
    assert_eq!(std::fs::metadata(wal_path).unwrap().len(), 0);
  }
}
