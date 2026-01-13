use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use chrono::Utc;
use parking_lot::{Mutex, RwLock};

use crate::api::types::{Document, IndexOptions, StorageType};
use crate::index::directory::ensure_root;
use crate::index::manifest::{Manifest, Schema};
use crate::index::segment::SegmentWriter;
use crate::index::wal::Wal;
use crate::storage::{FsStorage, InMemoryStorage, Storage};

pub mod codec;
pub mod directory;
pub mod docstore;
pub mod fastfields;
pub mod highlight;
pub mod manifest;
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
}

impl Index {
  pub fn create(path: &Path, schema: Schema, opts: IndexOptions) -> Result<Self> {
    let storage = storage_from_options(&opts);
    Self::create_with_storage(path, schema, opts, storage)
  }

  pub fn create_with_storage(
    path: &Path,
    schema: Schema,
    opts: IndexOptions,
    storage: Arc<dyn Storage>,
  ) -> Result<Self> {
    let mut opts = opts;
    opts.path = path.to_path_buf();
    schema.validate_config()?;
    ensure_root(storage.as_ref(), path)?;
    let manifest = Manifest::new(schema);
    manifest.store(storage.as_ref(), &Manifest::manifest_path(path))?;
    let inner = Arc::new(InnerIndex {
      path: path.to_path_buf(),
      storage,
      options: opts,
      manifest: RwLock::new(manifest),
      writer_lock: Mutex::new(()),
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
    let manifest = if storage.exists(&manifest_path) {
      Manifest::load(storage.as_ref(), &manifest_path)?
    } else if opts.create_if_missing {
      let schema = Schema::default_text_body();
      let m = Manifest::new(schema);
      m.store(storage.as_ref(), &manifest_path)?;
      m
    } else {
      bail!("index does not exist at {:?}", manifest_path);
    };
    let inner = Arc::new(InnerIndex {
      path: opts.path.clone(),
      storage,
      options: opts,
      manifest: RwLock::new(manifest),
      writer_lock: Mutex::new(()),
    });
    Ok(Self { inner })
  }

  pub fn writer(&self) -> Result<crate::api::writer::IndexWriter> {
    crate::api::writer::IndexWriter::new(self.inner.clone())
  }

  pub fn reader(&self) -> Result<crate::api::reader::IndexReader> {
    crate::api::reader::IndexReader::open(self.inner.clone())
  }

  pub fn compact(&self) -> Result<()> {
    let _writer_guard = self.inner.writer_lock.lock();
    let reader = self.reader()?;
    let manifest_snapshot = reader.manifest.clone();
    if manifest_snapshot.segments.len() <= 1 {
      return Ok(());
    }
    ensure_compact_safe(&manifest_snapshot.schema)?;
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

fn cleanup_segments(
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
}
