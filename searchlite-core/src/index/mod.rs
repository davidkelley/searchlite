use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Result};
use chrono::Utc;
use parking_lot::{Mutex, RwLock};

use crate::api::types::{IndexOptions, StorageType};
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
    let reader = self.reader()?;
    let manifest_snapshot = reader.manifest.clone();
    if manifest_snapshot.segments.len() <= 1 {
      return Ok(());
    }
    let mut all_docs = Vec::new();
    for seg in reader.segments.iter() {
      for doc_id in 0..seg.meta.doc_count {
        let doc_json = seg.get_doc(doc_id)?;
        if let Some(map) = doc_json.as_object() {
          let fields = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
          all_docs.push(crate::api::types::Document { fields });
        }
      }
    }
    let inner = &self.inner;
    let schema = manifest_snapshot.schema.clone();
    let mut manifest_guard = inner.manifest.write();
    let generation = manifest_guard
      .segments
      .iter()
      .map(|s| s.generation)
      .max()
      .unwrap_or(0)
      + 1;
    let writer = SegmentWriter::new(
      &inner.path,
      &schema,
      inner.options.enable_positions,
      cfg!(feature = "zstd"),
      inner.storage.clone(),
    );
    let new_seg = writer.write_segment(&all_docs, generation)?;
    manifest_guard.segments = vec![new_seg];
    manifest_guard.committed_at = Utc::now().to_rfc3339();
    manifest_guard.store(
      inner.storage.as_ref(),
      &Manifest::manifest_path(&inner.path),
    )?;
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
