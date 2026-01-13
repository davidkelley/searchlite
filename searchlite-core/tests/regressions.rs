use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use searchlite_core::api::types::{
  Document, ExecutionStrategy, IndexOptions, KeywordField, Schema, SearchRequest, StorageType,
  TextField,
};
use searchlite_core::api::{Filter, Index};
use searchlite_core::storage::Storage;
use serde_json::json;
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

fn base_request(query: &str, filter: Option<Filter>) -> SearchRequest {
  SearchRequest {
    query: query.into(),
    fields: None,
    filter,
    filters: vec![],
    limit: 10,
    return_hits: true,
    candidate_size: None,
    sort: Vec::new(),
    cursor: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    fuzzy: None,
    #[cfg(feature = "vectors")]
    vector_query: None,
    #[cfg(feature = "vectors")]
    vector_filter: None,
    return_stored: true,
    highlight_field: None,
    highlight: None,
    collapse: None,
    aggs: BTreeMap::new(),
    suggest: BTreeMap::new(),
    rescore: None,
    explain: false,
    profile: false,
  }
}

fn doc(id: &str, fields: Vec<(&str, serde_json::Value)>) -> Document {
  let mut map = BTreeMap::new();
  map.insert("_id".to_string(), json!(id));
  for (k, v) in fields {
    map.insert(k.to_string(), v);
  }
  Document { fields: map }
}

#[test]
fn compact_rejects_fast_only_fields() {
  let dir = tempdir().unwrap();
  let schema = Schema {
    doc_id_field: "_id".to_string(),
    analyzers: Vec::new(),
    text_fields: vec![TextField {
      name: "body".into(),
      analyzer: "default".into(),
      search_analyzer: None,
      stored: true,
      indexed: true,
      nullable: false,
      search_as_you_type: None,
    }],
    keyword_fields: vec![KeywordField {
      name: "tag".into(),
      stored: false,
      indexed: false,
      fast: true,
      nullable: false,
    }],
    numeric_fields: Vec::new(),
    nested_fields: Vec::new(),
    #[cfg(feature = "vectors")]
    vector_fields: Vec::new(),
  };
  let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "1",
        vec![("body", json!("first")), ("tag", json!("keep"))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "2",
        vec![("body", json!("second")), ("tag", json!("other"))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  let reader = idx.reader().unwrap();
  let filter = Filter::KeywordEq {
    field: "tag".to_string(),
    value: "keep".to_string(),
  };
  let req = base_request("first", Some(filter));
  let hits_before = reader.search(&req).unwrap().hits.len();
  assert_eq!(hits_before, 1, "baseline query should find the document");

  let err = idx.compact().unwrap_err();
  assert!(
    err.to_string().contains("indexed/fast but not stored"),
    "unexpected compaction error: {err}"
  );

  let reader_after = idx.reader().unwrap();
  let hits_after = reader_after.search(&req).unwrap().hits.len();
  assert_eq!(
    hits_after, 1,
    "compaction attempt must not drop fast-only field data"
  );
}

struct FailingManifestStorage {
  inner: searchlite_core::storage::InMemoryStorage,
  fail_manifest: std::sync::atomic::AtomicBool,
}

impl FailingManifestStorage {
  fn new(root: PathBuf) -> Self {
    Self {
      inner: searchlite_core::storage::InMemoryStorage::new(root),
      fail_manifest: std::sync::atomic::AtomicBool::new(false),
    }
  }

  fn fail_next_manifest_store(&self) {
    self
      .fail_manifest
      .store(true, std::sync::atomic::Ordering::SeqCst);
  }

  fn should_fail(&self, path: &Path) -> bool {
    path.ends_with("MANIFEST.json")
      && self
        .fail_manifest
        .swap(false, std::sync::atomic::Ordering::SeqCst)
  }
}

impl Storage for FailingManifestStorage {
  fn root(&self) -> &Path {
    self.inner.root()
  }

  fn ensure_dir(&self, path: &Path) -> anyhow::Result<()> {
    self.inner.ensure_dir(path)
  }

  fn exists(&self, path: &Path) -> bool {
    self.inner.exists(path)
  }

  fn open_read(&self, path: &Path) -> anyhow::Result<searchlite_core::storage::DynFile> {
    self.inner.open_read(path)
  }

  fn open_write(&self, path: &Path) -> anyhow::Result<searchlite_core::storage::DynFile> {
    self.inner.open_write(path)
  }

  fn open_append(&self, path: &Path) -> anyhow::Result<searchlite_core::storage::DynFile> {
    self.inner.open_append(path)
  }

  fn read_to_end(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
    self.inner.read_to_end(path)
  }

  fn write_all(&self, path: &Path, data: &[u8]) -> anyhow::Result<()> {
    self.inner.write_all(path, data)
  }

  fn atomic_write(&self, path: &Path, data: &[u8]) -> anyhow::Result<()> {
    if self.should_fail(path) {
      return Err(anyhow::anyhow!("manifest write failed"));
    }
    self.inner.atomic_write(path, data)
  }

  fn remove(&self, path: &Path) -> anyhow::Result<()> {
    self.inner.remove(path)
  }

  fn remove_dir_all(&self, path: &Path) -> anyhow::Result<()> {
    self.inner.remove_dir_all(path)
  }
}

#[test]
fn failed_manifest_persistence_does_not_publish_in_memory_state() {
  let dir = tempdir().unwrap();
  let storage = Arc::new(FailingManifestStorage::new(dir.path().to_path_buf()));
  let mut opts = opts(dir.path());
  opts.storage = StorageType::InMemory;
  let idx = Index::create_with_storage(
    dir.path(),
    Schema::default_text_body(),
    opts,
    storage.clone(),
  )
  .unwrap();
  let mut writer = idx.writer().unwrap();
  writer
    .add_document(&doc(
      "1",
      vec![("body", json!("commit failure should rollback"))],
    ))
    .unwrap();
  storage.fail_next_manifest_store();
  let err = writer.commit().unwrap_err();
  assert!(
    err.to_string().contains("manifest write failed")
      || err.to_string().contains("writing manifest"),
    "unexpected error: {err}"
  );
  // Manifest in memory should not show the failed segment.
  assert_eq!(idx.manifest().segments.len(), 0);

  // WAL should still be replayable by a fresh writer.
  let mut restored = idx.writer().unwrap();
  restored.commit().unwrap();
  assert_eq!(idx.manifest().segments.len(), 1);
}

#[test]
fn concurrent_writers_refresh_manifest_before_commit() {
  let dir = tempdir().unwrap();
  let idx = Index::create(dir.path(), Schema::default_text_body(), opts(dir.path())).unwrap();
  let mut writer1 = idx.writer().unwrap();
  let mut writer2 = idx.writer().unwrap(); // Created before writer1 commits; stale snapshot.

  writer1
    .add_document(&doc("1", vec![("body", json!("first body"))]))
    .unwrap();
  writer1.commit().unwrap();

  writer2
    .add_document(&doc("1", vec![("body", json!("updated body"))]))
    .unwrap();
  writer2.commit().unwrap();

  let reader = idx.reader().unwrap();
  let req_first = base_request("first", None);
  let req_updated = base_request("updated", None);

  let hits_first = reader.search(&req_first).unwrap().hits.len();
  let hits_updated = reader.search(&req_updated).unwrap().hits.len();
  assert_eq!(hits_first, 0, "stale writer should tombstone old doc");
  assert_eq!(hits_updated, 1, "new version must be visible");
}
