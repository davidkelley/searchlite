use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use searchlite_core::api::types::{
  CollapseRequest, Document, ExecutionStrategy, IndexOptions, KeywordField, Query, QueryNode,
  Schema, SearchRequest, StorageType, TextField,
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
    limit: 10,
    from: 0,
    return_hits: true,
    candidate_size: None,
    #[cfg(feature = "vectors")]
    max_global_vector_candidates: None,
    sort: Vec::new(),
    cursor: None,
    search_after: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    fuzzy: None,
    track_total_hits: None,
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
fn search_request_deserializes_defaults() {
  let raw = r#"{"query":"rust"}"#;
  let req: SearchRequest = serde_json::from_str(raw).expect("parse request with defaults");
  assert_eq!(req.limit, 10);
  assert!(!req.return_stored);
  assert!(req.return_hits);
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

#[test]
fn collapse_rejects_multivalued_fast_field() {
  let dir = tempdir().unwrap();
  let schema = Schema {
    doc_id_field: "_id".into(),
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
      stored: true,
      indexed: true,
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
        vec![("body", json!("first")), ("tag", json!(["foo", "bar"]))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  let reader = idx.reader().unwrap();
  let mut req = base_request("first", None);
  req.collapse = Some(CollapseRequest {
    field: "tag".into(),
    inner_hits: None,
  });
  let err = reader.search(&req).unwrap_err();
  assert!(err.to_string().contains("single-valued"));
}

struct FailingManifestStorage {
  inner: searchlite_core::storage::InMemoryStorage,
  fail_pending_manifest: std::sync::atomic::AtomicBool,
}

impl FailingManifestStorage {
  fn new(root: PathBuf) -> Self {
    Self {
      inner: searchlite_core::storage::InMemoryStorage::new(root),
      fail_pending_manifest: std::sync::atomic::AtomicBool::new(false),
    }
  }

  fn fail_next_pending_manifest_store(&self) {
    self
      .fail_pending_manifest
      .store(true, std::sync::atomic::Ordering::SeqCst);
  }

  fn should_fail(&self, path: &Path) -> bool {
    let name = path
      .file_name()
      .and_then(|n| n.to_str())
      .unwrap_or_default();
    name == "MANIFEST.json.pending"
      && self
        .fail_pending_manifest
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
  // The pre-fence manifest stage (`MANIFEST.json.pending`) is the last
  // recoverable failure point in the BUG-018 ordering — anything past it
  // crosses the WAL durability fence and is treated as a successful
  // commit. This test pins the original "if persistence fails, in-memory
  // state stays put and the WAL replays cleanly" contract on the
  // pre-fence path, where it still applies.
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
  storage.fail_next_pending_manifest_store();
  let err = writer.commit().unwrap_err();
  let msg = format!("{err:#}");
  assert!(
    msg.contains("manifest write failed")
      || msg.contains("staging manifest")
      || msg.contains("writing manifest"),
    "unexpected error: {msg}"
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

/// Regression test for the keyword case-folding divergence reported in
/// davidkelley/searchlite#212.
///
/// Before the fix, keyword indexing and `match`/`term` queries lowercased
/// with `str::to_ascii_lowercase`, which only rewrites the 26 ASCII
/// uppercase letters and leaves every other byte untouched. The fast-field
/// filter path (`Filter::KeywordEq`) already used Unicode-aware folding, so
/// documents whose keyword uppercase form contained non-ASCII code points
/// (e.g. `RÉSUMÉ`) were matched by the filter but silently missed by the
/// equivalent `term` query. After the fix both paths go through the shared
/// `fold_keyword` helper and agree for ASCII and non-ASCII input alike.
#[test]
fn keyword_match_and_filter_agree_on_non_ascii_case() {
  let dir = tempdir().unwrap();
  let path = dir.path().to_path_buf();
  let schema = Schema {
    doc_id_field: "_id".into(),
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
      stored: true,
      indexed: true,
      fast: true,
      nullable: false,
    }],
    numeric_fields: Vec::new(),
    nested_fields: Vec::new(),
    #[cfg(feature = "vectors")]
    vector_fields: Vec::new(),
  };
  let idx = Index::create(&path, schema, opts(&path)).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    // doc_a's tag lowercases to "résumé" under either ASCII or Unicode folding.
    writer
      .add_document(&doc(
        "doc_a",
        vec![("body", json!("alpha")), ("tag", json!("Résumé"))],
      ))
      .unwrap();
    // doc_b's tag only lowercases to "résumé" under Unicode folding; ASCII
    // folding would leave it as "rÉsumÉ". This is the document that the
    // pre-fix postings path silently missed.
    writer
      .add_document(&doc(
        "doc_b",
        vec![("body", json!("beta")), ("tag", json!("RÉSUMÉ"))],
      ))
      .unwrap();
    // A Cyrillic control: uppercase ЖУК should match lowercase жук under
    // Unicode folding and miss under ASCII folding.
    writer
      .add_document(&doc(
        "doc_c",
        vec![("body", json!("gamma")), ("tag", json!("ЖУК"))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  let reader = idx.reader().unwrap();

  // Filter path (fast field): both résumé docs match, the beetle does not.
  let filter_req = SearchRequest {
    query: Query::Node(QueryNode::MatchAll { boost: None }),
    filter: Some(Filter::KeywordEq {
      field: "tag".into(),
      value: "résumé".into(),
    }),
    ..base_request("", None)
  };
  let mut filter_ids: Vec<String> = reader
    .search(&filter_req)
    .unwrap()
    .hits
    .into_iter()
    .map(|h| h.doc_id)
    .collect();
  filter_ids.sort();
  assert_eq!(filter_ids, vec!["doc_a".to_string(), "doc_b".to_string()]);

  // Query/postings path (term query over the keyword field): same docs as
  // the filter path. Before the fix, doc_b was missing from this result set.
  let term_req = SearchRequest {
    query: Query::Node(QueryNode::Term {
      field: "tag".into(),
      value: "résumé".into(),
      boost: None,
    }),
    ..base_request("", None)
  };
  let mut term_ids: Vec<String> = reader
    .search(&term_req)
    .unwrap()
    .hits
    .into_iter()
    .map(|h| h.doc_id)
    .collect();
  term_ids.sort();
  assert_eq!(term_ids, filter_ids);

  // Cyrillic spot-check. Filter and query agree here too.
  let cyrillic_filter = SearchRequest {
    query: Query::Node(QueryNode::MatchAll { boost: None }),
    filter: Some(Filter::KeywordEq {
      field: "tag".into(),
      value: "жук".into(),
    }),
    ..base_request("", None)
  };
  let cyrillic_filter_ids: Vec<String> = reader
    .search(&cyrillic_filter)
    .unwrap()
    .hits
    .into_iter()
    .map(|h| h.doc_id)
    .collect();
  assert_eq!(cyrillic_filter_ids, vec!["doc_c".to_string()]);

  let cyrillic_query = SearchRequest {
    query: Query::Node(QueryNode::Term {
      field: "tag".into(),
      value: "жук".into(),
      boost: None,
    }),
    ..base_request("", None)
  };
  let cyrillic_query_ids: Vec<String> = reader
    .search(&cyrillic_query)
    .unwrap()
    .hits
    .into_iter()
    .map(|h| h.doc_id)
    .collect();
  assert_eq!(cyrillic_query_ids, cyrillic_filter_ids);
}
