use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Document, ExecutionStrategy, IndexOptions, NumericField, Schema, SearchRequest, StorageType,
};
use searchlite_core::api::Filter;
use searchlite_core::api::Index;

#[test]
fn index_and_search() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "year".to_string(),
    i64: true,
    fast: true,
    stored: true,
  });
  let opts = IndexOptions {
    path: path.clone(),
    create_if_missing: true,
    enable_positions: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::Filesystem,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let idx = IndexBuilder::create(&path, schema, opts).expect("create index");
  let mut writer = idx.writer().expect("writer");
  let doc = Document {
    fields: [
      ("title".to_string(), serde_json::json!("Rust systems")),
      (
        "body".to_string(),
        serde_json::json!("Rust is a systems programming language"),
      ),
      ("year".to_string(), serde_json::json!(2023)),
    ]
    .into_iter()
    .collect(),
  };
  writer.add_document(&doc).unwrap();
  writer.commit().unwrap();

  let reader = idx.reader().unwrap();
  let resp = reader
    .search(&SearchRequest {
      query: "systems programming".to_string(),
      fields: Some(vec!["body".to_string(), "title".to_string()]),
      filters: vec![Filter::I64Range {
        field: "year".to_string(),
        min: 2010,
        max: 2024,
      }],
      limit: 5,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: true,
      highlight_field: Some("body".to_string()),
    })
    .unwrap();
  assert!(!resp.hits.is_empty());
}

#[test]
fn in_memory_storage_keeps_disk_clean() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf().join("mem_idx");
  let opts = IndexOptions {
    path: path.clone(),
    create_if_missing: true,
    enable_positions: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::InMemory,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let idx = Index::create(&path, Schema::default_text_body(), opts).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&Document {
        fields: [("body".into(), serde_json::json!("in memory wal"))]
          .into_iter()
          .collect(),
      })
      .unwrap();
    writer.commit().unwrap();
  }
  let reader = idx.reader().unwrap();
  let resp = reader
    .search(&SearchRequest {
      query: "memory".to_string(),
      fields: None,
      filters: vec![],
      limit: 5,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: true,
      highlight_field: None,
    })
    .unwrap();
  assert_eq!(resp.hits.len(), 1);
  assert!(std::fs::metadata(&path).is_err());
  assert!(std::fs::metadata(path.join("wal.log")).is_err());
  assert!(std::fs::metadata(path.join("MANIFEST.json")).is_err());
}
