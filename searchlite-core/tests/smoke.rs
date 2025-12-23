use std::collections::BTreeMap;

use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Document, ExecutionStrategy, IndexOptions, KeywordField, NestedField, NestedProperty,
  NumericField, Schema, SearchRequest, StorageType, TextField,
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
  schema.text_fields.push(TextField {
    name: "title".to_string(),
    tokenizer: "default".to_string(),
    stored: true,
    indexed: true,
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
      aggs: BTreeMap::new(),
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
      aggs: BTreeMap::new(),
    })
    .unwrap();
  assert_eq!(resp.hits.len(), 1);
  assert!(std::fs::metadata(&path).is_err());
  assert!(std::fs::metadata(path.join("wal.log")).is_err());
  assert!(std::fs::metadata(path.join("MANIFEST.json")).is_err());
}

#[test]
fn nested_filters_scope_to_object_and_preserve_stored_shape() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.nested_fields.push(NestedField {
    name: "comment".into(),
    fields: vec![
      NestedProperty::Keyword(KeywordField {
        name: "author".into(),
        stored: true,
        indexed: true,
        fast: true,
      }),
      NestedProperty::Keyword(KeywordField {
        name: "tag".into(),
        stored: true,
        indexed: true,
        fast: true,
      }),
      NestedProperty::Keyword(KeywordField {
        name: "note".into(),
        stored: false,
        indexed: true,
        fast: false,
      }),
    ],
  });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();

  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("body".into(), serde_json::json!("rust nested filters")),
          (
            "comment".into(),
            serde_json::json!([
              { "author": "alice", "tag": "x", "note": "keep" },
              { "author": "bob", "tag": "y" }
            ]),
          ),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("body".into(), serde_json::json!("rust nested filters")),
          (
            "comment".into(),
            serde_json::json!([{ "author": "alice", "tag": "y", "note": "secret" }]),
          ),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    writer.commit().unwrap();
  }

  let filters = vec![
    Filter::Nested {
      path: "comment".into(),
      filter: Box::new(Filter::KeywordEq {
        field: "author".into(),
        value: "alice".into(),
      }),
    },
    Filter::Nested {
      path: "comment".into(),
      filter: Box::new(Filter::KeywordEq {
        field: "tag".into(),
        value: "y".into(),
      }),
    },
  ];

  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filters,
      limit: 5,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: true,
      highlight_field: None,
      aggs: BTreeMap::new(),
    })
    .unwrap();

  assert_eq!(resp.hits.len(), 1);
  let stored = resp.hits[0].fields.as_ref().expect("stored doc");
  let comment = stored
    .get("comment")
    .and_then(|v| v.as_array())
    .expect("comment array");
  assert_eq!(comment.len(), 1);
  let first = comment[0].as_object().expect("comment object");
  assert_eq!(first.get("author"), Some(&serde_json::json!("alice")));
  assert_eq!(first.get("tag"), Some(&serde_json::json!("y")));
  assert!(first.get("note").is_none());
}
