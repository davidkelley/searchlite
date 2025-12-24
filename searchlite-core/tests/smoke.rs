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
    nullable: false,
  });
  schema.text_fields.push(TextField {
    name: "title".to_string(),
    tokenizer: "default".to_string(),
    stored: true,
    indexed: true,
    nullable: false,
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
      cursor: None,
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
fn cursor_paginates_ordered_hits() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
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
  let idx = Index::create(&path, Schema::default_text_body(), opts).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for i in 0..3 {
      let repeats = 6 - i;
      writer
        .add_document(&Document {
          fields: [(
            "body".to_string(),
            serde_json::json!("rust ".repeat(repeats)),
          )]
          .into_iter()
          .collect(),
        })
        .unwrap();
    }
    writer.commit().unwrap();
  }
  {
    let mut writer = idx.writer().unwrap();
    for i in 3..6 {
      let repeats = 6 - i;
      writer
        .add_document(&Document {
          fields: [(
            "body".to_string(),
            serde_json::json!("rust ".repeat(repeats)),
          )]
          .into_iter()
          .collect(),
        })
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let reader = idx.reader().unwrap();
  let mut req = SearchRequest {
    query: "rust".to_string(),
    fields: None,
    filters: vec![],
    limit: 2,
    cursor: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    #[cfg(feature = "vectors")]
    vector_query: None,
    return_stored: true,
    highlight_field: None,
    aggs: BTreeMap::new(),
  };

  let mut bodies = Vec::new();

  let first = reader.search(&req).unwrap();
  assert_eq!(first.hits.len(), 2);
  assert!(first.next_cursor.is_some());
  bodies.extend(extract_bodies(&first));

  req.cursor = first.next_cursor.clone();
  let second = reader.search(&req).unwrap();
  assert_eq!(second.hits.len(), 2);
  assert!(second.next_cursor.is_some());
  bodies.extend(extract_bodies(&second));

  req.cursor = second.next_cursor.clone();
  let third = reader.search(&req).unwrap();
  assert_eq!(third.hits.len(), 2);
  assert!(third.next_cursor.is_none());
  bodies.extend(extract_bodies(&third));

  bodies.sort();
  bodies.dedup();
  assert_eq!(bodies.len(), 6);
}

fn extract_bodies(res: &searchlite_core::api::reader::SearchResult) -> Vec<String> {
  res
    .hits
    .iter()
    .filter_map(|hit| {
      hit
        .fields
        .as_ref()
        .and_then(|doc| doc.get("body"))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
    })
    .collect()
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
      cursor: None,
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
        nullable: false,
      }),
      NestedProperty::Keyword(KeywordField {
        name: "tag".into(),
        stored: true,
        indexed: true,
        fast: true,
        nullable: false,
      }),
      NestedProperty::Keyword(KeywordField {
        name: "note".into(),
        stored: false,
        indexed: true,
        fast: false,
        nullable: false,
      }),
    ],
    nullable: false,
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
      cursor: None,
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

#[test]
fn nested_numeric_filters_bind_to_object_values() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.nested_fields.push(NestedField {
    name: "review".into(),
    fields: vec![
      NestedProperty::Keyword(KeywordField {
        name: "user".into(),
        stored: true,
        indexed: true,
        fast: true,
        nullable: false,
      }),
      NestedProperty::Numeric(NumericField {
        name: "rating".into(),
        i64: true,
        fast: true,
        stored: true,
        nullable: false,
      }),
      NestedProperty::Numeric(NumericField {
        name: "score".into(),
        i64: false,
        fast: true,
        stored: false,
        nullable: false,
      }),
    ],
    nullable: false,
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
          ("body".into(), serde_json::json!("rust nested numbers")),
          (
            "review".into(),
            serde_json::json!([
              { "user": "alice", "rating": 5, "score": 0.72 },
              { "user": "bob", "rating": 9, "score": 0.25 }
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
          ("body".into(), serde_json::json!("rust nested numbers")),
          (
            "review".into(),
            serde_json::json!([{ "user": "alice", "rating": 2, "score": 0.95 }]),
          ),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("body".into(), serde_json::json!("rust nested numbers")),
          (
            "review".into(),
            serde_json::json!([
              { "user": "alice", "rating": 2, "score": 0.9 },
              { "user": "bob", "rating": 6, "score": 0.72 }
            ]),
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
      path: "review".into(),
      filter: Box::new(Filter::KeywordEq {
        field: "user".into(),
        value: "alice".into(),
      }),
    },
    Filter::Nested {
      path: "review".into(),
      filter: Box::new(Filter::I64Range {
        field: "rating".into(),
        min: 5,
        max: 8,
      }),
    },
    Filter::Nested {
      path: "review".into(),
      filter: Box::new(Filter::F64Range {
        field: "score".into(),
        min: 0.7,
        max: 0.8,
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
      cursor: None,
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
  let reviews = stored
    .get("review")
    .and_then(|v| v.as_array())
    .expect("review array");
  assert_eq!(reviews.len(), 2);
  let alice = reviews
    .iter()
    .find(|r| r.get("user") == Some(&serde_json::json!("alice")))
    .and_then(|v| v.as_object())
    .expect("alice review object");
  assert_eq!(alice.get("rating"), Some(&serde_json::json!(5)));
  assert!(alice.get("score").is_none());
}
