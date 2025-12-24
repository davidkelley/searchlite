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

#[test]
fn cursor_rejects_invalid_hex() {
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
    writer
      .add_document(&Document {
        fields: [("body".to_string(), serde_json::json!("rust"))]
          .into_iter()
          .collect(),
      })
      .unwrap();
    writer.commit().unwrap();
  }

  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    query: "rust".to_string(),
    fields: None,
    filters: vec![],
    limit: 1,
    cursor: Some("not-a-valid-cursor".to_string()),
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    #[cfg(feature = "vectors")]
    vector_query: None,
    return_stored: true,
    highlight_field: None,
    aggs: BTreeMap::new(),
  };

  assert!(reader.search(&req).is_err());
}

#[test]
fn cursor_rejects_when_limit_zero() {
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
    writer
      .add_document(&Document {
        fields: [("body".to_string(), serde_json::json!("rust"))]
          .into_iter()
          .collect(),
      })
      .unwrap();
    writer.commit().unwrap();
  }

  let reader = idx.reader().unwrap();
  let err = reader
    .search(&SearchRequest {
      query: "rust".to_string(),
      fields: None,
      filters: vec![],
      limit: 0,
      cursor: Some("00000000000000000000000000000000".to_string()),
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      aggs: BTreeMap::new(),
    })
    .unwrap_err();

  assert!(err
    .to_string()
    .contains("cursor is not supported when limit is 0"));
}

#[test]
fn cursor_rejects_excessive_advance() {
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
    writer
      .add_document(&Document {
        fields: [("body".to_string(), serde_json::json!("rust"))]
          .into_iter()
          .collect(),
      })
      .unwrap();
    writer.commit().unwrap();
  }

  let reader = idx.reader().unwrap();
  let manifest_generation = idx
    .manifest()
    .segments
    .iter()
    .map(|s| s.generation)
    .max()
    .unwrap_or(0);
  let req = SearchRequest {
    query: "rust".to_string(),
    fields: None,
    filters: vec![],
    limit: 1,
    cursor: Some(encode_cursor_with_returned(60_000, manifest_generation)),
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    #[cfg(feature = "vectors")]
    vector_query: None,
    return_stored: true,
    highlight_field: None,
    aggs: BTreeMap::new(),
  };

  assert!(reader.search(&req).is_err());
}

#[test]
fn cursor_rejects_mismatched_position() {
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
    for _ in 0..3 {
      writer
        .add_document(&Document {
          fields: [("body".to_string(), serde_json::json!("rust"))]
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

  let first = reader.search(&req).unwrap();
  assert!(first.next_cursor.is_some());

  req.cursor = first.next_cursor.as_ref().map(|c| tamper_cursor(c));
  assert!(reader.search(&req).is_err());
}

#[test]
fn cursor_orders_stably_across_segments() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.keyword_fields.push(KeywordField {
    name: "tag".to_string(),
    stored: true,
    indexed: true,
    fast: false,
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
  let idx = Index::create(&path, schema, opts).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for i in 0..3 {
      writer
        .add_document(&Document {
          fields: [
            ("body".to_string(), serde_json::json!("rust")),
            ("tag".to_string(), serde_json::json!(format!("s0-{i}"))),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
    }
    writer.commit().unwrap();
  }
  {
    let mut writer = idx.writer().unwrap();
    for i in 0..3 {
      writer
        .add_document(&Document {
          fields: [
            ("body".to_string(), serde_json::json!("rust")),
            ("tag".to_string(), serde_json::json!(format!("s1-{i}"))),
          ]
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
    return_stored: false,
    highlight_field: None,
    aggs: BTreeMap::new(),
  };

  let mut doc_ids = Vec::new();
  for _ in 0..4 {
    let res = reader.search(&req).unwrap();
    doc_ids.extend(res.hits.iter().map(|h| h.doc_id));
    if res.next_cursor.is_none() {
      break;
    }
    req.cursor = res.next_cursor.clone();
  }

  assert_eq!(doc_ids, vec![0, 1, 2, 0, 1, 2]);
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

const TEST_CURSOR_VERSION: u8 = 1;
const TEST_CURSOR_BYTES: usize = 21;

fn encode_cursor_with_returned(returned: u32, generation: u32) -> String {
  let mut buf = [0u8; TEST_CURSOR_BYTES];
  buf[0] = TEST_CURSOR_VERSION;
  buf[1..5].copy_from_slice(&generation.to_be_bytes());
  buf[17..].copy_from_slice(&returned.to_be_bytes());
  let mut encoded = String::with_capacity(TEST_CURSOR_BYTES * 2);
  const HEX: &[u8; 16] = b"0123456789abcdef";
  for byte in buf {
    encoded.push(HEX[(byte >> 4) as usize] as char);
    encoded.push(HEX[(byte & 0x0f) as usize] as char);
  }
  encoded
}

fn tamper_cursor(cursor: &str) -> String {
  let mut chars: Vec<char> = cursor.chars().collect();
  if let Some(first) = chars.first_mut() {
    *first = if *first == 'a' { 'b' } else { 'a' };
  }
  chars.into_iter().collect()
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
