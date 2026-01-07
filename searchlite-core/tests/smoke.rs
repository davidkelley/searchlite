use std::collections::BTreeMap;

use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Document, ExecutionStrategy, FuzzyOptions, IndexOptions, KeywordField, NestedField,
  NestedProperty, NumericField, Schema, SearchRequest, StorageType, TextField,
};
use searchlite_core::api::Filter;
use searchlite_core::api::Index;
use serde_json::json;

fn doc(id: &str, fields: Vec<(&str, serde_json::Value)>) -> Document {
  let mut map = BTreeMap::new();
  map.insert("_id".to_string(), json!(id));
  for (k, v) in fields {
    map.insert(k.to_string(), v);
  }
  Document { fields: map }
}

fn base_search_request(query: &str) -> SearchRequest {
  SearchRequest {
    query: query.into(),
    fields: None,
    filter: None,
    filters: vec![],
    limit: 10,
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

fn build_index_with_docs(docs: Vec<Document>) -> (tempfile::TempDir, Index) {
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
    for doc in docs {
      writer.add_document(&doc).unwrap();
    }
    writer.commit().unwrap();
  }
  (tmp, idx)
}

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
    analyzer: "default".to_string(),
    search_analyzer: None,
    stored: true,
    indexed: true,
    nullable: false,
    search_as_you_type: None,
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
  let doc = doc(
    "1",
    vec![
      ("title", json!("Rust systems")),
      ("body", json!("Rust is a systems programming language")),
      ("year", json!(2023)),
    ],
  );
  writer.add_document(&doc).unwrap();
  writer.commit().unwrap();

  let reader = idx.reader().unwrap();
  let resp = reader
    .search(&SearchRequest {
      query: "systems programming".into(),
      fields: Some(vec!["body".to_string(), "title".to_string()]),
      filter: None,
      filters: vec![Filter::I64Range {
        field: "year".to_string(),
        min: 2010,
        max: 2024,
      }],
      limit: 5,
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
      highlight_field: Some("body".to_string()),
      highlight: None,
      collapse: None,
      aggs: BTreeMap::new(),
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  assert!(!resp.hits.is_empty());
}

#[test]
fn fuzzy_matches_typos() {
  let (_tmp, idx) =
    build_index_with_docs(vec![doc("doc-1", vec![("body", json!("Rust is fast"))])]);
  let reader = idx.reader().unwrap();
  let fuzzy_req = SearchRequest {
    fuzzy: Some(FuzzyOptions {
      max_edits: 1,
      prefix_length: 1,
      max_expansions: 20,
      min_length: 3,
    }),
    ..base_search_request("rusk")
  };
  let fuzzy_resp = reader.search(&fuzzy_req).unwrap();
  assert_eq!(fuzzy_resp.hits.len(), 1);

  let exact_req = SearchRequest {
    fuzzy: None,
    ..fuzzy_req
  };
  let exact_resp = reader.search(&exact_req).unwrap();
  assert!(exact_resp.hits.is_empty());
}

#[test]
fn fuzzy_expands_multiple_terms() {
  let (_tmp, idx) = build_index_with_docs(vec![
    doc("doc-1", vec![("body", json!("Rust"))]),
    doc("doc-2", vec![("body", json!("Systems"))]),
  ]);
  let reader = idx.reader().unwrap();
  let fuzzy_req = SearchRequest {
    fuzzy: Some(FuzzyOptions {
      max_edits: 1,
      prefix_length: 1,
      max_expansions: 20,
      min_length: 3,
    }),
    ..base_search_request("rusk systms")
  };
  let fuzzy_resp = reader.search(&fuzzy_req).unwrap();
  let mut ids: Vec<_> = fuzzy_resp
    .hits
    .iter()
    .map(|hit| hit.doc_id.clone())
    .collect();
  ids.sort();
  assert_eq!(ids, vec!["doc-1".to_string(), "doc-2".to_string()]);

  let exact_resp = reader.search(&base_search_request("rusk systms")).unwrap();
  assert!(exact_resp.hits.is_empty());
}

#[test]
fn fuzzy_respects_min_length() {
  let (_tmp, idx) = build_index_with_docs(vec![doc("doc-1", vec![("body", json!("Rust"))])]);
  let reader = idx.reader().unwrap();
  let resp = reader
    .search(&SearchRequest {
      fuzzy: Some(FuzzyOptions {
        max_edits: 1,
        prefix_length: 1,
        max_expansions: 20,
        min_length: 3,
      }),
      ..base_search_request("ru")
    })
    .unwrap();
  assert!(resp.hits.is_empty());
}

#[test]
fn fuzzy_respects_max_expansions() {
  let (_tmp, idx) = build_index_with_docs(vec![
    doc("doc-1", vec![("body", json!("Rush"))]),
    doc("doc-2", vec![("body", json!("Rust"))]),
  ]);
  let reader = idx.reader().unwrap();
  let limited_resp = reader
    .search(&SearchRequest {
      fuzzy: Some(FuzzyOptions {
        max_edits: 1,
        prefix_length: 1,
        max_expansions: 1,
        min_length: 3,
      }),
      ..base_search_request("rusk")
    })
    .unwrap();
  assert_eq!(limited_resp.hits.len(), 1);

  let expanded_resp = reader
    .search(&SearchRequest {
      fuzzy: Some(FuzzyOptions {
        max_edits: 1,
        prefix_length: 1,
        max_expansions: 2,
        min_length: 3,
      }),
      ..base_search_request("rusk")
    })
    .unwrap();
  assert_eq!(expanded_resp.hits.len(), 2);
}

#[test]
fn fuzzy_respects_prefix_length() {
  let (_tmp, idx) = build_index_with_docs(vec![doc("doc-1", vec![("body", json!("Dusk"))])]);
  let reader = idx.reader().unwrap();
  let loose_resp = reader
    .search(&SearchRequest {
      fuzzy: Some(FuzzyOptions {
        max_edits: 1,
        prefix_length: 0,
        max_expansions: 20,
        min_length: 3,
      }),
      ..base_search_request("rusk")
    })
    .unwrap();
  assert_eq!(loose_resp.hits.len(), 1);

  let strict_resp = reader
    .search(&SearchRequest {
      fuzzy: Some(FuzzyOptions {
        max_edits: 1,
        prefix_length: 1,
        max_expansions: 20,
        min_length: 3,
      }),
      ..base_search_request("rusk")
    })
    .unwrap();
  assert!(strict_resp.hits.is_empty());
}

#[test]
fn fuzzy_allows_two_edits() {
  let (_tmp, idx) = build_index_with_docs(vec![doc("doc-1", vec![("body", json!("Rust"))])]);
  let reader = idx.reader().unwrap();
  let one_edit_resp = reader
    .search(&SearchRequest {
      fuzzy: Some(FuzzyOptions {
        max_edits: 1,
        prefix_length: 1,
        max_expansions: 20,
        min_length: 3,
      }),
      ..base_search_request("rsut")
    })
    .unwrap();
  assert!(one_edit_resp.hits.is_empty());

  let two_edit_resp = reader
    .search(&SearchRequest {
      fuzzy: Some(FuzzyOptions {
        max_edits: 2,
        prefix_length: 1,
        max_expansions: 20,
        min_length: 3,
      }),
      ..base_search_request("rsut")
    })
    .unwrap();
  assert_eq!(two_edit_resp.hits.len(), 1);
}

#[test]
fn upsert_and_delete_by_id() {
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
      .add_document(&doc("doc-1", vec![("body", json!("first version"))]))
      .unwrap();
    writer.commit().unwrap();
  }
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc("doc-1", vec![("body", json!("second version"))]))
      .unwrap();
    writer.commit().unwrap();
  }
  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    query: "second".into(),
    fields: None,
    filter: None,
    filters: vec![],
    limit: 5,
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
  };
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 1);
  assert_eq!(resp.hits[0].doc_id, "doc-1".to_string());
  assert_eq!(
    resp.hits[0]
      .fields
      .as_ref()
      .unwrap()
      .get("body")
      .and_then(|v| v.as_str())
      .unwrap(),
    "second version"
  );

  {
    let mut writer = idx.writer().unwrap();
    writer.delete_document("doc-1").unwrap();
    writer.commit().unwrap();
  }
  let after_delete = idx.reader().unwrap().search(&req).unwrap();
  assert!(after_delete.hits.is_empty());
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
        .add_document(&doc(
          &format!("{}", i),
          vec![("body", json!("rust ".repeat(repeats)))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }
  {
    let mut writer = idx.writer().unwrap();
    for i in 3..6 {
      let repeats = 6 - i;
      writer
        .add_document(&doc(
          &format!("{}", i),
          vec![("body", json!("rust ".repeat(repeats)))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let reader = idx.reader().unwrap();
  let mut req = SearchRequest {
    query: "rust".into(),
    fields: None,
    filter: None,
    filters: vec![],
    limit: 2,
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
      .add_document(&doc("cursor-1", vec![("body", json!("rust"))]))
      .unwrap();
    writer.commit().unwrap();
  }

  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    query: "rust".into(),
    fields: None,
    filter: None,
    filters: vec![],
    limit: 1,
    sort: Vec::new(),
    cursor: Some("not-a-valid-cursor".to_string()),
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
      .add_document(&doc("cursor-2", vec![("body", json!("rust"))]))
      .unwrap();
    writer.commit().unwrap();
  }

  let reader = idx.reader().unwrap();
  let err = reader
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      filters: vec![],
      limit: 0,
      sort: Vec::new(),
      cursor: Some("00000000000000000000000000000000".to_string()),
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,

      #[cfg(feature = "vectors")]
      vector_filter: None,
      return_stored: false,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs: BTreeMap::new(),
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
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
      .add_document(&doc("cursor-3", vec![("body", json!("rust"))]))
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
    query: "rust".into(),
    fields: None,
    filter: None,
    filters: vec![],
    limit: 1,
    sort: Vec::new(),
    cursor: Some(encode_cursor_with_returned(60_000, manifest_generation)),
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
    for i in 0..3 {
      writer
        .add_document(&doc(&format!("stable-{i}"), vec![("body", json!("rust"))]))
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let reader = idx.reader().unwrap();
  let mut req = SearchRequest {
    query: "rust".into(),
    fields: None,
    filter: None,
    filters: vec![],
    limit: 2,
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
        .add_document(&doc(
          &format!("doc-s0-{i}"),
          vec![("body", json!("rust")), ("tag", json!(format!("s0-{i}")))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }
  {
    let mut writer = idx.writer().unwrap();
    for i in 0..3 {
      writer
        .add_document(&doc(
          &format!("doc-s1-{i}"),
          vec![("body", json!("rust")), ("tag", json!(format!("s1-{i}")))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let reader = idx.reader().unwrap();
  let mut req = SearchRequest {
    query: "rust".into(),
    fields: None,
    filter: None,
    filters: vec![],
    limit: 2,
    sort: Vec::new(),
    cursor: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    fuzzy: None,
    #[cfg(feature = "vectors")]
    vector_query: None,

    #[cfg(feature = "vectors")]
    vector_filter: None,
    return_stored: false,
    highlight_field: None,
    highlight: None,
    collapse: None,
    aggs: BTreeMap::new(),
    suggest: BTreeMap::new(),
    rescore: None,
    explain: false,
    profile: false,
  };

  let mut doc_ids = Vec::new();
  for _ in 0..4 {
    let res = reader.search(&req).unwrap();
    doc_ids.extend(res.hits.iter().map(|h| h.doc_id.clone()));
    if res.next_cursor.is_none() {
      break;
    }
    req.cursor = res.next_cursor.clone();
  }

  assert_eq!(
    doc_ids,
    vec![
      "doc-s0-0".to_string(),
      "doc-s0-1".to_string(),
      "doc-s0-2".to_string(),
      "doc-s1-0".to_string(),
      "doc-s1-1".to_string(),
      "doc-s1-2".to_string()
    ]
  );
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
      .add_document(&doc("mem-1", vec![("body", json!("in memory wal"))]))
      .unwrap();
    writer.commit().unwrap();
  }
  let reader = idx.reader().unwrap();
  let resp = reader
    .search(&SearchRequest {
      query: "memory".into(),
      fields: None,
      filter: None,
      filters: vec![],
      limit: 5,
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
      .add_document(&doc(
        "nested-1",
        vec![
          ("body", json!("rust nested filters")),
          (
            "comment",
            json!([
              { "author": "alice", "tag": "x", "note": "keep" },
              { "author": "bob", "tag": "y" }
            ]),
          ),
        ],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "nested-2",
        vec![
          ("body", json!("rust nested filters")),
          (
            "comment",
            json!([{ "author": "alice", "tag": "y", "note": "secret" }]),
          ),
        ],
      ))
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
      filter: None,
      filters,
      limit: 5,
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
      .add_document(&doc(
        "nested-num-1",
        vec![
          ("body", json!("rust nested numbers")),
          (
            "review",
            json!([
              { "user": "alice", "rating": 5, "score": 0.72 },
              { "user": "bob", "rating": 9, "score": 0.25 }
            ]),
          ),
        ],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "nested-num-2",
        vec![
          ("body", json!("rust nested numbers")),
          (
            "review",
            json!([{ "user": "alice", "rating": 2, "score": 0.95 }]),
          ),
        ],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "nested-num-3",
        vec![
          ("body", json!("rust nested numbers")),
          (
            "review",
            json!([
              { "user": "alice", "rating": 2, "score": 0.9 },
              { "user": "bob", "rating": 6, "score": 0.72 }
            ]),
          ),
        ],
      ))
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
      filter: None,
      filters,
      limit: 5,
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
