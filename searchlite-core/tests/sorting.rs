use std::collections::BTreeMap;

use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Document, ExecutionStrategy, IndexOptions, KeywordField, NumericField, Schema, SearchRequest,
  SortOrder, SortSpec, StorageType,
};
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

fn base_options(path: &std::path::Path) -> IndexOptions {
  IndexOptions {
    path: path.to_path_buf(),
    create_if_missing: true,
    enable_positions: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::Filesystem,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  }
}

#[test]
fn sorts_numeric_and_missing_last() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "rating".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
  });
  let idx = Index::create(&path, schema, base_options(&path)).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "a",
        vec![("body", json!("rust alpha")), ("rating", json!(10))],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "b",
        vec![("body", json!("rust beta")), ("rating", json!(3))],
      ))
      .unwrap();
    writer
      .add_document(&doc("c", vec![("body", json!("rust gamma"))]))
      .unwrap();
    writer
      .add_document(&doc(
        "d",
        vec![("body", json!("rust delta")), ("rating", json!(7))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }

  let reader = idx.reader().unwrap();
  let resp = reader
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      filters: vec![],
      limit: 10,
      sort: vec![SortSpec {
        field: "rating".into(),
        order: Some(SortOrder::Asc),
      }],
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: true,
      highlight_field: None,
      aggs: BTreeMap::new(),
      suggest: BTreeMap::new(),
    })
    .unwrap();

  let ratings: Vec<Option<i64>> = resp
    .hits
    .iter()
    .map(|hit| {
      hit
        .fields
        .as_ref()
        .and_then(|doc| doc.get("rating"))
        .and_then(|v| v.as_i64())
    })
    .collect();
  assert_eq!(ratings, vec![Some(3), Some(7), Some(10), None]);
}

#[test]
fn sorts_keywords_descending_with_multivalue_mode() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.keyword_fields.push(KeywordField {
    name: "tag".into(),
    stored: true,
    indexed: true,
    fast: true,
    nullable: false,
  });
  let idx = IndexBuilder::create(&path, schema, base_options(&path)).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "t1",
        vec![
          ("body", json!("rust tags zulu")),
          ("tag", json!(["alpha", "omega"])),
        ],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "t2",
        vec![
          ("body", json!("rust tags kappa")),
          ("tag", json!(["kappa"])),
        ],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "t3",
        vec![("body", json!("rust tags zeta")), ("tag", json!(["zeta"]))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }

  let reader = idx.reader().unwrap();
  let resp = reader
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      filters: vec![],
      limit: 5,
      sort: vec![SortSpec {
        field: "tag".into(),
        order: Some(SortOrder::Desc),
      }],
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: true,
      highlight_field: None,
      aggs: BTreeMap::new(),
      suggest: BTreeMap::new(),
    })
    .unwrap();

  let tags: Vec<String> = resp
    .hits
    .iter()
    .map(|hit| {
      let values: Vec<String> = hit
        .fields
        .as_ref()
        .and_then(|doc| doc.get("tag"))
        .map(|v| {
          if let Some(arr) = v.as_array() {
            arr
              .iter()
              .filter_map(|val| val.as_str().map(|s| s.to_string()))
              .collect::<Vec<_>>()
          } else if let Some(s) = v.as_str() {
            vec![s.to_string()]
          } else {
            Vec::new()
          }
        })
        .unwrap_or_default();
      values
        .into_iter()
        .max()
        .unwrap_or_else(|| "<missing>".to_string())
    })
    .collect();
  assert_eq!(tags, vec!["zeta", "omega", "kappa"]);
}

#[test]
fn paginates_with_sorted_cursor_across_segments() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "rank".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
  });
  let idx = Index::create(&path, schema, base_options(&path)).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for val in [30, 10, 20] {
      writer
        .add_document(&doc(
          &format!("r{val}"),
          vec![("body", json!("rust paging")), ("rank", json!(val))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }
  {
    let mut writer = idx.writer().unwrap();
    for val in [15, 5] {
      writer
        .add_document(&doc(
          &format!("r{val}"),
          vec![("body", json!("rust paging")), ("rank", json!(val))],
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
    sort: vec![SortSpec {
      field: "rank".into(),
      order: Some(SortOrder::Asc),
    }],
    cursor: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    fuzzy: None,
    #[cfg(feature = "vectors")]
    vector_query: None,
    return_stored: true,
    highlight_field: None,
    aggs: BTreeMap::new(),
    suggest: BTreeMap::new(),
  };

  let mut ranks = Vec::new();
  for _ in 0..3 {
    let resp = reader.search(&req).unwrap();
    ranks.extend(extract_ranks(&resp));
    if resp.next_cursor.is_none() {
      break;
    }
    req.cursor = resp.next_cursor.clone();
  }
  assert_eq!(ranks, vec![5, 10, 15, 20, 30]);
}

fn extract_ranks(res: &searchlite_core::api::reader::SearchResult) -> Vec<i64> {
  res
    .hits
    .iter()
    .filter_map(|hit| {
      hit
        .fields
        .as_ref()
        .and_then(|doc| doc.get("rank"))
        .and_then(|v| v.as_i64())
    })
    .collect()
}
