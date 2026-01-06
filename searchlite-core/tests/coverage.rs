use std::collections::BTreeMap;

use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Document, ExecutionStrategy, Filter, IndexOptions, KeywordField, NumericField, Schema,
  SearchRequest, StorageType,
};
use tempfile::tempdir;

fn opts(path: &std::path::Path) -> IndexOptions {
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

fn doc(body: &str, category: &str, year: i64) -> Document {
  Document {
    fields: [
      (
        "_id".into(),
        serde_json::json!(format!("{category}-{year}-{body}")),
      ),
      ("body".into(), serde_json::json!(body)),
      ("category".into(), serde_json::json!(category)),
      ("year".into(), serde_json::json!(year)),
    ]
    .into_iter()
    .collect(),
  }
}

#[test]
fn search_with_phrase_filters_and_compaction() {
  let dir = tempdir().unwrap();
  let mut schema = Schema::default_text_body();
  schema.keyword_fields.push(KeywordField {
    name: "category".into(),
    stored: true,
    indexed: true,
    fast: true,
    nullable: false,
  });
  schema.numeric_fields.push(NumericField {
    name: "year".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
  });
  let index = IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();

  let mut writer = index.writer().unwrap();
  writer
    .add_document(&doc(
      "rust is a systems programming language",
      "search",
      2024,
    ))
    .unwrap();
  writer.commit().unwrap();

  let mut writer = index.writer().unwrap();
  writer
    .add_document(&doc(
      "boring systems programming rust writing",
      "search",
      2021,
    ))
    .unwrap();
  writer.commit().unwrap();

  {
    let mut writer = index.writer().unwrap();
    writer
      .add_document(&doc("rust systems programming pending wal", "search", 2022))
      .unwrap();
  }
  let mut writer = index.writer().unwrap();
  writer.commit().unwrap();

  let reader = index.reader().unwrap();
  let result = reader
    .search(&SearchRequest {
      query: "\"systems programming\" rust -boring".into(),
      fields: None,
      filter: None,
      filters: vec![
        Filter::KeywordEq {
          field: "category".into(),
          value: "search".into(),
        },
        Filter::I64Range {
          field: "year".into(),
          min: 2010,
          max: 2025,
        },
      ],
      limit: 10,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: true,
      highlight_field: Some("body".into()),
      aggs: BTreeMap::new(),
      suggest: BTreeMap::new(),
    })
    .unwrap();
  assert_eq!(result.hits.len(), 2);
  for hit in result.hits.iter() {
    let fields = hit.fields.as_ref().unwrap();
    assert_eq!(fields["category"], "search");
    assert!(hit.snippet.as_ref().unwrap().contains("rust"));
  }

  index.compact().unwrap();
  assert_eq!(index.manifest().segments.len(), 1);
}
