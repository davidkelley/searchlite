use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{Document, IndexOptions, NumericField, Schema, SearchRequest};
use searchlite_core::api::Filter;

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
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: true,
      highlight_field: Some("body".to_string()),
    })
    .unwrap();
  assert!(!resp.hits.is_empty());
}
