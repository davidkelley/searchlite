use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  AggregationOp, AggregationRequest, Document, ExecutionStrategy, FacetKind, FacetRequest,
  IndexOptions, NumericField, NumericRange, Schema, SearchRequest, StorageType,
};
use searchlite_core::api::Filter;
use searchlite_core::api::Index;

fn build_schema() -> Schema {
  Schema {
    text_fields: vec![searchlite_core::api::TextField {
      name: "body".into(),
      tokenizer: "default".into(),
      stored: true,
      indexed: true,
    }],
    keyword_fields: vec![searchlite_core::api::KeywordField {
      name: "category".into(),
      stored: true,
      indexed: true,
      fast: true,
    }],
    numeric_fields: vec![
      NumericField {
        name: "year".into(),
        i64: true,
        fast: true,
        stored: true,
      },
      NumericField {
        name: "score".into(),
        i64: false,
        fast: true,
        stored: true,
      },
    ],
    #[cfg(feature = "vectors")]
    vector_fields: Vec::new(),
  }
}

fn add_docs(idx: &Index) {
  let mut writer = idx.writer().unwrap();
  let docs = vec![
    ("Rust for search", "news", 2021, 1.0_f64),
    ("Rust and performance", "sports", 2020, 2.5_f64),
    ("Advanced rust systems", "news", 2023, 3.5_f64),
  ];
  for (body, category, year, score) in docs.into_iter() {
    writer
      .add_document(&Document {
        fields: [
          ("body".into(), serde_json::json!(body)),
          ("category".into(), serde_json::json!(category)),
          ("year".into(), serde_json::json!(year)),
          ("score".into(), serde_json::json!(score)),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
  }
  writer.commit().unwrap();
}

fn opts(path: &std::path::Path) -> IndexOptions {
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
fn keyword_and_range_facets_respect_filters() {
  let dir = tempfile::tempdir().unwrap();
  let idx = IndexBuilder::create(dir.path(), build_schema(), opts(dir.path())).unwrap();
  add_docs(&idx);
  let reader = idx.reader().unwrap();
  let resp = reader
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filters: vec![Filter::I64Range {
        field: "year".into(),
        min: 2020,
        max: 2022,
      }],
      limit: 5,
      execution: ExecutionStrategy::Bm25,
      bmw_block_size: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      facets: vec![
        FacetRequest {
          field: "category".into(),
          facet: FacetKind::Keyword,
        },
        FacetRequest {
          field: "year".into(),
          facet: FacetKind::Range {
            ranges: vec![NumericRange {
              min: 2019.0,
              max: 2022.0,
            }],
          },
        },
      ],
      aggregations: Vec::new(),
    })
    .unwrap();

  let cat_counts = resp.facets.iter().find(|f| f.field == "category").unwrap();
  assert_eq!(cat_counts.counts.get("news"), Some(&1));
  assert_eq!(cat_counts.counts.get("sports"), Some(&1));

  let range_counts = resp.facets.iter().find(|f| f.field == "year").unwrap();
  assert_eq!(range_counts.counts.get("2019..2022"), Some(&2));
}

#[test]
fn histogram_and_aggregations_cover_results() {
  let dir = tempfile::tempdir().unwrap();
  let idx = Index::create(dir.path(), build_schema(), opts(dir.path())).unwrap();
  add_docs(&idx);
  let reader = idx.reader().unwrap();
  let resp = reader
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filters: vec![Filter::I64Range {
        field: "year".into(),
        min: 2020,
        max: 2025,
      }],
      limit: 1,
      execution: ExecutionStrategy::Bm25,
      bmw_block_size: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      facets: vec![FacetRequest {
        field: "score".into(),
        facet: FacetKind::Histogram {
          interval: 1.0,
          min: Some(1.0),
        },
      }],
      aggregations: vec![AggregationRequest {
        field: "score".into(),
        operations: vec![
          AggregationOp::Min,
          AggregationOp::Max,
          AggregationOp::Sum,
          AggregationOp::Avg,
        ],
      }],
    })
    .unwrap();

  let hist = resp.facets.iter().find(|f| f.field == "score").unwrap();
  assert_eq!(hist.counts.get("1..2"), Some(&1));
  assert_eq!(hist.counts.get("2..3"), Some(&1));
  assert_eq!(hist.counts.get("3..4"), Some(&1));

  let agg = resp
    .aggregations
    .iter()
    .find(|a| a.field == "score")
    .unwrap();
  assert_eq!(agg.doc_count, 3);
  assert_eq!(agg.min, Some(1.0));
  assert_eq!(agg.max, Some(3.5));
  assert!(agg.sum.unwrap() - 7.0 < 1e-6);
  assert!(agg.avg.unwrap() - (7.0 / 3.0) < 1e-6);
}
