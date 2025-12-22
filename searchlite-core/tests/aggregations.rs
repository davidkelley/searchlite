use std::collections::BTreeMap;

use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Aggregation, Document, ExecutionStrategy, HistogramAggregation, IndexOptions, MetricAggregation,
  NumericField, RangeAggregation, Schema, SearchRequest, StorageType, TermsAggregation,
};
use searchlite_core::api::Index;
use serde_json::json;

#[test]
fn terms_and_stats_aggregations() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
      name: "tag".into(),
      stored: true,
      indexed: true,
      fast: true,
    });
  schema.numeric_fields.push(NumericField {
    name: "views".into(),
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
  let docs = vec![
    Document {
      fields: [
        ("body".into(), json!("rust systems")),
        ("tag".into(), json!("tech")),
        ("views".into(), json!(10)),
      ]
      .into_iter()
      .collect(),
    },
    Document {
      fields: [
        ("body".into(), json!("rust programming")),
        ("tag".into(), json!("tech")),
        ("views".into(), json!(15)),
      ]
      .into_iter()
      .collect(),
    },
    Document {
      fields: [
        ("body".into(), json!("gardening")),
        ("tag".into(), json!("hobby")),
        ("views".into(), json!(2)),
      ]
      .into_iter()
      .collect(),
    },
  ];
  for doc in docs.iter() {
    writer.add_document(doc).unwrap();
  }
  writer.commit().unwrap();

  let reader = idx.reader().unwrap();
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "tags".to_string(),
    Aggregation::Terms(Box::new(TermsAggregation {
      field: "tag".into(),
      size: Some(5),
      shard_size: None,
      min_doc_count: None,
      missing: None,
      aggs: BTreeMap::new(),
    })),
  );
  aggs.insert(
    "view_stats".to_string(),
    Aggregation::Stats(MetricAggregation {
      field: "views".into(),
      missing: None,
    }),
  );

  let resp = reader
    .search(&SearchRequest {
      query: "rust".to_string(),
      fields: None,
      filters: vec![],
      limit: 0,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      aggs,
    })
    .unwrap();

  let tags = resp.aggregations.get("tags").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Terms { buckets } = tags {
    assert_eq!(buckets[0].key, json!("tech"));
    assert_eq!(buckets[0].doc_count, 2);
  }

  let stats = resp.aggregations.get("view_stats").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Stats(stats) = stats {
    assert_eq!(stats.count, 3);
    assert_eq!(stats.min, 2.0);
    assert_eq!(stats.max, 15.0);
    assert_eq!(stats.sum, 27.0);
  }
}

#[test]
fn aggregation_requires_fast_field() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
      name: "tag".into(),
      stored: true,
      indexed: true,
      fast: false,
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
  let reader = idx.reader().unwrap();
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "tags".into(),
    Aggregation::Terms(Box::new(TermsAggregation {
      field: "tag".into(),
      size: Some(5),
      shard_size: None,
      min_doc_count: None,
      missing: None,
      aggs: BTreeMap::new(),
    })),
  );
  let resp = reader.search(&SearchRequest {
    query: "rust".into(),
    fields: None,
    filters: vec![],
    limit: 0,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    #[cfg(feature = "vectors")]
    vector_query: None,
    return_stored: false,
    highlight_field: None,
    aggs,
  });
  assert!(resp.is_err());
  let msg = resp.err().unwrap().to_string();
  assert!(msg.contains("fast field `tag`"));
}

#[test]
fn histogram_bucket_generation() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "views".into(),
    i64: true,
    fast: true,
    stored: true,
  });
  let idx = Index::create(
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
    for val in [1, 2, 7, 11] {
      writer
        .add_document(&Document {
          fields: [("body".into(), json!("rust")), ("views".into(), json!(val))]
            .into_iter()
            .collect(),
        })
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "views_hist".into(),
    Aggregation::Histogram(Box::new(HistogramAggregation {
      field: "views".into(),
      interval: 5.0,
      offset: None,
      min_doc_count: Some(1),
      extended_bounds: None,
      hard_bounds: None,
      missing: None,
      aggs: BTreeMap::new(),
    })),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filters: vec![],
      limit: 0,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      aggs,
    })
    .unwrap();
  let hist = resp.aggregations.get("views_hist").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Histogram { buckets } = hist {
    assert_eq!(buckets.len(), 3);
    assert_eq!(buckets[0].doc_count, 2);
  }
}

#[test]
fn range_aggregation_counts() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "score".into(),
    i64: true,
    fast: true,
    stored: true,
  });
  let idx = Index::create(
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
    for val in [1, 5, 10, 20] {
      writer
        .add_document(&Document {
          fields: [("body".into(), json!("rust")), ("score".into(), json!(val))]
            .into_iter()
            .collect(),
        })
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "score_ranges".into(),
    Aggregation::Range(Box::new(RangeAggregation {
      field: "score".into(),
      keyed: true,
      ranges: vec![
        searchlite_core::api::types::RangeBound {
          key: Some("low".into()),
          from: None,
          to: Some(5.0),
        },
        searchlite_core::api::types::RangeBound {
          key: Some("mid".into()),
          from: Some(5.0),
          to: Some(15.0),
        },
      ],
      missing: None,
      aggs: BTreeMap::new(),
    })),
  );

  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filters: vec![],
      limit: 0,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      aggs,
    })
    .unwrap();
  let range = resp.aggregations.get("score_ranges").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Range { buckets, .. } = range {
    assert_eq!(buckets[0].doc_count, 2);
    assert_eq!(buckets[1].doc_count, 2);
  }
}
