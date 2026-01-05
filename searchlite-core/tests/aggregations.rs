use std::collections::BTreeMap;

use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Aggregation, DateHistogramAggregation, Document, ExecutionStrategy, HistogramAggregation,
  IndexOptions, MetricAggregation, NumericField, RangeAggregation, Schema, SearchRequest,
  StorageType, TermsAggregation,
};
use searchlite_core::api::Index;
use serde_json::json;

fn doc(id: &str, fields: Vec<(&str, serde_json::Value)>) -> Document {
  let mut map = std::collections::BTreeMap::new();
  map.insert("_id".to_string(), json!(id));
  for (k, v) in fields {
    map.insert(k.to_string(), v);
  }
  Document { fields: map }
}

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
      nullable: false,
    });
  schema.numeric_fields.push(NumericField {
    name: "views".into(),
    i64: true,
    fast: true,
    stored: true,
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
  let docs = [
    doc(
      "agg-1",
      vec![
        ("body", json!("rust systems")),
        ("tag", json!("tech")),
        ("views", json!(10)),
      ],
    ),
    doc(
      "agg-2",
      vec![
        ("body", json!("rust programming")),
        ("tag", json!("tech")),
        ("views", json!(15)),
      ],
    ),
    doc(
      "agg-3",
      vec![
        ("body", json!("gardening")),
        ("tag", json!("hobby")),
        ("views", json!(2)),
      ],
    ),
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
      query: "rust".into(),
      fields: None,
      filter: None,
      filters: vec![],
      limit: 0,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
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
    assert_eq!(stats.count, 2);
    assert_eq!(stats.min, 10.0);
    assert_eq!(stats.max, 15.0);
    assert_eq!(stats.sum, 25.0);
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
    filter: None,
    filters: vec![],
    limit: 0,
    sort: Vec::new(),
    cursor: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    fuzzy: None,
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
    nullable: false,
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
        .add_document(&doc(
          &format!("hist-{val}"),
          vec![("body", json!("rust")), ("views", json!(val))],
        ))
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
      filter: None,
      filters: vec![],
      limit: 0,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
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
fn histogram_uses_floor_for_bucket_boundaries() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "views".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
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
    for val in [0, 4, 5] {
      writer
        .add_document(&doc(
          &format!("hist2-{val}"),
          vec![("body", json!("rust")), ("views", json!(val))],
        ))
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
      min_doc_count: Some(0),
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
      filter: None,
      filters: vec![],
      limit: 0,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      aggs,
    })
    .unwrap();
  let hist = resp.aggregations.get("views_hist").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Histogram { buckets } = hist {
    assert_eq!(buckets.len(), 2);
    assert_eq!(buckets[0].key, json!(0.0));
    assert_eq!(buckets[0].doc_count, 2); // both 0 and 4 land in the first bucket
    assert_eq!(buckets[1].key, json!(5.0));
    assert_eq!(buckets[1].doc_count, 1);
  } else {
    panic!("expected histogram response");
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
    nullable: false,
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
        .add_document(&doc(
          &format!("score-{val}"),
          vec![("body", json!("rust")), ("score", json!(val))],
        ))
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
      filter: None,
      filters: vec![],
      limit: 0,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
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

#[test]
fn date_range_missing_and_keyed() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "ts".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
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
    writer
      .add_document(&doc(
        "date-1",
        vec![("body", json!("rust")), ("ts", json!(1_000))],
      ))
      .unwrap();
    // missing ts should be counted in missing bucket
    writer
      .add_document(&doc("date-missing", vec![("body", json!("rust missing"))]))
      .unwrap();
    writer.commit().unwrap();
  }
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "ranges".into(),
    Aggregation::DateRange(Box::new(
      searchlite_core::api::types::DateRangeAggregation {
        field: "ts".into(),
        keyed: true,
        format: None,
        ranges: vec![
          searchlite_core::api::types::DateRangeBound {
            key: Some("early".into()),
            from: Some("1970-01-01T00:00:00Z".into()),
            to: Some("1970-01-01T00:00:02Z".into()),
          },
          searchlite_core::api::types::DateRangeBound {
            key: Some("late".into()),
            from: Some("1970-01-01T00:00:02Z".into()),
            to: Some("1970-01-01T00:00:03Z".into()),
          },
        ],
        missing: Some(json!("1970-01-01T00:00:01Z")),
        aggs: BTreeMap::new(),
      },
    )),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      filters: vec![],
      limit: 0,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      aggs,
    })
    .unwrap();
  let range = resp.aggregations.get("ranges").unwrap();
  if let searchlite_core::api::types::AggregationResponse::DateRange { buckets, keyed } = range {
    assert!(keyed);
    assert_eq!(buckets.len(), 2);
    assert_eq!(buckets[0].doc_count, 2); // early bucket includes missing
    assert_eq!(buckets[1].doc_count, 0);
  } else {
    panic!("expected date range response");
  }
}

#[test]
fn extended_stats_and_value_count_include_missing() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "score".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
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
    for (idx, val) in [Some(1), Some(2), None].into_iter().enumerate() {
      let mut fields = [("body".into(), json!("rust"))]
        .into_iter()
        .collect::<BTreeMap<_, _>>();
      fields.insert("_id".into(), json!(format!("stats-{idx}")));
      if let Some(v) = val {
        fields.insert("score".into(), json!(v));
      }
      writer.add_document(&Document { fields }).unwrap();
    }
    writer.commit().unwrap();
  }
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "stats".into(),
    Aggregation::ExtendedStats(MetricAggregation {
      field: "score".into(),
      missing: Some(json!(5)),
    }),
  );
  aggs.insert(
    "count".into(),
    Aggregation::ValueCount(MetricAggregation {
      field: "score".into(),
      missing: Some(json!(0)),
    }),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      filters: vec![],
      limit: 1, // ensure aggregations still see all docs
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      aggs,
    })
    .unwrap();
  let stats = resp.aggregations.get("stats").unwrap();
  if let searchlite_core::api::types::AggregationResponse::ExtendedStats(es) = stats {
    assert_eq!(es.count, 3);
    assert_eq!(es.sum, 8.0);
    assert_eq!(es.max, 5.0);
    assert_eq!(es.min, 1.0);
  } else {
    panic!("expected extended stats");
  }
  let count = resp.aggregations.get("count").unwrap();
  if let searchlite_core::api::types::AggregationResponse::ValueCount(vc) = count {
    assert_eq!(vc.value, 3);
  } else {
    panic!("expected value count");
  }
}

#[test]
fn date_histogram_fixed_interval_respects_offset_and_missing() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "ts".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
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
    for ts in [0, 1_000, 1_600] {
      writer
        .add_document(&doc(
          &format!("hist-ts-{ts}"),
          vec![("body", json!("rust")), ("ts", json!(ts))],
        ))
        .unwrap();
    }
    // one doc missing ts to exercise "missing"
    writer
      .add_document(&doc(
        "hist-ts-missing",
        vec![("body", json!("rust missing ts"))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hist".into(),
    Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
      field: "ts".into(),
      calendar_interval: None,
      fixed_interval: Some("1s".into()),
      offset: Some("0.5s".into()),
      format: None,
      min_doc_count: Some(0),
      extended_bounds: Some(searchlite_core::api::types::DateHistogramBounds {
        min: "1970-01-01T00:00:00Z".into(),
        max: "1970-01-01T00:00:03Z".into(),
      }),
      hard_bounds: None,
      missing: Some("500".into()),
      aggs: BTreeMap::new(),
    })),
  );

  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      filters: vec![],
      limit: 0,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      aggs,
    })
    .unwrap();

  let hist = resp.aggregations.get("hist").unwrap();
  if let searchlite_core::api::types::AggregationResponse::DateHistogram { buckets } = hist {
    let keys: Vec<_> = buckets.iter().map(|b| b.key.clone()).collect();
    assert_eq!(
      keys,
      vec![json!(500), json!(1500), json!(2500), json!(3500)]
    );
    assert_eq!(buckets[0].doc_count, 2); // ts=0 and missing->500
    assert_eq!(buckets[1].doc_count, 1);
    assert_eq!(buckets[2].doc_count, 1);
    assert_eq!(buckets[3].doc_count, 0);
  } else {
    panic!("expected date histogram response");
  }
}

#[test]
fn date_histogram_hard_bounds_filter_out_of_range() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "ts".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
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
    // within bounds
    writer
      .add_document(&doc(
        "hard-1",
        vec![("body", json!("rust")), ("ts", json!(1_000))],
      ))
      .unwrap();
    // below hard bounds
    writer
      .add_document(&doc(
        "hard-0",
        vec![("body", json!("rust")), ("ts", json!(0))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hist".into(),
    Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
      field: "ts".into(),
      calendar_interval: None,
      fixed_interval: Some("1s".into()),
      offset: None,
      format: None,
      min_doc_count: Some(0),
      extended_bounds: None,
      hard_bounds: Some(searchlite_core::api::types::DateHistogramBounds {
        min: "1970-01-01T00:00:01Z".into(),
        max: "1970-01-01T00:00:02Z".into(),
      }),
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
      filter: None,
      filters: vec![],
      limit: 0,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      aggs,
    })
    .unwrap();
  let hist = resp.aggregations.get("hist").unwrap();
  if let searchlite_core::api::types::AggregationResponse::DateHistogram { buckets } = hist {
    let keys: Vec<_> = buckets.iter().map(|b| b.key.clone()).collect();
    assert_eq!(keys, vec![json!(1_000), json!(2_000)]);
    assert_eq!(buckets[0].doc_count, 1);
    assert_eq!(buckets[1].doc_count, 0);
  } else {
    panic!("expected date histogram response");
  }
}

#[test]
fn terms_size_applied_after_merge() {
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
    for i in 0..2 {
      writer
        .add_document(&doc(
          &format!("t-a-{i}"),
          vec![("body", json!("rust")), ("tag", json!("a"))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }
  {
    let mut writer = idx.writer().unwrap();
    for i in 0..4 {
      writer
        .add_document(&doc(
          &format!("t-b-{i}"),
          vec![("body", json!("rust")), ("tag", json!("b"))],
        ))
        .unwrap();
    }
    writer
      .add_document(&doc(
        "t-a-last",
        vec![("body", json!("rust")), ("tag", json!("a"))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "tags".into(),
    Aggregation::Terms(Box::new(TermsAggregation {
      field: "tag".into(),
      size: Some(1),
      shard_size: None,
      min_doc_count: None,
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
      filter: None,
      filters: vec![],
      limit: 0,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      aggs,
    })
    .unwrap();

  let agg = resp.aggregations.get("tags").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Terms { buckets } = agg {
    assert_eq!(buckets.len(), 1);
    assert_eq!(buckets[0].key, json!("b"));
    assert_eq!(buckets[0].doc_count, 4);
  } else {
    panic!("expected terms response");
  }
}
