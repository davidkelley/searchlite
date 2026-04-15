use std::collections::BTreeMap;

use chrono::DateTime;
use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Aggregation, CompositeAggregation, CompositeSource, DateHistogramAggregation,
  DateHistogramBounds, Document, ExecutionStrategy, HistogramAggregation, HistogramBounds,
  IndexOptions, KeywordField, MetricAggregation, NumericField, Schema, SearchRequest, SortOrder,
  SortSpec, StorageType, TermsAggregation, TopHitsAggregation,
};
use searchlite_core::api::Index;
use serde_json::json;

fn build_base_options(path: &std::path::Path) -> IndexOptions {
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

fn doc(id: &str, fields: Vec<(&str, serde_json::Value)>) -> Document {
  let mut map = BTreeMap::new();
  map.insert("_id".to_string(), json!(id));
  for (k, v) in fields {
    map.insert(k.to_string(), v);
  }
  Document { fields: map }
}

#[test]
fn histogram_respects_extended_bounds_and_empty_buckets() {
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
  let idx = Index::create(&path, schema, build_base_options(&path)).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for val in [5, 15] {
      writer
        .add_document(&doc(
          &format!("hist-{val}"),
          vec![("body", json!("rust")), ("score", json!(val))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hist".into(),
    Aggregation::Histogram(Box::new(HistogramAggregation {
      field: "score".into(),
      interval: 10.0,
      offset: None,
      min_doc_count: None,
      extended_bounds: Some(HistogramBounds {
        min: 0.0,
        max: 30.0,
      }),
      hard_bounds: None,
      missing: None,
      sampling: None,
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
      limit: 1,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
      cursor: None,
      search_after: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      track_total_hits: None,
      #[cfg(feature = "vectors")]
      vector_query: None,

      #[cfg(feature = "vectors")]
      vector_filter: None,
      return_stored: false,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();

  let hist = resp.aggregations.get("hist").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Histogram { buckets, .. } = hist {
    let keys: Vec<_> = buckets.iter().map(|b| b.key.clone()).collect();
    assert_eq!(
      keys,
      vec![json!(0.0), json!(10.0), json!(20.0), json!(30.0)]
    );
    assert_eq!(buckets[0].doc_count, 1);
    assert_eq!(buckets[1].doc_count, 1);
    assert_eq!(buckets[2].doc_count, 0);
    assert_eq!(buckets[3].doc_count, 0);
  } else {
    panic!("unexpected histogram response");
  }
}

/// End-to-end coverage for the validator-permitted configuration where
/// `extended_bounds` is nested inside `hard_bounds`. The `collect()` path
/// still applies the `hard_bounds` gate per-document, and the fill path
/// populates empty buckets across `extended_bounds`. The direct BUG-188
/// clipping invariant (when `extended_bounds` extends past `hard_bounds`) is
/// pinned by the `intersect_fill_range_*` unit tests in
/// `searchlite-core/src/query/aggs/mod.rs` — that case is unreachable from
/// here because the request validator rejects it up front.
#[test]
fn histogram_nested_bounds_produce_expected_buckets() {
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
  let idx = Index::create(&path, schema, build_base_options(&path)).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    // 25 is inside hard_bounds [20, 80] but outside extended_bounds [30, 70]
    // — it seeds a collected bucket at key 20 that sits outside the fill
    // window. 60 is inside both. 95 is past hard_bounds.max and must be
    // discarded by the collect-path gate.
    for val in [25_i64, 60, 95] {
      writer
        .add_document(&doc(
          &format!("hist-{val}"),
          vec![("body", json!("rust")), ("score", json!(val))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hist".into(),
    Aggregation::Histogram(Box::new(HistogramAggregation {
      field: "score".into(),
      interval: 10.0,
      offset: None,
      min_doc_count: Some(0),
      extended_bounds: Some(HistogramBounds {
        min: 30.0,
        max: 70.0,
      }),
      hard_bounds: Some(HistogramBounds {
        min: 20.0,
        max: 80.0,
      }),
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );

  let mut req = SearchRequest::new("rust");
  req.aggs = aggs;
  let resp = idx.reader().unwrap().search(&req).unwrap();

  let hist = resp.aggregations.get("hist").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Histogram { buckets, .. } = hist {
    let keys: Vec<_> = buckets.iter().map(|b| b.key.clone()).collect();
    // Collected bucket at 20 (from the score=25 doc), the collected bucket
    // at 60 (from score=60), and empty-fill buckets across
    // `extended_bounds` [30, 70]. No bucket at key 80/90 must appear
    // because the score=95 doc was dropped by the hard_bounds gate and
    // the fill range never reaches into that grid cell.
    assert_eq!(
      keys,
      vec![
        json!(20.0),
        json!(30.0),
        json!(40.0),
        json!(50.0),
        json!(60.0),
        json!(70.0),
      ]
    );
    assert_eq!(buckets[0].doc_count, 1); // key 20 — collected from score=25
    assert_eq!(buckets[1].doc_count, 0); // key 30 — empty fill
    assert_eq!(buckets[2].doc_count, 0); // key 40 — empty fill
    assert_eq!(buckets[3].doc_count, 0); // key 50 — empty fill
    assert_eq!(buckets[4].doc_count, 1); // key 60 — collected from score=60
    assert_eq!(buckets[5].doc_count, 0); // key 70 — empty fill
  } else {
    panic!("unexpected histogram response");
  }
}

#[test]
fn histogram_requires_positive_interval() {
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
  let idx = Index::create(&path, schema, build_base_options(&path)).unwrap();

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hist".into(),
    Aggregation::Histogram(Box::new(HistogramAggregation {
      field: "score".into(),
      interval: -10.0,
      offset: None,
      min_doc_count: None,
      extended_bounds: None,
      hard_bounds: None,
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );

  let resp = idx.reader().unwrap().search(&SearchRequest {
    query: "rust".into(),
    fields: None,
    filter: None,
    limit: 1,
    from: 0,
    return_hits: true,
    candidate_size: None,
    #[cfg(feature = "vectors")]
    max_global_vector_candidates: None,
    sort: Vec::new(),
    cursor: None,
    search_after: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    fuzzy: None,
    track_total_hits: None,
    #[cfg(feature = "vectors")]
    vector_query: None,

    #[cfg(feature = "vectors")]
    vector_filter: None,
    return_stored: false,
    highlight_field: None,
    highlight: None,
    collapse: None,
    aggs,
    suggest: BTreeMap::new(),
    rescore: None,
    explain: false,
    profile: false,
  });
  assert!(resp.is_err());
  let msg = resp.err().unwrap().to_string();
  assert!(
    msg.contains("finite positive number"),
    "expected error to mention finite positive interval, got: {msg}"
  );
}

#[test]
fn nested_terms_stats_aggregation() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.keyword_fields.push(KeywordField {
    name: "lang".into(),
    stored: true,
    indexed: true,
    fast: true,
    nullable: false,
  });
  schema.numeric_fields.push(NumericField {
    name: "stars".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
  });
  let idx = IndexBuilder::create(&path, schema, build_base_options(&path)).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    let docs = vec![("rust", 10), ("rust", 8), ("go", 7)];
    for (idx, (lang, stars)) in docs.into_iter().enumerate() {
      writer
        .add_document(&doc(
          &format!("agg-l-{idx}"),
          vec![
            ("body", json!("systems")),
            ("lang", json!(lang)),
            ("stars", json!(stars)),
          ],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let mut sub_aggs = BTreeMap::new();
  sub_aggs.insert(
    "stars".into(),
    Aggregation::Stats(MetricAggregation {
      field: "stars".into(),
      missing: None,
    }),
  );

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "langs".into(),
    Aggregation::Terms(Box::new(TermsAggregation {
      field: "lang".into(),
      size: Some(10),
      shard_size: None,
      min_doc_count: None,
      missing: None,
      sampling: None,
      aggs: sub_aggs,
    })),
  );

  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "systems".into(),
      fields: None,
      filter: None,
      limit: 1,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
      cursor: None,
      search_after: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      track_total_hits: None,
      #[cfg(feature = "vectors")]
      vector_query: None,

      #[cfg(feature = "vectors")]
      vector_filter: None,
      return_stored: false,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();

  let terms = resp.aggregations.get("langs").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Terms { buckets, .. } = terms {
    let rust_bucket = buckets
      .iter()
      .find(|b| b.key == json!("rust"))
      .expect("rust bucket");
    assert_eq!(rust_bucket.doc_count, 2);
    let stats = rust_bucket
      .aggregations
      .get("stars")
      .and_then(|agg| {
        if let searchlite_core::api::types::AggregationResponse::Stats(stats) = agg {
          Some(stats)
        } else {
          None
        }
      })
      .expect("stats sub-aggregation");
    assert_eq!(stats.count, 2);
    assert_eq!(stats.sum, 18.0);
    assert_eq!(stats.max, 10.0);
  } else {
    panic!("unexpected terms response");
  }
}

#[test]
fn date_histogram_rejects_invalid_config() {
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
  let idx = IndexBuilder::create(&path, schema, build_base_options(&path)).unwrap();

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hist".into(),
    Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
      field: "ts".into(),
      calendar_interval: Some("fortnight".into()),
      fixed_interval: None,
      offset: Some("bogus".into()),
      format: None,
      min_doc_count: None,
      extended_bounds: None,
      hard_bounds: None,
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );

  let resp = idx.reader().unwrap().search(&SearchRequest {
    query: "rust".into(),
    fields: None,
    filter: None,
    limit: 1,
    from: 0,
    return_hits: true,
    candidate_size: None,
    #[cfg(feature = "vectors")]
    max_global_vector_candidates: None,
    sort: Vec::new(),
    cursor: None,
    search_after: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    fuzzy: None,
    track_total_hits: None,
    #[cfg(feature = "vectors")]
    vector_query: None,

    #[cfg(feature = "vectors")]
    vector_filter: None,
    return_stored: false,
    highlight_field: None,
    highlight: None,
    collapse: None,
    aggs,
    suggest: BTreeMap::new(),
    rescore: None,
    explain: false,
    profile: false,
  });
  assert!(resp.is_err());
  let msg = resp.err().unwrap().to_string();
  assert!(msg.contains("calendar_interval"));

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hist".into(),
    Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
      field: "ts".into(),
      calendar_interval: Some("day".into()),
      fixed_interval: None,
      offset: None,
      format: None,
      min_doc_count: None,
      extended_bounds: Some(DateHistogramBounds {
        min: "2024-01-03T00:00:00Z".into(),
        max: "2024-01-02T00:00:00Z".into(),
      }),
      hard_bounds: Some(DateHistogramBounds {
        min: "2024-01-05T00:00:00Z".into(),
        max: "2024-01-01T00:00:00Z".into(),
      }),
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );

  let resp = idx.reader().unwrap().search(&SearchRequest {
    query: "rust".into(),
    fields: None,
    filter: None,
    limit: 1,
    from: 0,
    return_hits: true,
    candidate_size: None,
    #[cfg(feature = "vectors")]
    max_global_vector_candidates: None,
    sort: Vec::new(),
    cursor: None,
    search_after: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    fuzzy: None,
    track_total_hits: None,
    #[cfg(feature = "vectors")]
    vector_query: None,

    #[cfg(feature = "vectors")]
    vector_filter: None,
    return_stored: false,
    highlight_field: None,
    highlight: None,
    collapse: None,
    aggs,
    suggest: BTreeMap::new(),
    rescore: None,
    explain: false,
    profile: false,
  });
  assert!(resp.is_err());
  let msg = resp.err().unwrap().to_string();
  assert!(msg.contains("extended_bounds") || msg.contains("hard_bounds"));
}

#[test]
fn top_hits_returns_requested_docs() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.keyword_fields.push(KeywordField {
    name: "tag".into(),
    stored: true,
    indexed: true,
    fast: false,
    nullable: false,
  });
  let idx = IndexBuilder::create(&path, schema, build_base_options(&path)).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for i in 0..4 {
      writer
        .add_document(&doc(
          &format!("top-{i}"),
          vec![("body", json!(format!("rust {i}"))), ("tag", json!("dev"))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hits".into(),
    Aggregation::TopHits(TopHitsAggregation {
      size: 2,
      from: 0,
      fields: Some(vec!["tag".into()]),
      sort: Vec::new(),
      highlight_field: Some("body".into()),
    }),
  );

  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
      cursor: None,
      search_after: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      track_total_hits: None,
      #[cfg(feature = "vectors")]
      vector_query: None,

      #[cfg(feature = "vectors")]
      vector_filter: None,
      return_stored: false,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();

  let agg = resp.aggregations.get("hits").unwrap();
  if let searchlite_core::api::types::AggregationResponse::TopHits(top_hits) = agg {
    assert_eq!(top_hits.total, 4);
    assert_eq!(top_hits.hits.len(), 2);
    assert!(top_hits.hits[0].score.is_some());
    assert!(top_hits.hits.iter().all(|h| h.fields.is_some()));
    // fields projection and snippet should be present
    assert!(top_hits
      .hits
      .iter()
      .all(|h| h.fields.as_ref().unwrap().get("tag").is_some()));
    assert!(top_hits.hits.iter().all(|h| h.snippet.is_some()));
  } else {
    panic!("expected top hits response");
  }
}

#[test]
fn top_hits_applies_sort_spec() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "priority".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
  });
  let idx = IndexBuilder::create(&path, schema, build_base_options(&path)).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for priority in [2, 5] {
      writer
        .add_document(&doc(
          &format!("priority-{priority}"),
          vec![
            ("body", json!("rust systems")),
            ("priority", json!(priority)),
          ],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
    writer
      .add_document(&doc(
        "priority-1",
        vec![("body", json!("rust systems")), ("priority", json!(1))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hits".into(),
    Aggregation::TopHits(TopHitsAggregation {
      size: 3,
      from: 0,
      fields: Some(vec!["priority".into()]),
      sort: vec![SortSpec {
        field: "priority".into(),
        order: Some(SortOrder::Asc),
      }],
      highlight_field: None,
    }),
  );

  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
      cursor: None,
      search_after: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      track_total_hits: None,
      #[cfg(feature = "vectors")]
      vector_query: None,

      #[cfg(feature = "vectors")]
      vector_filter: None,
      return_stored: false,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();

  let agg = resp.aggregations.get("hits").unwrap();
  if let searchlite_core::api::types::AggregationResponse::TopHits(top_hits) = agg {
    assert_eq!(top_hits.total, 3);
    let priorities: Vec<_> = top_hits
      .hits
      .iter()
      .map(|h| {
        h.fields
          .as_ref()
          .and_then(|f| f.get("priority"))
          .and_then(|v| v.as_i64())
          .unwrap()
      })
      .collect();
    assert_eq!(priorities, vec![1, 2, 5]);
    assert!(top_hits.hits.iter().all(|h| h.score.is_some()));
  } else {
    panic!("expected top hits response");
  }
}

#[test]
fn date_histogram_calendar_month_interval() {
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
  let idx = IndexBuilder::create(&path, schema, build_base_options(&path)).unwrap();

  let ts = |s: &str| DateTime::parse_from_rfc3339(s).unwrap().timestamp_millis();
  {
    let mut writer = idx.writer().unwrap();
    for t in [
      "2024-01-02T00:00:00Z",
      "2024-01-15T12:00:00Z",
      "2024-02-05T00:00:00Z",
    ] {
      writer
        .add_document(&doc(
          &format!("ts-{t}"),
          vec![("body", json!("rust")), ("ts", json!(ts(t)))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "dates".into(),
    Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
      field: "ts".into(),
      calendar_interval: Some("month".into()),
      fixed_interval: None,
      offset: None,
      format: None,
      min_doc_count: Some(0),
      extended_bounds: Some(DateHistogramBounds {
        min: "2024-01-01T00:00:00Z".into(),
        max: "2024-03-01T00:00:00Z".into(),
      }),
      hard_bounds: None,
      missing: None,
      sampling: None,
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
      limit: 1,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
      cursor: None,
      search_after: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      track_total_hits: None,
      #[cfg(feature = "vectors")]
      vector_query: None,

      #[cfg(feature = "vectors")]
      vector_filter: None,
      return_stored: false,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();

  let agg = resp.aggregations.get("dates").unwrap();
  if let searchlite_core::api::types::AggregationResponse::DateHistogram { buckets, .. } = agg {
    let keys: Vec<_> = buckets.iter().map(|b| b.key.clone()).collect();
    assert_eq!(
      keys,
      vec![
        json!(ts("2024-01-01T00:00:00Z")),
        json!(ts("2024-02-01T00:00:00Z")),
        json!(ts("2024-03-01T00:00:00Z"))
      ]
    );
    assert_eq!(buckets[0].doc_count, 2);
    assert_eq!(buckets[1].doc_count, 1);
    assert_eq!(buckets[2].doc_count, 0);
  } else {
    panic!("expected date histogram response");
  }
}

/// End-to-end coverage for `DateHistogramCollector` with `extended_bounds`
/// nested inside `hard_bounds` — the configuration the validator permits.
/// Like the numeric counterpart, the direct BUG-188 clipping invariant is
/// exercised by the `intersect_fill_range_*` unit tests; this test pins the
/// search-path contract that the `hard_bounds` gate still drops
/// out-of-range docs and the fill range stays inside `extended_bounds`.
#[test]
fn date_histogram_nested_bounds_produce_expected_buckets() {
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
  let idx = IndexBuilder::create(&path, schema, build_base_options(&path)).unwrap();

  let ts = |s: &str| DateTime::parse_from_rfc3339(s).unwrap().timestamp_millis();
  {
    let mut writer = idx.writer().unwrap();
    // Jan 10 is inside hard_bounds [Jan 1, May 1] but outside extended_bounds
    // [Feb 1, Apr 1] — it seeds a collected Jan 1 bucket outside the fill
    // window. Feb 5 / Mar 10 fall inside both. Jun 1 is past hard_bounds.max
    // and must be dropped by the collect-path gate.
    for t in [
      "2024-01-10T00:00:00Z",
      "2024-02-05T00:00:00Z",
      "2024-03-10T00:00:00Z",
      "2024-06-01T00:00:00Z",
    ] {
      writer
        .add_document(&doc(
          &format!("ts-{t}"),
          vec![("body", json!("rust")), ("ts", json!(ts(t)))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "dates".into(),
    Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
      field: "ts".into(),
      calendar_interval: Some("month".into()),
      fixed_interval: None,
      offset: None,
      format: None,
      min_doc_count: Some(0),
      extended_bounds: Some(DateHistogramBounds {
        min: "2024-02-01T00:00:00Z".into(),
        max: "2024-04-01T00:00:00Z".into(),
      }),
      hard_bounds: Some(DateHistogramBounds {
        min: "2024-01-01T00:00:00Z".into(),
        max: "2024-05-01T00:00:00Z".into(),
      }),
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );

  let mut req = SearchRequest::new("rust");
  req.aggs = aggs;
  let resp = idx.reader().unwrap().search(&req).unwrap();

  let agg = resp.aggregations.get("dates").unwrap();
  if let searchlite_core::api::types::AggregationResponse::DateHistogram { buckets, .. } = agg {
    let keys: Vec<_> = buckets.iter().map(|b| b.key.clone()).collect();
    // Collected Jan 1 (from the Jan 10 doc), Feb 1 + Mar 1 (from Feb 5 /
    // Mar 10), and an empty Apr 1 from the fill. No bucket for May 1, and
    // nothing for Jun 1 — the latter was dropped by the hard_bounds gate.
    assert_eq!(
      keys,
      vec![
        json!(ts("2024-01-01T00:00:00Z")),
        json!(ts("2024-02-01T00:00:00Z")),
        json!(ts("2024-03-01T00:00:00Z")),
        json!(ts("2024-04-01T00:00:00Z")),
      ]
    );
    assert_eq!(buckets[0].doc_count, 1);
    assert_eq!(buckets[1].doc_count, 1);
    assert_eq!(buckets[2].doc_count, 1);
    assert_eq!(buckets[3].doc_count, 0);
  } else {
    panic!("expected date histogram response");
  }
}

/// Regression tests for BUG-027 — `HistogramAggregation` with a degenerate
/// (zero / NaN / infinite) `interval` combined with `extended_bounds` or
/// `hard_bounds` previously drove `HistogramCollector::finish` into an
/// unbounded bucket-insertion loop that exhausted memory.
mod bug_027 {
  use super::*;

  fn numeric_score_index(path: &std::path::Path) -> searchlite_core::api::Index {
    let mut schema = Schema::default_text_body();
    schema.numeric_fields.push(NumericField {
      name: "score".into(),
      i64: true,
      fast: true,
      stored: true,
      nullable: false,
    });
    let idx = Index::create(path, schema, build_base_options(path)).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&doc(
          "doc-1",
          vec![("body", json!("rust")), ("score", json!(1))],
        ))
        .unwrap();
      writer.commit().unwrap();
    }
    idx
  }

  fn search_with_agg(
    idx: &searchlite_core::api::Index,
    aggs: BTreeMap<String, Aggregation>,
  ) -> anyhow::Result<()> {
    let mut req = SearchRequest::new("rust");
    req.aggs = aggs;
    idx.reader().unwrap().search(&req)?;
    Ok(())
  }

  fn assert_invalid_histogram(err: anyhow::Error, expected_substr: &str) {
    let msg = err.to_string();
    assert!(
      msg.contains(expected_substr),
      "expected error message to contain {expected_substr:?}, got: {msg}"
    );
  }

  #[test]
  fn zero_interval_with_extended_bounds_is_rejected_without_infinite_loop() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = numeric_score_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "h".into(),
      Aggregation::Histogram(Box::new(HistogramAggregation {
        field: "score".into(),
        interval: 0.0,
        offset: None,
        min_doc_count: None,
        extended_bounds: Some(HistogramBounds {
          min: 0.0,
          max: 100.0,
        }),
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs).expect_err("zero interval must be rejected");
    assert_invalid_histogram(err, "finite positive number");
  }

  #[test]
  fn nan_interval_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = numeric_score_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "h".into(),
      Aggregation::Histogram(Box::new(HistogramAggregation {
        field: "score".into(),
        interval: f64::NAN,
        offset: None,
        min_doc_count: None,
        extended_bounds: None,
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs).expect_err("NaN interval must be rejected");
    assert_invalid_histogram(err, "finite positive number");
  }

  #[test]
  fn positive_infinite_interval_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = numeric_score_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "h".into(),
      Aggregation::Histogram(Box::new(HistogramAggregation {
        field: "score".into(),
        interval: f64::INFINITY,
        offset: None,
        min_doc_count: None,
        extended_bounds: None,
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs).expect_err("infinite interval must be rejected");
    assert_invalid_histogram(err, "finite positive number");
  }

  #[test]
  fn non_finite_bounds_are_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = numeric_score_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "h".into(),
      Aggregation::Histogram(Box::new(HistogramAggregation {
        field: "score".into(),
        interval: 10.0,
        offset: None,
        min_doc_count: None,
        extended_bounds: Some(HistogramBounds {
          min: 0.0,
          max: f64::INFINITY,
        }),
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs).expect_err("non-finite bounds must be rejected");
    assert_invalid_histogram(err, "finite");
  }

  #[test]
  fn bounds_span_exceeding_bucket_cap_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = numeric_score_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "h".into(),
      Aggregation::Histogram(Box::new(HistogramAggregation {
        field: "score".into(),
        // 1_000_000 / 1.0 = 1_000_000 buckets — well above the cap.
        interval: 1.0,
        offset: None,
        min_doc_count: None,
        extended_bounds: Some(HistogramBounds {
          min: 0.0,
          max: 1_000_000.0,
        }),
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs).expect_err("excessive bounds span must be rejected");
    assert_invalid_histogram(err, "too many empty buckets");
  }

  /// Codex P1 regression: validation and the runtime `MAX_BUCKETS` cap must
  /// agree, otherwise a request sitting between the two caps would pass
  /// validation and then silently truncate at collection time. 15_000 buckets
  /// is above the 10_000 runtime cap but below the previous (draft) 65_536
  /// validation cap — it must be rejected, not truncated.
  #[test]
  fn bounds_span_between_runtime_cap_and_draft_cap_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = numeric_score_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "h".into(),
      Aggregation::Histogram(Box::new(HistogramAggregation {
        field: "score".into(),
        interval: 1.0,
        offset: None,
        min_doc_count: None,
        extended_bounds: Some(HistogramBounds {
          min: 0.0,
          max: 15_000.0,
        }),
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs)
      .expect_err("bounds span above the runtime cap must be rejected, not silently truncated");
    assert_invalid_histogram(err, "too many empty buckets");
  }

  /// Copilot regression: the bucket-count check must mirror the collector's
  /// `bucket_key(min)..=bucket_key(max)` (inclusive) formula so the
  /// fence-post bucket is not double-counted as "within cap". With
  /// interval=1.0 and bounds=[0.0, 10000.0], the collector would materialize
  /// 10_001 buckets; a naïve `(max-min)/interval = 10_000` check would
  /// (incorrectly) allow it.
  #[test]
  fn bounds_span_fence_post_respects_cap() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = numeric_score_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "h".into(),
      Aggregation::Histogram(Box::new(HistogramAggregation {
        field: "score".into(),
        interval: 1.0,
        offset: None,
        min_doc_count: None,
        extended_bounds: Some(HistogramBounds {
          min: 0.0,
          max: 10_000.0,
        }),
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs)
      .expect_err("10_001 inclusive buckets must be rejected (fence-post), not accepted as 10_000");
    assert_invalid_histogram(err, "too many empty buckets");
  }

  /// Copilot regression: `offset` shifts the bucket grid, which the old
  /// formula ignored. This test locks in that a request sitting exactly at
  /// the cap (accounting for offset) is still accepted.
  #[test]
  fn bounds_span_with_offset_at_cap_boundary_is_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = numeric_score_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "h".into(),
      Aggregation::Histogram(Box::new(HistogramAggregation {
        field: "score".into(),
        interval: 1.0,
        // floor((0 - 0.5)/1) = -1, floor((9998.5 - 0.5)/1) = 9998,
        // span = 9998 - (-1) + 1 = 10_000 — exactly at the cap.
        offset: Some(0.5),
        min_doc_count: None,
        extended_bounds: Some(HistogramBounds {
          min: 0.0,
          max: 9998.5,
        }),
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    search_with_agg(&idx, aggs).expect("bounds span exactly at cap must be accepted");
  }

  /// Guard against a non-finite `offset` slipping through — otherwise the
  /// bucket-count computation becomes NaN and the cap check becomes
  /// meaningless.
  #[test]
  fn non_finite_offset_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = numeric_score_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "h".into(),
      Aggregation::Histogram(Box::new(HistogramAggregation {
        field: "score".into(),
        interval: 10.0,
        offset: Some(f64::NAN),
        min_doc_count: None,
        extended_bounds: None,
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs).expect_err("NaN offset must be rejected");
    assert_invalid_histogram(err, "offset");
  }

  #[test]
  fn composite_histogram_source_rejects_zero_interval() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = numeric_score_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "c".into(),
      Aggregation::Composite(Box::new(CompositeAggregation {
        sources: vec![CompositeSource::Histogram {
          name: "score_buckets".into(),
          field: "score".into(),
          interval: 0.0,
        }],
        size: 10,
        after: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs)
      .expect_err("composite histogram source with zero interval must be rejected");
    assert_invalid_histogram(err, "finite positive number");
  }

  #[test]
  fn composite_histogram_source_rejects_nan_interval() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = numeric_score_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "c".into(),
      Aggregation::Composite(Box::new(CompositeAggregation {
        sources: vec![CompositeSource::Histogram {
          name: "score_buckets".into(),
          field: "score".into(),
          interval: f64::NAN,
        }],
        size: 10,
        after: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs)
      .expect_err("composite histogram source with NaN interval must be rejected");
    assert_invalid_histogram(err, "finite positive number");
  }
}

/// Regression tests for BUG-030 (#186) — `bucket_start` for
/// `DateInterval::Fixed` used `.ceil()` instead of `.floor()`, causing every
/// timestamp that did not fall exactly on a bucket boundary to be mis-assigned
/// to the *next* bucket. A noon timestamp was placed in the following day's
/// bucket; a timestamp 1ms past midnight was placed in the day-after-next's
/// bucket.
mod bug_030 {
  use super::*;

  fn timestamp_index(path: &std::path::Path) -> searchlite_core::api::Index {
    let mut schema = Schema::default_text_body();
    schema.numeric_fields.push(NumericField {
      name: "ts".into(),
      i64: true,
      fast: true,
      stored: true,
      nullable: false,
    });
    Index::create(path, schema, build_base_options(path)).unwrap()
  }

  fn run_date_histogram(
    idx: &searchlite_core::api::Index,
    aggs: BTreeMap<String, Aggregation>,
  ) -> searchlite_core::api::types::AggregationResponse {
    let resp = idx
      .reader()
      .unwrap()
      .search(&SearchRequest {
        query: "rust".into(),
        fields: None,
        filter: None,
        limit: 1,
        from: 0,
        return_hits: true,
        candidate_size: None,
        #[cfg(feature = "vectors")]
        max_global_vector_candidates: None,
        sort: Vec::new(),
        cursor: None,
        search_after: None,
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        fuzzy: None,
        track_total_hits: None,
        #[cfg(feature = "vectors")]
        vector_query: None,
        #[cfg(feature = "vectors")]
        vector_filter: None,
        return_stored: false,
        highlight_field: None,
        highlight: None,
        collapse: None,
        aggs,
        suggest: BTreeMap::new(),
        rescore: None,
        explain: false,
        profile: false,
      })
      .unwrap();
    resp.aggregations.get("hist").cloned().unwrap()
  }

  const DAY_MS: i64 = 86_400_000;

  /// A timestamp at noon on day 1 must land in day 1's bucket, not day 2's.
  /// Timestamp 1ms past midnight on day 2 must land in day 2's bucket, not
  /// day 3's. Boundary timestamps (exactly on midnight) stay put.
  #[test]
  fn fixed_interval_places_non_boundary_timestamps_in_current_bucket() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = timestamp_index(tmp.path());
    {
      let mut writer = idx.writer().unwrap();
      for (id, ts) in [
        ("day1-midnight", 0_i64),
        ("day1-noon", DAY_MS / 2),
        ("day2-midnight", DAY_MS),
        ("day2-1ms", DAY_MS + 1),
        ("day2-noon", DAY_MS + DAY_MS / 2),
      ] {
        writer
          .add_document(&doc(id, vec![("body", json!("rust")), ("ts", json!(ts))]))
          .unwrap();
      }
      writer.commit().unwrap();
    }

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "hist".into(),
      Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
        field: "ts".into(),
        calendar_interval: None,
        fixed_interval: Some("1d".into()),
        offset: None,
        format: None,
        min_doc_count: Some(0),
        extended_bounds: None,
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let agg = run_date_histogram(&idx, aggs);
    if let searchlite_core::api::types::AggregationResponse::DateHistogram { buckets, .. } = agg {
      let observed: Vec<(serde_json::Value, u64)> = buckets
        .iter()
        .map(|b| (b.key.clone(), b.doc_count))
        .collect();
      assert_eq!(
        observed,
        vec![
          // day 1: midnight + noon
          (json!(0), 2),
          // day 2: midnight + 1ms + noon
          (json!(DAY_MS), 3),
        ]
      );
    } else {
      panic!("expected date histogram response");
    }
  }

  /// A non-zero offset must still place non-boundary timestamps into the
  /// current bucket. With `offset = 500ms` and `interval = 1s`, bucket
  /// boundaries are at `..., -500, 500, 1500, 2500, ...`. A timestamp of
  /// `1000ms` belongs in the `[500, 1500)` bucket (key = 500), not the
  /// `[1500, 2500)` bucket.
  #[test]
  fn fixed_interval_with_offset_floors_rather_than_ceils() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = timestamp_index(tmp.path());
    {
      let mut writer = idx.writer().unwrap();
      // ts=600ms falls inside [500, 1500); ts=1400ms also falls inside
      // [500, 1500). ts=1600ms falls inside [1500, 2500).
      for (id, ts) in [("a", 600_i64), ("b", 1_400_i64), ("c", 1_600_i64)] {
        writer
          .add_document(&doc(id, vec![("body", json!("rust")), ("ts", json!(ts))]))
          .unwrap();
      }
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
        min_doc_count: Some(1),
        extended_bounds: None,
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let agg = run_date_histogram(&idx, aggs);
    if let searchlite_core::api::types::AggregationResponse::DateHistogram { buckets, .. } = agg {
      let observed: Vec<(serde_json::Value, u64)> = buckets
        .iter()
        .map(|b| (b.key.clone(), b.doc_count))
        .collect();
      assert_eq!(observed, vec![(json!(500), 2), (json!(1500), 1)]);
    } else {
      panic!("expected date histogram response");
    }
  }

  /// `extended_bounds` drives empty-bucket fill via `bucket_start`. With the
  /// previous `.ceil()` implementation, the bounds themselves were rounded
  /// up, so fills spilled into the bucket *beyond* `max`. With `.floor()`
  /// the range is clipped to the enclosing bucket keys, as expected.
  #[test]
  fn extended_bounds_fill_range_uses_floor() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = timestamp_index(tmp.path());
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&doc(
          "seed",
          vec![("body", json!("rust")), ("ts", json!(DAY_MS / 2))],
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
        fixed_interval: Some("1d".into()),
        offset: None,
        format: None,
        min_doc_count: Some(0),
        extended_bounds: Some(DateHistogramBounds {
          min: "1970-01-01T00:00:00Z".into(),
          // Noon on day 3. With `.floor()` this clamps to `bucket_start =
          // 2 * DAY_MS` (day 3's bucket key). With the old `.ceil()` it
          // rounded up to day 4's key (`3 * DAY_MS`), emitting a spurious
          // empty bucket past `max`.
          max: "1970-01-03T12:00:00Z".into(),
        }),
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let agg = run_date_histogram(&idx, aggs);
    if let searchlite_core::api::types::AggregationResponse::DateHistogram { buckets, .. } = agg {
      let keys: Vec<_> = buckets.iter().map(|b| b.key.clone()).collect();
      assert_eq!(
        keys,
        vec![json!(0), json!(DAY_MS), json!(2 * DAY_MS)],
        "extended_bounds max at noon day 3 must clip to day 3's key, \
         not spill into day 4"
      );
      // Seed doc is at noon on day 1 -> day 1's bucket.
      assert_eq!(buckets[0].doc_count, 1);
      assert_eq!(buckets[1].doc_count, 0);
      assert_eq!(buckets[2].doc_count, 0);
    } else {
      panic!("expected date histogram response");
    }
  }

  /// Defense-in-depth: `parse_interval_seconds` accepts `"0ms"` and returns
  /// `Some(0.0)`, which would previously produce a `DateInterval::Fixed(0)`
  /// — dividing by zero in `bucket_start` and never advancing in
  /// `add_interval` during empty-bucket fill. Config validation must reject
  /// non-positive fixed intervals up front.
  ///
  /// The collector also materializes intervals as integer milliseconds via
  /// `(secs * 1000.0) as i64`, so sub-millisecond specs like `"0.5ms"`
  /// previously slipped past a `secs > 0.0` gate but truncated to
  /// `Fixed(0)` — producing an empty-result silent regression. Validation
  /// now rejects anything that doesn't survive the ms conversion with at
  /// least a 1ms step.
  #[test]
  fn invalid_fixed_interval_is_rejected_by_validation() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = timestamp_index(tmp.path());

    for spec in ["0ms", "0s", "0d", "0.5ms", "0.0009s"] {
      let mut aggs = BTreeMap::new();
      aggs.insert(
        "hist".into(),
        Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
          field: "ts".into(),
          calendar_interval: None,
          fixed_interval: Some(spec.into()),
          offset: None,
          format: None,
          min_doc_count: Some(0),
          extended_bounds: None,
          hard_bounds: None,
          missing: None,
          sampling: None,
          aggs: BTreeMap::new(),
        })),
      );

      let mut req = SearchRequest::new("rust");
      req.aggs = aggs;
      let err = idx
        .reader()
        .unwrap()
        .search(&req)
        .expect_err(&format!("fixed_interval {spec:?} must be rejected"));
      let msg = err.to_string();
      assert!(
        msg.contains("at least 1ms"),
        "expected at-least-1ms error for {spec:?}, got: {msg}"
      );
    }
  }
}

/// Regression tests for BUG-200 — `DateHistogramCollector::finish` materialized
/// empty buckets between `extended_bounds.min` and `extended_bounds.max` with
/// no `MAX_BUCKETS` cap, and `validate_date_histogram_config` never rejected
/// pathological spans. A small `fixed_interval` (down to `1ms`, accepted by
/// validation) combined with a wide `extended_bounds` drove the unbounded
/// `HashMap` insert loop into OOM before the request could return. Both halves
/// of the `HistogramAggregation` defense — a validator-side span check and a
/// runtime cap inside `finish` — now apply to `DateHistogramAggregation`.
mod bug_200 {
  use super::*;

  fn timestamp_index(path: &std::path::Path) -> searchlite_core::api::Index {
    let mut schema = Schema::default_text_body();
    schema.numeric_fields.push(NumericField {
      name: "ts".into(),
      i64: true,
      fast: true,
      stored: true,
      nullable: false,
    });
    let idx = Index::create(path, schema, build_base_options(path)).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&doc(
          "seed",
          vec![("body", json!("rust")), ("ts", json!(0_i64))],
        ))
        .unwrap();
      writer.commit().unwrap();
    }
    idx
  }

  fn search_with_agg(
    idx: &searchlite_core::api::Index,
    aggs: BTreeMap<String, Aggregation>,
  ) -> anyhow::Result<()> {
    let mut req = SearchRequest::new("rust");
    req.aggs = aggs;
    idx.reader().unwrap().search(&req)?;
    Ok(())
  }

  /// The original issue repro: `fixed_interval=1ms` + a 4-year
  /// `extended_bounds` would materialize ~10^11 buckets. Validation must
  /// reject it before any collector loop runs.
  #[test]
  fn fixed_interval_1ms_with_wide_extended_bounds_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = timestamp_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "evil".into(),
      Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
        field: "ts".into(),
        calendar_interval: None,
        fixed_interval: Some("1ms".into()),
        offset: None,
        format: None,
        min_doc_count: Some(0),
        extended_bounds: Some(DateHistogramBounds {
          min: "2020-01-01T00:00:00Z".into(),
          max: "2024-01-01T00:00:00Z".into(),
        }),
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs).expect_err(
      "extended_bounds spanning ~10^11 empty buckets must be rejected, not materialized",
    );
    let msg = err.to_string();
    assert!(
      msg.contains("too many empty buckets"),
      "expected bucket-span error, got: {msg}"
    );
  }

  /// A `hard_bounds` range (without any `extended_bounds`) also drives
  /// empty-bucket materialization in the collector, so the validator must
  /// apply the same span check whenever either bound is present.
  #[test]
  fn fixed_interval_wide_hard_bounds_without_extended_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = timestamp_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "evil".into(),
      Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
        field: "ts".into(),
        calendar_interval: None,
        fixed_interval: Some("1ms".into()),
        offset: None,
        format: None,
        min_doc_count: Some(0),
        extended_bounds: None,
        hard_bounds: Some(DateHistogramBounds {
          min: "2020-01-01T00:00:00Z".into(),
          max: "2024-01-01T00:00:00Z".into(),
        }),
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs)
      .expect_err("hard_bounds alone can drive empty-bucket fill and must be span-checked");
    let msg = err.to_string();
    assert!(
      msg.contains("too many empty buckets"),
      "expected bucket-span error, got: {msg}"
    );
  }

  /// The collector's `bucket_start(min)..=bucket_start(max)` is *inclusive*
  /// of both endpoints. With `interval=1s` and a 10_000-second window, the
  /// inclusive span is 10_001 buckets — one above `MAX_BUCKETS`. A naïve
  /// `(max - min) / interval = 10_000` computation would silently allow it.
  #[test]
  fn bounds_span_fence_post_respects_cap() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = timestamp_index(tmp.path());

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
        extended_bounds: Some(DateHistogramBounds {
          min: "1970-01-01T00:00:00Z".into(),
          // 10_000 seconds after min → inclusive bucket count = 10_001.
          max: "1970-01-01T02:46:40Z".into(),
        }),
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs)
      .expect_err("10_001 inclusive buckets must be rejected (fence-post), not accepted as 10_000");
    let msg = err.to_string();
    assert!(
      msg.contains("too many empty buckets"),
      "expected bucket-span error, got: {msg}"
    );
  }

  /// A span that sits exactly at the cap must still be accepted — the check
  /// is `> MAX_BUCKETS`, not `>= MAX_BUCKETS`. With `interval=1s` and a
  /// 9_999-second window, the inclusive span is 10_000 — the cap exactly.
  #[test]
  fn bounds_span_exactly_at_cap_is_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = timestamp_index(tmp.path());

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
        extended_bounds: Some(DateHistogramBounds {
          min: "1970-01-01T00:00:00Z".into(),
          // 9_999 seconds after min → inclusive bucket count = 10_000.
          max: "1970-01-01T02:46:39Z".into(),
        }),
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    search_with_agg(&idx, aggs).expect("bounds span exactly at the cap must be accepted");
  }

  /// Calendar intervals floor at `Day`, so the realistic attack window is
  /// narrower — but it is still reachable via absurd year ranges. The
  /// validator must walk the calendar the same way the collector does so an
  /// overly wide `calendar_interval=day` span is also rejected up front.
  #[test]
  fn calendar_interval_day_with_wide_span_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = timestamp_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "evil".into(),
      Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
        field: "ts".into(),
        // Day buckets over a ~100-year span → ~36_525 buckets, well above cap.
        calendar_interval: Some("day".into()),
        fixed_interval: None,
        offset: None,
        format: None,
        min_doc_count: Some(0),
        extended_bounds: Some(DateHistogramBounds {
          min: "1900-01-01T00:00:00Z".into(),
          max: "2000-01-01T00:00:00Z".into(),
        }),
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let err = search_with_agg(&idx, aggs)
      .expect_err("day-interval calendar spans beyond the cap must be rejected");
    let msg = err.to_string();
    assert!(
      msg.contains("too many empty buckets"),
      "expected bucket-span error, got: {msg}"
    );
  }

  /// End-to-end wall-clock guard: a pathological date_histogram request that
  /// would otherwise drive ~10^11 `HashMap` inserts in `finish` must return
  /// quickly through the public API — either because validation rejects it
  /// up front (the load-bearing path) or because the runtime cap inside
  /// `DateHistogramCollector::finish` breaks the loop at `MAX_BUCKETS`
  /// (defense-in-depth). This test passes regardless of which guard fires
  /// first, and asserts both shape (`Err`) and wall-clock bound so an
  /// accidental reintroduction of the unbounded loop is caught.
  ///
  /// Runtime-cap-only coverage that bypasses validation lives in
  /// `searchlite-core/src/query/aggs/mod.rs` as a unit test of the span
  /// helper (`date_histogram_span_exceeds_cap`).
  #[test]
  fn pathological_bounds_return_quickly_through_public_api() {
    use std::time::{Duration, Instant};

    let tmp = tempfile::tempdir().unwrap();
    let idx = timestamp_index(tmp.path());

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "evil".into(),
      Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
        field: "ts".into(),
        calendar_interval: None,
        fixed_interval: Some("1ms".into()),
        offset: None,
        format: None,
        min_doc_count: Some(0),
        extended_bounds: Some(DateHistogramBounds {
          min: "2020-01-01T00:00:00Z".into(),
          max: "2024-01-01T00:00:00Z".into(),
        }),
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );

    let start = Instant::now();
    let result = search_with_agg(&idx, aggs);
    let elapsed = start.elapsed();
    assert!(
      result.is_err(),
      "pathological date_histogram request must be rejected, not silently accepted"
    );
    assert!(
      elapsed < Duration::from_secs(5),
      "pathological date_histogram must not loop: took {elapsed:?}"
    );
  }
}
