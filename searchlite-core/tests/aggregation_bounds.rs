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
        // 1_000_000 / 1.0 = 1_000_000 buckets — well above the 65_536 cap.
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
