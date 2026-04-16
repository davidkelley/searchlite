use std::collections::BTreeMap;

use chrono::DateTime;
use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Aggregation, CompositeAggregation, CompositeSource, DateHistogramAggregation,
  DateHistogramBounds, Document, ExecutionStrategy, HistogramAggregation, HistogramBounds,
  IndexOptions, KeywordField, MetricAggregation, MovingAvgAggregation, NumericField, Schema,
  SearchRequest, SortOrder, SortSpec, StorageType, TermsAggregation, TopHitsAggregation,
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

/// Regression test for BUG-269 — `hard_bounds` was applied against the raw
/// document value instead of the computed bucket key, and used an inclusive
/// upper bound (`val > max`) instead of exclusive on the bucket key
/// (`bucket_val >= max`).
///
/// **Lower bound:** `interval = 10`, `hard_bounds = { min: 25, max: 80 }`.
/// A document with value 27 has `bucket_key = 20`. Since `20 < 25`, the
/// bucket must be dropped. Before the fix the raw-value check `27 >= 25`
/// passed, producing a spurious bucket at key 20.
///
/// **Upper bound:** `interval = 10`, `hard_bounds = { min: 0, max: 30 }`.
/// A document with value 30 has `bucket_key = 30`. Since `30 >= 30`
/// (exclusive upper), the bucket must be dropped. Before the fix the
/// raw-value check `30 > 30` evaluated to false, letting the bucket
/// through.
#[test]
fn histogram_hard_bounds_filters_on_bucket_key_not_raw_value() {
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
    // 27 → bucket_key 20 (below hard_bounds.min 25 → excluded)
    // 60 → bucket_key 60 (inside [25, 80) → included)
    for val in [27_i64, 60] {
      writer
        .add_document(&doc(
          &format!("hb-{val}"),
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
      extended_bounds: None,
      hard_bounds: Some(HistogramBounds {
        min: 25.0,
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
    // hard_bounds [25, 80) produces filled buckets at keys 30..=70 (key 20
    // is below min, key 80 is at the exclusive upper bound). The doc with
    // value 27 maps to bucket_key 20 which is outside hard_bounds, so only
    // the doc with value 60 contributes a hit.
    assert_eq!(
      keys,
      vec![
        json!(30.0),
        json!(40.0),
        json!(50.0),
        json!(60.0),
        json!(70.0)
      ]
    );
    assert_eq!(buckets[0].doc_count, 0); // key 30 — empty fill
    assert_eq!(buckets[1].doc_count, 0); // key 40 — empty fill
    assert_eq!(buckets[2].doc_count, 0); // key 50 — empty fill
    assert_eq!(buckets[3].doc_count, 1); // key 60 — collected from val 60
    assert_eq!(buckets[4].doc_count, 0); // key 70 — empty fill
  } else {
    panic!("unexpected histogram response");
  }
}

/// Companion to the lower-bound test above: verifies that the upper bound
/// is exclusive on the bucket key (`bucket_key >= max` → drop).
#[test]
fn histogram_hard_bounds_upper_bound_is_exclusive_on_bucket_key() {
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
    // 25 → bucket_key 20 (inside [0, 30) → included)
    // 30 → bucket_key 30 (30 >= 30, exclusive upper → excluded)
    for val in [25_i64, 30] {
      writer
        .add_document(&doc(
          &format!("hbu-{val}"),
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
      extended_bounds: None,
      hard_bounds: Some(HistogramBounds {
        min: 0.0,
        max: 30.0,
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
    // hard_bounds [0, 30) fills keys 0, 10, 20. The doc with value 30 maps
    // to bucket_key 30 which is at the exclusive upper bound, so it is
    // dropped. Only the doc with value 25 (bucket_key 20) contributes.
    assert_eq!(keys, vec![json!(0.0), json!(10.0), json!(20.0)]);
    assert_eq!(buckets[0].doc_count, 0); // key 0 — empty fill
    assert_eq!(buckets[1].doc_count, 0); // key 10 — empty fill
    assert_eq!(buckets[2].doc_count, 1); // key 20 — collected from val 25
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

/// Regression test for BUG-269 — `DateHistogramCollector` applied
/// `hard_bounds` against the raw timestamp instead of the computed bucket
/// start, producing buckets whose keys fall outside `hard_bounds`.
///
/// `calendar_interval = "month"`, `hard_bounds = { min: "2024-01-15", max:
/// "2024-03-15" }`. A document dated 2024-01-20 has `bucket_start =
/// 2024-01-01T00:00:00Z`. Since Jan 1 < Jan 15, the bucket must be
/// dropped. Before the fix the raw-value check passed because Jan 20 >=
/// Jan 15, producing a spurious bucket at Jan 1.
#[test]
fn date_histogram_hard_bounds_filters_on_bucket_start_not_raw_value() {
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
    // Jan 20 → bucket_start Jan 1 (Jan 1 < Jan 15 → excluded)
    // Feb 10 → bucket_start Feb 1 (Feb 1 >= Jan 15 and Feb 1 < Mar 15 → included)
    // Mar 20 → bucket_start Mar 1 (Mar 1 >= Mar 15 is false, so Mar 1 < Mar 15 → included)
    // Apr 5  → bucket_start Apr 1 (Apr 1 >= Mar 15 exclusive upper → excluded)
    for t in [
      "2024-01-20T00:00:00Z",
      "2024-02-10T00:00:00Z",
      "2024-03-20T00:00:00Z",
      "2024-04-05T00:00:00Z",
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
      min_doc_count: None,
      extended_bounds: None,
      hard_bounds: Some(DateHistogramBounds {
        min: "2024-01-15T00:00:00Z".into(),
        max: "2024-03-15T00:00:00Z".into(),
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
    // Only Feb 1 and Mar 1 buckets should survive.
    // Jan 1 is below hard_bounds.min (Jan 15), Apr 1 is at/above hard_bounds.max (Mar 15).
    assert_eq!(
      keys,
      vec![
        json!(ts("2024-02-01T00:00:00Z")),
        json!(ts("2024-03-01T00:00:00Z")),
      ]
    );
    assert_eq!(buckets[0].doc_count, 1); // Feb 10 doc
    assert_eq!(buckets[1].doc_count, 1); // Mar 20 doc
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

/// Regression tests for BUG-221 — `MovingAvgAggregation::predict` is fed
/// straight into `vec![last_avg; predict]` inside `apply_moving_avg_pipeline`.
/// Without a request-time bound, a tiny request body (well under the HTTP
/// 50 MiB cap) could request multi-gigabyte allocations during response
/// finalization and OOM the server.
mod bug_221 {
  use super::*;
  use searchlite_core::api::types::GapPolicy;

  fn views_index(path: &std::path::Path) -> searchlite_core::api::Index {
    let mut schema = Schema::default_text_body();
    schema.numeric_fields.push(NumericField {
      name: "n".into(),
      i64: true,
      fast: true,
      stored: true,
      nullable: false,
    });
    let idx = Index::create(path, schema, build_base_options(path)).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      // Two docs in distinct histogram buckets so the bucketing agg has at
      // least one non-empty bucket — the precondition for the `predict`
      // branch in `apply_moving_avg_pipeline` to allocate.
      writer
        .add_document(&doc(
          "a",
          vec![("body", json!("rust")), ("n", json!(1_i64))],
        ))
        .unwrap();
      writer
        .add_document(&doc(
          "b",
          vec![("body", json!("rust")), ("n", json!(2_i64))],
        ))
        .unwrap();
      writer.commit().unwrap();
    }
    idx
  }

  fn moving_avg_request(predict: Option<usize>, window: usize) -> BTreeMap<String, Aggregation> {
    let mut hist_aggs = BTreeMap::new();
    hist_aggs.insert(
      "mov".into(),
      Aggregation::MovingAvg(MovingAvgAggregation {
        buckets_path: "_count".into(),
        window,
        predict,
        gap_policy: Some(GapPolicy::Skip),
      }),
    );
    let mut aggs = BTreeMap::new();
    aggs.insert(
      "hist".into(),
      Aggregation::Histogram(Box::new(HistogramAggregation {
        field: "n".into(),
        interval: 1.0,
        offset: None,
        min_doc_count: Some(0),
        extended_bounds: None,
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: hist_aggs,
      })),
    );
    aggs
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

  #[test]
  fn huge_predict_is_rejected_without_allocation() {
    use std::time::{Duration, Instant};

    let tmp = tempfile::tempdir().unwrap();
    let idx = views_index(tmp.path());

    // ~8 GiB of `f64` if it ever reached `vec![..; predict]`.
    let aggs = moving_avg_request(Some(1_073_741_824), 1);

    let start = Instant::now();
    let err = search_with_agg(&idx, aggs)
      .expect_err("moving_avg with predict above MAX_PREDICTIONS must be rejected");
    let elapsed = start.elapsed();
    let msg = err.to_string();
    assert!(
      msg.contains("predict") && msg.contains("exceeds limit"),
      "expected predict-bound error, got: {msg}"
    );
    // The validator runs before any allocation; rejection must be effectively
    // instantaneous, never paying the cost of `vec![..; predict]`.
    assert!(
      elapsed < Duration::from_secs(2),
      "moving_avg validation must reject quickly without allocating: took {elapsed:?}"
    );
  }

  #[test]
  fn predict_just_above_cap_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = views_index(tmp.path());

    // `MAX_PREDICTIONS` is 10_000; exercise the strict `>` boundary.
    let aggs = moving_avg_request(Some(10_001), 1);
    let err =
      search_with_agg(&idx, aggs).expect_err("predict = MAX_PREDICTIONS + 1 must be rejected");
    let msg = err.to_string();
    assert!(
      msg.contains("predict") && msg.contains("10001"),
      "expected predict bound error mentioning the offending value, got: {msg}"
    );
  }

  #[test]
  fn predict_at_cap_is_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = views_index(tmp.path());

    // 10_000 forecast points of an `f64` is ~80 KiB — well within budget and
    // intentionally accepted so legitimate clients keep working.
    let aggs = moving_avg_request(Some(10_000), 1);
    search_with_agg(&idx, aggs).expect("predict at the cap must be accepted");
  }

  #[test]
  fn small_predict_is_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = views_index(tmp.path());

    let aggs = moving_avg_request(Some(3), 2);
    search_with_agg(&idx, aggs).expect("typical predict values must be accepted");
  }

  #[test]
  fn huge_window_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = views_index(tmp.path());

    // `window` is not itself an unbounded allocation today, but a runaway
    // value is meaningless and must be rejected so future maintainers cannot
    // accidentally turn it into one.
    let aggs = moving_avg_request(None, 1_000_000);
    let err = search_with_agg(&idx, aggs)
      .expect_err("moving_avg window above MAX_BUCKETS must be rejected");
    let msg = err.to_string();
    assert!(
      msg.contains("window") && msg.contains("exceeds limit"),
      "expected window-bound error, got: {msg}"
    );
  }

  #[test]
  fn zero_window_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = views_index(tmp.path());

    let aggs = moving_avg_request(None, 0);
    let err = search_with_agg(&idx, aggs).expect_err("window = 0 must be rejected");
    let msg = err.to_string();
    assert!(
      msg.contains("window") && msg.contains(">= 1"),
      "expected zero-window error, got: {msg}"
    );
  }
}

/// Regression coverage for BUG-215: `top_hits` with `from > 0` on a
/// multi-segment index must return the globally `from`-th through
/// `(from + size - 1)`-th best documents.
///
/// Before the fix, `TopHitsCollector::finish` dropped items at per-segment
/// ranks `[0, from)` *before* `merge_top_hits` could compare them across
/// segments. On a two-segment index that produced an answer drawn from the
/// per-segment `[from, from + size)` window of each segment, which is not
/// the same as the global top `(from + size)` window.
mod bug_215 {
  use super::*;

  #[test]
  fn top_hits_from_offset_is_global_across_segments() {
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
    let idx = IndexBuilder::create(&path, schema, build_base_options(&path)).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      // Segment A: scores 10, 8, 6, 4, 2.
      for (id, s) in [(1_i64, 10_i64), (2, 8), (3, 6), (4, 4), (5, 2)] {
        writer
          .add_document(&doc(
            &format!("a-{id}"),
            vec![("body", json!("rust")), ("score", json!(s))],
          ))
          .unwrap();
      }
      writer.commit().unwrap();
      // Segment B: scores 9, 7, 5, 3, 1. Separate commit creates a second
      // segment, which is the precondition for the bug.
      for (id, s) in [(6_i64, 9_i64), (7, 7), (8, 5), (9, 3), (10, 1)] {
        writer
          .add_document(&doc(
            &format!("b-{id}"),
            vec![("body", json!("rust")), ("score", json!(s))],
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
        from: 1,
        fields: Some(vec!["score".into()]),
        sort: vec![SortSpec {
          field: "score".into(),
          order: Some(SortOrder::Desc),
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
      assert_eq!(top_hits.total, 10);
      let scores: Vec<_> = top_hits
        .hits
        .iter()
        .map(|h| {
          h.fields
            .as_ref()
            .and_then(|f| f.get("score"))
            .and_then(|v| v.as_i64())
            .unwrap()
        })
        .collect();
      // Global sorted order is [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]; skipping
      // `from = 1` and taking `size = 2` yields [9, 8]. The pre-fix
      // behaviour returned [7, 6] because segment B's top-ranked doc
      // (score 9) was discarded by the per-segment `from` skip before the
      // cross-segment merge ever saw it.
      assert_eq!(scores, vec![9, 8]);
    } else {
      panic!("expected top hits response");
    }
  }

  /// Deep `from` (larger than any single segment's per-segment `from`
  /// window alone) still returns globally-correct results.
  #[test]
  fn top_hits_deep_from_across_segments() {
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
    let idx = IndexBuilder::create(&path, schema, build_base_options(&path)).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      // Segment A: odd scores 1, 3, ..., 19.
      let odd: Vec<i64> = (1..=19).filter(|s| s % 2 == 1).collect();
      for s in &odd {
        writer
          .add_document(&doc(
            &format!("a-{s}"),
            vec![("body", json!("rust")), ("score", json!(*s))],
          ))
          .unwrap();
      }
      writer.commit().unwrap();
      // Segment B: even scores 2, 4, ..., 20.
      let even: Vec<i64> = (2..=20).filter(|s| s % 2 == 0).collect();
      for s in &even {
        writer
          .add_document(&doc(
            &format!("b-{s}"),
            vec![("body", json!("rust")), ("score", json!(*s))],
          ))
          .unwrap();
      }
      writer.commit().unwrap();
    }

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "hits".into(),
      Aggregation::TopHits(TopHitsAggregation {
        size: 3,
        from: 4,
        fields: Some(vec!["score".into()]),
        sort: vec![SortSpec {
          field: "score".into(),
          order: Some(SortOrder::Desc),
        }],
        highlight_field: None,
      }),
    );

    let mut req = SearchRequest::new("rust");
    req.limit = 1;
    req.aggs = aggs;
    let resp = idx.reader().unwrap().search(&req).unwrap();

    let agg = resp.aggregations.get("hits").unwrap();
    if let searchlite_core::api::types::AggregationResponse::TopHits(top_hits) = agg {
      assert_eq!(top_hits.total, 20);
      let scores: Vec<_> = top_hits
        .hits
        .iter()
        .map(|h| {
          h.fields
            .as_ref()
            .and_then(|f| f.get("score"))
            .and_then(|v| v.as_i64())
            .unwrap()
        })
        .collect();
      // Global descending order is 20, 19, ..., 1. Skipping `from = 4`
      // and taking `size = 3` yields [16, 15, 14].
      assert_eq!(scores, vec![16, 15, 14]);
    } else {
      panic!("expected top hits response");
    }
  }
}

/// Regression tests for BUG-222 — `TopHitsAggregation::size` and `from` are
/// forwarded straight into `TopHitsCollector::new`, which uses them to size a
/// per-segment `BinaryHeap<RankedDoc>`. Without a request-time bound, a tiny
/// request body (well under the HTTP 50 MiB cap) could ask for `size = 10^10`
/// and grow the heap until the segment is exhausted or the process OOMs.
mod bug_222 {
  use super::*;

  fn corpus_index(path: &std::path::Path) -> searchlite_core::api::Index {
    let mut schema = Schema::default_text_body();
    schema.numeric_fields.push(NumericField {
      name: "n".into(),
      i64: true,
      fast: true,
      stored: true,
      nullable: false,
    });
    let idx = IndexBuilder::create(path, schema, build_base_options(path)).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      // Two docs is enough to exercise the heap-growth branch in `collect`;
      // the bug is about how the heap is *sized*, not about the number of
      // matching docs.
      writer
        .add_document(&doc(
          "a",
          vec![("body", json!("rust")), ("n", json!(1_i64))],
        ))
        .unwrap();
      writer
        .add_document(&doc(
          "b",
          vec![("body", json!("rust")), ("n", json!(2_i64))],
        ))
        .unwrap();
      writer.commit().unwrap();
    }
    idx
  }

  fn top_hits_request(size: usize, from: usize) -> BTreeMap<String, Aggregation> {
    let mut aggs = BTreeMap::new();
    aggs.insert(
      "hits".into(),
      Aggregation::TopHits(TopHitsAggregation {
        size,
        from,
        fields: None,
        sort: Vec::new(),
        highlight_field: None,
      }),
    );
    aggs
  }

  fn search_with_agg(
    idx: &searchlite_core::api::Index,
    aggs: BTreeMap<String, Aggregation>,
  ) -> anyhow::Result<()> {
    let mut req = SearchRequest::new("rust");
    req.limit = 1;
    req.aggs = aggs;
    idx.reader().unwrap().search(&req)?;
    Ok(())
  }

  #[test]
  fn huge_size_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = corpus_index(tmp.path());

    // `usize::MAX` is the largest possible value the deserializer can hand us;
    // it is also portable across 32- and 64-bit targets, where a literal like
    // `10_000_000_000` would overflow on 32-bit. The validator must reject it
    // outright — without the bound, the per-segment heap would grow until the
    // segment is exhausted or the process OOMs.
    let aggs = top_hits_request(usize::MAX, 0);
    let err = search_with_agg(&idx, aggs)
      .expect_err("top_hits with size above MAX_TOP_HITS must be rejected");
    let msg = err.to_string();
    assert!(
      msg.contains("size") && msg.contains("exceeds limit"),
      "expected size-bound error, got: {msg}"
    );
  }

  #[test]
  fn huge_from_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = corpus_index(tmp.path());

    // See `huge_size_is_rejected` — `usize::MAX` keeps the test 32-bit safe.
    let aggs = top_hits_request(1, usize::MAX);
    let err = search_with_agg(&idx, aggs)
      .expect_err("top_hits with from above MAX_TOP_HITS must be rejected");
    let msg = err.to_string();
    assert!(
      msg.contains("from") && msg.contains("exceeds limit"),
      "expected from-bound error, got: {msg}"
    );
  }

  #[test]
  fn size_just_above_cap_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = corpus_index(tmp.path());

    // `MAX_TOP_HITS` is 10_000; exercise the strict `>` boundary on `size`.
    let aggs = top_hits_request(10_001, 0);
    let err = search_with_agg(&idx, aggs).expect_err("size = MAX_TOP_HITS + 1 must be rejected");
    let msg = err.to_string();
    assert!(
      msg.contains("size") && msg.contains("10001"),
      "expected size bound error mentioning the offending value, got: {msg}"
    );
  }

  #[test]
  fn from_just_above_cap_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = corpus_index(tmp.path());

    let aggs = top_hits_request(0, 10_001);
    let err = search_with_agg(&idx, aggs).expect_err("from = MAX_TOP_HITS + 1 must be rejected");
    let msg = err.to_string();
    assert!(
      msg.contains("from") && msg.contains("10001"),
      "expected from bound error mentioning the offending value, got: {msg}"
    );
  }

  #[test]
  fn size_plus_from_above_cap_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = corpus_index(tmp.path());

    // Each value is below the cap, but the sum exceeds it. Without the
    // additive check, an attacker could pick `size = cap` and `from = cap`
    // to size the heap at `2 * cap` and bypass the per-dimension bound.
    let aggs = top_hits_request(10_000, 1);
    let err =
      search_with_agg(&idx, aggs).expect_err("size + from above MAX_TOP_HITS must be rejected");
    let msg = err.to_string();
    assert!(
      msg.contains("size + from") && msg.contains("exceeds limit"),
      "expected combined bound error, got: {msg}"
    );
  }

  #[test]
  fn size_at_cap_is_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = corpus_index(tmp.path());

    // 10_000 hits is intentionally accepted so legitimate clients keep working.
    let aggs = top_hits_request(10_000, 0);
    search_with_agg(&idx, aggs).expect("size at the cap must be accepted");
  }

  #[test]
  fn size_plus_from_at_cap_is_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = corpus_index(tmp.path());

    // The combined bound is `<= MAX_TOP_HITS`, not `<`. Exercise the boundary
    // so a future tightening of the check does not silently break clients
    // that already rely on `size + from = cap`.
    let aggs = top_hits_request(9_000, 1_000);
    search_with_agg(&idx, aggs).expect("size + from at the cap must be accepted");
  }

  #[test]
  fn small_top_hits_request_is_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = corpus_index(tmp.path());

    let aggs = top_hits_request(2, 1);
    search_with_agg(&idx, aggs).expect("typical top_hits values must be accepted");
  }
}

/// Regression for BUG-233: calendar_interval "quarter" silently dropped
/// documents dated May 31 because truncate_calendar changed the month
/// before normalizing the day.
mod bug_233 {
  use super::*;

  fn timestamp_index(path: &std::path::Path) -> Index {
    let mut schema = Schema::default_text_body();
    schema.numeric_fields.push(NumericField {
      name: "ts".into(),
      i64: true,
      fast: true,
      stored: true,
      nullable: false,
    });
    let idx = IndexBuilder::create(path, schema, build_base_options(path)).unwrap();
    let ts = |s: &str| DateTime::parse_from_rfc3339(s).unwrap().timestamp_millis();
    let mut writer = idx.writer().unwrap();
    for (id, t) in [
      ("d1", "2024-04-15T10:00:00Z"),
      ("d2", "2024-05-31T12:00:00Z"),
      ("d3", "2024-07-10T08:00:00Z"),
    ] {
      writer
        .add_document(&doc(
          id,
          vec![("body", json!("text")), ("ts", json!(ts(t)))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
    idx
  }

  fn run_quarter_histogram(idx: &Index) -> Vec<(serde_json::Value, u64)> {
    let mut aggs = BTreeMap::new();
    aggs.insert(
      "q".into(),
      Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
        field: "ts".into(),
        calendar_interval: Some("quarter".into()),
        fixed_interval: None,
        offset: None,
        format: None,
        min_doc_count: Some(1),
        extended_bounds: None,
        hard_bounds: None,
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );
    let req = SearchRequest {
      query: "text".into(),
      fields: None,
      filter: None,
      limit: 0,
      from: 0,
      return_hits: false,
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
    };
    let reader = idx.reader().unwrap();
    let resp = reader.search(&req).unwrap();
    if let Some(searchlite_core::api::types::AggregationResponse::DateHistogram {
      buckets, ..
    }) = resp.aggregations.get("q")
    {
      buckets
        .iter()
        .map(|b| (b.key.clone(), b.doc_count))
        .collect()
    } else {
      panic!("expected DateHistogram response");
    }
  }

  #[test]
  fn quarter_calendar_interval_counts_may_31_documents() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = timestamp_index(tmp.path());
    let buckets = run_quarter_histogram(&idx);

    assert_eq!(
      buckets.len(),
      2,
      "expected exactly 2 quarter buckets (Q2 + Q3); got: {buckets:?}"
    );

    // Q2 (2024-04-01) must have 2 docs (April 15 + May 31).
    assert_eq!(
      buckets[0].1, 2,
      "Q2 bucket must contain 2 docs; got: {buckets:?}"
    );

    // Q3 (2024-07-01) must have 1 doc.
    assert_eq!(
      buckets[1].1, 1,
      "Q3 bucket must contain 1 doc; got: {buckets:?}"
    );

    // Total doc_count across all buckets must equal 3 (no silent drops).
    let total: u64 = buckets.iter().map(|(_, c)| c).sum();
    assert_eq!(
      total, 3,
      "total doc_count must equal indexed docs; got: {buckets:?}"
    );
  }
}

/// Regression tests for #251: add_calendar must preserve the sub-day time
/// component so that the fill loop produces bucket keys aligned with the
/// offset applied by bucket_start.
mod bug_251 {
  use super::*;

  /// Calendar month interval with offset=1h and extended_bounds spanning
  /// April–June. The fill loop must produce keys at T01:00:00Z (not midnight)
  /// for every bucket, and there must be no phantom duplicate buckets.
  #[test]
  fn date_histogram_calendar_month_with_offset_produces_aligned_keys() {
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
    let idx = Index::create(&path, schema, build_base_options(&path)).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      // April 15 doc
      let apr = DateTime::parse_from_rfc3339("2024-04-15T10:00:00Z")
        .unwrap()
        .timestamp_millis();
      writer
        .add_document(&doc(
          "d1",
          vec![("body", json!("test")), ("ts", json!(apr))],
        ))
        .unwrap();
      // May 20 doc
      let may = DateTime::parse_from_rfc3339("2024-05-20T10:00:00Z")
        .unwrap()
        .timestamp_millis();
      writer
        .add_document(&doc(
          "d2",
          vec![("body", json!("test")), ("ts", json!(may))],
        ))
        .unwrap();
      writer.commit().unwrap();
    }

    let mut aggs = BTreeMap::new();
    aggs.insert(
      "hist".into(),
      Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
        field: "ts".into(),
        calendar_interval: Some("month".into()),
        fixed_interval: None,
        offset: Some("1h".into()),
        format: None,
        min_doc_count: Some(0),
        extended_bounds: Some(DateHistogramBounds {
          min: "2024-04-01T02:00:00Z".into(),
          max: "2024-06-01T02:00:00Z".into(),
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
        query: "test".into(),
        fields: None,
        filter: None,
        limit: 0,
        from: 0,
        return_hits: false,
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
    let buckets = match hist {
      searchlite_core::api::types::AggregationResponse::DateHistogram { buckets, .. } => buckets,
      other => panic!("expected DateHistogram, got: {other:?}"),
    };

    // Collect (key_ms, doc_count) pairs
    let entries: Vec<(i64, u64)> = buckets
      .iter()
      .map(|b| {
        let key = b.key.as_i64().unwrap();
        (key, b.doc_count)
      })
      .collect();

    // Expected bucket keys: all at T01:00:00Z due to 1h offset
    let apr_key = DateTime::parse_from_rfc3339("2024-04-01T01:00:00Z")
      .unwrap()
      .timestamp_millis();
    let may_key = DateTime::parse_from_rfc3339("2024-05-01T01:00:00Z")
      .unwrap()
      .timestamp_millis();
    let jun_key = DateTime::parse_from_rfc3339("2024-06-01T01:00:00Z")
      .unwrap()
      .timestamp_millis();

    // Must have exactly 3 buckets (Apr, May, Jun) — no phantom midnight buckets
    assert_eq!(
      entries.len(),
      3,
      "expected 3 buckets (Apr/May/Jun), got {}: {entries:?}",
      entries.len()
    );

    // Verify all keys are at T01:00:00Z
    let keys: Vec<i64> = entries.iter().map(|(k, _)| *k).collect();
    assert_eq!(
      keys,
      vec![apr_key, may_key, jun_key],
      "bucket keys must be at T01:00:00Z, not midnight; got: {entries:?}"
    );

    // April: 1 doc, May: 1 doc, June: 0 docs
    assert_eq!(entries[0].1, 1, "April bucket should have 1 doc");
    assert_eq!(entries[1].1, 1, "May bucket should have 1 doc");
    assert_eq!(entries[2].1, 0, "June bucket should have 0 docs");

    // Total doc_count must equal indexed docs (no double-counting)
    let total: u64 = entries.iter().map(|(_, c)| c).sum();
    assert_eq!(total, 2, "total doc_count must equal indexed docs");
  }
}
