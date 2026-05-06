use std::collections::{BTreeMap, HashSet};

use searchlite_core::analysis::analyzer::{AnalyzerDef, StopwordsConfig, TokenFilterDef};
use searchlite_core::api::reader::SearchResult;
use searchlite_core::api::types::{
  Document, ExecutionStrategy, FieldSpec, IndexOptions, KeywordField, MatchOperator,
  MultiMatchFuzziness, MultiMatchType, NumericField, Query, QueryNode, Schema, SearchRequest,
  StorageType, TextField,
};
use searchlite_core::api::Index;
use tempfile::TempDir;

fn doc(id: &str, title: &str, body: &str) -> Document {
  Document {
    fields: [
      ("_id".to_string(), serde_json::json!(id)),
      ("title".to_string(), serde_json::json!(title)),
      ("body".to_string(), serde_json::json!(body)),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>(),
  }
}

fn setup_reader() -> (TempDir, searchlite_core::api::IndexReader) {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx");
  let mut schema = Schema::default_text_body();
  schema.text_fields.push(TextField {
    name: "title".into(),
    analyzer: "default".into(),
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
    checksum_policy: Default::default(),
    checksum_audit_failure_hook: None,
    read_only: false,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let idx = Index::create(&path, schema, opts).unwrap();
  let mut writer = idx.writer().unwrap();
  let docs = vec![
    doc("doc-1", "rust search", "fast"),
    doc("doc-2", "rust", "search"),
    doc("doc-3", "rust", "rust search"),
    doc("doc-4", "boring", "rust"),
    doc("doc-5", "none", "rust fast search"),
  ];
  for doc in docs {
    writer.add_document(&doc).unwrap();
  }
  writer.commit().unwrap();
  let reader = idx.reader().unwrap();
  (dir, reader)
}

fn request(query: impl Into<Query>) -> SearchRequest {
  SearchRequest {
    query: query.into(),
    fields: None,
    filter: None,
    limit: 10,
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
    aggs: BTreeMap::new(),
    suggest: BTreeMap::new(),
    rescore: None,
    explain: false,
    profile: false,
  }
}

fn score_for(result: &SearchResult, id: &str) -> Option<f32> {
  result
    .hits
    .iter()
    .find(|hit| hit.doc_id == id)
    .map(|hit| hit.score)
}

fn ids(result: &SearchResult) -> HashSet<String> {
  result.hits.iter().map(|hit| hit.doc_id.clone()).collect()
}

fn ranked(result: &SearchResult) -> Vec<(String, f32)> {
  result
    .hits
    .iter()
    .map(|hit| (hit.doc_id.clone(), hit.score))
    .collect()
}

#[test]
fn multi_match_most_fields_counts_across_fields() {
  let (_tmp, reader) = setup_reader();
  let fields = vec![
    FieldSpec {
      field: "title".into(),
      boost: None,
    },
    FieldSpec {
      field: "body".into(),
      boost: None,
    },
  ];
  let best = QueryNode::MultiMatch {
    query: "rust search".into(),
    fields: fields.clone(),
    match_type: MultiMatchType::BestFields,
    fuzziness: None,
    tie_breaker: None,
    operator: Some(MatchOperator::Or),
    minimum_should_match: None,
    boost: None,
  };
  let most = QueryNode::MultiMatch {
    query: "rust search".into(),
    fields,
    match_type: MultiMatchType::MostFields,
    fuzziness: None,
    tie_breaker: None,
    operator: Some(MatchOperator::Or),
    minimum_should_match: None,
    boost: None,
  };
  let body_only = QueryNode::MultiMatch {
    query: "rust search".into(),
    fields: vec![FieldSpec {
      field: "body".into(),
      boost: None,
    }],
    match_type: MultiMatchType::BestFields,
    fuzziness: None,
    tie_breaker: None,
    operator: Some(MatchOperator::Or),
    minimum_should_match: None,
    boost: None,
  };
  let body_ids = ids(&reader.search(&request(body_only)).unwrap());
  assert!(body_ids.contains("doc-3"), "{body_ids:?}");
  let best_res = reader.search(&request(best)).unwrap();
  let most_res = reader.search(&request(most)).unwrap();
  let best_ids = ids(&best_res);
  let most_ids = ids(&most_res);
  assert!(best_ids.contains("doc-2"), "best_fields ids: {best_ids:?}");
  assert!(most_ids.contains("doc-2"), "most_fields ids: {most_ids:?}");
  let best_score = score_for(&best_res, "doc-2").unwrap();
  let most_score = score_for(&most_res, "doc-2").unwrap();
  assert!(most_score > best_score);
}

#[test]
fn dis_max_tie_breaker_prefers_multi_field_hit() {
  let (_tmp, reader) = setup_reader();
  let query = QueryNode::DisMax {
    queries: vec![
      QueryNode::Term {
        field: "title".into(),
        value: "rust".into(),
        boost: None,
      },
      QueryNode::Term {
        field: "body".into(),
        value: "rust".into(),
        boost: None,
      },
    ],
    tie_breaker: Some(0.5),
    boost: None,
  };
  let result = reader.search(&request(query)).unwrap();
  assert_eq!(
    result.hits.first().map(|h| h.doc_id.as_str()),
    Some("doc-3")
  );
}

#[test]
fn field_boost_reshapes_best_field_ranking() {
  let (_tmp, reader) = setup_reader();
  let boosted = QueryNode::MultiMatch {
    query: "rust".into(),
    fields: vec![
      FieldSpec {
        field: "title".into(),
        boost: Some(2.0),
      },
      FieldSpec {
        field: "body".into(),
        boost: None,
      },
    ],
    match_type: MultiMatchType::BestFields,
    fuzziness: None,
    tie_breaker: None,
    operator: None,
    minimum_should_match: None,
    boost: None,
  };
  let result = reader.search(&request(boosted)).unwrap();
  let ids = ids(&result);
  assert!(ids.contains("doc-2"));
  assert!(ids.contains("doc-4"));
  let score_title = score_for(&result, "doc-2").unwrap();
  let score_body = score_for(&result, "doc-4").unwrap();
  assert!(score_title > score_body);
}

#[test]
fn cross_fields_operator_and_matches_split_terms() {
  let (_tmp, reader) = setup_reader();
  let rust_ids = ids(
    &reader
      .search(&request(QueryNode::Term {
        field: "body".into(),
        value: "rust".into(),
        boost: None,
      }))
      .unwrap(),
  );
  assert!(rust_ids.contains("doc-4"));
  let query = QueryNode::MultiMatch {
    query: "rust search".into(),
    fields: vec![
      FieldSpec {
        field: "title".into(),
        boost: None,
      },
      FieldSpec {
        field: "body".into(),
        boost: None,
      },
    ],
    match_type: MultiMatchType::CrossFields,
    fuzziness: None,
    tie_breaker: None,
    operator: Some(MatchOperator::And),
    minimum_should_match: None,
    boost: None,
  };
  let hits = ids(&reader.search(&request(query)).unwrap());
  assert!(hits.contains("doc-2"), "hits: {hits:?}");
  assert!(!hits.contains("doc-4"), "hits: {hits:?}");
}

#[test]
fn cross_fields_fuzziness_auto_recovers_typo() {
  let (_tmp, reader) = setup_reader();
  let mut exact = request(QueryNode::MultiMatch {
    query: "rust serch".into(),
    fields: vec![
      FieldSpec {
        field: "title".into(),
        boost: None,
      },
      FieldSpec {
        field: "body".into(),
        boost: None,
      },
    ],
    match_type: MultiMatchType::CrossFields,
    fuzziness: None,
    tie_breaker: None,
    operator: Some(MatchOperator::And),
    minimum_should_match: None,
    boost: None,
  });
  let exact_hits = ids(&reader.search(&exact).unwrap());
  assert!(exact_hits.is_empty(), "exact hits: {exact_hits:?}");

  exact.query = Query::Node(QueryNode::MultiMatch {
    query: "rust serch".into(),
    fields: vec![
      FieldSpec {
        field: "title".into(),
        boost: None,
      },
      FieldSpec {
        field: "body".into(),
        boost: None,
      },
    ],
    match_type: MultiMatchType::CrossFields,
    fuzziness: Some(MultiMatchFuzziness::Auto),
    tie_breaker: None,
    operator: Some(MatchOperator::And),
    minimum_should_match: None,
    boost: None,
  });
  let fuzzy_hits = ids(&reader.search(&exact).unwrap());
  assert!(fuzzy_hits.contains("doc-2"), "fuzzy hits: {fuzzy_hits:?}");
  assert!(!fuzzy_hits.contains("doc-4"), "fuzzy hits: {fuzzy_hits:?}");
}

#[test]
fn phrase_slop_matches_gapped_tokens() {
  let (_tmp, reader) = setup_reader();
  let exact = QueryNode::Phrase {
    field: Some("body".into()),
    terms: vec!["rust".into(), "search".into()],
    slop: Some(0),
    boost: None,
  };
  let sloppy = QueryNode::Phrase {
    field: Some("body".into()),
    terms: vec!["rust".into(), "search".into()],
    slop: Some(1),
    boost: None,
  };
  let exact_ids = ids(&reader.search(&request(exact)).unwrap());
  let sloppy_ids = ids(&reader.search(&request(sloppy)).unwrap());
  assert!(exact_ids.contains("doc-3"));
  assert!(!exact_ids.contains("doc-5"));
  assert!(sloppy_ids.contains("doc-5"));
}

#[test]
fn cross_fields_duplicate_fields_do_not_change_scores() {
  let (_tmp, reader) = setup_reader();
  let unique = QueryNode::MultiMatch {
    query: "rust search".into(),
    fields: vec![
      FieldSpec {
        field: "title".into(),
        boost: None,
      },
      FieldSpec {
        field: "body".into(),
        boost: None,
      },
    ],
    match_type: MultiMatchType::CrossFields,
    fuzziness: None,
    tie_breaker: None,
    operator: Some(MatchOperator::And),
    minimum_should_match: None,
    boost: None,
  };
  let with_dupes = QueryNode::MultiMatch {
    query: "rust search".into(),
    fields: vec![
      FieldSpec {
        field: "title".into(),
        boost: None,
      },
      FieldSpec {
        field: "title".into(),
        boost: None,
      },
      FieldSpec {
        field: "body".into(),
        boost: None,
      },
      FieldSpec {
        field: "body".into(),
        boost: None,
      },
    ],
    match_type: MultiMatchType::CrossFields,
    fuzziness: None,
    tie_breaker: None,
    operator: Some(MatchOperator::And),
    minimum_should_match: None,
    boost: None,
  };
  let unique_res = reader.search(&request(unique)).unwrap();
  let duped_res = reader.search(&request(with_dupes)).unwrap();
  assert_eq!(ranked(&duped_res), ranked(&unique_res));
}

#[test]
fn cross_fields_mixed_field_kinds_are_deterministic() {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx-mixed-kinds");
  let mut schema = Schema::default_text_body();
  schema.keyword_fields.push(KeywordField {
    name: "tag".into(),
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
  let opts = IndexOptions {
    path: path.clone(),
    create_if_missing: true,
    enable_positions: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::Filesystem,
    checksum_policy: Default::default(),
    checksum_audit_failure_hook: None,
    read_only: false,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let idx = Index::create(&path, schema, opts).unwrap();
  let mut writer = idx.writer().unwrap();
  writer
    .add_document(&Document {
      fields: [
        ("_id".to_string(), serde_json::json!("doc-1")),
        ("body".to_string(), serde_json::json!("rust systems")),
        ("tag".to_string(), serde_json::json!("alpha")),
        ("year".to_string(), serde_json::json!(2024)),
      ]
      .into_iter()
      .collect(),
    })
    .unwrap();
  writer
    .add_document(&Document {
      fields: [
        ("_id".to_string(), serde_json::json!("doc-2")),
        ("body".to_string(), serde_json::json!("rust systems")),
        ("tag".to_string(), serde_json::json!("beta")),
        ("year".to_string(), serde_json::json!(2023)),
      ]
      .into_iter()
      .collect(),
    })
    .unwrap();
  writer.commit().unwrap();
  let reader = idx.reader().unwrap();
  let query = QueryNode::MultiMatch {
    query: "rust alpha".into(),
    fields: vec![
      FieldSpec {
        field: "body".into(),
        boost: None,
      },
      FieldSpec {
        field: "tag".into(),
        boost: None,
      },
      FieldSpec {
        field: "year".into(),
        boost: None,
      },
    ],
    match_type: MultiMatchType::CrossFields,
    fuzziness: None,
    tie_breaker: None,
    operator: Some(MatchOperator::And),
    minimum_should_match: None,
    boost: None,
  };
  let first = reader.search(&request(query.clone())).unwrap();
  let second = reader.search(&request(query)).unwrap();
  assert_eq!(ranked(&first), ranked(&second));
  assert_eq!(first.hits.first().map(|h| h.doc_id.as_str()), Some("doc-1"));
}

#[test]
fn cross_fields_zero_token_analyzer_behavior_is_deterministic() {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx-zero-tokens");
  let mut schema = Schema::default_text_body();
  schema.analyzers.push(AnalyzerDef {
    name: "drop_all".into(),
    tokenizer: "default".into(),
    filters: vec![TokenFilterDef::Stopwords(StopwordsConfig::List(vec![
      "rust".into(),
      "alpha".into(),
      "systems".into(),
    ]))],
  });
  schema.text_fields.push(TextField {
    name: "title".into(),
    analyzer: "default".into(),
    search_analyzer: Some("drop_all".into()),
    stored: true,
    indexed: true,
    nullable: false,
    search_as_you_type: None,
  });
  schema.keyword_fields.push(KeywordField {
    name: "tag".into(),
    stored: true,
    indexed: true,
    fast: true,
    nullable: false,
  });
  let opts = IndexOptions {
    path: path.clone(),
    create_if_missing: true,
    enable_positions: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::Filesystem,
    checksum_policy: Default::default(),
    checksum_audit_failure_hook: None,
    read_only: false,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let idx = Index::create(&path, schema, opts).unwrap();
  let mut writer = idx.writer().unwrap();
  writer
    .add_document(&Document {
      fields: [
        ("_id".to_string(), serde_json::json!("doc-1")),
        ("title".to_string(), serde_json::json!("rust alpha")),
        ("body".to_string(), serde_json::json!("rust systems")),
        ("tag".to_string(), serde_json::json!("alpha")),
      ]
      .into_iter()
      .collect(),
    })
    .unwrap();
  writer
    .add_document(&Document {
      fields: [
        ("_id".to_string(), serde_json::json!("doc-2")),
        ("title".to_string(), serde_json::json!("rust alpha")),
        ("body".to_string(), serde_json::json!("rust systems")),
        ("tag".to_string(), serde_json::json!("beta")),
      ]
      .into_iter()
      .collect(),
    })
    .unwrap();
  writer.commit().unwrap();
  let reader = idx.reader().unwrap();
  let query = QueryNode::MultiMatch {
    query: "rust alpha".into(),
    fields: vec![
      FieldSpec {
        field: "title".into(),
        boost: None,
      },
      FieldSpec {
        field: "body".into(),
        boost: None,
      },
      FieldSpec {
        field: "tag".into(),
        boost: None,
      },
    ],
    match_type: MultiMatchType::CrossFields,
    fuzziness: None,
    tie_breaker: None,
    operator: Some(MatchOperator::And),
    minimum_should_match: None,
    boost: None,
  };
  let first = reader.search(&request(query.clone())).unwrap();
  let second = reader.search(&request(query)).unwrap();
  assert_eq!(ranked(&first), ranked(&second));
  assert_eq!(first.hits.first().map(|h| h.doc_id.as_str()), Some("doc-1"));
}

#[test]
fn multi_match_rejects_group_times_field_boost_overflow() {
  // BUG-396: `group.boost` (the combined query boost already validated by
  // `combine_boost` at plan time) and `field.boost` (per-field multi_match
  // boost validated by `validate_boost`) are each finite, but their f32
  // product in `expand_term_groups` can still overflow past `f32::MAX` to
  // `+inf`. Before the guard, that non-finite weight flowed into
  // `ScoredTerm.weight`, BM25 `score_tf`, and finally `Hit.score`, tripping
  // `serde_json` on the HTTP path (HTTP 500) or silently dropping the doc
  // under the in-flight BUG-381 heap guard. Expansion-time rejection
  // mirrors the `combine_boost` plan-time error shape for the same class
  // of bug on the next multiplication site.
  let (_tmp, reader) = setup_reader();
  let overflow = QueryNode::MultiMatch {
    query: "rust".into(),
    // Each per-field boost is finite on its own (f32::MAX ≈ 3.4e38), so
    // `validate_boost` accepts both. The query-level boost multiplies into
    // every per-field weight via `expand_term_groups`, and `1e20 * 1e20`
    // overflows f32.
    fields: vec![
      FieldSpec {
        field: "title".into(),
        boost: Some(1e20),
      },
      FieldSpec {
        field: "body".into(),
        boost: Some(1e20),
      },
    ],
    match_type: MultiMatchType::BestFields,
    fuzziness: None,
    tie_breaker: None,
    operator: None,
    minimum_should_match: None,
    boost: Some(1e20),
  };
  let err = reader.search(&request(overflow)).unwrap_err();
  assert!(
    err.to_string().contains("overflows"),
    "expected overflow error from the group.boost * field.boost guard, got: {err}",
  );
}

#[test]
fn multi_match_accepts_group_times_field_boost_within_range() {
  // Control case for BUG-396: a combined product that stays finite after
  // multiplication must search cleanly, so the new guard does not regress
  // legitimate per-field boosts.
  let (_tmp, reader) = setup_reader();
  let finite = QueryNode::MultiMatch {
    query: "rust".into(),
    fields: vec![
      FieldSpec {
        field: "title".into(),
        boost: Some(3.0),
      },
      FieldSpec {
        field: "body".into(),
        boost: Some(2.0),
      },
    ],
    match_type: MultiMatchType::BestFields,
    fuzziness: None,
    tie_breaker: None,
    operator: None,
    minimum_should_match: None,
    boost: Some(4.0),
  };
  let result = reader
    .search(&request(finite))
    .expect("finite group × field boost product must search cleanly");
  // Guard against vacuous pass: the corpus in `setup_reader` seeds
  // several docs whose `body`/`title` contain "rust", so this query must
  // return hits. A zero-hit result would skip the finitude loop below
  // and silently mask a guard that rejected every legitimate doc.
  assert!(
    !result.hits.is_empty(),
    "BUG-396 control case: legitimate in-range boosts must still return hits — the \
     new guard cannot accidentally reject every candidate",
  );
  let matching_ids = ids(&result);
  assert!(
    matching_ids.contains("doc-2") || matching_ids.contains("doc-4"),
    "BUG-396 control case: at least one known-\"rust\"-bearing doc must be returned, got {:?}",
    matching_ids,
  );
  for hit in result.hits.iter() {
    assert!(
      hit.score.is_finite(),
      "BUG-396: legitimate in-range boosts must yield finite scores (got {})",
      hit.score,
    );
  }
}

#[test]
fn multi_match_ignores_overflow_boost_on_numeric_field_kind() {
  // BUG-396: the `combine_boost(group.boost, field.boost)?` guard must be
  // scoped to Text/Keyword field kinds. `Numeric` and `Unknown` fields
  // are no-ops inside `expand_term_groups` (nothing consumes `weight`),
  // so an overflowing boost on one of those fields must not abort the
  // whole query. Raising there would be a behavior regression for
  // mixed / typoed field lists where Text fields still have legitimate
  // finite boosts. Verifies Codex's P2 comment on the initial patch.
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx-bug-396-numeric-nop");
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "year".into(),
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
    checksum_policy: Default::default(),
    checksum_audit_failure_hook: None,
    read_only: false,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let idx = Index::create(&path, schema, opts).unwrap();
  let mut writer = idx.writer().unwrap();
  writer
    .add_document(&Document {
      fields: [
        ("_id".to_string(), serde_json::json!("doc-1")),
        ("body".to_string(), serde_json::json!("rust systems")),
        ("year".to_string(), serde_json::json!(2024)),
      ]
      .into_iter()
      .collect(),
    })
    .unwrap();
  writer.commit().unwrap();
  let reader = idx.reader().unwrap();
  // `body` carries a sane 2.0 boost; `year` (numeric) carries an
  // overflow-trigger 1e20 with a query-level 1e20 boost. The guard must
  // not fire on the numeric field because its branch discards `weight`.
  let query = QueryNode::MultiMatch {
    query: "rust".into(),
    fields: vec![
      FieldSpec {
        field: "body".into(),
        boost: Some(2.0),
      },
      FieldSpec {
        field: "year".into(),
        boost: Some(1e20),
      },
    ],
    match_type: MultiMatchType::BestFields,
    fuzziness: None,
    tie_breaker: None,
    operator: None,
    minimum_should_match: None,
    boost: Some(1e20),
  };
  let result = reader
    .search(&request(query))
    .expect("overflow boost on Numeric field must be a no-op, not a hard error");
  assert!(
    !result.hits.is_empty(),
    "BUG-396: mixed field list must still surface the Text-field match",
  );
  for hit in result.hits.iter() {
    assert!(
      hit.score.is_finite(),
      "BUG-396: Text field's finite weight must still produce finite scores (got {})",
      hit.score,
    );
  }
}

#[test]
fn bool_rejects_accumulated_term_weight_overflow() {
  // BUG-401: `term_weights` in `IndexReader::search` sums per-term weights
  // with `entry.1 += term.weight` when multiple query clauses resolve to the
  // same `field:term` key. Each individual boost is validated finite by
  // `combine_boost` at plan/expansion time (BUG-381 / BUG-396), but the sum
  // of N finite f32 values can still overflow `f32::MAX` (~3.4e38) to `+Inf`.
  // Before the guard, that `Inf` propagated into `ScoredTerm.weight` and
  // through BM25 `score_tf` into the WAND/brute-force heap, corrupting
  // ranking (WAND+Sum silently clamped to 0.0 per BUG-364) or producing an
  // `Inf` in the serialised response (HTTP 500 on `serde_json::to_vec`).
  // The guard must fire at accumulation time and mirror the
  // `combine_boost` error shape.
  let (_tmp, reader) = setup_reader();
  // Two `should` clauses, each matching `body:rust` with a boost that is
  // individually finite (f32::MAX ≈ 3.4e38). Their accumulated sum
  // (3.6e38) exceeds f32::MAX and overflows to +Inf.
  let query = QueryNode::Bool {
    must: Vec::new(),
    should: vec![
      QueryNode::Term {
        field: "body".into(),
        value: "rust".into(),
        boost: Some(1.8e38),
      },
      QueryNode::Term {
        field: "body".into(),
        value: "rust".into(),
        boost: Some(1.8e38),
      },
    ],
    must_not: Vec::new(),
    filter: Vec::new(),
    minimum_should_match: None,
    boost: None,
  };
  let err = reader.search(&request(query)).unwrap_err();
  let msg = err.to_string();
  assert!(
    msg.contains("overflows"),
    "expected overflow error from the term_weights accumulation guard, got: {msg}",
  );
  assert!(
    msg.contains("body:rust"),
    "error should name the field:term key that overflowed, got: {msg}",
  );
}

#[test]
fn bool_accepts_accumulated_term_weight_within_range() {
  // Control case for BUG-401: two clauses sharing a key whose summed weight
  // stays finite must continue to search cleanly. Guards against a guard
  // that rejects every legitimate multi-clause query.
  let (_tmp, reader) = setup_reader();
  let query = QueryNode::Bool {
    must: Vec::new(),
    should: vec![
      QueryNode::Term {
        field: "body".into(),
        value: "rust".into(),
        boost: Some(2.0),
      },
      QueryNode::Term {
        field: "body".into(),
        value: "rust".into(),
        boost: Some(3.0),
      },
    ],
    must_not: Vec::new(),
    filter: Vec::new(),
    minimum_should_match: None,
    boost: None,
  };
  let result = reader
    .search(&request(query))
    .expect("finite summed weight must search cleanly");
  assert!(
    !result.hits.is_empty(),
    "BUG-401 control case: legitimate in-range boosts must still return hits",
  );
  for hit in result.hits.iter() {
    assert!(
      hit.score.is_finite(),
      "BUG-401 control case: legitimate summed weights must yield finite scores (got {})",
      hit.score,
    );
  }
}
