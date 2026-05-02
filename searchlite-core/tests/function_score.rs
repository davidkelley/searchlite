use std::collections::BTreeMap;

use searchlite_core::api::types::{
  DecayFunction, Document, ExecutionStrategy, FieldValueModifier, Filter, FunctionBoostMode,
  FunctionScoreMode, FunctionSpec, IndexOptions, KeywordField, NumericField, Query, QueryNode,
  RankFeatureModifier, RescoreMode, RescoreRequest, Schema, SearchRequest, StorageType,
};
use searchlite_core::api::{Index, SearchResult};

fn doc(id: &str, body: &str, popularity: i64, lang: &str) -> Document {
  Document {
    fields: [
      ("_id".to_string(), serde_json::json!(id)),
      ("body".to_string(), serde_json::json!(body)),
      ("popularity".to_string(), serde_json::json!(popularity)),
      ("lang".to_string(), serde_json::json!(lang)),
    ]
    .into_iter()
    .collect(),
  }
}

fn setup_reader() -> searchlite_core::api::IndexReader {
  let path = tempfile::tempdir().unwrap().path().join("idx");
  let mut schema = Schema::default_text_body();
  schema.keyword_fields.push(KeywordField {
    name: "lang".into(),
    stored: true,
    indexed: true,
    fast: true,
    nullable: false,
  });
  schema.numeric_fields.push(NumericField {
    name: "popularity".into(),
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
  let docs = vec![
    doc("doc-1", "rust fast", 10, "en"),
    doc("doc-2", "rust slow", 1, "en"),
    doc("doc-3", "boring", 5, "fr"),
  ];
  for d in docs {
    writer.add_document(&d).unwrap();
  }
  writer.commit().unwrap();
  idx.reader().unwrap()
}

fn base_request(query: impl Into<Query>) -> SearchRequest {
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

fn ids(result: &SearchResult) -> Vec<String> {
  result.hits.iter().map(|h| h.doc_id.clone()).collect()
}

#[test]
fn constant_score_applies_fixed_boost() {
  let reader = setup_reader();
  let req = base_request(QueryNode::ConstantScore {
    filter: Filter::KeywordEq {
      field: "lang".into(),
      value: "en".into(),
    },
    boost: Some(2.5),
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 2);
  for hit in resp.hits {
    assert!((hit.score - 2.5).abs() < 1e-6);
  }
}

#[test]
fn function_score_replaces_score_with_weight() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Weight {
      weight: 3.0,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  for hit in resp.hits {
    assert!((hit.score - 3.0).abs() < 1e-6);
  }
}

#[test]
fn field_value_factor_orders_by_field() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::FieldValueFactor {
      field: "popularity".into(),
      factor: 1.0,
      modifier: Some(FieldValueModifier::None),
      missing: None,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(ids(&resp), vec!["doc-1", "doc-3", "doc-2"]);
}

#[test]
fn rescore_reorders_within_window() {
  let reader = setup_reader();
  let mut req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Weight {
      weight: 1.0,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  req.rescore = Some(RescoreRequest {
    window_size: 2,
    query: QueryNode::Term {
      field: "body".into(),
      value: "fast".into(),
      boost: None,
    },
    score_mode: RescoreMode::Total,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(ids(&resp), vec!["doc-1", "doc-2", "doc-3"]);
  assert!(resp.hits[0].score > resp.hits[1].score);
}

#[test]
fn explain_returns_function_details() {
  let reader = setup_reader();
  let mut req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Weight {
      weight: 2.0,
      filter: Some(Filter::KeywordEq {
        field: "lang".into(),
        value: "en".into(),
      }),
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  req.explain = true;
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  let mut matched = 0;
  for hit in resp.hits {
    let expl = hit.explanation.expect("missing explanation");
    match hit.doc_id.as_str() {
      "doc-1" | "doc-2" => {
        matched += 1;
        assert!((expl.final_score - 2.0).abs() < 1e-6);
        assert!(!expl.functions.is_empty());
      }
      "doc-3" => {
        assert!(expl.functions.is_empty());
        assert!((expl.final_score - 1.0).abs() < 1e-6);
      }
      _ => panic!("unexpected doc {}", hit.doc_id),
    }
  }
  assert_eq!(matched, 2);
}

#[test]
fn field_value_modifier_variants_apply() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::FieldValueFactor {
      field: "popularity".into(),
      factor: 1.0,
      modifier: Some(FieldValueModifier::Reciprocal),
      missing: None,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  // Reciprocal yields higher scores for smaller values.
  assert_eq!(ids(&resp), vec!["doc-2", "doc-3", "doc-1"]);
  assert!(resp.hits[0].score > resp.hits[1].score);
  assert!(resp.hits[1].score > resp.hits[2].score);
}

#[test]
fn decay_function_orders_by_distance() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Decay {
      field: "popularity".into(),
      origin: 0.0,
      scale: 10.0,
      offset: Some(0.0),
      decay: Some(0.5),
      function: Some(DecayFunction::Linear),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  // popularity: doc-2=1, doc-3=5, doc-1=10; linear decay should rank by proximity to origin.
  assert_eq!(ids(&resp), vec!["doc-2", "doc-3", "doc-1"]);
  assert!(resp.hits[0].score > resp.hits[1].score);
  assert!(resp.hits[1].score > resp.hits[2].score);
}

// BUG-373: `serde_wasm_bindgen` (unlike `serde_json`) passes JS `NaN`/`Infinity`
// straight into `f64`, so a decay `origin`/`offset` arriving via the WASM
// binding could poison the scoring formula. `(value - NaN).abs() = NaN`,
// then `NaN.max(0.0) = 0.0` (IEEE-754 maxNum picks the non-NaN), which
// collapses every document to `decay^0 = 1.0` — a silently wrong result.
// `compile_functions` must reject non-finite `origin` and `offset` at the
// boundary, matching the existing `scale` guard.
#[test]
fn decay_rejects_non_finite_origin_nan() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Decay {
      field: "popularity".into(),
      origin: f64::NAN,
      scale: 10.0,
      offset: Some(0.0),
      decay: Some(0.5),
      function: Some(DecayFunction::Linear),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("decay origin must be finite"),
    "expected origin-finitude error, got {err}"
  );
}

#[test]
fn decay_rejects_non_finite_origin_infinity() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Decay {
      field: "popularity".into(),
      origin: f64::INFINITY,
      scale: 10.0,
      offset: Some(0.0),
      decay: Some(0.5),
      function: Some(DecayFunction::Linear),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("decay origin must be finite"),
    "expected origin-finitude error, got {err}"
  );
}

#[test]
fn decay_rejects_non_finite_offset_nan() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Decay {
      field: "popularity".into(),
      origin: 0.0,
      scale: 10.0,
      offset: Some(f64::NAN),
      decay: Some(0.5),
      function: Some(DecayFunction::Linear),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("decay offset must be finite"),
    "expected offset-finitude error, got {err}"
  );
}

#[test]
fn decay_rejects_non_finite_origin_negative_infinity() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Decay {
      field: "popularity".into(),
      origin: f64::NEG_INFINITY,
      scale: 10.0,
      offset: Some(0.0),
      decay: Some(0.5),
      function: Some(DecayFunction::Linear),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("decay origin must be finite"),
    "expected origin-finitude error, got {err}"
  );
}

#[test]
fn decay_rejects_non_finite_offset_infinity() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Decay {
      field: "popularity".into(),
      origin: 0.0,
      scale: 10.0,
      offset: Some(f64::INFINITY),
      decay: Some(0.5),
      function: Some(DecayFunction::Linear),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("decay offset must be finite"),
    "expected offset-finitude error, got {err}"
  );
}

#[test]
fn decay_rejects_non_finite_offset_negative_infinity() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Decay {
      field: "popularity".into(),
      origin: 0.0,
      scale: 10.0,
      offset: Some(f64::NEG_INFINITY),
      decay: Some(0.5),
      function: Some(DecayFunction::Linear),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("decay offset must be finite"),
    "expected offset-finitude error, got {err}"
  );
}

// BUG-379: the existing `decay <= 0.0 || decay > 1.0` range check uses
// ordered comparisons that are always `false` for NaN under IEEE-754, so a
// NaN `decay` factor arriving via the WASM binding (where
// `serde_wasm_bindgen` passes JS `NaN` straight into `f64`) slipped past
// the guard. For `Linear`, `NaN.max(0.0) = 0.0` collapses every document to
// score 0.0; for `Exp`/`Gauss`, `NaN.powf(norm)` is NaN and documents are
// silently dropped by the `is_finite()` evaluation guard. Both outcomes
// are silently wrong — `compile_functions` must reject non-finite `decay`
// at the boundary, matching the origin/scale/offset guards.
#[test]
fn decay_rejects_non_finite_decay_factor_nan() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Decay {
      field: "popularity".into(),
      origin: 0.0,
      scale: 10.0,
      offset: Some(0.0),
      decay: Some(f64::NAN),
      function: Some(DecayFunction::Linear),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("decay factor must be finite"),
    "expected decay-factor-finitude error, got {err}"
  );
}

#[test]
fn decay_rejects_non_finite_decay_factor_infinity() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Decay {
      field: "popularity".into(),
      origin: 0.0,
      scale: 10.0,
      offset: Some(0.0),
      decay: Some(f64::INFINITY),
      function: Some(DecayFunction::Linear),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("decay factor must be finite"),
    "expected decay-factor-finitude error, got {err}"
  );
}

#[test]
fn decay_rejects_non_finite_decay_factor_negative_infinity() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Decay {
      field: "popularity".into(),
      origin: 0.0,
      scale: 10.0,
      offset: Some(0.0),
      decay: Some(f64::NEG_INFINITY),
      function: Some(DecayFunction::Linear),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("decay factor must be finite"),
    "expected decay-factor-finitude error, got {err}"
  );
}

// BUG-392: `compile_functions` validates `FieldValueFactor.factor` for
// finiteness but not `FieldValueFactor.missing`. A non-finite `missing`
// (NaN, ±Infinity) propagates through `raw * factor` for docs that omit the
// numeric field; the downstream `scaled.is_finite()` guard then silently
// drops the doc's function contribution. Mirrors the symmetric guard on
// `rank_feature.missing` and the `decay` parameter guards: reject
// non-finite `missing` at the plan boundary with an actionable error.
#[test]
fn field_value_factor_rejects_non_finite_missing_nan() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::FieldValueFactor {
      field: "popularity".into(),
      factor: 1.0,
      modifier: Some(FieldValueModifier::None),
      missing: Some(f64::NAN),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("field_value_factor `missing` must be finite"),
    "expected missing-finitude error, got {err}"
  );
}

#[test]
fn field_value_factor_rejects_non_finite_missing_infinity() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::FieldValueFactor {
      field: "popularity".into(),
      factor: 1.0,
      modifier: Some(FieldValueModifier::None),
      missing: Some(f64::INFINITY),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("field_value_factor `missing` must be finite"),
    "expected missing-finitude error, got {err}"
  );
}

#[test]
fn field_value_factor_rejects_non_finite_missing_negative_infinity() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::FieldValueFactor {
      field: "popularity".into(),
      factor: 1.0,
      modifier: Some(FieldValueModifier::None),
      missing: Some(f64::NEG_INFINITY),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("field_value_factor `missing` must be finite"),
    "expected missing-finitude error, got {err}"
  );
}

#[test]
fn field_value_factor_accepts_finite_missing() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::FieldValueFactor {
      field: "popularity".into(),
      factor: 1.0,
      modifier: Some(FieldValueModifier::None),
      missing: Some(0.0),
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
}

#[test]
fn min_score_branch_does_not_drop_other_clauses() {
  let reader = setup_reader();
  let req = base_request(QueryNode::Bool {
    must: Vec::new(),
    should: vec![
      QueryNode::Term {
        field: "body".into(),
        value: "fast".into(),
        boost: None,
      },
      QueryNode::FunctionScore {
        query: Box::new(QueryNode::MatchAll { boost: None }),
        functions: vec![FunctionSpec::Weight {
          weight: 1.0,
          filter: None,
        }],
        score_mode: Some(FunctionScoreMode::Sum),
        boost_mode: Some(FunctionBoostMode::Multiply),
        max_boost: None,
        min_score: Some(10.0),
        boost: None,
      },
    ],
    must_not: Vec::new(),
    filter: Vec::new(),
    minimum_should_match: Some(1),
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(ids(&resp), vec!["doc-1"]);
  assert!(resp.hits[0].score > 0.0);
}

#[test]
fn rescore_min_score_filters_hits() {
  let reader = setup_reader();
  let mut req = base_request(QueryNode::MatchAll { boost: None });
  req.rescore = Some(RescoreRequest {
    window_size: 3,
    query: QueryNode::FunctionScore {
      query: Box::new(QueryNode::MatchAll { boost: None }),
      functions: vec![FunctionSpec::Weight {
        weight: 2.0,
        filter: Some(Filter::KeywordEq {
          field: "lang".into(),
          value: "en".into(),
        }),
      }],
      score_mode: Some(FunctionScoreMode::Sum),
      boost_mode: Some(FunctionBoostMode::Multiply),
      max_boost: None,
      min_score: Some(2.0),
      boost: None,
    },
    score_mode: RescoreMode::Total,
  });
  let resp = reader.search(&req).unwrap();
  let ids = ids(&resp);
  assert_eq!(ids.len(), 2);
  assert!(ids.contains(&"doc-1".to_string()));
  assert!(ids.contains(&"doc-2".to_string()));
  assert!(!ids.contains(&"doc-3".to_string()));
}

#[test]
fn rank_feature_uses_numeric_fast_field() {
  let reader = setup_reader();
  let req = base_request(QueryNode::RankFeature {
    field: "popularity".into(),
    boost: Some(1.0),
    modifier: Some(RankFeatureModifier::Sqrt),
    missing: Some(0.0),
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(ids(&resp), vec!["doc-1", "doc-3", "doc-2"]);
  let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();
  assert!(scores[0] > scores[1] && scores[1] > scores[2]);
}

#[test]
fn rank_feature_log_modifier_uses_log10_of_value() {
  let reader = setup_reader();
  let req = base_request(QueryNode::RankFeature {
    field: "popularity".into(),
    boost: Some(1.0),
    modifier: Some(RankFeatureModifier::Log),
    missing: Some(0.0),
  });
  let resp = reader.search(&req).unwrap();
  // popularity: doc-1=10, doc-3=5, doc-2=1
  // Log = log10(value): log10(10) = 1.0, log10(5) ≈ 0.699, log10(1) = 0.0
  assert_eq!(ids(&resp), vec!["doc-1", "doc-3", "doc-2"]);
  let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();
  assert!(
    (scores[0] - 10_f32.log10()).abs() < 1e-6,
    "doc-1: {}",
    scores[0]
  );
  assert!(
    (scores[1] - 5_f32.log10()).abs() < 1e-6,
    "doc-3: {}",
    scores[1]
  );
  assert!(
    (scores[2] - 1_f32.log10()).abs() < 1e-6,
    "doc-2: {}",
    scores[2]
  );
}

#[test]
fn rank_feature_log1p_modifier_uses_log10_of_one_plus_value() {
  let reader = setup_reader();
  let req = base_request(QueryNode::RankFeature {
    field: "popularity".into(),
    boost: Some(1.0),
    modifier: Some(RankFeatureModifier::Log1p),
    missing: Some(0.0),
  });
  let resp = reader.search(&req).unwrap();
  // popularity: doc-1=10, doc-3=5, doc-2=1
  // Log1p = log10(1 + value): log10(11) ≈ 1.041, log10(6) ≈ 0.778, log10(2) ≈ 0.301
  assert_eq!(ids(&resp), vec!["doc-1", "doc-3", "doc-2"]);
  let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();
  assert!(
    (scores[0] - 11_f32.log10()).abs() < 1e-6,
    "doc-1: {}",
    scores[0]
  );
  assert!(
    (scores[1] - 6_f32.log10()).abs() < 1e-6,
    "doc-3: {}",
    scores[1]
  );
  assert!(
    (scores[2] - 2_f32.log10()).abs() < 1e-6,
    "doc-2: {}",
    scores[2]
  );
}

#[test]
fn log2p_modifier_uses_log10_of_value_plus_two() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::FieldValueFactor {
      field: "popularity".into(),
      factor: 1.0,
      modifier: Some(FieldValueModifier::Log2p),
      missing: None,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  // popularity: doc-1=10, doc-3=5, doc-2=1
  // Log2p = log10(value + 2): log10(12) ≈ 1.079, log10(7) ≈ 0.845, log10(3) ≈ 0.477
  assert_eq!(ids(&resp), vec!["doc-1", "doc-3", "doc-2"]);
  let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();
  assert!(
    (scores[0] - 12_f32.log10()).abs() < 1e-6,
    "doc-1: {}",
    scores[0]
  );
  assert!(
    (scores[1] - 7_f32.log10()).abs() < 1e-6,
    "doc-3: {}",
    scores[1]
  );
  assert!(
    (scores[2] - 3_f32.log10()).abs() < 1e-6,
    "doc-2: {}",
    scores[2]
  );
}

#[test]
fn log_modifier_uses_log10_of_value() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::FieldValueFactor {
      field: "popularity".into(),
      factor: 1.0,
      modifier: Some(FieldValueModifier::Log),
      missing: None,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  // popularity: doc-1=10, doc-3=5, doc-2=1
  // Log = log10(value): log10(10) = 1.0, log10(5) ≈ 0.699, log10(1) = 0.0
  assert_eq!(ids(&resp), vec!["doc-1", "doc-3", "doc-2"]);
  let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();
  assert!(
    (scores[0] - 10_f32.log10()).abs() < 1e-6,
    "doc-1: {}",
    scores[0]
  );
  assert!(
    (scores[1] - 5_f32.log10()).abs() < 1e-6,
    "doc-3: {}",
    scores[1]
  );
  assert!(
    (scores[2] - 1_f32.log10()).abs() < 1e-6,
    "doc-2: {}",
    scores[2]
  );
}

#[test]
fn log1p_modifier_uses_log10_of_one_plus_value() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::FieldValueFactor {
      field: "popularity".into(),
      factor: 1.0,
      modifier: Some(FieldValueModifier::Log1p),
      missing: None,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  // popularity: doc-1=10, doc-3=5, doc-2=1
  // Log1p = log10(1 + value): log10(11) ≈ 1.041, log10(6) ≈ 0.778, log10(2) ≈ 0.301
  assert_eq!(ids(&resp), vec!["doc-1", "doc-3", "doc-2"]);
  let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();
  assert!(
    (scores[0] - 11_f32.log10()).abs() < 1e-6,
    "doc-1: {}",
    scores[0]
  );
  assert!(
    (scores[1] - 6_f32.log10()).abs() < 1e-6,
    "doc-3: {}",
    scores[1]
  );
  assert!(
    (scores[2] - 2_f32.log10()).abs() < 1e-6,
    "doc-2: {}",
    scores[2]
  );
}

#[test]
fn script_score_evaluates_expression_with_score_and_field() {
  let reader = setup_reader();
  let req = base_request(QueryNode::ScriptScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    script: "_score + popularity * 0.1".into(),
    params: None,
    boost: Some(1.0),
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(ids(&resp), vec!["doc-1", "doc-3", "doc-2"]);
  let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();
  assert!(scores[0] > scores[1] && scores[1] > scores[2]);
}

// Regression: unary `+` in operand position was emitted as binary `Add`,
// which underflowed the RPN stack. `CompiledScript::evaluate()` returned
// `None`, so the script_score function contributed no score and the hit
// was effectively dropped from the result set. See BUG-309.
#[test]
fn script_score_accepts_unary_plus_on_number_literal() {
  let reader = setup_reader();
  let req = base_request(QueryNode::ScriptScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    script: "+1 + +popularity".into(),
    params: None,
    boost: Some(1.0),
  });
  let resp = reader.search(&req).unwrap();
  // Expected: 1 + popularity → doc-1=11, doc-3=6, doc-2=2
  assert_eq!(ids(&resp), vec!["doc-1", "doc-3", "doc-2"]);
  let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();
  assert!((scores[0] - 11.0).abs() < 1e-6, "doc-1: {}", scores[0]);
  assert!((scores[1] - 6.0).abs() < 1e-6, "doc-3: {}", scores[1]);
  assert!((scores[2] - 2.0).abs() < 1e-6, "doc-2: {}", scores[2]);
}

#[test]
fn script_score_accepts_unary_plus_in_parenthesized_group() {
  let reader = setup_reader();
  let req = base_request(QueryNode::ScriptScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    script: "(-popularity) + (+popularity) + 100".into(),
    params: None,
    boost: Some(1.0),
  });
  let resp = reader.search(&req).unwrap();
  // (-pop) + (+pop) + 100 = 100 for every doc
  let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();
  assert_eq!(scores.len(), 3);
  for s in &scores {
    assert!((s - 100.0).abs() < 1e-6, "expected 100.0, got {s}");
  }
}

#[test]
fn script_score_accepts_binary_plus_after_unary_plus() {
  let reader = setup_reader();
  let req = base_request(QueryNode::ScriptScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    script: "1 + +2".into(),
    params: None,
    boost: Some(1.0),
  });
  let resp = reader.search(&req).unwrap();
  // 1 + (+2) = 3 for every doc
  let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();
  assert_eq!(scores.len(), 3);
  for s in &scores {
    assert!((s - 3.0).abs() < 1e-6, "expected 3.0, got {s}");
  }
}

#[test]
fn rescore_multiply_zeros_non_matching_docs() {
  let reader = setup_reader();
  // Give every doc a constant score of 5.0 via function_score + Replace.
  let mut req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Weight {
      weight: 5.0,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  // Rescore with a term that only matches doc-1 ("rust fast") and doc-2 ("rust slow").
  // doc-3 ("boring") does not match and should receive 5.0 * 0.0 = 0.0.
  req.rescore = Some(RescoreRequest {
    window_size: 10,
    query: QueryNode::Term {
      field: "body".into(),
      value: "rust".into(),
      boost: None,
    },
    score_mode: RescoreMode::Multiply,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  // Find doc-3's score — it must be zero because it did not match the rescore query.
  let doc3 = resp.hits.iter().find(|h| h.doc_id == "doc-3").unwrap();
  assert!(
    doc3.score.abs() < 1e-6,
    "expected doc-3 score ≈ 0.0 under Multiply, got {}",
    doc3.score
  );
  // doc-1 and doc-2 matched the rescore query, so they should have positive scores.
  for hit in resp.hits.iter().filter(|h| h.doc_id != "doc-3") {
    assert!(
      hit.score > 0.0,
      "expected positive rescore score for {}, got {}",
      hit.doc_id,
      hit.score
    );
  }
}

#[test]
fn rescore_min_zeros_non_matching_docs() {
  let reader = setup_reader();
  let mut req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Weight {
      weight: 5.0,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  // Rescore with Min mode — non-matching docs get min(5.0, 0.0) = 0.0.
  req.rescore = Some(RescoreRequest {
    window_size: 10,
    query: QueryNode::Term {
      field: "body".into(),
      value: "rust".into(),
      boost: None,
    },
    score_mode: RescoreMode::Min,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  let doc3 = resp.hits.iter().find(|h| h.doc_id == "doc-3").unwrap();
  assert!(
    doc3.score.abs() < 1e-6,
    "expected doc-3 score ≈ 0.0 under Min, got {}",
    doc3.score
  );
}

#[test]
fn rescore_total_preserves_score_for_non_matching_docs() {
  let reader = setup_reader();
  let mut req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Weight {
      weight: 5.0,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  // Rescore with Total mode — non-matching docs get 5.0 + 0.0 = 5.0 (unchanged).
  req.rescore = Some(RescoreRequest {
    window_size: 10,
    query: QueryNode::Term {
      field: "body".into(),
      value: "rust".into(),
      boost: None,
    },
    score_mode: RescoreMode::Total,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  let doc3 = resp.hits.iter().find(|h| h.doc_id == "doc-3").unwrap();
  assert!(
    (doc3.score - 5.0).abs() < 1e-6,
    "expected doc-3 score ≈ 5.0 under Total, got {}",
    doc3.score
  );
}

#[test]
fn rescore_max_preserves_score_for_non_matching_docs() {
  let reader = setup_reader();
  let mut req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Weight {
      weight: 5.0,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  // Rescore with Max mode — non-matching docs get max(5.0, 0.0) = 5.0 (unchanged).
  req.rescore = Some(RescoreRequest {
    window_size: 10,
    query: QueryNode::Term {
      field: "body".into(),
      value: "rust".into(),
      boost: None,
    },
    score_mode: RescoreMode::Max,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  let doc3 = resp.hits.iter().find(|h| h.doc_id == "doc-3").unwrap();
  assert!(
    (doc3.score - 5.0).abs() < 1e-6,
    "expected doc-3 score ≈ 5.0 under Max, got {}",
    doc3.score
  );
}

#[test]
fn rescore_explain_consistent_for_non_matching_docs() {
  let reader = setup_reader();
  let mut req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Weight {
      weight: 5.0,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  req.rescore = Some(RescoreRequest {
    window_size: 10,
    query: QueryNode::Term {
      field: "body".into(),
      value: "rust".into(),
      boost: None,
    },
    score_mode: RescoreMode::Multiply,
  });
  req.explain = true;
  let resp = reader.search(&req).unwrap();
  // doc-3 does not match the rescore query — its explanation must reflect
  // the zero-contribution rescore and the combined score must match hit.score.
  let doc3 = resp.hits.iter().find(|h| h.doc_id == "doc-3").unwrap();
  let expl = doc3
    .explanation
    .as_ref()
    .expect("missing explanation for doc-3");
  let rescore_expl = expl
    .rescore
    .as_ref()
    .expect("missing rescore explanation for doc-3");
  assert!(
    rescore_expl.rescore_score.abs() < 1e-6,
    "non-matching rescore_score should be 0.0, got {}",
    rescore_expl.rescore_score
  );
  assert!(
    (rescore_expl.combined_score - doc3.score).abs() < 1e-6,
    "combined_score ({}) should match hit.score ({})",
    rescore_expl.combined_score,
    doc3.score
  );
  assert!(
    (expl.final_score - doc3.score).abs() < 1e-6,
    "final_score ({}) should match hit.score ({})",
    expl.final_score,
    doc3.score
  );
}

#[test]
fn max_boost_caps_function_score_before_boost_mode_sum() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Weight {
      weight: 10.0,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Sum),
    max_boost: Some(5.0),
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  // base=1.0, func=10.0, max_boost=5.0 → capped_func=5.0 → 1.0+5.0=6.0
  for hit in &resp.hits {
    assert!(
      (hit.score - 6.0).abs() < 1e-6,
      "expected 6.0 (base + capped func), got {}",
      hit.score
    );
  }
}

#[test]
fn max_boost_caps_function_score_before_boost_mode_multiply() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::ConstantScore {
      filter: Filter::KeywordEq {
        field: "lang".into(),
        value: "en".into(),
      },
      boost: Some(2.0),
    }),
    functions: vec![FunctionSpec::Weight {
      weight: 10.0,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Multiply),
    max_boost: Some(5.0),
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 2);
  // base=2.0 (ConstantScore), func=10.0, max_boost=5.0
  // Correct: 2.0 * min(10.0, 5.0) = 2.0 * 5.0 = 10.0
  // Old buggy code: min(2.0 * 10.0, 5.0) = 5.0
  for hit in &resp.hits {
    assert!(
      (hit.score - 10.0).abs() < 1e-6,
      "expected 2.0 * capped(5.0) = 10.0, got {}",
      hit.score
    );
  }
}

// Regression: `combine_function_scores` and the `FunctionScore` branch of
// `evaluate_compiled_score` previously lacked finitude guards. When
// individually finite function values overflow `f32` during combine (e.g.
// Multiply of two 1e20 weights → 1e40 > f32::MAX), the result leaked out
// as `Some(f32::INFINITY)` and corrupted sort ordering. The fix is to
// reject non-finite combined scores (return `None`) so overflowing hits
// are excluded from the result set, mirroring the RankFeature guard. See
// BUG-315.
#[test]
fn function_score_multiply_overflow_is_excluded() {
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![
      FunctionSpec::Weight {
        weight: 1.0e20,
        filter: None,
      },
      FunctionSpec::Weight {
        weight: 1.0e20,
        filter: None,
      },
    ],
    score_mode: Some(FunctionScoreMode::Multiply),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  // 1e20 * 1e20 = 1e40 overflows f32 to infinity; the combined score is
  // non-finite after capping, so evaluate_compiled_score returns None and
  // every hit is excluded from the result set.
  assert!(
    resp.hits.is_empty(),
    "expected no hits when combine overflows to infinity, got {} hits with scores {:?}",
    resp.hits.len(),
    resp.hits.iter().map(|h| h.score).collect::<Vec<_>>()
  );
}

#[test]
fn function_score_sum_overflow_is_excluded() {
  let reader = setup_reader();
  // Two Weight functions whose Sum overflows f32::MAX (≈ 3.4e38).
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![
      FunctionSpec::Weight {
        weight: f32::MAX,
        filter: None,
      },
      FunctionSpec::Weight {
        weight: f32::MAX,
        filter: None,
      },
    ],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert!(
    resp.hits.is_empty(),
    "expected no hits when Sum combine overflows to infinity, got {} hits with scores {:?}",
    resp.hits.len(),
    resp.hits.iter().map(|h| h.score).collect::<Vec<_>>()
  );
}

#[test]
fn function_score_boost_multiplier_overflow_is_excluded() {
  let reader = setup_reader();
  // A single function value that is finite, but the final `combined *= boost`
  // step overflows. The second finitude guard (after boost) must catch it.
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::Weight {
      weight: 1.0e30,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: Some(1.0e30),
  });
  let resp = reader.search(&req).unwrap();
  assert!(
    resp.hits.is_empty(),
    "expected no hits when final boost multiply overflows, got {} hits with scores {:?}",
    resp.hits.len(),
    resp.hits.iter().map(|h| h.score).collect::<Vec<_>>()
  );
}

#[test]
fn function_score_max_boost_caps_combine_overflow_to_finite() {
  let reader = setup_reader();
  // When `max_boost` is set, it must cap the combined function score even
  // if the combine step overflowed to `f32::INFINITY`. This works because
  // `f32::INFINITY.min(finite) == finite`, so the doc survives with a
  // finite, capped score. This test documents that `max_boost` protects
  // against combine overflow, which was the existing workaround noted in
  // BUG-315 for users who were aware of the issue.
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![
      FunctionSpec::Weight {
        weight: 1.0e20,
        filter: None,
      },
      FunctionSpec::Weight {
        weight: 1.0e20,
        filter: None,
      },
    ],
    score_mode: Some(FunctionScoreMode::Multiply),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: Some(100.0),
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  for hit in &resp.hits {
    assert!(
      hit.score.is_finite(),
      "expected finite score after max_boost capping, got {}",
      hit.score
    );
    assert!(
      (hit.score - 100.0).abs() < 1e-6,
      "expected score to be capped at max_boost=100.0, got {}",
      hit.score
    );
  }
}

#[test]
fn rescore_sort_window_excludes_non_rescored_after_removal() {
  // Regression test for BUG-291: when rescore drops hits from within the
  // window (via function_score + min_score causing evaluate_compiled_score
  // to return None), the sort window must shrink by the number of removed
  // hits. Otherwise non-rescored hits that shifted left into the original
  // window bounds get re-sorted against rescored hits whose scores are on
  // a different scale, producing incorrect ranking.
  let reader = setup_reader();
  // Base query: MatchAll scored by popularity (field_value_factor, Replace).
  // Raw scores → doc-1: 10, doc-3: 5, doc-2: 1. Initial order: [doc-1, doc-3, doc-2].
  let mut req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::FieldValueFactor {
      field: "popularity".into(),
      factor: 1.0,
      modifier: Some(FieldValueModifier::None),
      missing: None,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  // Rescore window covers only the first two hits (doc-1 and doc-3). The
  // rescore query yields 0.01 for doc-1 and 0.005 for doc-3; min_score=0.008
  // drops doc-3 (returns None) but keeps doc-1. With RescoreMode::Multiply:
  // doc-1 combined = 10 * 0.01 = 0.1. doc-2 sits outside the window and
  // keeps its raw score of 1.0. If the sort window is not shrunk, the
  // post-removal sort would re-order doc-2 (1.0) above doc-1 (0.1), even
  // though doc-2 was never rescored.
  req.rescore = Some(RescoreRequest {
    window_size: 2,
    query: QueryNode::FunctionScore {
      query: Box::new(QueryNode::MatchAll { boost: None }),
      functions: vec![FunctionSpec::FieldValueFactor {
        field: "popularity".into(),
        factor: 0.001,
        modifier: Some(FieldValueModifier::None),
        missing: None,
        filter: None,
      }],
      score_mode: Some(FunctionScoreMode::Sum),
      boost_mode: Some(FunctionBoostMode::Replace),
      max_boost: None,
      min_score: Some(0.008),
      boost: None,
    },
    score_mode: RescoreMode::Multiply,
  });
  let resp = reader.search(&req).unwrap();
  // doc-3 was dropped by rescore; doc-1 stays first (rescored), doc-2 stays
  // second (non-rescored raw score preserved at its original rank).
  assert_eq!(ids(&resp), vec!["doc-1", "doc-2"]);
  let doc1 = resp.hits.iter().find(|h| h.doc_id == "doc-1").unwrap();
  let doc2 = resp.hits.iter().find(|h| h.doc_id == "doc-2").unwrap();
  assert!(
    (doc1.score - 0.1).abs() < 1e-6,
    "doc-1 combined score should be 10 * 0.01 = 0.1, got {}",
    doc1.score
  );
  assert!(
    (doc2.score - 1.0).abs() < 1e-6,
    "doc-2 raw score should be preserved at 1.0, got {}",
    doc2.score
  );
}

// BUG-352: `read_number_literal` called `num.parse::<f64>()` without
// validating that the parsed value was finite. Rust's `str::parse::<f64>`
// returns `Ok(f64::INFINITY)` for decimal strings whose magnitude exceeds
// `f64::MAX` (~1.8e308) instead of surfacing an error, so a 309+ digit
// literal embedded in a script compiled to `Instruction::PushConst(
// f64::INFINITY)`. The eval-time guards in `CompiledScript::evaluate` then
// rejected the value and returned `None`, silently dropping every matching
// document with no error surfaced to the caller. The fix rejects the
// literal at compile time, matching the policy already used for
// `script_score` `params` validation and the BUG-334/BUG-338/BUG-344
// sibling `str::parse::<f64>` fixes.

fn overflow_literal(extra_zeros: usize) -> String {
  // Construct `1` followed by `extra_zeros` zeros (i.e. `10^extra_zeros`).
  // For this specific pattern overflow to f64::INFINITY starts at
  // `extra_zeros >= 309` (310 digits total, since `10^309 > f64::MAX`);
  // pad further so we are well above the boundary and still well under
  // `MAX_SCRIPT_LENGTH = 512`.
  let mut s = String::with_capacity(1 + extra_zeros);
  s.push('1');
  s.extend(std::iter::repeat_n('0', extra_zeros));
  s
}

#[test]
fn script_score_overflow_number_literal_is_rejected_at_compile_time() {
  let reader = setup_reader();
  let req = base_request(QueryNode::ScriptScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    script: overflow_literal(310),
    params: None,
    boost: Some(1.0),
  });
  let err = reader.search(&req).expect_err(
    "script_score with an f64-overflow number literal must surface a clear error, not silently drop hits",
  );
  let msg = format!("{err:#}");
  assert!(
    msg.contains("overflows to infinity"),
    "error should mention overflow to infinity, got: {msg}"
  );
}

#[test]
fn script_score_overflow_number_literal_in_expression_is_rejected_at_compile_time() {
  // The overflow surfaces regardless of where the literal appears in the
  // script: an operator-embedded literal would previously have compiled
  // cleanly and been caught only by an eval-time op guard, silently
  // dropping the hit.
  let reader = setup_reader();
  let script = format!("{} - popularity", overflow_literal(310));
  let req = base_request(QueryNode::ScriptScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    script,
    params: None,
    boost: Some(1.0),
  });
  let err = reader
    .search(&req)
    .expect_err("operator-embedded f64-overflow literal must also surface a compile-time error");
  let msg = format!("{err:#}");
  assert!(
    msg.contains("overflows to infinity"),
    "error should mention overflow to infinity, got: {msg}"
  );
}

#[test]
fn script_score_negative_overflow_number_literal_is_rejected_at_compile_time() {
  // Unary `-` consumes the digit string via `read_number_literal` before
  // negating, so the finitude check in `read_number_literal` catches the
  // overflow before the negation site ever sees the infinity.
  let reader = setup_reader();
  let script = format!("-{}", overflow_literal(310));
  let req = base_request(QueryNode::ScriptScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    script,
    params: None,
    boost: Some(1.0),
  });
  let err = reader
    .search(&req)
    .expect_err("negated f64-overflow literal must surface a compile-time error");
  let msg = format!("{err:#}");
  assert!(
    msg.contains("overflows to infinity"),
    "error should mention overflow to infinity, got: {msg}"
  );
}

// Regression: the `Sum` and `DisMax` arms of `evaluate_compiled_score`
// previously accumulated child scores without a finitude guard. Every other
// scoring path (FunctionScore, RankFeature, ScriptScore, rescore, hybrid)
// rejects non-finite accumulated scores, but the top-level aggregation
// nodes did not. When individually-finite children summed past f32::MAX,
// the accumulator overflowed to ±INFINITY (or NaN, in the DisMax path
// where `sum - max` is `∞ - ∞`) and leaked into the sort key heap,
// silently corrupting ordering. See BUG-364.
#[test]
fn sum_node_rejects_document_when_children_overflow_to_infinity() {
  let reader = setup_reader();
  // Two FunctionScore children in a Bool `must`. Each child returns a
  // finite `f32::MAX` via Replace of a weight function, but their Sum
  // overflows to +INFINITY. The outer Sum must now return None and the
  // document must be excluded.
  let req = base_request(QueryNode::Bool {
    must: vec![
      QueryNode::FunctionScore {
        query: Box::new(QueryNode::MatchAll { boost: None }),
        functions: vec![FunctionSpec::Weight {
          weight: f32::MAX,
          filter: None,
        }],
        score_mode: Some(FunctionScoreMode::Sum),
        boost_mode: Some(FunctionBoostMode::Replace),
        max_boost: None,
        min_score: None,
        boost: None,
      },
      QueryNode::FunctionScore {
        query: Box::new(QueryNode::MatchAll { boost: None }),
        functions: vec![FunctionSpec::Weight {
          weight: f32::MAX,
          filter: None,
        }],
        score_mode: Some(FunctionScoreMode::Sum),
        boost_mode: Some(FunctionBoostMode::Replace),
        max_boost: None,
        min_score: None,
        boost: None,
      },
    ],
    should: Vec::new(),
    must_not: Vec::new(),
    filter: Vec::new(),
    minimum_should_match: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert!(
    resp.hits.is_empty(),
    "expected no hits when Sum of children overflows to infinity, got {} hits with scores {:?}",
    resp.hits.len(),
    resp.hits.iter().map(|h| h.score).collect::<Vec<_>>()
  );
}

#[test]
fn dismax_node_rejects_document_when_accumulated_sum_overflows() {
  let reader = setup_reader();
  // Two FunctionScore children under a DisMax with tie_breaker > 0. Each
  // child returns a finite `f32::MAX`. `max` is finite but `sum`
  // overflows to +INFINITY, so `max + tie_breaker * (sum - max)` is
  // +INFINITY. The outer DisMax must now reject the document.
  let req = base_request(QueryNode::DisMax {
    queries: vec![
      QueryNode::FunctionScore {
        query: Box::new(QueryNode::MatchAll { boost: None }),
        functions: vec![FunctionSpec::Weight {
          weight: f32::MAX,
          filter: None,
        }],
        score_mode: Some(FunctionScoreMode::Sum),
        boost_mode: Some(FunctionBoostMode::Replace),
        max_boost: None,
        min_score: None,
        boost: None,
      },
      QueryNode::FunctionScore {
        query: Box::new(QueryNode::MatchAll { boost: None }),
        functions: vec![FunctionSpec::Weight {
          weight: f32::MAX,
          filter: None,
        }],
        score_mode: Some(FunctionScoreMode::Sum),
        boost_mode: Some(FunctionBoostMode::Replace),
        max_boost: None,
        min_score: None,
        boost: None,
      },
    ],
    tie_breaker: Some(0.5),
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert!(
    resp.hits.is_empty(),
    "expected no hits when DisMax accumulated sum overflows to infinity, got {} hits with scores {:?}",
    resp.hits.len(),
    resp.hits.iter().map(|h| h.score).collect::<Vec<_>>()
  );
}

#[test]
fn dismax_node_preserves_max_when_tie_breaker_is_zero_and_sum_overflows() {
  let reader = setup_reader();
  // With `tie_breaker == 0` the DisMax formula reduces to `max` by
  // definition. Naïve evaluation of `max + 0 * (sum - max)` becomes
  // `max + 0 * ∞ = max + NaN = NaN` when `sum` overflows — even though
  // semantically the hit should be preserved with score `max`. The
  // short-circuit must return `max` directly so the hit survives.
  let req = base_request(QueryNode::DisMax {
    queries: vec![
      QueryNode::FunctionScore {
        query: Box::new(QueryNode::MatchAll { boost: None }),
        functions: vec![FunctionSpec::Weight {
          weight: f32::MAX,
          filter: None,
        }],
        score_mode: Some(FunctionScoreMode::Sum),
        boost_mode: Some(FunctionBoostMode::Replace),
        max_boost: None,
        min_score: None,
        boost: None,
      },
      QueryNode::FunctionScore {
        query: Box::new(QueryNode::MatchAll { boost: None }),
        functions: vec![FunctionSpec::Weight {
          weight: f32::MAX,
          filter: None,
        }],
        score_mode: Some(FunctionScoreMode::Sum),
        boost_mode: Some(FunctionBoostMode::Replace),
        max_boost: None,
        min_score: None,
        boost: None,
      },
    ],
    tie_breaker: Some(0.0),
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  for hit in &resp.hits {
    assert!(
      hit.score.is_finite(),
      "{} score must be finite, got {}",
      hit.doc_id,
      hit.score
    );
    assert!(
      (hit.score - f32::MAX).abs() < f32::EPSILON * f32::MAX,
      "{} score must equal max=f32::MAX, got {}",
      hit.doc_id,
      hit.score
    );
  }
}

#[test]
fn sum_node_keeps_document_when_children_sum_stays_finite() {
  let reader = setup_reader();
  // Boundary check: children whose Sum fits within f32 must still produce
  // hits. Guards against an over-eager finitude check that would reject
  // legitimate scores. Two children at `f32::MAX / 4.0` sum to
  // ~`f32::MAX / 2.0`, comfortably finite.
  let req = base_request(QueryNode::Bool {
    must: vec![
      QueryNode::FunctionScore {
        query: Box::new(QueryNode::MatchAll { boost: None }),
        functions: vec![FunctionSpec::Weight {
          weight: f32::MAX / 4.0,
          filter: None,
        }],
        score_mode: Some(FunctionScoreMode::Sum),
        boost_mode: Some(FunctionBoostMode::Replace),
        max_boost: None,
        min_score: None,
        boost: None,
      },
      QueryNode::FunctionScore {
        query: Box::new(QueryNode::MatchAll { boost: None }),
        functions: vec![FunctionSpec::Weight {
          weight: f32::MAX / 4.0,
          filter: None,
        }],
        score_mode: Some(FunctionScoreMode::Sum),
        boost_mode: Some(FunctionBoostMode::Replace),
        max_boost: None,
        min_score: None,
        boost: None,
      },
    ],
    should: Vec::new(),
    must_not: Vec::new(),
    filter: Vec::new(),
    minimum_should_match: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  for hit in &resp.hits {
    assert!(
      hit.score.is_finite(),
      "{} score must be finite, got {}",
      hit.doc_id,
      hit.score
    );
  }
}

#[test]
fn dismax_node_keeps_document_when_accumulated_sum_stays_finite() {
  let reader = setup_reader();
  // Boundary check for DisMax: two finite children whose `sum` stays
  // below f32::MAX must still produce hits and finite scores.
  let req = base_request(QueryNode::DisMax {
    queries: vec![
      QueryNode::FunctionScore {
        query: Box::new(QueryNode::MatchAll { boost: None }),
        functions: vec![FunctionSpec::Weight {
          weight: f32::MAX / 4.0,
          filter: None,
        }],
        score_mode: Some(FunctionScoreMode::Sum),
        boost_mode: Some(FunctionBoostMode::Replace),
        max_boost: None,
        min_score: None,
        boost: None,
      },
      QueryNode::FunctionScore {
        query: Box::new(QueryNode::MatchAll { boost: None }),
        functions: vec![FunctionSpec::Weight {
          weight: f32::MAX / 4.0,
          filter: None,
        }],
        score_mode: Some(FunctionScoreMode::Sum),
        boost_mode: Some(FunctionBoostMode::Replace),
        max_boost: None,
        min_score: None,
        boost: None,
      },
    ],
    tie_breaker: Some(0.5),
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  for hit in &resp.hits {
    assert!(
      hit.score.is_finite(),
      "{} score must be finite, got {}",
      hit.doc_id,
      hit.score
    );
  }
}

#[test]
fn script_score_large_but_finite_literal_is_accepted() {
  // Boundary check: a large-but-finite literal within f64 range must still
  // compile and execute. Guards against an over-eager finitude check that
  // would reject legitimate scripts. The literal used here is ~1e50 which
  // parses to a finite f64; multiplying by 0 collapses it before the f32
  // narrowing cast, so the final score is `popularity` which fits in f32.
  let reader = setup_reader();
  let script = format!("{} * 0 + popularity", "1".to_string() + &"0".repeat(50));
  let req = base_request(QueryNode::ScriptScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    script,
    params: None,
    boost: Some(1.0),
  });
  let resp = reader
    .search(&req)
    .expect("finite large literal must compile and execute cleanly");
  assert_eq!(resp.hits.len(), 3);
  for hit in &resp.hits {
    assert!(
      hit.score.is_finite(),
      "{} score must be finite, got {}",
      hit.doc_id,
      hit.score
    );
  }
}

// BUG-362: `evaluate_compiled_score` unconditionally rewrote a near-zero
// base query score to `1.0` whenever function values were present, but the
// rewrite is only correct for `boost_mode: multiply` (where `0 * func = 0`
// would erase the function contribution). For Sum, Max, and Min boost
// modes, rewriting the base from `0.0` to `1.0` adds an artificial bias:
// Sum becomes `1 + func` instead of `0 + func`, Max clamps at `1.0` when
// `func < 1.0`, and Min produces a `1.0` floor when `func >= 1.0`. The fix
// gates the rewrite on `FunctionBoostMode::Multiply`, leaving other modes
// to preserve the base naturally.

fn zero_base_function_score(func_weight: f32, boost_mode: FunctionBoostMode) -> QueryNode {
  QueryNode::FunctionScore {
    query: Box::new(QueryNode::ConstantScore {
      filter: Filter::KeywordEq {
        field: "lang".into(),
        value: "en".into(),
      },
      boost: Some(0.0),
    }),
    functions: vec![FunctionSpec::Weight {
      weight: func_weight,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(boost_mode),
    max_boost: None,
    min_score: None,
    boost: None,
  }
}

#[test]
fn function_score_sum_boost_mode_preserves_zero_base() {
  // Sum: `0 + 5 = 5`. The pre-fix implementation rewrote the base to 1.0,
  // producing `1 + 5 = 6` — a +1.0 bias on every doc whose base query
  // legitimately scored zero.
  let reader = setup_reader();
  let req = base_request(zero_base_function_score(5.0, FunctionBoostMode::Sum));
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 2);
  for hit in &resp.hits {
    assert!(
      (hit.score - 5.0).abs() < 1e-6,
      "expected 5.0 (0.0 + 5.0), got {}",
      hit.score
    );
  }
}

#[test]
fn function_score_max_boost_mode_preserves_zero_base() {
  // Max: `max(0, 0.5) = 0.5`. The pre-fix implementation rewrote the base
  // to 1.0, producing `max(1, 0.5) = 1.0` — clamping the function value
  // whenever it was below 1.0.
  let reader = setup_reader();
  let req = base_request(zero_base_function_score(0.5, FunctionBoostMode::Max));
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 2);
  for hit in &resp.hits {
    assert!(
      (hit.score - 0.5).abs() < 1e-6,
      "expected 0.5 (max(0.0, 0.5)), got {}",
      hit.score
    );
  }
}

#[test]
fn function_score_min_boost_mode_preserves_zero_base() {
  // Min: `min(0, 5) = 0`. The pre-fix implementation rewrote the base to
  // 1.0, producing `min(1, 5) = 1.0` — a 1.0 floor whenever the function
  // value was ≥ 1.0.
  let reader = setup_reader();
  let req = base_request(zero_base_function_score(5.0, FunctionBoostMode::Min));
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 2);
  for hit in &resp.hits {
    assert!(
      hit.score.abs() < 1e-6,
      "expected 0.0 (min(0.0, 5.0)), got {}",
      hit.score
    );
  }
}

#[test]
fn function_score_multiply_boost_mode_still_rewrites_zero_base() {
  // Regression lock: Multiply still needs the `0 -> 1` rewrite because
  // `0 * func = 0` would erase the function contribution entirely. After
  // the fix, Multiply must continue to produce `1 * 5 = 5`.
  let reader = setup_reader();
  let req = base_request(zero_base_function_score(5.0, FunctionBoostMode::Multiply));
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 2);
  for hit in &resp.hits {
    assert!(
      (hit.score - 5.0).abs() < 1e-6,
      "expected 5.0 (rewritten 1.0 * 5.0), got {}",
      hit.score
    );
  }
}

#[test]
fn function_score_replace_boost_mode_preserves_function_value_over_zero_base() {
  // Replace: ignores the base entirely. Behaviour is identical before and
  // after the fix; included as a regression lock so the base rewrite gate
  // cannot accidentally alter Replace semantics.
  let reader = setup_reader();
  let req = base_request(zero_base_function_score(5.0, FunctionBoostMode::Replace));
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 2);
  for hit in &resp.hits {
    assert!(
      (hit.score - 5.0).abs() < 1e-6,
      "expected 5.0 (Replace drops base), got {}",
      hit.score
    );
  }
}

#[test]
fn function_score_sum_boost_mode_with_nonzero_base_unchanged() {
  // Regression lock: when the base is not near zero, the rewrite gate does
  // not fire regardless of boost_mode, and Sum still combines the real
  // base score with the function value.
  let reader = setup_reader();
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::ConstantScore {
      filter: Filter::KeywordEq {
        field: "lang".into(),
        value: "en".into(),
      },
      boost: Some(2.0),
    }),
    functions: vec![FunctionSpec::Weight {
      weight: 5.0,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Sum),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 2);
  for hit in &resp.hits {
    assert!(
      (hit.score - 7.0).abs() < 1e-6,
      "expected 7.0 (2.0 + 5.0), got {}",
      hit.score
    );
  }
}

// BUG-336: the f64 -> f32 narrowing cast in `FieldValueFactor::evaluate`,
// `RankFeature` evaluation, and `CompiledScript::evaluate` saturates any
// finite f64 whose magnitude exceeds `f32::MAX` (~3.4e38) to
// `±f32::INFINITY`. Downstream non-finite guards in
// `evaluate_compiled_score` then reject the hit, silently dropping it from
// the result set. The fix clamps the f64 to the f32 representable range
// before the cast so the document survives with the closest representable
// score (`f32::MAX` / `f32::MIN`), matching the `finite_or_zero` policy used
// in aggregations.

fn weight_doc(id: &str, body: &str, weight: f64) -> Document {
  Document {
    fields: [
      ("_id".to_string(), serde_json::json!(id)),
      ("body".to_string(), serde_json::json!(body)),
      ("weight".to_string(), serde_json::json!(weight)),
    ]
    .into_iter()
    .collect(),
  }
}

fn setup_reader_with_weight_field(
  weights: &[(&str, f64)],
) -> (tempfile::TempDir, searchlite_core::api::IndexReader) {
  // Keep the `TempDir` alive for the reader's lifetime so the index
  // directory survives for the duration of the test and is cleaned up
  // when the caller drops the returned guard.
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().join("idx");
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "weight".into(),
    i64: false,
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
  for (id, weight) in weights {
    writer
      .add_document(&weight_doc(id, "rust", *weight))
      .unwrap();
  }
  writer.commit().unwrap();
  (tmp, idx.reader().unwrap())
}

#[test]
fn field_value_factor_reciprocal_overflow_preserves_document_with_f32_max() {
  // weight = 1e-40 is a finite f64 well within the f64 range. Taking its
  // reciprocal yields 1e40, which is finite as f64 but exceeds f32::MAX
  // (~3.4e38). Before the fix, `modified as f32` saturated to
  // f32::INFINITY, which the downstream non-finite guards rejected,
  // dropping the hit from the result set. After the fix, the value is
  // clamped to f32::MAX and the hit survives.
  let (_tmp, reader) =
    setup_reader_with_weight_field(&[("doc-small", 1.0e-40), ("doc-normal", 1.0)]);
  let req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    functions: vec![FunctionSpec::FieldValueFactor {
      field: "weight".into(),
      factor: 1.0,
      modifier: Some(FieldValueModifier::Reciprocal),
      missing: None,
      filter: None,
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Replace),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  let resp = reader.search(&req).unwrap();
  let hit_ids = ids(&resp);
  assert!(
    hit_ids.contains(&"doc-small".to_string()),
    "doc-small must survive the f64->f32 narrowing cast, got hits {hit_ids:?}"
  );
  let doc_small = resp
    .hits
    .iter()
    .find(|h| h.doc_id == "doc-small")
    .expect("doc-small present");
  assert!(
    doc_small.score.is_finite(),
    "doc-small score must be finite after clamp, got {}",
    doc_small.score
  );
  assert_eq!(
    doc_small.score,
    f32::MAX,
    "doc-small score must be clamped to f32::MAX, got {}",
    doc_small.score
  );
}

#[test]
fn rank_feature_reciprocal_overflow_preserves_document_with_f32_max() {
  // Same overflow trigger as above but routed through the RankFeature
  // score node, which performs the `modified as f32` cast independently of
  // the FunctionScore path.
  let (_tmp, reader) =
    setup_reader_with_weight_field(&[("doc-small", 1.0e-40), ("doc-normal", 1.0)]);
  let req = base_request(QueryNode::RankFeature {
    field: "weight".into(),
    boost: Some(1.0),
    modifier: Some(RankFeatureModifier::Reciprocal),
    missing: Some(1.0),
  });
  let resp = reader.search(&req).unwrap();
  let hit_ids = ids(&resp);
  assert!(
    hit_ids.contains(&"doc-small".to_string()),
    "doc-small must survive the f64->f32 narrowing cast, got hits {hit_ids:?}"
  );
  let doc_small = resp
    .hits
    .iter()
    .find(|h| h.doc_id == "doc-small")
    .expect("doc-small present");
  assert!(
    doc_small.score.is_finite(),
    "doc-small score must be finite after clamp, got {}",
    doc_small.score
  );
  assert_eq!(
    doc_small.score,
    f32::MAX,
    "doc-small score must be clamped to f32::MAX, got {}",
    doc_small.score
  );
}

#[test]
fn script_score_large_literal_overflow_preserves_document_with_f32_max() {
  // The script tokenizer accepts params (validated finite at compile time)
  // but a finite `f64` param larger than `f32::MAX` saturates to
  // `f32::INFINITY` on the final `value as f32` cast at the end of
  // `CompiledScript::evaluate`. Before the fix the hit was dropped; after
  // the fix the score is clamped to `f32::MAX`.
  let reader = setup_reader();
  let mut params = BTreeMap::new();
  params.insert("big".to_string(), 1.0e40_f64);
  let req = base_request(QueryNode::ScriptScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    script: "big".into(),
    params: Some(params),
    boost: Some(1.0),
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  for hit in &resp.hits {
    assert!(
      hit.score.is_finite(),
      "{} score must be finite after clamp, got {}",
      hit.doc_id,
      hit.score
    );
    assert_eq!(
      hit.score,
      f32::MAX,
      "{} score must be clamped to f32::MAX, got {}",
      hit.doc_id,
      hit.score
    );
  }
}

#[test]
fn script_score_large_negative_literal_overflow_clamps_to_f32_min() {
  // Symmetric negative overflow: a finite f64 below `f32::MIN` saturates
  // to `f32::NEG_INFINITY` on the narrowing cast. The clamp should floor
  // the result at `f32::MIN` so the hit survives with the closest
  // representable negative score.
  let reader = setup_reader();
  let mut params = BTreeMap::new();
  params.insert("big".to_string(), -1.0e40_f64);
  let req = base_request(QueryNode::ScriptScore {
    query: Box::new(QueryNode::MatchAll { boost: None }),
    script: "big".into(),
    params: Some(params),
    boost: Some(1.0),
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 3);
  for hit in &resp.hits {
    assert!(
      hit.score.is_finite(),
      "{} score must be finite after clamp, got {}",
      hit.doc_id,
      hit.score
    );
    assert_eq!(
      hit.score,
      f32::MIN,
      "{} score must be clamped to f32::MIN, got {}",
      hit.doc_id,
      hit.score
    );
  }
}

// Regression: the `Constant` arm of `evaluate_compiled_score` previously
// returned `Some(*score)` unconditionally. Every other variant (Sum, DisMax,
// FunctionScore, RankFeature, ScriptScore) rejects non-finite results, but
// Constant did not. When `boost * node_boost` accumulated across nested
// scopes overflowed `f32::MAX` to `+INFINITY`, the non-finite score leaked
// into the WAND heap (corrupting ordering) and into `Hit.score`, where
// `serde_json` rejects it as invalid JSON and the HTTP endpoint returns 500.
// See BUG-370.
//
// The planner now catches this class of overflow at build time (the
// `combine_boost` helper bails when `boost * node_boost` is non-finite),
// so the request is rejected with an actionable error before it reaches
// the WAND loop. The evaluator guard remains as defense-in-depth for any
// non-finite score that might be produced by later arithmetic.
#[test]
fn constant_score_rejects_request_when_boost_product_overflows_to_infinity() {
  let reader = setup_reader();
  // `Bool` with `boost = 1e38` containing a single `ConstantScore` with
  // `boost = 1e38`. Both factors are individually finite and pass
  // `validate_boost`, but their product `1e38 * 1e38 = 1e76` overflows
  // `f32::MAX ≈ 3.4e38` and saturates to `+INFINITY`. With the planner
  // guard in place, `search` surfaces the overflow as a validation error
  // rather than silently dropping the document.
  let req = base_request(QueryNode::Bool {
    must: vec![QueryNode::ConstantScore {
      filter: Filter::KeywordEq {
        field: "lang".into(),
        value: "en".into(),
      },
      boost: Some(1e38),
    }],
    should: Vec::new(),
    must_not: Vec::new(),
    filter: Vec::new(),
    minimum_should_match: None,
    boost: Some(1e38),
  });
  let err = reader
    .search(&req)
    .expect_err("overflowing boost product must be rejected at plan time");
  assert!(
    err.to_string().contains("overflows"),
    "expected overflow validation error, got: {err}",
  );
}

// Regression test for BUG-376: when `score_fast_path` is active (sort by
// `_score` desc, no `track_total_hits`, no aggs) and a score hook
// (function_score/script_score/rank_feature) amplifies scores above BM25
// upper bounds, WAND's dynamic threshold pruning terminates early because
// the heap threshold — now an adjusted-score minimum — exceeds the sum of
// remaining BM25 term upper bounds. Documents that would have ranked in the
// top-k after adjustment are silently dropped.
#[test]
fn wand_does_not_prune_against_amplified_scores_on_fast_path() {
  // Bind `TempDir` to a local so the directory survives the whole test and
  // is cleaned up on drop; calling `.path()` on a temporary would delete the
  // directory before the index could be created.
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().join("idx");
  let mut schema = Schema::default_text_body();
  schema.keyword_fields.push(KeywordField {
    name: "boosted".into(),
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
  // All documents match the boost filter (boosted = "yes") and contain
  // "common". Term frequency increases with doc_id so doc-2 has the highest
  // BM25, indexed last. With weight=100 multiplicative boost, the early
  // docs' adjusted scores fill the heap with a threshold far above any
  // term's BM25 upper bound. The buggy WAND terminates before reaching the
  // latest doc (doc-2), returning doc-1 instead of doc-2 as the top hit.
  for d in [
    Document {
      fields: [
        ("_id".to_string(), serde_json::json!("doc-0")),
        ("body".to_string(), serde_json::json!("common")),
        ("boosted".to_string(), serde_json::json!("yes")),
      ]
      .into_iter()
      .collect(),
    },
    Document {
      fields: [
        ("_id".to_string(), serde_json::json!("doc-1")),
        ("body".to_string(), serde_json::json!("common common")),
        ("boosted".to_string(), serde_json::json!("yes")),
      ]
      .into_iter()
      .collect(),
    },
    Document {
      fields: [
        ("_id".to_string(), serde_json::json!("doc-2")),
        (
          "body".to_string(),
          serde_json::json!("common common common common"),
        ),
        ("boosted".to_string(), serde_json::json!("yes")),
      ]
      .into_iter()
      .collect(),
    },
  ] {
    writer.add_document(&d).unwrap();
  }
  writer.commit().unwrap();
  let reader = idx.reader().unwrap();

  let mut req = base_request(QueryNode::FunctionScore {
    query: Box::new(QueryNode::Term {
      field: "body".into(),
      value: "common".into(),
      boost: None,
    }),
    functions: vec![FunctionSpec::Weight {
      weight: 100.0,
      filter: Some(Filter::KeywordEq {
        field: "boosted".into(),
        value: "yes".into(),
      }),
    }],
    score_mode: Some(FunctionScoreMode::Sum),
    boost_mode: Some(FunctionBoostMode::Multiply),
    max_boost: None,
    min_score: None,
    boost: None,
  });
  // size=1 (top_k=2 internally) is the minimal setting that forces the
  // heap to fill before the last document is examined. Once the heap
  // holds two adjusted-score hits, the heap threshold climbs well above
  // any BM25 upper bound. `track_total_hits = false` (the default) is
  // what activates `score_fast_path` — the code path with the bug.
  req.limit = 1;

  let resp = reader.search(&req).unwrap();
  let ids: Vec<String> = resp.hits.iter().map(|h| h.doc_id.clone()).collect();

  assert_eq!(
    ids,
    vec!["doc-2".to_string()],
    "top-1 must be doc-2 (highest BM25 * 100); got {:?}",
    ids
  );
  let top = &resp.hits[0];
  assert!(
    top.score.is_finite(),
    "score must be finite, got {}",
    top.score
  );

  // Compare against the `track_total_hits = true` path, which disables
  // `score_fast_path` entirely and therefore has never been affected by
  // the WAND pruning bug. The two paths must agree on the top hit.
  let mut req_total = req.clone();
  req_total.track_total_hits = Some(true);
  let resp_total = reader.search(&req_total).unwrap();
  let ids_total: Vec<String> = resp_total.hits.iter().map(|h| h.doc_id.clone()).collect();
  assert_eq!(
    ids, ids_total,
    "fast-path top-k must match the track_total_hits=true path (which doesn't trigger WAND pruning); fast={:?} full={:?}",
    ids, ids_total
  );
}

#[test]
fn constant_score_keeps_document_when_boost_product_stays_finite() {
  let reader = setup_reader();
  // Boundary check: the new finitude guard must not over-reject legitimate
  // large-but-finite scores. `1e10 * 1e10 = 1e20` is comfortably within
  // `f32::MAX`, so the Constant evaluator must return the product as the
  // document's score.
  let req = base_request(QueryNode::Bool {
    must: vec![QueryNode::ConstantScore {
      filter: Filter::KeywordEq {
        field: "lang".into(),
        value: "en".into(),
      },
      boost: Some(1e10),
    }],
    should: Vec::new(),
    must_not: Vec::new(),
    filter: Vec::new(),
    minimum_should_match: None,
    boost: Some(1e10),
  });
  let resp = reader.search(&req).unwrap();
  assert_eq!(resp.hits.len(), 2);
  for hit in &resp.hits {
    assert!(
      hit.score.is_finite(),
      "{} score must be finite, got {}",
      hit.doc_id,
      hit.score
    );
    assert!(
      (hit.score - 1e20).abs() < 1e20 * 1e-5,
      "{} score must equal 1e20, got {}",
      hit.doc_id,
      hit.score
    );
  }
}
