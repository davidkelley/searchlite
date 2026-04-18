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
fn log2p_modifier_uses_natural_log_of_value_plus_two() {
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
  // Log2p = ln(value + 2): ln(12) ≈ 2.485, ln(7) ≈ 1.946, ln(3) ≈ 1.099
  assert_eq!(ids(&resp), vec!["doc-1", "doc-3", "doc-2"]);
  let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();
  assert!(
    (scores[0] - 12_f32.ln()).abs() < 1e-6,
    "doc-1: {}",
    scores[0]
  );
  assert!(
    (scores[1] - 7_f32.ln()).abs() < 1e-6,
    "doc-3: {}",
    scores[1]
  );
  assert!(
    (scores[2] - 3_f32.ln()).abs() < 1e-6,
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
