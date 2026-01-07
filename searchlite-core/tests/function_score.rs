use std::collections::BTreeMap;

use searchlite_core::api::types::{
  DecayFunction, Document, ExecutionStrategy, FieldValueModifier, Filter, FunctionBoostMode,
  FunctionScoreMode, FunctionSpec, IndexOptions, KeywordField, NumericField, Query, QueryNode,
  RescoreMode, RescoreRequest, Schema, SearchRequest, StorageType,
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
    filters: vec![],
    limit: 10,
    sort: Vec::new(),
    cursor: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    fuzzy: None,
    #[cfg(feature = "vectors")]
    vector_query: None,

    #[cfg(feature = "vectors")]
    vector_filter: None,
    return_stored: false,
    highlight_field: None,
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
