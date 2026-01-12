use std::collections::{BTreeMap, HashSet};

use searchlite_core::api::reader::SearchResult;
use searchlite_core::api::types::{
  Document, ExecutionStrategy, FieldSpec, IndexOptions, MatchOperator, MultiMatchType, Query,
  QueryNode, Schema, SearchRequest, StorageType, TextField,
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
    filters: vec![],
    limit: 10,
    return_hits: true,
    candidate_size: None,
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
    tie_breaker: None,
    operator: Some(MatchOperator::Or),
    minimum_should_match: None,
    boost: None,
  };
  let most = QueryNode::MultiMatch {
    query: "rust search".into(),
    fields,
    match_type: MultiMatchType::MostFields,
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
    tie_breaker: None,
    operator: Some(MatchOperator::Or),
    minimum_should_match: None,
    boost: None,
  };
  let body_ids = ids(&reader.search(&request(body_only)).unwrap());
  assert!(body_ids.contains("doc-3"), "{:?}", body_ids);
  let best_res = reader.search(&request(best)).unwrap();
  let most_res = reader.search(&request(most)).unwrap();
  let best_ids = ids(&best_res);
  let most_ids = ids(&most_res);
  assert!(
    best_ids.contains("doc-2"),
    "best_fields ids: {:?}",
    best_ids
  );
  assert!(
    most_ids.contains("doc-2"),
    "most_fields ids: {:?}",
    most_ids
  );
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
    tie_breaker: None,
    operator: Some(MatchOperator::And),
    minimum_should_match: None,
    boost: None,
  };
  let hits = ids(&reader.search(&request(query)).unwrap());
  assert!(hits.contains("doc-2"), "hits: {:?}", hits);
  assert!(!hits.contains("doc-4"), "hits: {:?}", hits);
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
