use std::collections::{BTreeMap, HashSet};

use searchlite_core::api::types::{
  Document, ExecutionStrategy, IndexOptions, KeywordField, MultiMatchFuzziness, NumericField,
  Query, QueryNode, Schema, SearchRequest, StorageType,
};
use searchlite_core::api::{Filter, Index, SearchResult};
use serde_json::json;
use tempfile::TempDir;

fn doc(id: &str, body: &str, lang: &str, year: i64) -> Document {
  Document {
    fields: [
      ("_id".to_string(), serde_json::json!(id)),
      ("body".to_string(), serde_json::json!(body)),
      ("lang".to_string(), serde_json::json!(lang)),
      ("year".to_string(), serde_json::json!(year)),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>(),
  }
}

fn setup_reader() -> (TempDir, searchlite_core::api::IndexReader) {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx");
  let mut schema = Schema::default_text_body();
  schema.keyword_fields.push(KeywordField {
    name: "lang".into(),
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
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let idx = Index::create(&path, schema, opts).unwrap();
  let mut writer = idx.writer().unwrap();
  let docs = vec![
    doc("doc-1", "rust engine fast", "en", 2024),
    doc("doc-2", "rust database tiny", "en", 2022),
    doc("doc-3", "rust search", "fr", 2021),
    doc("doc-4", "rust boring engine", "en", 2020),
    doc("doc-5", "fast tiny search", "fr", 2023),
  ];
  for doc in docs {
    writer.add_document(&doc).unwrap();
  }
  writer.commit().unwrap();
  let reader = idx.reader().unwrap();
  (dir, reader)
}

fn ids(result: &SearchResult) -> HashSet<String> {
  result.hits.iter().map(|hit| hit.doc_id.clone()).collect()
}

fn term(field: &str, value: &str) -> QueryNode {
  QueryNode::Term {
    field: field.into(),
    value: value.into(),
    boost: None,
  }
}

fn bool_query(
  must: Vec<QueryNode>,
  should: Vec<QueryNode>,
  must_not: Vec<QueryNode>,
  filter: Vec<Filter>,
  minimum_should_match: Option<usize>,
) -> QueryNode {
  QueryNode::Bool {
    must,
    should,
    must_not,
    filter,
    minimum_should_match,
    boost: None,
  }
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

#[test]
fn bool_grouped_or_matches() {
  let (_tmp, reader) = setup_reader();
  let grouped_or = bool_query(
    vec![term("body", "rust")],
    vec![term("body", "engine"), term("body", "database")],
    vec![],
    vec![],
    Some(1),
  );
  let query = bool_query(vec![grouped_or], vec![], vec![], vec![], None);
  let resp = reader.search(&request(query)).unwrap();
  let got = ids(&resp);
  let expected: HashSet<String> = ["doc-1", "doc-2", "doc-4"]
    .into_iter()
    .map(String::from)
    .collect();
  assert_eq!(got, expected);
}

#[test]
fn bool_must_not_excludes() {
  let (_tmp, reader) = setup_reader();
  let query = bool_query(
    vec![term("body", "rust")],
    vec![],
    vec![term("body", "boring")],
    vec![],
    None,
  );
  let resp = reader.search(&request(query)).unwrap();
  let got = ids(&resp);
  let expected: HashSet<String> = ["doc-1", "doc-2", "doc-3"]
    .into_iter()
    .map(String::from)
    .collect();
  assert_eq!(got, expected);
}

#[test]
fn minimum_should_match_enforced() {
  let (_tmp, reader) = setup_reader();
  let query = bool_query(
    vec![],
    vec![
      term("body", "fast"),
      term("body", "tiny"),
      term("body", "engine"),
    ],
    vec![],
    vec![],
    Some(2),
  );
  let resp = reader.search(&request(query)).unwrap();
  let got = ids(&resp);
  let expected: HashSet<String> = ["doc-1", "doc-5"].into_iter().map(String::from).collect();
  assert_eq!(got, expected);
}

#[test]
fn filter_only_query_matches() {
  let (_tmp, reader) = setup_reader();
  let filter = Filter::Or(vec![
    Filter::KeywordEq {
      field: "lang".into(),
      value: "fr".into(),
    },
    Filter::I64Range {
      field: "year".into(),
      min: 2024,
      max: 2024,
    },
  ]);
  let query = bool_query(vec![], vec![], vec![], vec![filter], None);
  let resp = reader.search(&request(query)).unwrap();
  let got = ids(&resp);
  let expected: HashSet<String> = ["doc-1", "doc-3", "doc-5"]
    .into_iter()
    .map(String::from)
    .collect();
  assert_eq!(got, expected);
}

#[test]
fn mixed_query_and_filter_matches() {
  let (_tmp, reader) = setup_reader();
  let mut req = request("rust");
  req.filter = Some(Filter::I64Range {
    field: "year".into(),
    min: 2023,
    max: 2024,
  });
  let resp = reader.search(&req).unwrap();
  let got = ids(&resp);
  let expected: HashSet<String> = ["doc-1"].into_iter().map(String::from).collect();
  assert_eq!(got, expected);
}

#[test]
fn query_string_phrase_only_matches() {
  let (_tmp, reader) = setup_reader();
  let query = QueryNode::QueryString {
    query: "\"rust engine\"".into(),
    fields: None,
    boost: None,
  };
  let resp = reader.search(&request(query)).unwrap();
  let got = ids(&resp);
  let expected: HashSet<String> = ["doc-1"].into_iter().map(String::from).collect();
  assert_eq!(got, expected);
}

#[test]
fn query_string_negated_only_matches() {
  let (_tmp, reader) = setup_reader();
  let query = QueryNode::QueryString {
    query: "-boring".into(),
    fields: None,
    boost: None,
  };
  let resp = reader.search(&request(query)).unwrap();
  let got = ids(&resp);
  let expected: HashSet<String> = ["doc-1", "doc-2", "doc-3", "doc-5"]
    .into_iter()
    .map(String::from)
    .collect();
  assert_eq!(got, expected);
}

#[test]
fn multi_match_fuzziness_parses_auto_and_numeric() {
  let auto: QueryNode = serde_json::from_value(json!({
    "type": "multi_match",
    "query": "rust",
    "fields": ["body"],
    "fuzziness": "AUTO"
  }))
  .unwrap();
  match auto {
    QueryNode::MultiMatch { fuzziness, .. } => {
      assert!(matches!(fuzziness, Some(MultiMatchFuzziness::Auto)))
    }
    _ => panic!("expected multi_match"),
  }

  let numeric: QueryNode = serde_json::from_value(json!({
    "type": "multi_match",
    "query": "rust",
    "fields": ["body"],
    "fuzziness": 1
  }))
  .unwrap();
  match numeric {
    QueryNode::MultiMatch { fuzziness, .. } => {
      assert!(matches!(fuzziness, Some(MultiMatchFuzziness::Edits(1))))
    }
    _ => panic!("expected multi_match"),
  }
}

#[test]
fn multi_match_fuzziness_rejects_out_of_range_value() {
  let err = serde_json::from_value::<QueryNode>(json!({
    "type": "multi_match",
    "query": "rust",
    "fields": ["body"],
    "fuzziness": 3
  }))
  .unwrap_err();
  let msg = err.to_string();
  assert!(msg.contains("between 0 and 2"), "error: {msg}");
}

#[test]
fn fuzziness_on_non_multi_match_query_is_ignored() {
  let term: QueryNode = serde_json::from_value(json!({
    "type": "term",
    "field": "body",
    "value": "rust",
    "fuzziness": "AUTO"
  }))
  .unwrap();
  match term {
    QueryNode::Term { field, value, .. } => {
      assert_eq!(field, "body");
      assert_eq!(value, "rust");
    }
    _ => panic!("expected term query"),
  }
}
