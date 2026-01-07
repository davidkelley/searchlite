use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use searchlite_core::api::types::{
  Document, ExecutionStrategy, FieldSpec, FuzzyOptions, IndexOptions, Schema, SearchAsYouType,
  SearchRequest, SuggestRequest, TextField,
};
use searchlite_core::api::{Index, Query, QueryNode, StorageType};

fn opts(path: &Path) -> IndexOptions {
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

#[test]
fn prefix_query_matches_docs() {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx-prefix");
  let idx = Index::create(&path, Schema::default_text_body(), opts(&path)).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for (id, body) in [
      ("1", "rust search"),
      ("2", "ruby language"),
      ("3", "python"),
    ] {
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(id)),
            ("body".into(), serde_json::json!(body)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
    }
    writer.commit().unwrap();
  }
  let reader = idx.reader().unwrap();
  let res = reader
    .search(&SearchRequest {
      query: Query::Node(QueryNode::Prefix {
        field: "body".into(),
        value: "ru".into(),
        max_expansions: None,
        boost: None,
      }),
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
      highlight: None,
      collapse: None,
      aggs: BTreeMap::new(),
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  let ids: HashSet<_> = res.hits.iter().map(|h| h.doc_id.as_str()).collect();
  assert_eq!(ids.len(), 2);
  assert!(ids.contains("1"));
  assert!(ids.contains("2"));
}

#[test]
fn search_as_you_type_matches_partial_tokens() {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx-sayt");
  let schema = Schema {
    doc_id_field: "_id".into(),
    analyzers: Vec::new(),
    text_fields: vec![TextField {
      name: "title".into(),
      analyzer: "default".into(),
      search_analyzer: None,
      stored: true,
      indexed: true,
      nullable: false,
      search_as_you_type: Some(SearchAsYouType {
        min_gram: 1,
        max_gram: 5,
      }),
    }],
    keyword_fields: Vec::new(),
    numeric_fields: Vec::new(),
    nested_fields: Vec::new(),
    #[cfg(feature = "vectors")]
    vector_fields: Vec::new(),
  };
  let idx = Index::create(&path, schema, opts(&path)).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!("doc-1")),
          ("title".into(), serde_json::json!("rustacean handbook")),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    writer.commit().unwrap();
  }
  let reader = idx.reader().unwrap();
  let res = reader
    .search(&SearchRequest {
      query: Query::Node(QueryNode::QueryString {
        query: "ru".into(),
        fields: Some(vec![FieldSpec {
          field: "title".into(),
          boost: None,
        }]),
        boost: None,
      }),
      fields: None,
      filter: None,
      filters: vec![],
      limit: 5,
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
    })
    .unwrap();
  assert_eq!(res.hits.len(), 1);
  assert_eq!(res.hits[0].doc_id, "doc-1");
}

#[test]
fn fuzzy_completion_suggests_close_terms() {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx-suggest");
  let idx = Index::create(&path, Schema::default_text_body(), opts(&path)).unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for (id, body) in [("1", "rust"), ("2", "bust"), ("3", "trust")] {
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(id)),
            ("body".into(), serde_json::json!(body)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
    }
    writer.commit().unwrap();
  }
  let reader = idx.reader().unwrap();
  let mut suggest = BTreeMap::new();
  suggest.insert(
    "complete".into(),
    SuggestRequest::Completion {
      field: "body".into(),
      prefix: "rsut".into(),
      size: 3,
      fuzzy: Some(FuzzyOptions {
        max_edits: 2,
        prefix_length: 1,
        max_expansions: 10,
        min_length: 2,
      }),
    },
  );
  let res = reader
    .search(&SearchRequest {
      query: Query::Node(QueryNode::MatchAll { boost: None }),
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

      #[cfg(feature = "vectors")]
      vector_filter: None,
      return_stored: false,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs: BTreeMap::new(),
      suggest,
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  let options = res
    .suggest
    .get("complete")
    .expect("missing suggestion")
    .options
    .clone();
  assert!(!options.is_empty());
  assert_eq!(options[0].text, "rust");
}
