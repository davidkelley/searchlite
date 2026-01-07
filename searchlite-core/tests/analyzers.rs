use std::collections::BTreeMap;
use std::path::Path;

use searchlite_core::analysis::analyzer::{
  AnalyzerDef, EdgeNgramConfig, StemmerConfig, StopwordsConfig, SynonymRule, TokenFilterDef,
};
use searchlite_core::api::types::{
  Document, ExecutionStrategy, IndexOptions, Query, QueryNode, Schema, SearchRequest, StorageType,
  TextField,
};
use searchlite_core::api::{Index, SearchResult};

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

fn request(query: impl Into<Query>) -> SearchRequest {
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

fn ids(resp: &SearchResult) -> Vec<String> {
  resp.hits.iter().map(|h| h.doc_id.clone()).collect()
}

#[test]
fn search_analyzer_expands_synonyms() {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx");
  let schema = Schema {
    doc_id_field: "_id".into(),
    analyzers: vec![AnalyzerDef {
      name: "synonyms".into(),
      tokenizer: "default".into(),
      filters: vec![TokenFilterDef::Synonyms(vec![SynonymRule {
        from: vec!["nyc".into()],
        to: vec!["new".into(), "york".into()],
      }])],
    }],
    text_fields: vec![TextField {
      name: "body".into(),
      analyzer: "default".into(),
      search_analyzer: Some("synonyms".into()),
      stored: true,
      indexed: true,
      nullable: false,
      search_as_you_type: None,
    }],
    keyword_fields: Vec::new(),
    numeric_fields: Vec::new(),
    nested_fields: Vec::new(),
    #[cfg(feature = "vectors")]
    vector_fields: Vec::new(),
  };
  let idx = Index::create(&path, schema, opts(&path)).unwrap();
  let mut writer = idx.writer().unwrap();
  let docs = vec![
    Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("new york subway")),
      ]
      .into_iter()
      .collect(),
    },
    Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-2")),
        ("body".into(), serde_json::json!("boston metro")),
      ]
      .into_iter()
      .collect(),
    },
  ];
  for doc in docs {
    writer.add_document(&doc).unwrap();
  }
  writer.commit().unwrap();
  let reader = idx.reader().unwrap();
  let resp = reader.search(&request("nyc subway")).unwrap();
  let got = ids(&resp);
  assert_eq!(got, vec!["doc-1".to_string()]);
}

#[test]
fn edge_ngram_index_analyzer_supports_prefix_queries() {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx");
  let schema = Schema {
    doc_id_field: "_id".into(),
    analyzers: vec![AnalyzerDef {
      name: "edge".into(),
      tokenizer: "default".into(),
      filters: vec![TokenFilterDef::EdgeNgram(EdgeNgramConfig {
        min: 1,
        max: 3,
      })],
    }],
    text_fields: vec![TextField {
      name: "title".into(),
      analyzer: "edge".into(),
      search_analyzer: Some("default".into()),
      stored: true,
      indexed: true,
      nullable: false,
      search_as_you_type: None,
    }],
    keyword_fields: Vec::new(),
    numeric_fields: Vec::new(),
    nested_fields: Vec::new(),
    #[cfg(feature = "vectors")]
    vector_fields: Vec::new(),
  };
  let idx = Index::create(&path, schema, opts(&path)).unwrap();
  let mut writer = idx.writer().unwrap();
  let docs = vec![
    Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("title".into(), serde_json::json!("rustacean")),
      ]
      .into_iter()
      .collect(),
    },
    Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-2")),
        ("title".into(), serde_json::json!("go coder")),
      ]
      .into_iter()
      .collect(),
    },
  ];
  for doc in docs {
    writer.add_document(&doc).unwrap();
  }
  writer.commit().unwrap();
  let reader = idx.reader().unwrap();
  let resp = reader
    .search(&request(QueryNode::Term {
      field: "title".into(),
      value: "ru".into(),
      boost: None,
    }))
    .unwrap();
  let got = ids(&resp);
  assert_eq!(got, vec!["doc-1".to_string()]);
}

#[test]
fn stopwords_and_stemming_apply_to_phrases_and_terms() {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx");
  let schema = Schema {
    doc_id_field: "_id".into(),
    analyzers: vec![AnalyzerDef {
      name: "stop_stem".into(),
      tokenizer: "default".into(),
      filters: vec![
        TokenFilterDef::Stopwords(StopwordsConfig::Named("en".into())),
        TokenFilterDef::Stemmer(StemmerConfig::Language("english".into())),
      ],
    }],
    text_fields: vec![TextField {
      name: "body".into(),
      analyzer: "stop_stem".into(),
      search_analyzer: None,
      stored: true,
      indexed: true,
      nullable: false,
      search_as_you_type: None,
    }],
    keyword_fields: Vec::new(),
    numeric_fields: Vec::new(),
    nested_fields: Vec::new(),
    #[cfg(feature = "vectors")]
    vector_fields: Vec::new(),
  };
  let idx = Index::create(&path, schema, opts(&path)).unwrap();
  let mut writer = idx.writer().unwrap();
  let docs = vec![
    Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        (
          "body".into(),
          serde_json::json!("state of the art search engine"),
        ),
      ]
      .into_iter()
      .collect(),
    },
    Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-2")),
        ("body".into(), serde_json::json!("runners running uphill")),
      ]
      .into_iter()
      .collect(),
    },
  ];
  for doc in docs {
    writer.add_document(&doc).unwrap();
  }
  writer.commit().unwrap();
  let reader = idx.reader().unwrap();

  let phrase = reader.search(&request("\"state art\"")).unwrap();
  let phrase_ids = ids(&phrase);
  assert_eq!(phrase_ids, vec!["doc-1".to_string()]);

  let stem = reader
    .search(&request(QueryNode::Term {
      field: "body".into(),
      value: "runner".into(),
      boost: None,
    }))
    .unwrap();
  let stem_ids = ids(&stem);
  assert_eq!(stem_ids, vec!["doc-2".to_string()]);
}

#[test]
fn default_schema_remains_compatible() {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx");
  let schema = Schema::default_text_body();
  let idx = Index::create(&path, schema, opts(&path)).unwrap();
  let mut writer = idx.writer().unwrap();
  let docs = vec![
    Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        (
          "body".into(),
          serde_json::json!("Rust: systems programming language"),
        ),
      ]
      .into_iter()
      .collect(),
    },
    Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-2")),
        ("body".into(), serde_json::json!("Go programming language")),
      ]
      .into_iter()
      .collect(),
    },
  ];
  for doc in docs {
    writer.add_document(&doc).unwrap();
  }
  writer.commit().unwrap();
  let reader = idx.reader().unwrap();
  let resp = reader.search(&request("rust language")).unwrap();
  let got = ids(&resp);
  assert_eq!(got, vec!["doc-1".to_string(), "doc-2".to_string()]);
}
