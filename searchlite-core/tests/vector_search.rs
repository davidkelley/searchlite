#![cfg(feature = "vectors")]

use std::collections::BTreeMap;
use std::path::Path;

use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Aggregation, Document, ExecutionStrategy, Filter, IndexOptions, Query, QueryNode, SearchRequest,
  SortSpec, StorageType, VectorQuery,
};
use searchlite_core::{Index, Schema};
use serde_json::json;
use tempfile::tempdir;

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

fn schema() -> Schema {
  serde_json::from_value(json!({
    "doc_id_field": "_id",
    "text_fields": [
      { "name": "body", "analyzer": "default", "stored": true, "indexed": true, "nullable": false }
    ],
    "keyword_fields": [
      { "name": "tag", "stored": true, "indexed": true, "fast": true, "nullable": true }
    ],
    "numeric_fields": [],
    "nested_fields": [],
    "vector_fields": [
      { "name": "embedding", "dim": 2, "metric": "Cosine" }
    ]
  }))
  .expect("schema")
}

fn add_docs(idx: &Index, docs: &[Document]) {
  let mut writer = idx.writer().expect("writer");
  for doc in docs {
    writer.add_document(doc).expect("add doc");
  }
  writer.commit().expect("commit");
}

fn base_request(query: Query, limit: usize) -> SearchRequest {
  SearchRequest {
    query,
    fields: None,
    filter: None,
    filters: Vec::new(),
    limit,
    sort: Vec::<SortSpec>::new(),
    cursor: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    fuzzy: None,
    #[cfg(feature = "vectors")]
    vector_query: None,
    #[cfg(feature = "vectors")]
    vector_filter: None,
    return_stored: true,
    highlight_field: None,
    aggs: BTreeMap::<String, Aggregation>::new(),
    suggest: BTreeMap::new(),
    rescore: None,
    explain: false,
    profile: false,
  }
}

#[test]
fn vector_only_search_skips_missing_vectors() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[
      Document {
        fields: [
          ("_id".into(), serde_json::json!("vec-1")),
          ("body".into(), serde_json::json!("rust search")),
          ("embedding".into(), serde_json::json!([1.0, 0.0])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), serde_json::json!("vec-2")),
          ("body".into(), serde_json::json!("other body")),
          ("embedding".into(), serde_json::json!([0.0, 1.0])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), serde_json::json!("no-vector")),
          ("body".into(), serde_json::json!("no embedding here")),
        ]
        .into_iter()
        .collect(),
      },
    ],
  );
  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    query: Query::Node(QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![1.0, 0.0],
      k: Some(3),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: Some(3),
      boost: None,
    })),
    #[cfg(feature = "vectors")]
    vector_query: None,
    #[cfg(feature = "vectors")]
    vector_filter: None,
    ..base_request(Query::String("".into()), 3)
  };
  let res = reader.search(&req).unwrap();
  assert_eq!(res.hits.len(), 2);
  assert_eq!(res.hits[0].doc_id, "vec-1");
  assert!(res.hits[0].vector_score.is_some());
  assert!(res.hits.iter().all(|h| h.doc_id != "no-vector"));
}

#[test]
fn hybrid_blends_text_and_vector() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[
      Document {
        fields: [
          ("_id".into(), serde_json::json!("short")),
          ("body".into(), serde_json::json!("rust")),
          ("embedding".into(), serde_json::json!([1.0, 0.0])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), serde_json::json!("long")),
          ("body".into(), serde_json::json!("rust rust rust")),
          ("embedding".into(), serde_json::json!([0.0, 1.0])),
        ]
        .into_iter()
        .collect(),
      },
    ],
  );
  let reader = idx.reader().unwrap();
  let bm25_only = SearchRequest {
    #[cfg(feature = "vectors")]
    vector_query: Some(("embedding".into(), vec![1.0, 0.0], 1.0)),
    #[cfg(feature = "vectors")]
    vector_filter: None,
    ..base_request(
      QueryNode::QueryString {
        query: "rust".into(),
        fields: None,
        boost: None,
      }
      .into(),
      2,
    )
  };
  let blended = SearchRequest {
    #[cfg(feature = "vectors")]
    vector_query: Some(("embedding".into(), vec![1.0, 0.0], 0.2)),
    #[cfg(feature = "vectors")]
    vector_filter: None,
    ..bm25_only.clone()
  };
  let bm25_hits = reader.search(&bm25_only).unwrap().hits;
  let blended_hits = reader.search(&blended).unwrap().hits;
  assert_eq!(bm25_hits.first().map(|h| h.doc_id.as_str()), Some("long"));
  assert_eq!(
    blended_hits.first().map(|h| h.doc_id.as_str()),
    Some("short")
  );
  assert!(blended_hits[0].score > blended_hits[1].score);
  assert!(blended_hits[0].vector_score.is_some());
}

#[test]
fn hybrid_applies_alpha_to_docs_without_vectors() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[
      Document {
        fields: [
          ("_id".into(), serde_json::json!("with-vector")),
          ("body".into(), serde_json::json!("rust")),
          ("embedding".into(), serde_json::json!([1.0, 0.0])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), serde_json::json!("bm25-heavy")),
          ("body".into(), serde_json::json!("rust rust rust rust rust")),
        ]
        .into_iter()
        .collect(),
      },
    ],
  );
  let reader = idx.reader().unwrap();
  let bm25_heavy = SearchRequest {
    #[cfg(feature = "vectors")]
    vector_query: Some(("embedding".into(), vec![1.0, 0.0], 1.0)),
    #[cfg(feature = "vectors")]
    vector_filter: None,
    ..base_request(
      QueryNode::QueryString {
        query: "rust".into(),
        fields: None,
        boost: None,
      }
      .into(),
      2,
    )
  };
  let blended = SearchRequest {
    #[cfg(feature = "vectors")]
    vector_query: Some(("embedding".into(), vec![1.0, 0.0], 0.2)),
    #[cfg(feature = "vectors")]
    vector_filter: None,
    ..bm25_heavy.clone()
  };
  let bm25_hits = reader.search(&bm25_heavy).unwrap().hits;
  let blended_hits = reader.search(&blended).unwrap().hits;
  assert_eq!(
    bm25_hits.first().map(|h| h.doc_id.as_str()),
    Some("bm25-heavy")
  );
  assert_eq!(
    blended_hits.first().map(|h| h.doc_id.as_str()),
    Some("with-vector")
  );
}

#[test]
fn vector_filter_limits_results() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[
      Document {
        fields: [
          ("_id".into(), serde_json::json!("keep")),
          ("body".into(), serde_json::json!("rust keep")),
          ("tag".into(), serde_json::json!("keep")),
          ("embedding".into(), serde_json::json!([1.0, 0.0])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), serde_json::json!("drop")),
          ("body".into(), serde_json::json!("rust drop")),
          ("tag".into(), serde_json::json!("drop")),
          ("embedding".into(), serde_json::json!([1.0, 0.0])),
        ]
        .into_iter()
        .collect(),
      },
    ],
  );
  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    query: Query::Node(QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![1.0, 0.0],
      k: Some(5),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: None,
      boost: None,
    })),
    #[cfg(feature = "vectors")]
    vector_query: None,
    #[cfg(feature = "vectors")]
    vector_filter: Some(Filter::KeywordEq {
      field: "tag".into(),
      value: "keep".into(),
    }),
    ..base_request(Query::String("".into()), 5)
  };
  let hits = reader.search(&req).unwrap().hits;
  assert_eq!(hits.len(), 1);
  assert_eq!(hits[0].doc_id, "keep");
}

#[test]
fn vector_search_caps_to_available_vectors() {
  let dir = tempdir().unwrap();
  let mut schema = schema();
  schema.vector_fields[0].dim = 3;
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[
      Document {
        fields: [
          ("_id".into(), serde_json::json!("only-one")),
          ("body".into(), serde_json::json!("rust caps k")),
          ("embedding".into(), serde_json::json!([1.0, 0.0, 0.0])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), serde_json::json!("no-vector")),
          ("body".into(), serde_json::json!("rust none")),
        ]
        .into_iter()
        .collect(),
      },
    ],
  );
  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    query: Query::Node(QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![1.0, 0.0, 0.0],
      k: Some(10),
      alpha: Some(0.0),
      ef_search: Some(50),
      candidate_size: Some(20),
      boost: None,
    })),
    #[cfg(feature = "vectors")]
    vector_query: None,
    #[cfg(feature = "vectors")]
    vector_filter: None,
    ..base_request(Query::String("".into()), 10)
  };
  let hits = reader.search(&req).unwrap().hits;
  assert_eq!(hits.len(), 1);
  assert_eq!(hits[0].doc_id, "only-one");
}
