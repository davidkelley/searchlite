#![cfg(feature = "vectors")]

use std::collections::BTreeMap;
use std::path::Path;

use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Aggregation, Document, ExecutionStrategy, Filter, IndexOptions, LegacyVectorQuery, Query,
  QueryNode, RescoreMode, RescoreRequest, SearchRequest, SortSpec, StorageType, VectorQuery,
  VectorQuerySpec,
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
    checksum_policy: Default::default(),
    checksum_audit_failure_hook: None,
    read_only: false,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  }
}

fn schema() -> Schema {
  serde_json::from_value(json!({
    "type": "object",
    "properties": {
      "body": { "type": "string" },
      "tag": { "type": ["string", "null"], "searchlite:kind": "keyword" },
      "embedding": { "type": "array", "items": { "type": "number" }, "searchlite:vector": { "dim": 2, "metric": "Cosine" } }
    }
  }))
  .expect("schema")
}

fn multi_vector_schema() -> Schema {
  serde_json::from_value(json!({
    "type": "object",
    "properties": {
      "body": { "type": "string" },
      "vec_a": { "type": "array", "items": { "type": "number" }, "searchlite:vector": { "dim": 2, "metric": "Cosine" } },
      "vec_b": { "type": "array", "items": { "type": "number" }, "searchlite:vector": { "dim": 2, "metric": "Cosine" } }
    }
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
    limit,
    from: 0,
    return_hits: true,
    candidate_size: None,
    #[cfg(feature = "vectors")]
    max_global_vector_candidates: None,
    sort: Vec::<SortSpec>::new(),
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
    return_stored: true,
    highlight_field: None,
    highlight: None,
    collapse: None,
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
fn vector_query_with_limit_zero_succeeds_without_hits() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[Document {
      fields: [
        ("_id".into(), serde_json::json!("only")),
        ("body".into(), serde_json::json!("rust search")),
        ("embedding".into(), serde_json::json!([1.0, 0.0])),
      ]
      .into_iter()
      .collect(),
    }],
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
    ..base_request(Query::String("".into()), 0)
  };
  let res = reader.search(&req).unwrap();
  assert!(res.hits.is_empty());
  assert_eq!(res.next_cursor, None);
  assert!(res.total_hits_estimate > 0);
}

// BUG-411 follow-up (Codex P1 / Copilot review on #501): the text/hybrid
// rescore + score-sort cursor guard must not regress vector-only queries.
// `search_vector_only` never calls `rescore_hits`, so its cursor is built
// from the base vector score and is not corrupted by rescore even when a
// caller provides a (silently-ignored) `rescore` block. The inbound
// validation and outbound suppression in `IndexReader::search` therefore
// run *after* the vector-only early return; this test pins that scoping
// in place so page 2 of a vector-only paginated request still succeeds.
#[test]
fn vector_only_paginated_cursor_works_when_request_carries_rescore() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[
      Document {
        fields: [
          ("_id".into(), json!("vec-1")),
          ("body".into(), json!("rust search engine")),
          ("embedding".into(), json!([1.0, 0.0])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), json!("vec-2")),
          ("body".into(), json!("rust traits")),
          ("embedding".into(), json!([0.95, 0.05])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), json!("vec-3")),
          ("body".into(), json!("rust query")),
          ("embedding".into(), json!([0.9, 0.1])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), json!("vec-4")),
          ("body".into(), json!("rust borrow checker")),
          ("embedding".into(), json!([0.8, 0.2])),
        ]
        .into_iter()
        .collect(),
      },
    ],
  );
  let reader = idx.reader().unwrap();
  // Vector-only request: alpha=0.0 means pure vector ranking, no text
  // contribution. The included `rescore` block would matter for a
  // text/hybrid search, but `search_vector_only` silently ignores it —
  // we still expect a working cursor.
  let mut req = SearchRequest {
    query: Query::Node(QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![1.0, 0.0],
      k: Some(4),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: Some(4),
      boost: None,
    })),
    vector_query: None,
    vector_filter: None,
    rescore: Some(RescoreRequest {
      window_size: 2,
      query: QueryNode::Phrase {
        field: Some("body".into()),
        terms: vec!["rust".into(), "search".into()],
        slop: Some(2),
        boost: None,
      },
      score_mode: RescoreMode::Total,
    }),
    ..base_request(Query::String("".into()), 2)
  };
  let first = reader.search(&req).expect("page 1 must succeed");
  assert_eq!(first.hits.len(), 2, "expected limit=2 hits on page 1");
  let cursor = first
    .next_cursor
    .clone()
    .expect("vector-only + rescore must still emit next_cursor");

  req.cursor = Some(cursor);
  let second = reader.search(&req).expect("page 2 must succeed");
  assert!(!second.hits.is_empty(), "page 2 must return hits");
  let first_ids: std::collections::HashSet<&str> =
    first.hits.iter().map(|h| h.doc_id.as_str()).collect();
  for hit in &second.hits {
    assert!(
      !first_ids.contains(hit.doc_id.as_str()),
      "page 1 and page 2 must be disjoint; {} appeared in both",
      hit.doc_id
    );
  }
}

#[test]
fn hybrid_vector_query_with_limit_zero_returns_no_hits() {
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
    ],
  );
  let reader = idx.reader().unwrap();
  let mut req = base_request(Query::String("rust".into()), 0);
  req.vector_query = Some(VectorQuerySpec::Structured(VectorQuery {
    field: "embedding".into(),
    vector: vec![1.0, 0.0],
    k: Some(3),
    alpha: Some(0.5),
    ef_search: None,
    candidate_size: Some(3),
    boost: None,
  }));
  let res = reader.search(&req).unwrap();
  assert!(res.hits.is_empty());
  assert_eq!(res.next_cursor, None);
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
    vector_query: Some(VectorQuerySpec::Legacy(LegacyVectorQuery(
      "embedding".into(),
      vec![1.0, 0.0],
      1.0,
    ))),
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
    vector_query: Some(VectorQuerySpec::Legacy(LegacyVectorQuery(
      "embedding".into(),
      vec![1.0, 0.0],
      0.2,
    ))),
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

fn schema_l2() -> Schema {
  serde_json::from_value(json!({
    "type": "object",
    "properties": {
      "body": { "type": "string" },
      "embedding": { "type": "array", "items": { "type": "number" }, "searchlite:vector": { "dim": 2, "metric": "L2" } }
    }
  }))
  .expect("schema")
}

#[test]
fn hybrid_l2_penalizes_missing_vectors() {
  let dir = tempdir().unwrap();
  let schema = schema_l2();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[
      Document {
        fields: [
          ("_id".into(), serde_json::json!("with-vector")),
          ("body".into(), serde_json::json!("rust vector")),
          ("embedding".into(), serde_json::json!([0.0, 0.0])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), serde_json::json!("bm25-only")),
          ("body".into(), serde_json::json!("rust vector")),
        ]
        .into_iter()
        .collect(),
      },
    ],
  );
  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    #[cfg(feature = "vectors")]
    vector_query: Some(VectorQuerySpec::Legacy(LegacyVectorQuery(
      "embedding".into(),
      vec![1.0, 1.0],
      0.2,
    ))),
    #[cfg(feature = "vectors")]
    vector_filter: None,
    ..base_request("rust".into(), 2)
  };
  let hits = reader.search(&req).unwrap().hits;
  assert_eq!(hits.first().map(|h| h.doc_id.as_str()), Some("with-vector"));
  assert!(hits.iter().any(|h| h.doc_id == "bm25-only"));
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
    vector_query: Some(VectorQuerySpec::Legacy(LegacyVectorQuery(
      "embedding".into(),
      vec![1.0, 0.0],
      1.0,
    ))),
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
    vector_query: Some(VectorQuerySpec::Legacy(LegacyVectorQuery(
      "embedding".into(),
      vec![1.0, 0.0],
      0.2,
    ))),
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

#[test]
fn multiple_vector_clauses_merge_candidates() {
  let dir = tempdir().unwrap();
  let schema = multi_vector_schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[
      Document {
        fields: [
          ("_id".into(), serde_json::json!("doc-1")),
          ("body".into(), serde_json::json!("first")),
          ("vec_a".into(), serde_json::json!([1.0, 0.0])),
          ("vec_b".into(), serde_json::json!([0.0, 1.0])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), serde_json::json!("doc-2")),
          ("body".into(), serde_json::json!("second")),
          ("vec_a".into(), serde_json::json!([0.0, 1.0])),
          ("vec_b".into(), serde_json::json!([0.0, 1.0])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), serde_json::json!("doc-3")),
          ("body".into(), serde_json::json!("third")),
          ("vec_a".into(), serde_json::json!([0.0, 1.0])),
          ("vec_b".into(), serde_json::json!([1.0, 0.0])),
        ]
        .into_iter()
        .collect(),
      },
    ],
  );
  let reader = idx.reader().unwrap();
  let query = QueryNode::Bool {
    must: Vec::new(),
    should: vec![
      QueryNode::Vector(VectorQuery {
        field: "vec_a".into(),
        vector: vec![1.0, 0.0],
        k: Some(3),
        alpha: Some(0.0),
        ef_search: None,
        candidate_size: Some(3),
        boost: Some(1.0),
      }),
      QueryNode::Vector(VectorQuery {
        field: "vec_b".into(),
        vector: vec![0.0, 1.0],
        k: Some(3),
        alpha: Some(0.0),
        ef_search: None,
        candidate_size: Some(3),
        boost: Some(1.0),
      }),
    ],
    must_not: Vec::new(),
    filter: Vec::new(),
    minimum_should_match: None,
    boost: None,
  };
  let req = SearchRequest {
    query: Query::Node(query),
    limit: 3,
    from: 0,
    return_hits: true,
    #[cfg(feature = "vectors")]
    vector_query: None,
    #[cfg(feature = "vectors")]
    vector_filter: None,
    ..base_request(Query::String("".into()), 3)
  };
  let hits = reader.search(&req).unwrap().hits;
  let ids: Vec<_> = hits.iter().map(|h| h.doc_id.as_str()).collect();
  assert_eq!(ids, vec!["doc-1", "doc-2", "doc-3"]);
}

#[test]
fn rejects_global_cap_below_clause_count() {
  let dir = tempdir().unwrap();
  let schema = multi_vector_schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("first")),
        ("vec_a".into(), serde_json::json!([1.0, 0.0])),
        ("vec_b".into(), serde_json::json!([0.0, 1.0])),
      ]
      .into_iter()
      .collect(),
    }],
  );
  let reader = idx.reader().unwrap();
  let query = QueryNode::Bool {
    must: Vec::new(),
    should: vec![
      QueryNode::Vector(VectorQuery {
        field: "vec_a".into(),
        vector: vec![1.0, 0.0],
        k: Some(1),
        alpha: Some(0.0),
        ef_search: None,
        candidate_size: Some(1),
        boost: Some(1.0),
      }),
      QueryNode::Vector(VectorQuery {
        field: "vec_b".into(),
        vector: vec![0.0, 1.0],
        k: Some(1),
        alpha: Some(0.0),
        ef_search: None,
        candidate_size: Some(1),
        boost: Some(1.0),
      }),
    ],
    must_not: Vec::new(),
    filter: Vec::new(),
    minimum_should_match: None,
    boost: None,
  };
  let req = SearchRequest {
    query: Query::Node(query),
    limit: 2,
    from: 0,
    return_hits: true,
    max_global_vector_candidates: Some(1), // smaller than clause count (2)
    vector_query: None,
    vector_filter: None,
    ..base_request(Query::String("".into()), 2)
  };
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("max_global_vector_candidates"),
    "expected validation error, got {err}"
  );
}

// BUG-330: `collect_vector_value` previously cast each JSON `f64` component
// to `f32` via `num as f32` without validating the resulting `f32`. A finite
// `f64` whose magnitude exceeds `f32::MAX` (~3.4e38) saturates to
// `±f32::INFINITY` under Rust's `as` cast and was persisted into the segment,
// then propagated through `metric_similarity` (and through `normalize_in_place`
// on cosine, which produces an all-`NaN` vector). The fix bails when the
// pending document is flushed to a segment, surfacing the bad input to the
// writer instead of corrupting reads forever after.
#[test]
fn commit_rejects_vector_component_overflowing_f32_to_positive_inf() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  let mut writer = idx.writer().expect("writer");
  // 1e40 is a valid JSON number; it parses to a finite f64 ≈ 1e40 but
  // `1e40_f64 as f32` saturates to f32::INFINITY.
  let doc = Document {
    fields: [
      ("_id".into(), serde_json::json!("inf-pos")),
      ("body".into(), serde_json::json!("body")),
      ("embedding".into(), serde_json::json!([1.0e40, 0.0])),
    ]
    .into_iter()
    .collect(),
  };
  writer.add_document(&doc).expect("staging accepts the doc");
  let err = writer.commit().unwrap_err().to_string();
  assert!(
    err.contains("non-finite"),
    "expected non-finite component error, got {err}"
  );
}

#[test]
fn commit_rejects_vector_component_overflowing_f32_to_negative_inf() {
  let dir = tempdir().unwrap();
  let schema = schema_l2();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  let mut writer = idx.writer().expect("writer");
  let doc = Document {
    fields: [
      ("_id".into(), serde_json::json!("inf-neg")),
      ("body".into(), serde_json::json!("body")),
      ("embedding".into(), serde_json::json!([-1.0e40, 1.0])),
    ]
    .into_iter()
    .collect(),
  };
  writer.add_document(&doc).expect("staging accepts the doc");
  let err = writer.commit().unwrap_err().to_string();
  assert!(
    err.contains("non-finite"),
    "expected non-finite component error, got {err}"
  );
}

#[test]
fn commit_accepts_vector_component_with_finite_squared_magnitude() {
  // The boundary case for BUG-330 / BUG-386: an individually-finite component
  // whose squared magnitude also stays inside `f32::MAX` must still be
  // accepted so the new sum-of-squares guard does not over-reject legitimate
  // inputs. `(1e19)^2 = 1e38 < f32::MAX ≈ 3.4e38`, so this vector passes
  // both guards. A literal `f32::MAX` component is not accepted because
  // `(f32::MAX)^2` itself overflows to `+inf`; the per-component boundary
  // from pre-BUG-386 is therefore no longer the effective acceptance bound —
  // the sum-of-squares boundary is. No dedicated test exercises a literal
  // `f32::MAX` component because such a vector is now (correctly) rejected by
  // the BUG-386 guard at both commit and query time.
  let dir = tempdir().unwrap();
  let schema = schema_l2();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  let mut writer = idx.writer().expect("writer");
  let doc = Document {
    fields: [
      ("_id".into(), serde_json::json!("at-max")),
      ("body".into(), serde_json::json!("body")),
      ("embedding".into(), serde_json::json!([1.0e19_f64, 0.0])),
    ]
    .into_iter()
    .collect(),
  };
  writer
    .add_document(&doc)
    .expect("finite component with finite squared magnitude must be accepted");
  writer
    .commit()
    .expect("finite component with finite squared magnitude must commit");
}

#[test]
fn vector_query_with_positive_inf_component_is_rejected() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("rust search")),
        ("embedding".into(), serde_json::json!([1.0, 0.0])),
      ]
      .into_iter()
      .collect(),
    }],
  );
  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    query: Query::Node(QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![f32::INFINITY, 0.0],
      k: Some(3),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: Some(3),
      boost: None,
    })),
    vector_query: None,
    vector_filter: None,
    ..base_request(Query::String("".into()), 3)
  };
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("non-finite component") && err.contains("embedding"),
    "expected non-finite component error, got {err}"
  );
}

#[test]
fn vector_query_with_negative_inf_component_is_rejected_for_l2() {
  let dir = tempdir().unwrap();
  let schema = schema_l2();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("rust vector")),
        ("embedding".into(), serde_json::json!([0.0, 0.0])),
      ]
      .into_iter()
      .collect(),
    }],
  );
  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    vector_query: Some(VectorQuerySpec::Structured(VectorQuery {
      field: "embedding".into(),
      vector: vec![f32::NEG_INFINITY, 0.0],
      k: Some(3),
      alpha: Some(0.5),
      ef_search: None,
      candidate_size: Some(3),
      boost: None,
    })),
    vector_filter: None,
    ..base_request(Query::String("rust".into()), 3)
  };
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("non-finite component") && err.contains("embedding"),
    "expected non-finite component error, got {err}"
  );
}

#[test]
fn vector_query_with_nan_component_is_rejected() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("rust search")),
        ("embedding".into(), serde_json::json!([1.0, 0.0])),
      ]
      .into_iter()
      .collect(),
    }],
  );
  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    query: Query::Node(QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![0.0, f32::NAN],
      k: Some(3),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: Some(3),
      boost: None,
    })),
    vector_query: None,
    vector_filter: None,
    ..base_request(Query::String("".into()), 3)
  };
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("non-finite component") && err.contains("embedding"),
    "expected non-finite component error, got {err}"
  );
}

#[test]
fn vector_query_with_finite_boundary_components_is_accepted() {
  let dir = tempdir().unwrap();
  let schema = schema_l2();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("rust vector")),
        ("embedding".into(), serde_json::json!([0.0, 0.0])),
      ]
      .into_iter()
      .collect(),
    }],
  );
  let reader = idx.reader().unwrap();
  // Use a large-but-finite component that keeps the sum-of-squares within
  // `f32::MAX` (post-BUG-386): `(1e19)^2 = 1e38 < f32::MAX ≈ 3.4e38`. This
  // test exercises the sum-of-squares acceptance boundary for finite values.
  // Cases where `sum-of-squares` itself overflows are covered by the
  // dedicated `*_overflowing_sum_of_squares_*` tests below.
  let req = SearchRequest {
    query: Query::Node(QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![1.0e19_f32, 0.0],
      k: Some(3),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: Some(3),
      boost: None,
    })),
    vector_query: None,
    vector_filter: None,
    ..base_request(Query::String("".into()), 3)
  };
  let _ = reader.search(&req).expect("boundary f32 values are finite");
}

// BUG-384: cosine query vectors with individually-finite components whose
// squared magnitudes sum past `f32::MAX` used to reach `normalize_in_place`,
// where division by `+inf` silently turned them into an all-zero vector — so
// downstream cosine dot products returned 0 for every hit and the query
// matched nothing even though the caller supplied a well-formed vector. The
// defensive `is_finite()` guard now skips normalization instead of zeroing
// it, but an un-normalized cosine vector still violates the unit-length
// assumption and produces garbage scores. Reject the input at the same layer
// as BUG-340 so callers get an actionable error either way.
#[test]
fn cosine_query_vector_with_overflowing_sum_of_squares_is_rejected() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("rust search")),
        ("embedding".into(), serde_json::json!([1.0, 0.0])),
      ]
      .into_iter()
      .collect(),
    }],
  );
  let reader = idx.reader().unwrap();
  // Each component is finite (`3e19 < f32::MAX ≈ 3.4e38`) and passes the
  // BUG-340 per-component guard, but `(3e19)^2 + (3e19)^2 = 1.8e39` overflows
  // `f32::MAX` to `+inf`.
  let req = SearchRequest {
    query: Query::Node(QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![3.0e19_f32, 3.0e19_f32],
      k: Some(3),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: Some(3),
      boost: None,
    })),
    vector_query: None,
    vector_filter: None,
    ..base_request(Query::String("".into()), 3)
  };
  let err = reader.search(&req).unwrap_err().to_string();
  // BUG-386 unified cosine + L2 under the same guard and error shape.
  assert!(
    err.contains("sum-of-squares overflows") && err.contains("embedding"),
    "expected sum-of-squares overflow error, got {err}"
  );
}

// BUG-384: cosine-indexed documents whose squared magnitudes sum past
// `f32::MAX` used to be silently written as an all-zero normalized vector.
// Commit must refuse the document so the segment does not capture a corrupt
// vector that is invisible to every subsequent query.
#[test]
fn commit_rejects_cosine_vector_with_overflowing_sum_of_squares() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  let mut writer = idx.writer().expect("writer");
  // Each component is finite but their squared magnitudes sum to `+inf`.
  let doc = Document {
    fields: [
      ("_id".into(), serde_json::json!("overflow-norm")),
      ("body".into(), serde_json::json!("body")),
      (
        "embedding".into(),
        serde_json::json!([3.0e19_f64, 3.0e19_f64]),
      ),
    ]
    .into_iter()
    .collect(),
  };
  writer.add_document(&doc).expect("staging accepts the doc");
  let err = writer.commit().unwrap_err().to_string();
  // BUG-386 unified cosine + L2 under the same guard and error shape.
  assert!(
    err.contains("sum-of-squares overflows"),
    "expected sum-of-squares overflow error, got {err}"
  );
}

// Companion: a cosine-indexed document and query where the sum-of-squares is
// finite but close to the boundary still round-trip successfully — the new
// guard must not reject legitimate inputs.
#[test]
fn cosine_vector_with_finite_sum_of_squares_round_trips() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  // `(1e19)^2 = 1e38 < f32::MAX ≈ 3.4e38`, so the sum-of-squares stays finite.
  add_docs(
    &idx,
    &[Document {
      fields: [
        ("_id".into(), serde_json::json!("finite-norm")),
        ("body".into(), serde_json::json!("rust search")),
        ("embedding".into(), serde_json::json!([1.0e19_f64, 0.0])),
      ]
      .into_iter()
      .collect(),
    }],
  );
  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    query: Query::Node(QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![1.0e19_f32, 0.0],
      k: Some(3),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: Some(3),
      boost: None,
    })),
    vector_query: None,
    vector_filter: None,
    ..base_request(Query::String("".into()), 3)
  };
  let hits = reader
    .search(&req)
    .expect("finite sum-of-squares must be accepted")
    .hits;
  assert!(!hits.is_empty(), "expected the single doc to match");
  assert_eq!(hits[0].doc_id.as_str(), "finite-norm");
}

// BUG-386: L2-indexed documents with individually-finite components whose
// squared magnitudes sum past `f32::MAX` used to be persisted verbatim, then
// `l2_distance` would accumulate `v[i]^2` into `+inf` against an origin-like
// query and `-l2_distance` would return `-inf`, which `compute_hybrid_score`
// silently dropped via the BUG-328 guard — the caller saw an empty hit set
// for docs they had just indexed, with no actionable error. Extend the
// BUG-384 sum-of-squares check to L2 so commit refuses the doc instead of
// capturing a vector that cannot be meaningfully compared to distant queries.
#[test]
fn commit_rejects_l2_vector_with_overflowing_sum_of_squares() {
  let dir = tempdir().unwrap();
  let schema = schema_l2();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  let mut writer = idx.writer().expect("writer");
  // Each component is finite (`3e19 < f32::MAX`) but `(3e19)^2 + (3e19)^2 =
  // 1.8e39 > f32::MAX`.
  let doc = Document {
    fields: [
      ("_id".into(), serde_json::json!("l2-overflow")),
      ("body".into(), serde_json::json!("body")),
      (
        "embedding".into(),
        serde_json::json!([3.0e19_f64, 3.0e19_f64]),
      ),
    ]
    .into_iter()
    .collect(),
  };
  writer.add_document(&doc).expect("staging accepts the doc");
  let err = writer.commit().unwrap_err().to_string();
  assert!(
    err.contains("sum-of-squares overflows") && err.contains("embedding"),
    "expected sum-of-squares overflow error, got {err}"
  );
}

// BUG-386: symmetric query-side guard — an L2 query vector whose squared
// magnitudes sum past `f32::MAX` used to slip past the BUG-340 per-component
// check, drive `l2_distance` into `+inf`, and return an empty hit set. Reject
// at plan time with the same message shape as the cosine BUG-384 guard.
#[test]
fn l2_query_vector_with_overflowing_sum_of_squares_is_rejected() {
  let dir = tempdir().unwrap();
  let schema = schema_l2();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("rust vector")),
        ("embedding".into(), serde_json::json!([0.0, 0.0])),
      ]
      .into_iter()
      .collect(),
    }],
  );
  let reader = idx.reader().unwrap();
  // Each component is finite but their squared magnitudes sum to `+inf`.
  let req = SearchRequest {
    query: Query::Node(QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![3.0e19_f32, 3.0e19_f32],
      k: Some(3),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: Some(3),
      boost: None,
    })),
    vector_query: None,
    vector_filter: None,
    ..base_request(Query::String("".into()), 3)
  };
  let err = reader.search(&req).unwrap_err().to_string();
  assert!(
    err.contains("sum-of-squares overflows") && err.contains("embedding"),
    "expected sum-of-squares overflow error, got {err}"
  );
}

// BUG-386: boundary test — an L2-indexed doc and query whose sum-of-squares
// are each finite but close to `f32::MAX` must still round-trip. The new
// guard must not over-reject legitimate inputs just because the components
// are large.
#[test]
fn l2_vector_with_finite_sum_of_squares_round_trips() {
  let dir = tempdir().unwrap();
  let schema = schema_l2();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  // `(1e19)^2 = 1e38 < f32::MAX ≈ 3.4e38`, so the sum-of-squares stays
  // finite even when the component magnitudes are near the top of the f32
  // range.
  add_docs(
    &idx,
    &[Document {
      fields: [
        ("_id".into(), serde_json::json!("l2-finite")),
        ("body".into(), serde_json::json!("rust vector")),
        ("embedding".into(), serde_json::json!([1.0e19_f64, 0.0])),
      ]
      .into_iter()
      .collect(),
    }],
  );
  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    query: Query::Node(QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![1.0e19_f32, 0.0],
      k: Some(3),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: Some(3),
      boost: None,
    })),
    vector_query: None,
    vector_filter: None,
    ..base_request(Query::String("".into()), 3)
  };
  let hits = reader
    .search(&req)
    .expect("finite sum-of-squares must be accepted")
    .hits;
  assert!(!hits.is_empty(), "expected the single doc to match");
  assert_eq!(hits[0].doc_id.as_str(), "l2-finite");
}

// BUG-388: the BUG-384 / BUG-386 sum-of-squares guards bound every indexed
// and queried vector to `sum(v_i^2) <= f32::MAX`, which constrains
// `|v_i| <= sqrt(f32::MAX) ≈ 1.84e19` per component. But pairwise
// `|a_i - b_i|` can still reach `3.68e19`, so a single dimension's squared
// difference can overflow `f32::MAX`. Before the `l2_distance` saturation
// guard, `sum` saturated to `+inf`, `metric_similarity(L2) = -inf`, and the
// BUG-328 hybrid-score guard silently dropped the far doc from the result
// set with no error. After the guard, `l2_distance` returns a finite
// sentinel (`f32::MAX`) so the doc is returned (ranked last) instead of
// silently disappearing.
#[test]
fn l2_search_returns_far_doc_when_pairwise_squared_diff_overflows() {
  let dir = tempdir().unwrap();
  let schema = schema_l2();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  // Both vectors pass the per-vector sum-of-squares bound: `(1.5e19)^2 =
  // 2.25e38 < f32::MAX ≈ 3.4e38`. But their pairwise `d[0] = 3.0e19` and
  // `d[0]^2 = 9e38 > f32::MAX` overflows the `l2_distance` accumulator in a
  // single step.
  add_docs(
    &idx,
    &[
      Document {
        fields: [
          ("_id".into(), serde_json::json!("near")),
          ("body".into(), serde_json::json!("rust vector")),
          ("embedding".into(), serde_json::json!([1.5e19_f64, 0.0_f64])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), serde_json::json!("far")),
          ("body".into(), serde_json::json!("rust search")),
          (
            "embedding".into(),
            serde_json::json!([-1.5e19_f64, 0.0_f64]),
          ),
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
      vector: vec![1.5e19_f32, 0.0_f32],
      k: Some(10),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: Some(10),
      boost: None,
    })),
    vector_query: None,
    vector_filter: None,
    ..base_request(Query::String("".into()), 10)
  };
  let hits = reader
    .search(&req)
    .expect("pairwise-overflow distance must not fail the query")
    .hits;
  let ids: Vec<_> = hits.iter().map(|h| h.doc_id.as_str().to_string()).collect();
  assert!(
    ids.contains(&"far".to_string()),
    "`far` doc with overflowing pairwise distance must be returned, not silently dropped; got ids={ids:?}",
  );
  assert!(
    ids.contains(&"near".to_string()),
    "`near` doc must remain in results alongside `far`; got ids={ids:?}",
  );
  for hit in hits.iter() {
    assert!(
      hit.score.is_finite(),
      "every hit score must be finite; doc {} had score {}",
      hit.doc_id,
      hit.score,
    );
  }
}

// BUG-394: BUG-388 saturates `l2_distance` to `f32::MAX` so that
// `metric_similarity(L2)` returns the finite sentinel `f32::MIN` at the worst
// pair — avoiding the `-INF` that the BUG-328 hybrid-score guard would
// otherwise drop. That saturation is local to `l2_distance`, though. The very
// next step in `collect_vector_maps` is `vscore *= clause.boost`, and any
// user-supplied `boost > 1.0` turns the `f32::MIN` sentinel straight back
// into `-INF` via f32 multiplication overflow. The non-finite `vscore` then
// flows into `vector_scores`, `compute_hybrid_score` rejects it, and the
// `far` doc is silently dropped again — undoing BUG-388's fix for any query
// with a non-default boost. This regression is identical in shape to
// `l2_search_returns_far_doc_when_pairwise_squared_diff_overflows` except for
// `boost: Some(2.0)`; pre-fix, that single change causes `far` to disappear
// from the result set, matching exactly the failure mode BUG-388 was written
// to eliminate.
#[test]
fn l2_search_with_boost_gt_one_returns_far_doc_when_pairwise_squared_diff_overflows() {
  let dir = tempdir().unwrap();
  let schema = schema_l2();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[
      Document {
        fields: [
          ("_id".into(), serde_json::json!("near")),
          ("body".into(), serde_json::json!("rust vector")),
          ("embedding".into(), serde_json::json!([1.5e19_f64, 0.0_f64])),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".into(), serde_json::json!("far")),
          ("body".into(), serde_json::json!("rust search")),
          (
            "embedding".into(),
            serde_json::json!([-1.5e19_f64, 0.0_f64]),
          ),
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
      vector: vec![1.5e19_f32, 0.0_f32],
      k: Some(10),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: Some(10),
      // `boost > 1.0` is the only delta vs the BUG-388 regression: pre-fix,
      // `f32::MIN * 2.0` overflows to `-INF` and the far doc silently
      // disappears even though every pre-BUG-394 guard passes.
      boost: Some(2.0),
    })),
    vector_query: None,
    vector_filter: None,
    ..base_request(Query::String("".into()), 10)
  };
  let hits = reader
    .search(&req)
    .expect("pairwise-overflow distance with boost must not fail the query")
    .hits;
  let ids: Vec<_> = hits.iter().map(|h| h.doc_id.as_str().to_string()).collect();
  assert!(
    ids.contains(&"far".to_string()),
    "`far` doc must be returned even with boost > 1.0; got ids={ids:?}",
  );
  assert!(
    ids.contains(&"near".to_string()),
    "`near` doc must remain in results alongside `far`; got ids={ids:?}",
  );
  for hit in hits.iter() {
    assert!(
      hit.score.is_finite(),
      "every hit score must be finite; doc {} had score {}",
      hit.doc_id,
      hit.score,
    );
    let vs = hit
      .vector_score
      .expect("vector-only query hits must include vector_score");
    assert!(
      vs.is_finite(),
      "vector_score must be finite after saturating boost multiplication; doc {} had vector_score {}",
      hit.doc_id,
      vs,
    );
  }
}

// BUG-394: boundary test. The f64-then-clamp saturation must not silently
// perturb legitimate in-range products. A cosine query with `boost = 3.0`
// against a unit vector produces `vscore = 1.0 * 3.0 = 3.0`, which stays
// well inside the f32 range. The clamp must preserve the exact product.
#[test]
fn vector_boost_preserves_finite_vscore_product_within_f32_range() {
  let dir = tempdir().unwrap();
  let schema = schema();
  IndexBuilder::create(dir.path(), schema.clone(), opts(dir.path())).unwrap();
  let idx = Index::open(opts(dir.path())).unwrap();
  add_docs(
    &idx,
    &[Document {
      fields: [
        ("_id".into(), serde_json::json!("only-doc")),
        ("body".into(), serde_json::json!("rust search")),
        ("embedding".into(), serde_json::json!([1.0, 0.0])),
      ]
      .into_iter()
      .collect(),
    }],
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
      boost: Some(3.0),
    })),
    vector_query: None,
    vector_filter: None,
    ..base_request(Query::String("".into()), 3)
  };
  let hits = reader
    .search(&req)
    .expect("finite product must be preserved")
    .hits;
  assert_eq!(hits.len(), 1);
  let vs = hits[0]
    .vector_score
    .expect("cosine match must carry a vector_score");
  // Cosine similarity of identical unit vectors is 1.0; times boost=3.0 the
  // vscore is exactly 3.0. The f64 clamp must not perturb in-range products.
  assert!(
    (vs - 3.0).abs() < 1e-5,
    "expected vscore ≈ 3.0 (cosine 1.0 * boost 3.0), got {vs}"
  );
  assert!(hits[0].score.is_finite());
  // Vector-only hybrid query (`alpha = 0.0`) with a single clause must
  // collapse to `hit.score == hit.vector_score`. Asserting the relationship
  // catches any future divergence between the two fields end-to-end.
  assert!(
    (hits[0].score - vs).abs() < 1e-5,
    "expected score ≈ vector_score for a vector-only query, got score={} vector_score={vs}",
    hits[0].score
  );
}
