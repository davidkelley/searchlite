use std::collections::BTreeMap;

use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Aggregation, CardinalityAggregation, CompositeAggregation, CompositeSource,
  DateHistogramAggregation, DerivativeAggregation, Document, ExecutionStrategy, Filter, GapPolicy,
  HistogramAggregation, IndexOptions, MetricAggregation, MovingAvgAggregation, NumericField,
  PercentileRanksAggregation, PercentilesAggregation, RangeAggregation, RareTermsAggregation,
  Schema, SearchRequest, SignificantTermsAggregation, StorageType, TermsAggregation,
};
use searchlite_core::api::Index;
use serde_json::json;

fn doc(id: &str, fields: Vec<(&str, serde_json::Value)>) -> Document {
  let mut map = std::collections::BTreeMap::new();
  map.insert("_id".to_string(), json!(id));
  for (k, v) in fields {
    map.insert(k.to_string(), v);
  }
  Document { fields: map }
}

fn nested_keyword(name: &str) -> searchlite_core::api::types::KeywordField {
  searchlite_core::api::types::KeywordField {
    name: name.into(),
    stored: true,
    indexed: true,
    fast: true,
    nullable: false,
  }
}

#[test]
fn nested_terms_aggregation_counts_nested_objects() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "images".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
        nested_keyword("illustrator"),
      )],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "img-1",
        vec![
          ("body", json!("rust nested faceting")),
          (
            "images",
            json!([
              { "illustrator": "alice" },
              { "illustrator": "bob" }
            ]),
          ),
        ],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "img-2",
        vec![
          ("body", json!("rust nested faceting")),
          ("images", json!([{ "illustrator": "alice" }])),
        ],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "img-3",
        vec![
          ("body", json!("rust nested faceting")),
          ("images", json!([{ "illustrator": "carol" }])),
        ],
      ))
      .unwrap();
    writer.commit().unwrap();
  }

  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust nested",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "illustrators": {
        "type": "nested",
        "path": "images",
        "aggs": {
          "names": {
            "type": "terms",
            "field": "images.illustrator",
            "size": 10
          }
        }
      }
    }
  }))
  .unwrap();

  let resp = idx.reader().unwrap().search(&req).unwrap();
  let nested = resp
    .aggregations
    .get("illustrators")
    .expect("nested aggregation");
  if let searchlite_core::api::types::AggregationResponse::Nested {
    doc_count,
    aggregations,
    ..
  } = nested
  {
    assert_eq!(*doc_count, 4);
    if let Some(searchlite_core::api::types::AggregationResponse::Terms { buckets, .. }) =
      aggregations.get("names")
    {
      assert_eq!(buckets.len(), 3);
      assert_eq!(buckets[0].key, json!("alice"));
      assert_eq!(buckets[0].doc_count, 2);
    } else {
      panic!("expected names terms aggregation");
    }
  } else {
    panic!("expected nested aggregation response");
  }
}

#[test]
fn nested_terms_aggregation_respects_outer_filters() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
      name: "lang".into(),
      stored: true,
      indexed: true,
      fast: true,
      nullable: false,
    });
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "images".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
        nested_keyword("illustrator"),
      )],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "img-1",
        vec![
          ("body", json!("rust nested faceting")),
          ("lang", json!("en")),
          (
            "images",
            json!([
              { "illustrator": "alice" },
              { "illustrator": "bob" }
            ]),
          ),
        ],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "img-2",
        vec![
          ("body", json!("rust nested faceting")),
          ("lang", json!("en")),
          ("images", json!([{ "illustrator": "alice" }])),
        ],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "img-3",
        vec![
          ("body", json!("rust nested faceting")),
          ("lang", json!("fr")),
          ("images", json!([{ "illustrator": "carol" }])),
        ],
      ))
      .unwrap();
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "illustrators".into(),
    serde_json::from_value(json!({
      "type": "nested",
      "path": "images",
      "aggs": {
        "names": {
          "type": "terms",
          "field": "images.illustrator",
          "size": 10
        }
      }
    }))
    .unwrap(),
  );

  let req = SearchRequest {
    query: "rust".into(),
    fields: None,
    filter: Some(Filter::KeywordEq {
      field: "lang".into(),
      value: "en".into(),
    }),
    limit: 0,
    from: 0,
    return_hits: false,
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
    aggs,
    suggest: BTreeMap::new(),
    rescore: None,
    explain: false,
    profile: false,
  };
  let resp = idx.reader().unwrap().search(&req).unwrap();
  let nested = resp
    .aggregations
    .get("illustrators")
    .expect("nested aggregation");
  if let searchlite_core::api::types::AggregationResponse::Nested {
    doc_count,
    aggregations,
    ..
  } = nested
  {
    assert_eq!(*doc_count, 3);
    if let Some(searchlite_core::api::types::AggregationResponse::Terms { buckets, .. }) =
      aggregations.get("names")
    {
      assert_eq!(buckets.len(), 2);
      assert_eq!(buckets[0].key, json!("alice"));
      assert_eq!(buckets[0].doc_count, 2);
      assert_eq!(buckets[1].key, json!("bob"));
      assert_eq!(buckets[1].doc_count, 1);
    } else {
      panic!("expected names terms aggregation");
    }
  } else {
    panic!("expected nested aggregation response");
  }
}

#[test]
fn nested_terms_aggregation_skips_null_entries_in_nullable_arrays() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "images".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
        nested_keyword("illustrator"),
      )],
      nullable: true,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "img-1",
        vec![
          ("body", json!("rust nested faceting")),
          ("images", json!([null, { "illustrator": "alice" }])),
        ],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "img-2",
        vec![
          ("body", json!("rust nested faceting")),
          ("images", json!([null, null])),
        ],
      ))
      .unwrap();
    writer.commit().unwrap();
  }

  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust nested",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "illustrators": {
        "type": "nested",
        "path": "images",
        "aggs": {
          "names": {
            "type": "terms",
            "field": "images.illustrator",
            "size": 10
          }
        }
      }
    }
  }))
  .unwrap();

  let resp = idx.reader().unwrap().search(&req).unwrap();
  let nested = resp
    .aggregations
    .get("illustrators")
    .expect("nested aggregation");
  if let searchlite_core::api::types::AggregationResponse::Nested {
    doc_count,
    aggregations,
    ..
  } = nested
  {
    assert_eq!(*doc_count, 1);
    if let Some(searchlite_core::api::types::AggregationResponse::Terms { buckets, .. }) =
      aggregations.get("names")
    {
      assert_eq!(buckets.len(), 1);
      assert_eq!(buckets[0].key, json!("alice"));
      assert_eq!(buckets[0].doc_count, 1);
    } else {
      panic!("expected names terms aggregation");
    }
  } else {
    panic!("expected nested aggregation response");
  }
}

#[test]
fn nested_aggregation_rejects_unknown_path() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let idx = IndexBuilder::create(
    &path,
    Schema::default_text_body(),
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "bad": {
        "type": "nested",
        "path": "does_not_exist",
        "aggs": {
          "names": { "type": "terms", "field": "name", "size": 5 }
        }
      }
    }
  }))
  .unwrap();
  let err = idx.reader().unwrap().search(&req).unwrap_err();
  assert!(
    err.to_string().contains("nested path"),
    "unexpected error: {err}"
  );
}

#[test]
fn nested_aggregation_rejects_dotted_non_nested_prefix_path() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
      name: "metadata.key".into(),
      stored: true,
      indexed: true,
      fast: true,
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "bad": {
        "type": "nested",
        "path": "metadata",
        "aggs": {
          "names": { "type": "terms", "field": "metadata.key", "size": 5 }
        }
      }
    }
  }))
  .unwrap();
  let err = idx.reader().unwrap().search(&req).unwrap_err();
  assert!(
    err.to_string().contains("nested path"),
    "unexpected error: {err}"
  );
}

#[test]
fn root_terms_aggregation_rejects_nested_keyword_field_without_nested_scope() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "images".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
        nested_keyword("illustrator"),
      )],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "bad": {
        "type": "terms",
        "field": "images.illustrator",
        "size": 5
      }
    }
  }))
  .unwrap();
  let err = idx.reader().unwrap().search(&req).unwrap_err();
  assert!(
    err.to_string().contains("fast keyword field"),
    "unexpected error: {err}"
  );
}

#[test]
fn root_terms_aggregation_rejects_dotted_nested_keyword_field_without_nested_scope() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "images".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
        nested_keyword("source.name"),
      )],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "bad": {
        "type": "terms",
        "field": "images.source.name",
        "size": 5
      }
    }
  }))
  .unwrap();
  let err = idx.reader().unwrap().search(&req).unwrap_err();
  assert!(
    err.to_string().contains("fast keyword field"),
    "unexpected error: {err}"
  );
}

#[test]
fn root_terms_aggregation_allows_literal_dotted_top_level_keyword_field() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
      name: "images.source.name".into(),
      stored: true,
      indexed: true,
      fast: true,
      nullable: false,
    });
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "images".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
        nested_keyword("source.name"),
      )],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "img-1",
        vec![
          ("body", json!("rust")),
          ("images.source.name", json!("top-level")),
          ("images", json!([{ "source.name": "nested-only" }])),
        ],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "top": {
        "type": "terms",
        "field": "images.source.name",
        "size": 10
      }
    }
  }))
  .unwrap();
  let res = idx.reader().unwrap().search(&req);
  assert!(res.is_ok(), "unexpected error: {}", res.unwrap_err());
}

#[test]
fn root_numeric_aggregation_rejects_nested_numeric_field_without_nested_scope() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "images".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Numeric(
        NumericField {
          name: "score".into(),
          i64: false,
          fast: true,
          stored: true,
          nullable: false,
        },
      )],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "img-1",
        vec![
          ("body", json!("rust")),
          ("images", json!([{ "score": 99.0 }])),
        ],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "bad": {
        "type": "stats",
        "field": "images.score"
      }
    }
  }))
  .unwrap();
  let err = idx.reader().unwrap().search(&req).unwrap_err();
  assert!(
    err.to_string().contains("fast numeric field"),
    "unexpected error: {err}"
  );
}

#[test]
fn root_numeric_aggregation_allows_literal_dotted_top_level_numeric_field() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "images.score".into(),
    i64: false,
    fast: true,
    stored: true,
    nullable: false,
  });
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "images".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Numeric(
        NumericField {
          name: "score".into(),
          i64: false,
          fast: true,
          stored: true,
          nullable: false,
        },
      )],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "score_stats": {
        "type": "stats",
        "field": "images.score"
      }
    }
  }))
  .unwrap();
  let res = idx.reader().unwrap().search(&req);
  assert!(res.is_ok(), "unexpected error: {}", res.unwrap_err());
}

#[test]
fn collapse_rejects_nested_keyword_field_without_nested_scope() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "images".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
        nested_keyword("illustrator"),
      )],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 1,
    "return_hits": true,
    "collapse": {
      "field": "images.illustrator"
    },
    "aggs": {}
  }))
  .unwrap();
  let err = idx.reader().unwrap().search(&req).unwrap_err();
  assert!(
    err.to_string().contains("fast keyword field"),
    "unexpected error: {err}"
  );
}

#[test]
fn nested_terms_accept_direct_child_dotted_keyword_name() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "images".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
        nested_keyword("source.name"),
      )],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "img-1",
        vec![
          ("body", json!("rust nested faceting")),
          (
            "images",
            json!([
              { "source.name": "alice" },
              { "source.name": "bob" }
            ]),
          ),
        ],
      ))
      .unwrap();
    writer.commit().unwrap();
  }

  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "images_nested": {
        "type": "nested",
        "path": "images",
        "aggs": {
          "names": {
            "type": "terms",
            "field": "source.name",
            "size": 10
          }
        }
      }
    }
  }))
  .unwrap();
  let resp = idx.reader().unwrap().search(&req).unwrap();
  let nested = resp
    .aggregations
    .get("images_nested")
    .expect("images nested aggregation");
  if let searchlite_core::api::types::AggregationResponse::Nested {
    doc_count,
    aggregations,
    ..
  } = nested
  {
    assert_eq!(*doc_count, 2);
    if let Some(searchlite_core::api::types::AggregationResponse::Terms { buckets, .. }) =
      aggregations.get("names")
    {
      assert_eq!(buckets.len(), 2);
      assert_eq!(buckets[0].key, json!("alice"));
      assert_eq!(buckets[0].doc_count, 1);
      assert_eq!(buckets[1].key, json!("bob"));
      assert_eq!(buckets[1].doc_count, 1);
    } else {
      panic!("expected terms aggregation for dotted child name");
    }
  } else {
    panic!("expected nested aggregation response");
  }
}

#[test]
fn nested_aggregation_accepts_direct_child_dotted_object_name() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "comment".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Object(
        searchlite_core::api::types::NestedField {
          name: "reply.vote".into(),
          fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
            nested_keyword("tag"),
          )],
          nullable: false,
        },
      )],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "nested-1",
        vec![
          ("body", json!("rust nested reply votes")),
          (
            "comment",
            json!([
              {
                "reply.vote": [{ "tag": "up" }]
              }
            ]),
          ),
        ],
      ))
      .unwrap();
    writer.commit().unwrap();
  }

  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "comments": {
        "type": "nested",
        "path": "comment",
        "aggs": {
          "votes": {
            "type": "nested",
            "path": "reply.vote",
            "aggs": {
              "tags": { "type": "terms", "field": "tag", "size": 10 }
            }
          }
        }
      }
    }
  }))
  .unwrap();
  let resp = idx.reader().unwrap().search(&req).unwrap();
  let comments = resp
    .aggregations
    .get("comments")
    .expect("comments nested aggregation");
  let votes =
    if let searchlite_core::api::types::AggregationResponse::Nested { aggregations, .. } = comments
    {
      aggregations.get("votes").expect("votes nested agg")
    } else {
      panic!("expected comments nested aggregation");
    };
  if let searchlite_core::api::types::AggregationResponse::Nested {
    doc_count,
    aggregations,
    ..
  } = votes
  {
    assert_eq!(*doc_count, 1);
    if let Some(searchlite_core::api::types::AggregationResponse::Terms { buckets, .. }) =
      aggregations.get("tags")
    {
      assert_eq!(buckets.len(), 1);
      assert_eq!(buckets[0].key, json!("up"));
      assert_eq!(buckets[0].doc_count, 1);
    } else {
      panic!("expected tags terms aggregation");
    }
  } else {
    panic!("expected votes nested aggregation");
  }
}

#[test]
fn nested_aggregation_rejects_unsupported_child_aggregation() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "images".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
        nested_keyword("illustrator"),
      )],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "images_nested": {
        "type": "nested",
        "path": "images",
        "aggs": {
          "bad_top_hits": {
            "type": "top_hits",
            "size": 1
          }
        }
      }
    }
  }))
  .unwrap();
  let err = idx.reader().unwrap().search(&req).unwrap_err();
  assert!(
    err
      .to_string()
      .contains("not supported inside nested aggregations"),
    "unexpected error: {err}"
  );
}

#[test]
fn nested_terms_reject_descendant_fields_in_nested_scope() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "comment".into(),
      fields: vec![
        searchlite_core::api::types::NestedProperty::Keyword(nested_keyword("author")),
        searchlite_core::api::types::NestedProperty::Object(
          searchlite_core::api::types::NestedField {
            name: "reply".into(),
            fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
              nested_keyword("tag"),
            )],
            nullable: false,
          },
        ),
      ],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "comments": {
        "type": "nested",
        "path": "comment",
        "aggs": {
          "bad_terms": {
            "type": "terms",
            "field": "reply.tag",
            "size": 10
          }
        }
      }
    }
  }))
  .unwrap();
  let err = idx.reader().unwrap().search(&req).unwrap_err();
  assert!(
    err
      .to_string()
      .contains("direct child of nested scope `comment`"),
    "unexpected error: {err}"
  );
}

#[test]
fn nested_aggregation_rejects_non_direct_child_nested_path_in_scope() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "comment".into(),
      fields: vec![searchlite_core::api::types::NestedProperty::Object(
        searchlite_core::api::types::NestedField {
          name: "reply".into(),
          fields: vec![searchlite_core::api::types::NestedProperty::Object(
            searchlite_core::api::types::NestedField {
              name: "vote".into(),
              fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
                nested_keyword("tag"),
              )],
              nullable: false,
            },
          )],
          nullable: false,
        },
      )],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "comments": {
        "type": "nested",
        "path": "comment",
        "aggs": {
          "bad_nested": {
            "type": "nested",
            "path": "reply.vote",
            "aggs": {
              "tags": { "type": "terms", "field": "tag", "size": 10 }
            }
          }
        }
      }
    }
  }))
  .unwrap();
  let err = idx.reader().unwrap().search(&req).unwrap_err();
  assert!(
    err
      .to_string()
      .contains("direct child of nested scope `comment`"),
    "unexpected error: {err}"
  );
}

#[test]
fn nested_sub_aggregation_binds_to_parent_object() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "comment".into(),
      fields: vec![
        searchlite_core::api::types::NestedProperty::Keyword(nested_keyword("author")),
        searchlite_core::api::types::NestedProperty::Object(
          searchlite_core::api::types::NestedField {
            name: "reply".into(),
            fields: vec![searchlite_core::api::types::NestedProperty::Keyword(
              nested_keyword("tag"),
            )],
            nullable: false,
          },
        ),
      ],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "nested-bind-1",
        vec![
          ("body", json!("rust nested binding")),
          (
            "comment",
            json!([
              {
                "author": "alice",
                "reply": [{ "tag": "x" }]
              },
              {
                "author": "bob",
                "reply": [{ "tag": "y" }]
              }
            ]),
          ),
        ],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  let req: SearchRequest = serde_json::from_value(json!({
    "query": "rust",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "comments": {
        "type": "nested",
        "path": "comment",
        "aggs": {
          "authors": {
            "type": "terms",
            "field": "author",
            "size": 10,
            "aggs": {
              "replies": {
                "type": "nested",
                "path": "reply",
                "aggs": {
                  "tags": { "type": "terms", "field": "tag", "size": 10 }
                }
              }
            }
          }
        }
      }
    }
  }))
  .unwrap();
  let resp = idx.reader().unwrap().search(&req).unwrap();
  let comments = resp
    .aggregations
    .get("comments")
    .expect("comments nested agg");
  let (comments_doc_count, comments_aggs) =
    if let searchlite_core::api::types::AggregationResponse::Nested {
      doc_count,
      aggregations,
      ..
    } = comments
    {
      (*doc_count, aggregations)
    } else {
      panic!("expected nested response for comments");
    };
  assert_eq!(comments_doc_count, 2);
  let authors = comments_aggs.get("authors").expect("authors terms");
  if let searchlite_core::api::types::AggregationResponse::Terms { buckets, .. } = authors {
    assert_eq!(buckets.len(), 2);
    let alice = buckets.iter().find(|b| b.key == json!("alice")).unwrap();
    let bob = buckets.iter().find(|b| b.key == json!("bob")).unwrap();
    let alice_replies = alice
      .aggregations
      .get("replies")
      .expect("alice replies nested");
    let bob_replies = bob.aggregations.get("replies").expect("bob replies nested");
    let alice_tags =
      if let searchlite_core::api::types::AggregationResponse::Nested { aggregations, .. } =
        alice_replies
      {
        aggregations
          .get("tags")
          .and_then(|agg| match agg {
            searchlite_core::api::types::AggregationResponse::Terms { buckets, .. } => {
              buckets.first()
            }
            _ => None,
          })
          .map(|bucket| bucket.key.clone())
          .expect("alice tags")
      } else {
        panic!("expected nested replies response");
      };
    let bob_tags =
      if let searchlite_core::api::types::AggregationResponse::Nested { aggregations, .. } =
        bob_replies
      {
        aggregations
          .get("tags")
          .and_then(|agg| match agg {
            searchlite_core::api::types::AggregationResponse::Terms { buckets, .. } => {
              buckets.first()
            }
            _ => None,
          })
          .map(|bucket| bucket.key.clone())
          .expect("bob tags")
      } else {
        panic!("expected nested replies response");
      };
    assert_eq!(alice_tags, json!("x"));
    assert_eq!(bob_tags, json!("y"));
  } else {
    panic!("expected authors terms response");
  }
}

#[test]
fn nested_metadata_facets_bind_key_value_pairs() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .nested_fields
    .push(searchlite_core::api::types::NestedField {
      name: "metadata".into(),
      fields: vec![
        searchlite_core::api::types::NestedProperty::Keyword(nested_keyword("key")),
        searchlite_core::api::types::NestedProperty::Keyword(nested_keyword("value")),
      ],
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "meta-1",
        vec![
          ("body", json!("image metadata")),
          (
            "metadata",
            json!([
              { "key": "Category", "value": "Nature" },
              { "key": "Color", "value": "Blue" }
            ]),
          ),
        ],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "meta-2",
        vec![
          ("body", json!("image metadata")),
          (
            "metadata",
            json!([
              { "key": "Category", "value": "Portrait" },
              { "key": "Color", "value": "Red" }
            ]),
          ),
        ],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "meta-3",
        vec![
          ("body", json!("image metadata")),
          (
            "metadata",
            json!([{ "key": "Category", "value": "Nature" }]),
          ),
        ],
      ))
      .unwrap();
    writer.commit().unwrap();
  }

  let req: SearchRequest = serde_json::from_value(json!({
    "query": "image",
    "limit": 0,
    "return_hits": false,
    "aggs": {
      "metadata_nested": {
        "type": "nested",
        "path": "metadata",
        "aggs": {
          "by_key": {
            "type": "terms",
            "field": "key",
            "size": 10,
            "aggs": {
              "by_value": {
                "type": "terms",
                "field": "value",
                "size": 10
              }
            }
          }
        }
      }
    }
  }))
  .unwrap();

  let resp = idx.reader().unwrap().search(&req).unwrap();
  let nested = resp
    .aggregations
    .get("metadata_nested")
    .expect("metadata nested aggregation");
  let key_buckets = if let searchlite_core::api::types::AggregationResponse::Nested {
    doc_count,
    aggregations,
    ..
  } = nested
  {
    assert_eq!(*doc_count, 5);
    if let Some(searchlite_core::api::types::AggregationResponse::Terms { buckets, .. }) =
      aggregations.get("by_key")
    {
      buckets
    } else {
      panic!("expected by_key terms aggregation");
    }
  } else {
    panic!("expected nested aggregation response");
  };

  let category = key_buckets
    .iter()
    .find(|bucket| bucket.key == json!("Category"))
    .expect("Category bucket");
  assert_eq!(category.doc_count, 3);
  let category_values = match category.aggregations.get("by_value") {
    Some(searchlite_core::api::types::AggregationResponse::Terms { buckets, .. }) => buckets,
    _ => panic!("expected by_value terms under Category"),
  };
  assert_eq!(category_values.len(), 2);
  assert_eq!(category_values[0].key, json!("Nature"));
  assert_eq!(category_values[0].doc_count, 2);
  assert_eq!(category_values[1].key, json!("Portrait"));
  assert_eq!(category_values[1].doc_count, 1);

  let color = key_buckets
    .iter()
    .find(|bucket| bucket.key == json!("Color"))
    .expect("Color bucket");
  assert_eq!(color.doc_count, 2);
  let color_values = match color.aggregations.get("by_value") {
    Some(searchlite_core::api::types::AggregationResponse::Terms { buckets, .. }) => buckets,
    _ => panic!("expected by_value terms under Color"),
  };
  assert_eq!(color_values.len(), 2);
  assert_eq!(color_values[0].key, json!("Blue"));
  assert_eq!(color_values[0].doc_count, 1);
  assert_eq!(color_values[1].key, json!("Red"));
  assert_eq!(color_values[1].doc_count, 1);
}

#[test]
fn terms_and_stats_aggregations() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
      name: "tag".into(),
      stored: true,
      indexed: true,
      fast: true,
      nullable: false,
    });
  schema.numeric_fields.push(NumericField {
    name: "views".into(),
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
  let idx = IndexBuilder::create(&path, schema, opts).expect("create index");
  let mut writer = idx.writer().expect("writer");
  let docs = [
    doc(
      "agg-1",
      vec![
        ("body", json!("rust systems")),
        ("tag", json!("tech")),
        ("views", json!(10)),
      ],
    ),
    doc(
      "agg-2",
      vec![
        ("body", json!("rust programming")),
        ("tag", json!("tech")),
        ("views", json!(15)),
      ],
    ),
    doc(
      "agg-3",
      vec![
        ("body", json!("gardening")),
        ("tag", json!("hobby")),
        ("views", json!(2)),
      ],
    ),
  ];
  for doc in docs.iter() {
    writer.add_document(doc).unwrap();
  }
  writer.commit().unwrap();

  let reader = idx.reader().unwrap();
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "tags".to_string(),
    Aggregation::Terms(Box::new(TermsAggregation {
      field: "tag".into(),
      size: Some(5),
      shard_size: None,
      min_doc_count: None,
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );
  aggs.insert(
    "view_stats".to_string(),
    Aggregation::Stats(MetricAggregation {
      field: "views".into(),
      missing: None,
    }),
  );

  let resp = reader
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();

  let tags = resp.aggregations.get("tags").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Terms { buckets, .. } = tags {
    assert_eq!(buckets[0].key, json!("tech"));
    assert_eq!(buckets[0].doc_count, 2);
  }

  let stats = resp.aggregations.get("view_stats").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Stats(stats) = stats {
    assert_eq!(stats.count, 2);
    assert_eq!(stats.min, 10.0);
    assert_eq!(stats.max, 15.0);
    assert_eq!(stats.sum, 25.0);
  }
}

#[test]
fn significant_terms_respects_deletions() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
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
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let idx = IndexBuilder::create(&path, schema, opts).expect("create index");
  {
    let mut writer = idx.writer().expect("writer");
    writer
      .add_document(&doc(
        "1",
        vec![("body", json!("keep me")), ("tag", json!("foo"))],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "2",
        vec![("body", json!("delete me")), ("tag", json!("foo"))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  {
    let mut writer = idx.writer().expect("writer");
    writer.delete_document("2").unwrap();
    writer.commit().unwrap();
  }

  let reader = idx.reader().unwrap();
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "sig".to_string(),
    Aggregation::SignificantTerms(Box::new(SignificantTermsAggregation {
      field: "tag".into(),
      size: Some(5),
      min_doc_count: None,
      background_filter: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );

  let resp = reader
    .search(&SearchRequest {
      query: "keep".into(),
      fields: None,
      filter: None,
      limit: 5,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();

  let sig = resp.aggregations.get("sig").unwrap();
  if let searchlite_core::api::types::AggregationResponse::SignificantTerms { buckets, .. } = sig {
    assert_eq!(buckets.len(), 1);
    assert_eq!(buckets[0].key, json!("foo"));
    assert_eq!(
      buckets[0].doc_count, 1,
      "deleted docs must not contribute to significant_terms doc counts"
    );
  } else {
    panic!("expected significant_terms response");
  }
}

#[test]
fn aggregation_requires_fast_field() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
      name: "tag".into(),
      stored: true,
      indexed: true,
      fast: false,
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  let reader = idx.reader().unwrap();
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "tags".into(),
    Aggregation::Terms(Box::new(TermsAggregation {
      field: "tag".into(),
      size: Some(5),
      shard_size: None,
      min_doc_count: None,
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );
  let resp = reader.search(&SearchRequest {
    query: "rust".into(),
    fields: None,
    filter: None,
    limit: 1,
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
    aggs,
    suggest: BTreeMap::new(),
    rescore: None,
    explain: false,
    profile: false,
  });
  assert!(resp.is_err());
  let msg = resp.err().unwrap().to_string();
  assert!(msg.contains("fast field `tag`"));
}

#[test]
fn histogram_bucket_generation() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "views".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
  });
  let idx = Index::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for val in [1, 2, 7, 11] {
      writer
        .add_document(&doc(
          &format!("hist-{val}"),
          vec![("body", json!("rust")), ("views", json!(val))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "views_hist".into(),
    Aggregation::Histogram(Box::new(HistogramAggregation {
      field: "views".into(),
      interval: 5.0,
      offset: None,
      min_doc_count: Some(1),
      extended_bounds: None,
      hard_bounds: None,
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  let hist = resp.aggregations.get("views_hist").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Histogram { buckets, .. } = hist {
    assert_eq!(buckets.len(), 3);
    assert_eq!(buckets[0].doc_count, 2);
  }
}

#[test]
fn histogram_uses_floor_for_bucket_boundaries() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "views".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
  });
  let idx = Index::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for val in [0, 4, 5] {
      writer
        .add_document(&doc(
          &format!("hist2-{val}"),
          vec![("body", json!("rust")), ("views", json!(val))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "views_hist".into(),
    Aggregation::Histogram(Box::new(HistogramAggregation {
      field: "views".into(),
      interval: 5.0,
      offset: None,
      min_doc_count: Some(0),
      extended_bounds: None,
      hard_bounds: None,
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  let hist = resp.aggregations.get("views_hist").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Histogram { buckets, .. } = hist {
    assert_eq!(buckets.len(), 2);
    assert_eq!(buckets[0].key, json!(0.0));
    assert_eq!(buckets[0].doc_count, 2); // both 0 and 4 land in the first bucket
    assert_eq!(buckets[1].key, json!(5.0));
    assert_eq!(buckets[1].doc_count, 1);
  } else {
    panic!("expected histogram response");
  }
}

#[test]
fn range_aggregation_counts() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "score".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
  });
  let idx = Index::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for val in [1, 5, 10, 20] {
      writer
        .add_document(&doc(
          &format!("score-{val}"),
          vec![("body", json!("rust")), ("score", json!(val))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "score_ranges".into(),
    Aggregation::Range(Box::new(RangeAggregation {
      field: "score".into(),
      keyed: true,
      ranges: vec![
        searchlite_core::api::types::RangeBound {
          key: Some("low".into()),
          from: None,
          to: Some(5.0),
        },
        searchlite_core::api::types::RangeBound {
          key: Some("mid".into()),
          from: Some(5.0),
          to: Some(15.0),
        },
      ],
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );

  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  let range = resp.aggregations.get("score_ranges").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Range { buckets, .. } = range {
    // With `to` exclusive: score=1 is in low (1 < 5), score=5 is NOT in low (5 < 5 is false).
    assert_eq!(buckets[0].doc_count, 1);
    // score=5 and score=10 are in mid (5 >= 5 && 5 < 15, 10 >= 5 && 10 < 15).
    assert_eq!(buckets[1].doc_count, 2);
  }
}

#[test]
fn date_range_missing_and_keyed() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  // `ts` is intentionally nullable here so that one of the documents below can
  // omit it to exercise the aggregation-side `missing` default. See BUG-224:
  // omitting a non-nullable field is now rejected at validation time.
  schema.numeric_fields.push(NumericField {
    name: "ts".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: true,
  });
  let idx = Index::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc(
        "date-1",
        vec![("body", json!("rust")), ("ts", json!(1_000))],
      ))
      .unwrap();
    // missing ts should be counted in missing bucket
    writer
      .add_document(&doc("date-missing", vec![("body", json!("rust missing"))]))
      .unwrap();
    writer.commit().unwrap();
  }
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "ranges".into(),
    Aggregation::DateRange(Box::new(
      searchlite_core::api::types::DateRangeAggregation {
        field: "ts".into(),
        keyed: true,
        format: None,
        ranges: vec![
          searchlite_core::api::types::DateRangeBound {
            key: Some("early".into()),
            from: Some("1970-01-01T00:00:00Z".into()),
            to: Some("1970-01-01T00:00:02Z".into()),
          },
          searchlite_core::api::types::DateRangeBound {
            key: Some("late".into()),
            from: Some("1970-01-01T00:00:02Z".into()),
            to: Some("1970-01-01T00:00:03Z".into()),
          },
        ],
        missing: Some(json!("1970-01-01T00:00:01Z")),
        sampling: None,
        aggs: BTreeMap::new(),
      },
    )),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  let range = resp.aggregations.get("ranges").unwrap();
  if let searchlite_core::api::types::AggregationResponse::DateRange { buckets, keyed, .. } = range
  {
    assert!(keyed);
    assert_eq!(buckets.len(), 2);
    assert_eq!(buckets[0].doc_count, 2); // early bucket includes missing
    assert_eq!(buckets[1].doc_count, 0);
  } else {
    panic!("expected date range response");
  }
}

#[test]
fn extended_stats_and_value_count_include_missing() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  // `score` is intentionally nullable so one of the documents below can omit
  // it to exercise the metric-aggregation `missing` default. See BUG-224:
  // omitting a non-nullable field is now rejected at validation time.
  schema.numeric_fields.push(NumericField {
    name: "score".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: true,
  });
  let idx = Index::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for (idx, val) in [Some(1), Some(2), None].into_iter().enumerate() {
      let mut fields = [("body".into(), json!("rust"))]
        .into_iter()
        .collect::<BTreeMap<_, _>>();
      fields.insert("_id".into(), json!(format!("stats-{idx}")));
      if let Some(v) = val {
        fields.insert("score".into(), json!(v));
      }
      writer.add_document(&Document { fields }).unwrap();
    }
    writer.commit().unwrap();
  }
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "stats".into(),
    Aggregation::ExtendedStats(MetricAggregation {
      field: "score".into(),
      missing: Some(json!(5)),
    }),
  );
  aggs.insert(
    "count".into(),
    Aggregation::ValueCount(MetricAggregation {
      field: "score".into(),
      missing: Some(json!(0)),
    }),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1, // ensure aggregations still see all docs
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  let stats = resp.aggregations.get("stats").unwrap();
  if let searchlite_core::api::types::AggregationResponse::ExtendedStats(es) = stats {
    assert_eq!(es.count, 3);
    assert_eq!(es.sum, 8.0);
    assert_eq!(es.max, 5.0);
    assert_eq!(es.min, 1.0);
  } else {
    panic!("expected extended stats");
  }
  let count = resp.aggregations.get("count").unwrap();
  if let searchlite_core::api::types::AggregationResponse::ValueCount(vc) = count {
    assert_eq!(vc.value, 3);
  } else {
    panic!("expected value count");
  }
}

#[test]
fn date_histogram_fixed_interval_respects_offset_and_missing() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  // `ts` is intentionally nullable so the "missing ts" document below can
  // exercise the date-histogram `missing` default. See BUG-224: omitting a
  // non-nullable field is now rejected at validation time.
  schema.numeric_fields.push(NumericField {
    name: "ts".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: true,
  });
  let idx = Index::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for ts in [0, 1_000, 1_600] {
      writer
        .add_document(&doc(
          &format!("hist-ts-{ts}"),
          vec![("body", json!("rust")), ("ts", json!(ts))],
        ))
        .unwrap();
    }
    // one doc missing ts to exercise "missing"
    writer
      .add_document(&doc(
        "hist-ts-missing",
        vec![("body", json!("rust missing ts"))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hist".into(),
    Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
      field: "ts".into(),
      calendar_interval: None,
      fixed_interval: Some("1s".into()),
      offset: Some("0.5s".into()),
      format: None,
      min_doc_count: Some(0),
      extended_bounds: Some(searchlite_core::api::types::DateHistogramBounds {
        min: "1970-01-01T00:00:00Z".into(),
        max: "1970-01-01T00:00:03Z".into(),
      }),
      hard_bounds: None,
      missing: Some("500".into()),
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );

  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();

  let hist = resp.aggregations.get("hist").unwrap();
  if let searchlite_core::api::types::AggregationResponse::DateHistogram { buckets, .. } = hist {
    let keys: Vec<_> = buckets.iter().map(|b| b.key.clone()).collect();
    // With offset=500ms and interval=1s, buckets are the half-open ranges
    // [-500, 500), [500, 1500), [1500, 2500), [2500, 3500). A timestamp
    // should land in the bucket whose `key` is
    // `floor((value - offset) / interval) * interval + offset`:
    //   ts=0       -> bucket -500 (in [-500, 500))
    //   ts=500     -> bucket  500 (from the `missing` substitute)
    //   ts=1000    -> bucket  500 (in [500, 1500))
    //   ts=1600    -> bucket 1500 (in [1500, 2500))
    // extended_bounds [0, 3000] then fills empty buckets between
    // bucket_start(0)=-500 and bucket_start(3000)=2500.
    assert_eq!(
      keys,
      vec![json!(-500), json!(500), json!(1500), json!(2500)]
    );
    assert_eq!(buckets[0].doc_count, 1); // ts=0
    assert_eq!(buckets[1].doc_count, 2); // missing->500 and ts=1000
    assert_eq!(buckets[2].doc_count, 1); // ts=1600
    assert_eq!(buckets[3].doc_count, 0); // empty fill from extended_bounds
  } else {
    panic!("expected date histogram response");
  }
}

#[test]
fn date_histogram_hard_bounds_filter_out_of_range() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "ts".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
  });
  let idx = Index::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    // within bounds
    writer
      .add_document(&doc(
        "hard-1",
        vec![("body", json!("rust")), ("ts", json!(1_000))],
      ))
      .unwrap();
    // below hard bounds
    writer
      .add_document(&doc(
        "hard-0",
        vec![("body", json!("rust")), ("ts", json!(0))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hist".into(),
    Aggregation::DateHistogram(Box::new(DateHistogramAggregation {
      field: "ts".into(),
      calendar_interval: None,
      fixed_interval: Some("1s".into()),
      offset: None,
      format: None,
      min_doc_count: Some(0),
      extended_bounds: None,
      hard_bounds: Some(searchlite_core::api::types::DateHistogramBounds {
        min: "1970-01-01T00:00:01Z".into(),
        max: "1970-01-01T00:00:02Z".into(),
      }),
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  let hist = resp.aggregations.get("hist").unwrap();
  if let searchlite_core::api::types::AggregationResponse::DateHistogram { buckets, .. } = hist {
    let keys: Vec<_> = buckets.iter().map(|b| b.key.clone()).collect();
    assert_eq!(keys, vec![json!(1_000), json!(2_000)]);
    assert_eq!(buckets[0].doc_count, 1);
    assert_eq!(buckets[1].doc_count, 0);
  } else {
    panic!("expected date histogram response");
  }
}

#[test]
fn terms_size_applied_after_merge() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
      name: "tag".into(),
      stored: true,
      indexed: true,
      fast: true,
      nullable: false,
    });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for i in 0..2 {
      writer
        .add_document(&doc(
          &format!("t-a-{i}"),
          vec![("body", json!("rust")), ("tag", json!("a"))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }
  {
    let mut writer = idx.writer().unwrap();
    for i in 0..4 {
      writer
        .add_document(&doc(
          &format!("t-b-{i}"),
          vec![("body", json!("rust")), ("tag", json!("b"))],
        ))
        .unwrap();
    }
    writer
      .add_document(&doc(
        "t-a-last",
        vec![("body", json!("rust")), ("tag", json!("a"))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "tags".into(),
    Aggregation::Terms(Box::new(TermsAggregation {
      field: "tag".into(),
      size: Some(1),
      shard_size: None,
      min_doc_count: None,
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );

  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();

  let agg = resp.aggregations.get("tags").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Terms { buckets, .. } = agg {
    assert_eq!(buckets.len(), 1);
    assert_eq!(buckets[0].key, json!("b"));
    assert_eq!(buckets[0].doc_count, 4);
  } else {
    panic!("expected terms response");
  }
}

#[test]
fn filter_aggregation_counts_and_sub_aggs() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
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
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let idx = IndexBuilder::create(&path, schema, opts).expect("create index");
  let mut writer = idx.writer().expect("writer");
  let docs = [
    doc("f-1", vec![("body", json!("rust")), ("tag", json!("tech"))]),
    doc("f-2", vec![("body", json!("rust")), ("tag", json!("tech"))]),
    doc(
      "f-3",
      vec![("body", json!("rust")), ("tag", json!("hobby"))],
    ),
  ];
  for doc in docs.iter() {
    writer.add_document(doc).unwrap();
  }
  writer.commit().unwrap();
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "only_tech".into(),
    Aggregation::Filter(Box::new(searchlite_core::api::types::FilterAggregation {
      filter: searchlite_core::api::types::Filter::KeywordEq {
        field: "tag".into(),
        value: "tech".into(),
      },
      sampling: None,
      aggs: BTreeMap::from([(
        "tags".into(),
        Aggregation::Terms(Box::new(TermsAggregation {
          field: "tag".into(),
          size: Some(10),
          shard_size: None,
          min_doc_count: None,
          missing: None,
          sampling: None,
          aggs: BTreeMap::new(),
        })),
      )]),
    })),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  let filter = resp.aggregations.get("only_tech").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Filter {
    doc_count,
    aggregations,
    ..
  } = filter
  {
    assert_eq!(*doc_count, 2);
    if let Some(searchlite_core::api::types::AggregationResponse::Terms { buckets, .. }) =
      aggregations.get("tags")
    {
      assert_eq!(buckets.len(), 1);
      assert_eq!(buckets[0].key, json!("tech"));
    } else {
      panic!("expected nested terms agg");
    }
  } else {
    panic!("expected filter agg");
  }
}

#[test]
fn composite_aggregation_paginates() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
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
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let idx = IndexBuilder::create(&path, schema, opts).expect("create index");
  let mut writer = idx.writer().expect("writer");
  for tag in ["a", "b", "c"] {
    writer
      .add_document(&doc(
        &format!("c-{tag}"),
        vec![("body", json!("rust")), ("tag", json!(tag))],
      ))
      .unwrap();
  }
  writer.commit().unwrap();
  let make_req = |after: Option<serde_json::Value>| -> SearchRequest {
    let mut aggs = BTreeMap::new();
    aggs.insert(
      "cmp".into(),
      Aggregation::Composite(Box::new(CompositeAggregation {
        sources: vec![CompositeSource::Terms {
          name: "tag".into(),
          field: "tag".into(),
        }],
        size: 2,
        after,
        sampling: None,
        aggs: BTreeMap::new(),
      })),
    );
    SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    }
  };
  let first = idx.reader().unwrap().search(&make_req(None)).unwrap();
  let cmp = first.aggregations.get("cmp").unwrap();
  let after_key = match cmp {
    searchlite_core::api::types::AggregationResponse::Composite {
      buckets, after_key, ..
    } => {
      assert_eq!(buckets.len(), 2);
      after_key.clone()
    }
    _ => panic!("expected composite agg"),
  };
  let second = idx.reader().unwrap().search(&make_req(after_key)).unwrap();
  let cmp2 = second.aggregations.get("cmp").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Composite {
    buckets, after_key, ..
  } = cmp2
  {
    assert_eq!(buckets.len(), 1);
    assert!(after_key.is_none());
  } else {
    panic!("expected composite agg");
  }
}

#[test]
fn cardinality_and_percentiles_metrics() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "latency".into(),
    i64: false,
    fast: true,
    stored: false,
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
  {
    let mut writer = idx.writer().unwrap();
    for (i, val) in [10.0, 20.0, 30.0, 40.0].iter().enumerate() {
      writer
        .add_document(&doc(
          &format!("p-{i}"),
          vec![("body", json!("rust")), ("latency", json!(val))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "card".into(),
    Aggregation::Cardinality(CardinalityAggregation {
      field: "latency".into(),
      precision_threshold: None,
      missing: None,
    }),
  );
  aggs.insert(
    "pct".into(),
    Aggregation::Percentiles(PercentilesAggregation {
      field: "latency".into(),
      percents: Some(vec![50.0]),
      missing: None,
    }),
  );
  aggs.insert(
    "pct_ranks".into(),
    Aggregation::PercentileRanks(PercentileRanksAggregation {
      field: "latency".into(),
      values: vec![20.0, 35.0],
      missing: None,
    }),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  if let searchlite_core::api::types::AggregationResponse::Cardinality(val) =
    resp.aggregations.get("card").unwrap()
  {
    assert_eq!(val.value, 4);
  } else {
    panic!("expected cardinality agg");
  }
  if let searchlite_core::api::types::AggregationResponse::Percentiles(p) =
    resp.aggregations.get("pct").unwrap()
  {
    assert_eq!(p.values.get("50").copied().unwrap() as i64, 25);
  } else {
    panic!("expected percentiles agg");
  }
  if let searchlite_core::api::types::AggregationResponse::PercentileRanks(p) =
    resp.aggregations.get("pct_ranks").unwrap()
  {
    let v20 = p.values.get("20").unwrap();
    let v35 = p.values.get("35").unwrap();
    assert!(*v20 > 0.0);
    assert!(*v35 > *v20);
  } else {
    panic!("expected percentile ranks agg");
  }
}

/// Regression test for BUG-209: on the TDigest (approximate) path, a
/// `percentile_rank(target)` call where `target == min_val` used to short-circuit
/// to `0.0`, disagreeing with the exact path's inclusive `count of v <= target`
/// semantics. This fixture populates > `PERCENTILE_EXACT_LIMIT` (256) documents
/// with repeats at the observed minimum so the approximate path is selected,
/// and verifies the reported rank reflects the share of values equal to the
/// minimum instead of collapsing to zero.
#[test]
fn percentile_ranks_tdigest_path_includes_observed_minimum() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "latency".into(),
    i64: false,
    fast: true,
    stored: false,
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
  {
    let mut writer = idx.writer().unwrap();
    // 100 values at the observed minimum (0.0) plus 400 values in 1.0..=400.0.
    // Total 500 > PERCENTILE_EXACT_LIMIT (256), so `percentile_rank` takes the
    // approximate (TDigest) path.
    for i in 0..100u32 {
      writer
        .add_document(&doc(
          &format!("min-{i}"),
          vec![("body", json!("rust")), ("latency", json!(0.0_f64))],
        ))
        .unwrap();
    }
    for i in 1..=400u32 {
      writer
        .add_document(&doc(
          &format!("rest-{i}"),
          vec![("body", json!("rust")), ("latency", json!(i as f64))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "pct_ranks".into(),
    Aggregation::PercentileRanks(PercentileRanksAggregation {
      field: "latency".into(),
      values: vec![0.0],
      missing: None,
    }),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  if let searchlite_core::api::types::AggregationResponse::PercentileRanks(p) =
    resp.aggregations.get("pct_ranks").unwrap()
  {
    let rank = *p.values.get("0").unwrap();
    // The regression being guarded against is that the TDigest path
    // short-circuits to 0.0 when the target equals the observed minimum. A
    // strict `> 0.0` check is enough to catch that without being brittle to
    // TDigest parameterization changes that might shift the exact approximate
    // value (100/500 worth of mass at the minimum ≈ 20.0).
    assert!(
      rank > 0.0,
      "percentile_rank(0.0) on TDigest path must not short-circuit to 0 when values equal the minimum, got {rank}"
    );
    assert!(
      rank <= 100.0,
      "percentile_rank(0.0) must not exceed 100, got {rank}"
    );
  } else {
    panic!("expected percentile ranks agg");
  }
}

/// Regression test for BUG-209: when every observed value equals the target,
/// the approximate path must report 100.0 (not 0.0, which the buggy
/// `target <= min_val` short-circuit produced before reaching the `target >=
/// max_val` branch).
#[test]
fn percentile_ranks_tdigest_path_all_values_equal_target() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "latency".into(),
    i64: false,
    fast: true,
    stored: false,
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
  {
    let mut writer = idx.writer().unwrap();
    // 300 > PERCENTILE_EXACT_LIMIT (256) identical values forces the approximate
    // path, where `min_val == max_val == target == 42.0`.
    for i in 0..300u32 {
      writer
        .add_document(&doc(
          &format!("d-{i}"),
          vec![("body", json!("rust")), ("latency", json!(42.0_f64))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "pct_ranks".into(),
    Aggregation::PercentileRanks(PercentileRanksAggregation {
      field: "latency".into(),
      values: vec![42.0],
      missing: None,
    }),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  if let searchlite_core::api::types::AggregationResponse::PercentileRanks(p) =
    resp.aggregations.get("pct_ranks").unwrap()
  {
    let rank = *p.values.get("42").unwrap();
    assert!(
      (rank - 100.0).abs() < f64::EPSILON,
      "percentile_rank(target) where every value equals target must be 100.0, got {rank}"
    );
  } else {
    panic!("expected percentile ranks agg");
  }
}

#[test]
fn bucket_sort_and_avg_bucket_pipeline() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
      name: "tag".into(),
      stored: true,
      indexed: true,
      fast: true,
      nullable: false,
    });
  schema.numeric_fields.push(NumericField {
    name: "score".into(),
    i64: false,
    fast: true,
    stored: false,
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
  {
    let mut writer = idx.writer().unwrap();
    for (tag, sc) in [("a", 1.0), ("b", 5.0), ("c", 3.0)] {
      writer
        .add_document(&doc(
          &format!("bs-{tag}"),
          vec![
            ("body", json!("rust")),
            ("tag", json!(tag)),
            ("score", json!(sc)),
          ],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }
  let mut sub = BTreeMap::new();
  sub.insert(
    "score_stats".into(),
    Aggregation::Stats(MetricAggregation {
      field: "score".into(),
      missing: None,
    }),
  );
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "tags".into(),
    Aggregation::Terms(Box::new(TermsAggregation {
      field: "tag".into(),
      size: Some(10),
      shard_size: None,
      min_doc_count: None,
      missing: None,
      sampling: None,
      aggs: {
        let mut m = sub.clone();
        m.insert(
          "sorted".into(),
          Aggregation::BucketSort(searchlite_core::api::types::BucketSortAggregation {
            sort: vec![searchlite_core::api::types::BucketSortSpec {
              field: "score_stats.avg".into(),
              order: searchlite_core::api::types::SortOrder::Desc,
            }],
            from: Some(0),
            size: Some(2),
          }),
        );
        m.insert(
          "avg_scores".into(),
          Aggregation::AvgBucket(searchlite_core::api::types::BucketMetricAggregation {
            buckets_path: "score_stats.avg".into(),
          }),
        );
        m
      },
    })),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  if let searchlite_core::api::types::AggregationResponse::Terms {
    buckets,
    aggregations,
    ..
  } = resp.aggregations.get("tags").unwrap()
  {
    assert_eq!(buckets.len(), 2);
    assert_eq!(buckets[0].key, json!("b"));
    if let Some(searchlite_core::api::types::AggregationResponse::AvgBucket(val)) =
      aggregations.get("avg_scores")
    {
      assert!(val.value > 0.0);
    } else {
      panic!("expected avg_bucket");
    }
  } else {
    panic!("expected terms agg");
  }
}

#[test]
fn significant_and_rare_terms() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema
    .keyword_fields
    .push(searchlite_core::api::types::KeywordField {
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
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let idx = IndexBuilder::create(&path, schema, opts).expect("create index");
  let mut writer = idx.writer().expect("writer");
  let docs = [
    doc(
      "sig-1",
      vec![("body", json!("rust systems")), ("tag", json!("tech"))],
    ),
    doc(
      "sig-2",
      vec![("body", json!("rust lang")), ("tag", json!("tech"))],
    ),
    doc(
      "sig-3",
      vec![("body", json!("gardening tips")), ("tag", json!("hobby"))],
    ),
    doc(
      "sig-4",
      vec![("body", json!("news digest")), ("tag", json!("news"))],
    ),
  ];
  for d in docs.iter() {
    writer.add_document(d).unwrap();
  }
  writer.commit().unwrap();

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "sig".into(),
    Aggregation::SignificantTerms(Box::new(SignificantTermsAggregation {
      field: "tag".into(),
      size: Some(5),
      min_doc_count: None,
      background_filter: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );
  aggs.insert(
    "rare".into(),
    Aggregation::RareTerms(Box::new(RareTermsAggregation {
      field: "tag".into(),
      max_doc_count: Some(1),
      size: Some(5),
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );
  let req = SearchRequest {
    query: searchlite_core::api::types::Query::Node(
      searchlite_core::api::types::QueryNode::MatchAll { boost: None },
    ),
    fields: None,
    filter: None,
    limit: 1,
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
    aggs,
    suggest: BTreeMap::new(),
    rescore: None,
    explain: false,
    profile: false,
  };
  let resp = idx.reader().unwrap().search(&req).unwrap();
  if let Some(aggregation) = resp.aggregations.get("sig") {
    if let searchlite_core::api::types::AggregationResponse::SignificantTerms {
      buckets,
      doc_count,
      bg_count,
      ..
    } = aggregation
    {
      assert_eq!(*doc_count, 4);
      assert_eq!(*bg_count, 4);
      assert_eq!(buckets[0].key, json!("tech"));
      assert_eq!(buckets[0].bg_count, 2);
      assert!(buckets[0].score >= 1.0);
    } else {
      panic!("expected significant_terms agg");
    }
  } else {
    panic!("missing significant_terms agg");
  }
  if let Some(aggregation) = resp.aggregations.get("rare") {
    if let searchlite_core::api::types::AggregationResponse::RareTerms { buckets, .. } = aggregation
    {
      assert!(buckets.iter().all(|b| b.doc_count <= 1));
      let keys: Vec<_> = buckets.iter().map(|b| b.key.clone()).collect();
      assert!(keys.contains(&json!("hobby")));
      assert!(keys.contains(&json!("news")));
    } else {
      panic!("expected rare_terms agg");
    }
  } else {
    panic!("missing rare_terms agg");
  }
}

#[test]
fn derivative_and_moving_avg_pipeline() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "views".into(),
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
  let idx = IndexBuilder::create(&path, schema, opts).expect("create index");
  let mut writer = idx.writer().expect("writer");
  let docs = [
    doc("p1", vec![("body", json!("a")), ("views", json!(1))]),
    doc("p2", vec![("body", json!("b")), ("views", json!(2))]),
    doc("p3", vec![("body", json!("c")), ("views", json!(4))]),
  ];
  for d in docs.iter() {
    writer.add_document(d).unwrap();
  }
  writer.commit().unwrap();

  let mut hist_aggs = BTreeMap::new();
  hist_aggs.insert(
    "views_stats".into(),
    Aggregation::Stats(MetricAggregation {
      field: "views".into(),
      missing: None,
    }),
  );
  hist_aggs.insert(
    "delta".into(),
    Aggregation::Derivative(DerivativeAggregation {
      buckets_path: "views_stats.avg".into(),
      gap_policy: Some(GapPolicy::Skip),
      unit: Some(1.0),
    }),
  );
  hist_aggs.insert(
    "smooth".into(),
    Aggregation::MovingAvg(MovingAvgAggregation {
      buckets_path: "views_stats.avg".into(),
      window: 2,
      predict: Some(1),
      gap_policy: Some(GapPolicy::Skip),
    }),
  );
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hist".into(),
    Aggregation::Histogram(Box::new(HistogramAggregation {
      field: "views".into(),
      interval: 1.0,
      offset: None,
      min_doc_count: Some(0),
      extended_bounds: None,
      hard_bounds: None,
      missing: None,
      sampling: None,
      aggs: hist_aggs,
    })),
  );
  let req = SearchRequest {
    query: "a b c".into(),
    fields: None,
    filter: None,
    limit: 1,
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
    aggs,
    suggest: BTreeMap::new(),
    rescore: None,
    explain: false,
    profile: false,
  };
  let resp = idx.reader().unwrap().search(&req).unwrap();
  let hist = resp.aggregations.get("hist").expect("hist agg");
  if let searchlite_core::api::types::AggregationResponse::Histogram {
    buckets,
    aggregations,
    ..
  } = hist
  {
    assert!(buckets.len() >= 3);
    let delta = buckets[1]
      .aggregations
      .get("delta")
      .and_then(|agg| match agg {
        searchlite_core::api::types::AggregationResponse::Derivative(val) => val.value,
        _ => None,
      })
      .unwrap();
    assert!((delta - 1.0).abs() < 1e-6);
    let smooth_val = buckets[2]
      .aggregations
      .get("smooth")
      .and_then(|agg| match agg {
        searchlite_core::api::types::AggregationResponse::MovingAvg(val) => val.value,
        _ => None,
      })
      .unwrap();
    assert!(smooth_val > 0.0);
    if let Some(searchlite_core::api::types::AggregationResponse::MovingAvg(resp)) =
      aggregations.get("smooth")
    {
      assert_eq!(resp.predictions, vec![smooth_val]);
    } else {
      panic!("missing moving_avg pipeline response");
    }
  } else {
    panic!("expected histogram agg");
  }
}

#[test]
fn pipeline_missing_metric_path_with_gap_policy_inserts_zeros() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "views".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
  });
  let idx = IndexBuilder::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&doc("m1", vec![("body", json!("x")), ("views", json!(1))]))
      .unwrap();
    writer
      .add_document(&doc("m2", vec![("body", json!("y")), ("views", json!(3))]))
      .unwrap();
    writer.commit().unwrap();
  }

  let mut hist_aggs = BTreeMap::new();
  hist_aggs.insert(
    "deriv".into(),
    Aggregation::Derivative(DerivativeAggregation {
      buckets_path: "missing.metric".into(),
      gap_policy: Some(GapPolicy::InsertZeros),
      unit: Some(1.0),
    }),
  );
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "hist".into(),
    Aggregation::Histogram(Box::new(HistogramAggregation {
      field: "views".into(),
      interval: 1.0,
      offset: None,
      min_doc_count: None,
      extended_bounds: None,
      hard_bounds: None,
      missing: None,
      sampling: None,
      aggs: hist_aggs,
    })),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "x y".into(),
      fields: None,
      filter: None,
      limit: 1,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  if let searchlite_core::api::types::AggregationResponse::Histogram { buckets, .. } =
    resp.aggregations.get("hist").expect("hist agg")
  {
    assert!(buckets.len() >= 2);
    let deriv_second = buckets[1]
      .aggregations
      .get("deriv")
      .and_then(|agg| match agg {
        searchlite_core::api::types::AggregationResponse::Derivative(val) => val.value,
        _ => None,
      });
    assert_eq!(deriv_second, Some(0.0));
  } else {
    panic!("expected histogram agg");
  }
}

#[test]
fn range_aggregation_to_is_exclusive_at_boundary() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "price".into(),
    i64: false,
    fast: true,
    stored: true,
    nullable: false,
  });
  let idx = Index::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    for (i, price) in [25.0, 50.0, 75.0, 100.0, 150.0].iter().enumerate() {
      writer
        .add_document(&doc(
          &format!("p-{i}"),
          vec![("body", json!("item")), ("price", json!(price))],
        ))
        .unwrap();
    }
    writer.commit().unwrap();
  }
  // Disjoint ranges matching the searchlite-node README example: cheap/mid/premium.
  // With `to` exclusive, each boundary value belongs to exactly one bucket.
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "price_ranges".into(),
    Aggregation::Range(Box::new(RangeAggregation {
      field: "price".into(),
      keyed: false,
      ranges: vec![
        searchlite_core::api::types::RangeBound {
          key: Some("cheap".into()),
          from: None,
          to: Some(50.0),
        },
        searchlite_core::api::types::RangeBound {
          key: Some("mid".into()),
          from: Some(50.0),
          to: Some(100.0),
        },
        searchlite_core::api::types::RangeBound {
          key: Some("premium".into()),
          from: Some(100.0),
          to: None,
        },
      ],
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "item".into(),
      fields: None,
      filter: None,
      limit: 0,
      from: 0,
      return_hits: false,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  let range = resp.aggregations.get("price_ranges").unwrap();
  if let searchlite_core::api::types::AggregationResponse::Range { buckets, .. } = range {
    assert_eq!(buckets.len(), 3);
    // cheap: only 25 (50 is NOT included because to is exclusive)
    assert_eq!(
      buckets[0].doc_count, 1,
      "cheap should contain only price=25"
    );
    // mid: 50 and 75 (100 is NOT included because to is exclusive)
    assert_eq!(
      buckets[1].doc_count, 2,
      "mid should contain price=50 and price=75"
    );
    // premium: 100 and 150
    assert_eq!(
      buckets[2].doc_count, 2,
      "premium should contain price=100 and price=150"
    );
    // In this fixture each matching document has a single price value, so the bucket totals
    // should add up to the number of matching docs if boundary values are not double-counted.
    let total: u64 = buckets.iter().map(|b| b.doc_count).sum();
    assert_eq!(
      total, 5,
      "single-valued price docs should not be double-counted across exclusive range boundaries"
    );
  } else {
    panic!("expected range agg response");
  }
}

#[test]
fn date_range_to_is_exclusive_at_boundary() {
  let tmp = tempfile::tempdir().unwrap();
  let path = tmp.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.numeric_fields.push(NumericField {
    name: "ts".into(),
    i64: true,
    fast: true,
    stored: true,
    nullable: false,
  });
  let idx = Index::create(
    &path,
    schema,
    IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    },
  )
  .unwrap();
  {
    let mut writer = idx.writer().unwrap();
    // ts=2000 corresponds to the exact boundary between the two ranges.
    writer
      .add_document(&doc(
        "on-boundary",
        vec![("body", json!("event")), ("ts", json!(2000))],
      ))
      .unwrap();
    writer
      .add_document(&doc(
        "before-boundary",
        vec![("body", json!("event")), ("ts", json!(1000))],
      ))
      .unwrap();
    writer.commit().unwrap();
  }
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "ts_ranges".into(),
    Aggregation::DateRange(Box::new(
      searchlite_core::api::types::DateRangeAggregation {
        field: "ts".into(),
        keyed: false,
        format: None,
        ranges: vec![
          searchlite_core::api::types::DateRangeBound {
            key: Some("before".into()),
            from: Some("1970-01-01T00:00:00Z".into()),
            to: Some("1970-01-01T00:00:02Z".into()), // 2000 ms
          },
          searchlite_core::api::types::DateRangeBound {
            key: Some("after".into()),
            from: Some("1970-01-01T00:00:02Z".into()), // 2000 ms
            to: Some("1970-01-01T00:00:04Z".into()),
          },
        ],
        missing: None,
        sampling: None,
        aggs: BTreeMap::new(),
      },
    )),
  );
  let resp = idx
    .reader()
    .unwrap()
    .search(&SearchRequest {
      query: "event".into(),
      fields: None,
      filter: None,
      limit: 0,
      from: 0,
      return_hits: false,
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
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    })
    .unwrap();
  let range = resp.aggregations.get("ts_ranges").unwrap();
  if let searchlite_core::api::types::AggregationResponse::DateRange { buckets, .. } = range {
    assert_eq!(buckets.len(), 2);
    // ts=1000 is in "before" (1000 >= 0 && 1000 < 2000)
    assert_eq!(buckets[0].doc_count, 1, "before: only ts=1000");
    // ts=2000 is in "after" (2000 >= 2000 && 2000 < 4000), NOT in "before"
    assert_eq!(
      buckets[1].doc_count, 1,
      "after: only ts=2000 (boundary is exclusive in 'before')"
    );
    let total: u64 = buckets.iter().map(|b| b.doc_count).sum();
    assert_eq!(total, 2, "no double-counting at boundary");
  } else {
    panic!("expected date range agg response");
  }
}
