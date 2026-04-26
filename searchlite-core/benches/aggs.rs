use std::collections::BTreeMap;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Aggregation, Document, ExecutionStrategy, HistogramAggregation, IndexOptions, KeywordField,
  NestedAggregation, NestedField, NestedProperty, NumericField, Schema, SearchRequest, StorageType,
  TermsAggregation,
};
use searchlite_core::api::Index;

struct BenchIndex {
  index: Index,
  _dir: tempfile::TempDir,
}

fn build_bench_index(doc_count: usize, cardinality: usize) -> BenchIndex {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.keyword_fields.push(KeywordField {
    name: "tag".into(),
    stored: true,
    indexed: true,
    fast: true,
    nullable: false,
  });
  schema.numeric_fields.push(NumericField {
    name: "score".into(),
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
  let idx = IndexBuilder::create(&path, schema, opts).unwrap();

  let mut writer = idx.writer().unwrap();
  let mut rng = StdRng::seed_from_u64(42);
  for i in 0..doc_count {
    let tag_id = rng.random_range(0..cardinality);
    let score = rng.random_range(0..10_000i64);
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!(format!("bench-{i}"))),
          (
            "body".into(),
            serde_json::json!(format!("rust systems {i}")),
          ),
          ("tag".into(), serde_json::json!(format!("tag_{tag_id}"))),
          ("score".into(), serde_json::json!(score)),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
  }
  writer.commit().unwrap();

  BenchIndex {
    index: idx,
    _dir: dir,
  }
}

fn build_nested_bench_index(
  doc_count: usize,
  category_count: usize,
  value_cardinality: usize,
) -> BenchIndex {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().to_path_buf();
  let mut schema = Schema::default_text_body();
  schema.nested_fields.push(NestedField {
    name: "metadata".into(),
    fields: vec![
      NestedProperty::Keyword(KeywordField {
        name: "key".into(),
        stored: true,
        indexed: true,
        fast: true,
        nullable: false,
      }),
      NestedProperty::Keyword(KeywordField {
        name: "value".into(),
        stored: true,
        indexed: true,
        fast: true,
        nullable: false,
      }),
    ],
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
  let idx = IndexBuilder::create(&path, schema, opts).unwrap();

  let mut writer = idx.writer().unwrap();
  let mut rng = StdRng::seed_from_u64(1337);
  for i in 0..doc_count {
    let mut metadata = Vec::with_capacity(3);
    let category_seed = rng.random_range(0..category_count);
    for slot in 0..3 {
      let category = (category_seed + slot) % category_count;
      let value_id = rng.random_range(0..value_cardinality);
      metadata.push(serde_json::json!({
        "key": format!("Category_{category}"),
        "value": format!("Value_{value_id}")
      }));
    }
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!(format!("nested-bench-{i}"))),
          (
            "body".into(),
            serde_json::json!(format!("rust nested metadata {i}")),
          ),
          ("metadata".into(), serde_json::Value::Array(metadata)),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
  }
  writer.commit().unwrap();

  BenchIndex {
    index: idx,
    _dir: dir,
  }
}

fn bench_terms_aggregation(c: &mut Criterion) {
  let bench = build_bench_index(5_000, 500);
  let reader = bench.index.reader().unwrap();
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "tags".into(),
    Aggregation::Terms(Box::new(TermsAggregation {
      field: "tag".into(),
      size: Some(20),
      shard_size: Some(200),
      min_doc_count: Some(1),
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );
  let req = SearchRequest {
    query: "rust".into(),
    fields: None,
    filter: None,
    limit: 1,
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

  c.bench_function("aggs_terms_high_card", |b| {
    b.iter_batched(
      || req.clone(),
      |req| {
        let resp = reader.search(&req).unwrap();
        criterion::black_box(resp);
      },
      BatchSize::SmallInput,
    );
  });
}

fn bench_histogram_aggregation(c: &mut Criterion) {
  let bench = build_bench_index(5_000, 50);
  let reader = bench.index.reader().unwrap();
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "scores".into(),
    Aggregation::Histogram(Box::new(HistogramAggregation {
      field: "score".into(),
      interval: 250.0,
      offset: None,
      min_doc_count: Some(0),
      extended_bounds: None,
      hard_bounds: None,
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );
  let req = SearchRequest {
    query: "rust".into(),
    fields: None,
    filter: None,
    limit: 1,
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

  c.bench_function("aggs_histogram_numeric", |b| {
    b.iter_batched(
      || req.clone(),
      |req| {
        let resp = reader.search(&req).unwrap();
        criterion::black_box(resp);
      },
      BatchSize::SmallInput,
    );
  });
}

fn bench_nested_terms_aggregation(c: &mut Criterion) {
  let bench = build_nested_bench_index(5_000, 12, 1_000);
  let reader = bench.index.reader().unwrap();
  let mut nested_children = BTreeMap::new();
  nested_children.insert(
    "by_key".into(),
    Aggregation::Terms(Box::new(TermsAggregation {
      field: "key".into(),
      size: Some(12),
      shard_size: Some(48),
      min_doc_count: Some(1),
      missing: None,
      sampling: None,
      aggs: {
        let mut by_key_children = BTreeMap::new();
        by_key_children.insert(
          "by_value".into(),
          Aggregation::Terms(Box::new(TermsAggregation {
            field: "value".into(),
            size: Some(20),
            shard_size: Some(120),
            min_doc_count: Some(1),
            missing: None,
            sampling: None,
            aggs: BTreeMap::new(),
          })),
        );
        by_key_children
      },
    })),
  );
  let mut aggs = BTreeMap::new();
  aggs.insert(
    "metadata_nested".into(),
    Aggregation::Nested(Box::new(NestedAggregation {
      path: "metadata".into(),
      sampling: None,
      aggs: nested_children,
    })),
  );
  let req = SearchRequest {
    query: "rust".into(),
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
  };

  c.bench_function("aggs_nested_terms_metadata", |b| {
    b.iter_batched(
      || req.clone(),
      |req| {
        let resp = reader.search(&req).unwrap();
        criterion::black_box(resp);
      },
      BatchSize::SmallInput,
    );
  });
}

criterion_group!(
  benches,
  bench_terms_aggregation,
  bench_histogram_aggregation,
  bench_nested_terms_aggregation
);
criterion_main!(benches);
