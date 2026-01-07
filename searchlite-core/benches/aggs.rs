use std::collections::BTreeMap;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Aggregation, Document, ExecutionStrategy, HistogramAggregation, IndexOptions, KeywordField,
  NumericField, Schema, SearchRequest, StorageType, TermsAggregation,
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
    let tag_id = rng.gen_range(0..cardinality);
    let score = rng.gen_range(0..10_000i64);
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
      aggs: BTreeMap::new(),
    })),
  );
  let req = SearchRequest {
    query: "rust".into(),
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
      aggs: BTreeMap::new(),
    })),
  );
  let req = SearchRequest {
    query: "rust".into(),
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

criterion_group!(
  benches,
  bench_terms_aggregation,
  bench_histogram_aggregation
);
criterion_main!(benches);
