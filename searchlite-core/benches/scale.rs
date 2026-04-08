#[allow(dead_code)]
mod datagen;

use std::collections::BTreeMap;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Aggregation, ExecutionStrategy, HistogramAggregation, IndexOptions, SearchRequest, StorageType,
  TermsAggregation,
};
use searchlite_core::api::{Filter, Index};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build an in-memory index from `count` synthetic docs, split across
/// `segments` commits so the index has multiple segments.
/// Returns both the index and the TempDir handle to prevent premature cleanup.
fn build_index(count: usize, segments: usize, seed: u64) -> (Index, tempfile::TempDir) {
  let docs = datagen::generate_docs(count, seed);
  let schema = datagen::generate_schema();
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().to_path_buf();
  let opts = IndexOptions {
    path: path.clone(),
    create_if_missing: true,
    enable_positions: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::InMemory,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let idx = IndexBuilder::create(&path, schema, opts).unwrap();

  let chunk_size = (count + segments - 1) / segments; // ceil division
  for chunk in docs.chunks(chunk_size) {
    let mut writer = idx.writer().unwrap();
    for doc in chunk {
      writer.add_document(doc).unwrap();
    }
    writer.commit().unwrap();
  }
  (idx, dir)
}

/// Build a default SearchRequest with the given query, limit, and optional filter.
fn make_search_request(query: &str, limit: usize, filter: Option<Filter>) -> SearchRequest {
  SearchRequest {
    query: query.to_string().into(),
    fields: None,
    filter,
    limit,
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

// ---------------------------------------------------------------------------
// Indexing throughput benchmarks
// ---------------------------------------------------------------------------

fn bench_index_10k(c: &mut Criterion) {
  let docs = datagen::generate_docs(10_000, 1);
  let schema = datagen::generate_schema();

  c.bench_function("index_10k", |b| {
    b.iter_batched(
      || (docs.clone(), schema.clone()),
      |(docs, schema)| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        let opts = IndexOptions {
          path: path.clone(),
          create_if_missing: true,
          enable_positions: true,
          bm25_k1: 0.9,
          bm25_b: 0.4,
          storage: StorageType::InMemory,
          #[cfg(feature = "vectors")]
          vector_defaults: None,
        };
        let idx = IndexBuilder::create(&path, schema, opts).unwrap();
        let mut writer = idx.writer().unwrap();
        for doc in &docs {
          writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();
        black_box(idx);
      },
      BatchSize::LargeInput,
    );
  });
}

fn bench_index_100k(c: &mut Criterion) {
  let docs = datagen::generate_docs(100_000, 2);
  let schema = datagen::generate_schema();

  c.bench_function("index_100k", |b| {
    b.iter_batched(
      || (docs.clone(), schema.clone()),
      |(docs, schema)| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        let opts = IndexOptions {
          path: path.clone(),
          create_if_missing: true,
          enable_positions: true,
          bm25_k1: 0.9,
          bm25_b: 0.4,
          storage: StorageType::InMemory,
          #[cfg(feature = "vectors")]
          vector_defaults: None,
        };
        let idx = IndexBuilder::create(&path, schema, opts).unwrap();
        let mut writer = idx.writer().unwrap();
        for doc in &docs {
          writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();
        black_box(idx);
      },
      BatchSize::LargeInput,
    );
  });
}

// ---------------------------------------------------------------------------
// Search latency benchmarks (on a pre-built 100K-doc index)
// ---------------------------------------------------------------------------

fn bench_search_single_term_100k(c: &mut Criterion) {
  let (idx, _dir) = build_index(100_000, 1, 100);
  let reader = idx.reader().unwrap();
  let req = make_search_request("search", 10, None);

  c.bench_function("search_single_term_100k", |b| {
    b.iter_batched(
      || req.clone(),
      |req| {
        let resp = reader.search(&req).unwrap();
        black_box(resp);
      },
      BatchSize::LargeInput,
    );
  });
}

fn bench_search_bool_and_100k(c: &mut Criterion) {
  let (idx, _dir) = build_index(100_000, 1, 101);
  let reader = idx.reader().unwrap();
  let req = make_search_request("search engine", 10, None);

  c.bench_function("search_bool_and_100k", |b| {
    b.iter_batched(
      || req.clone(),
      |req| {
        let resp = reader.search(&req).unwrap();
        black_box(resp);
      },
      BatchSize::LargeInput,
    );
  });
}

fn bench_search_filtered_100k(c: &mut Criterion) {
  let (idx, _dir) = build_index(100_000, 1, 102);
  let reader = idx.reader().unwrap();
  let filter = Some(Filter::KeywordEq {
    field: "category".to_string(),
    value: "electronics".to_string(),
  });
  let req = make_search_request("search", 10, filter);

  c.bench_function("search_filtered_100k", |b| {
    b.iter_batched(
      || req.clone(),
      |req| {
        let resp = reader.search(&req).unwrap();
        black_box(resp);
      },
      BatchSize::LargeInput,
    );
  });
}

fn bench_search_top100_100k(c: &mut Criterion) {
  let (idx, _dir) = build_index(100_000, 1, 103);
  let reader = idx.reader().unwrap();
  let req = make_search_request("search", 100, None);

  c.bench_function("search_top100_100k", |b| {
    b.iter_batched(
      || req.clone(),
      |req| {
        let resp = reader.search(&req).unwrap();
        black_box(resp);
      },
      BatchSize::LargeInput,
    );
  });
}

// ---------------------------------------------------------------------------
// Aggregation latency benchmarks (on a pre-built 100K-doc index)
// ---------------------------------------------------------------------------

fn bench_agg_terms_100k(c: &mut Criterion) {
  let (idx, _dir) = build_index(100_000, 1, 200);
  let reader = idx.reader().unwrap();

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "categories".into(),
    Aggregation::Terms(Box::new(TermsAggregation {
      field: "category".into(),
      size: Some(50),
      shard_size: Some(200),
      min_doc_count: Some(1),
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );

  let mut req = make_search_request("search", 0, None);
  req.return_hits = false;
  req.aggs = aggs;

  c.bench_function("agg_terms_100k", |b| {
    b.iter_batched(
      || req.clone(),
      |req| {
        let resp = reader.search(&req).unwrap();
        black_box(resp);
      },
      BatchSize::LargeInput,
    );
  });
}

fn bench_agg_histogram_100k(c: &mut Criterion) {
  let (idx, _dir) = build_index(100_000, 1, 201);
  let reader = idx.reader().unwrap();

  let mut aggs = BTreeMap::new();
  aggs.insert(
    "prices".into(),
    Aggregation::Histogram(Box::new(HistogramAggregation {
      field: "price".into(),
      interval: 500.0,
      offset: None,
      min_doc_count: Some(0),
      extended_bounds: None,
      hard_bounds: None,
      missing: None,
      sampling: None,
      aggs: BTreeMap::new(),
    })),
  );

  let mut req = make_search_request("search", 0, None);
  req.return_hits = false;
  req.aggs = aggs;

  c.bench_function("agg_histogram_100k", |b| {
    b.iter_batched(
      || req.clone(),
      |req| {
        let resp = reader.search(&req).unwrap();
        black_box(resp);
      },
      BatchSize::LargeInput,
    );
  });
}

// ---------------------------------------------------------------------------
// Commit & compact benchmarks (on a 100K-doc index with 10 segments)
// ---------------------------------------------------------------------------

fn bench_commit_100k(c: &mut Criterion) {
  let extra_docs = datagen::generate_docs(1_000, 999);

  c.bench_function("commit_100k", |b| {
    b.iter_batched(
      || {
        let (idx, dir) = build_index(100_000, 10, 300);
        (idx, dir, extra_docs.clone())
      },
      |(idx, _dir, docs)| {
        let mut writer = idx.writer().unwrap();
        for doc in &docs {
          writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();
        black_box(&idx);
      },
      BatchSize::LargeInput,
    );
  });
}

fn bench_compact_100k(c: &mut Criterion) {
  c.bench_function("compact_100k", |b| {
    b.iter_batched(
      || build_index(100_000, 10, 301),
      |(idx, _dir)| {
        idx.compact().unwrap();
        black_box(&idx);
      },
      BatchSize::LargeInput,
    );
  });
}

// ---------------------------------------------------------------------------
// Criterion groups & main
// ---------------------------------------------------------------------------

criterion_group! {
  name = indexing;
  config = Criterion::default().sample_size(10);
  targets = bench_index_10k, bench_index_100k
}

criterion_group! {
  name = search;
  config = Criterion::default().sample_size(20);
  targets =
    bench_search_single_term_100k,
    bench_search_bool_and_100k,
    bench_search_filtered_100k,
    bench_search_top100_100k
}

criterion_group! {
  name = aggregations;
  config = Criterion::default().sample_size(20);
  targets = bench_agg_terms_100k, bench_agg_histogram_100k
}

criterion_group! {
  name = maintenance;
  config = Criterion::default().sample_size(10);
  targets = bench_commit_100k, bench_compact_100k
}

criterion_main!(indexing, search, aggregations, maintenance);
