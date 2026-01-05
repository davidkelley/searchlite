use std::time::{Duration, Instant};

use anyhow::Result;
use hashbrown::HashMap;
use rand::Rng;
use searchlite_core::api::query::parse_query;
use searchlite_core::api::types::{Document, ExecutionStrategy, IndexOptions, Schema, StorageType};
use searchlite_core::api::Index;
use searchlite_core::query::planner::expand_terms;
use searchlite_core::query::wand::{execute_top_k_with_stats, QueryStats, ScoredTerm};

fn main() -> Result<()> {
  let docs = 200usize;
  let query_count = 200usize;
  let k = 10usize;
  let vocab = [
    "rust", "search", "engine", "tiny", "sqlite", "wand", "bmw", "fast", "index",
  ];
  let dir = tempfile::tempdir()?;
  let path = dir.path().join("pruning_bench");
  let opts = IndexOptions {
    path: path.clone(),
    create_if_missing: true,
    enable_positions: true,
    bm25_k1: 1.2,
    bm25_b: 0.75,
    storage: StorageType::InMemory,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };

  let idx = Index::create(&path, Schema::default_text_body(), opts.clone())?;
  {
    let mut rng = rand::thread_rng();
    let mut writer = idx.writer()?;
    for i in 0..docs {
      let body_tokens: Vec<&str> = (0..12)
        .map(|_| vocab[rng.gen_range(0..vocab.len())])
        .collect();
      let body = body_tokens.join(" ");
      writer.add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!(format!("doc-{i}"))),
          ("body".into(), serde_json::json!(body)),
        ]
        .into_iter()
        .collect(),
      })?;
    }
    writer.commit()?;
  }

  let reader = idx.reader()?;
  let default_fields: Vec<String> = reader
    .manifest
    .schema
    .text_fields
    .iter()
    .map(|f| f.name.clone())
    .collect();
  let queries: Vec<String> = {
    let mut rng = rand::thread_rng();
    (0..query_count)
      .map(|_| {
        let tokens: Vec<&str> = (0..3)
          .map(|_| vocab[rng.gen_range(0..vocab.len())])
          .collect();
        tokens.join(" ")
      })
      .collect()
  };

  let bm25 = run_strategy(
    &reader,
    &default_fields,
    &opts,
    &queries,
    ExecutionStrategy::Bm25,
    None,
    k,
  );
  let wand = run_strategy(
    &reader,
    &default_fields,
    &opts,
    &queries,
    ExecutionStrategy::Wand,
    None,
    k,
  );
  let bmw = run_strategy(
    &reader,
    &default_fields,
    &opts,
    &queries,
    ExecutionStrategy::Bmw,
    Some(16),
    k,
  );

  print_summary("bm25", &bm25, query_count);
  print_summary("wand", &wand, query_count);
  print_summary("bmw", &bmw, query_count);
  Ok(())
}

struct BenchOutcome {
  duration: Duration,
  stats: QueryStats,
}

fn run_strategy(
  reader: &searchlite_core::api::reader::IndexReader,
  default_fields: &[String],
  opts: &IndexOptions,
  queries: &[String],
  strategy: ExecutionStrategy,
  block_size: Option<usize>,
  k: usize,
) -> BenchOutcome {
  let mut stats = QueryStats::default();
  let start = Instant::now();
  for query in queries.iter() {
    let parsed = parse_query(query);
    let term_keys = expand_terms(&parsed, default_fields);
    let qualified_terms: Vec<(String, String, String)> = term_keys
      .iter()
      .map(|(field, term)| {
        let mut key = String::with_capacity(field.len() + term.len() + 1);
        key.push_str(field);
        key.push(':');
        key.push_str(term);
        (field.clone(), term.clone(), key)
      })
      .collect();
    for seg in reader.segments.iter() {
      let mut term_weights: HashMap<String, (String, f32)> = HashMap::new();
      for (field, _, key) in qualified_terms.iter() {
        let entry = term_weights
          .entry(key.clone())
          .or_insert((field.clone(), 0.0));
        entry.1 += 1.0;
      }
      let docs = seg.meta.doc_count as f32;
      let mut terms = Vec::new();
      for (key, (field, weight)) in term_weights.into_iter() {
        if let Some(postings) = seg.postings(&key) {
          terms.push(ScoredTerm {
            postings,
            weight,
            avgdl: seg.avg_field_length(&field),
            docs,
            k1: opts.bm25_k1,
            b: opts.bm25_b,
            leaf: 0,
          });
        }
      }
      if terms.is_empty() {
        continue;
      }
      let mut accept = |_doc: searchlite_core::DocId, _score: f32| true;
      let _ =
        execute_top_k_with_stats::<_, searchlite_core::query::collector::MatchCountingCollector>(
          terms,
          k,
          strategy.clone(),
          block_size,
          &mut accept,
          None,
          Some(&mut stats),
        );
    }
  }
  BenchOutcome {
    duration: start.elapsed(),
    stats,
  }
}

fn print_summary(label: &str, outcome: &BenchOutcome, queries: usize) {
  let per_query = outcome.duration.as_secs_f64() * 1_000.0 / queries as f64;
  println!(
    "{label:>4}: {:>6.2} ms/q, scored {:>5} docs, advances {:>6}",
    per_query, outcome.stats.scored_docs, outcome.stats.postings_advanced
  );
}
