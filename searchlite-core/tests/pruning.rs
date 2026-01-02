use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use searchlite_core::api::types::{
  Document, ExecutionStrategy, IndexOptions, Schema, SearchRequest, StorageType,
};
use searchlite_core::api::Index;

fn opts(path: &std::path::Path) -> IndexOptions {
  IndexOptions {
    path: path.to_path_buf(),
    create_if_missing: true,
    enable_positions: true,
    bm25_k1: 1.2,
    bm25_b: 0.75,
    storage: StorageType::InMemory,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  }
}

fn add_random_docs(idx: &Index, vocab: &[&str], doc_count: usize, rng: &mut StdRng) {
  let mut writer = idx.writer().unwrap();
  for i in 0..doc_count {
    let body_tokens: Vec<&str> = (0..6)
      .map(|_| vocab[rng.gen_range(0..vocab.len())])
      .collect();
    let body = body_tokens.join(" ");
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!(format!("doc-{i}"))),
          ("body".into(), serde_json::json!(body)),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
  }
  writer.commit().unwrap();
}

#[test]
fn wand_and_bmw_match_bm25_on_random_corpora() {
  let vocab = ["rust", "search", "engine", "fast", "tiny", "wand", "bmw"];
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx");
  let idx = Index::create(&path, Schema::default_text_body(), opts(&path)).unwrap();
  let mut rng = StdRng::seed_from_u64(42);
  add_random_docs(&idx, &vocab, 40, &mut rng);
  let reader = idx.reader().unwrap();

  for _ in 0..5 {
    let mut terms = vocab.to_vec();
    terms.shuffle(&mut rng);
    let query_terms: Vec<&str> = terms.iter().take(3).copied().collect();
    let query = query_terms.join(" ");
    let mut req = SearchRequest {
      query: query.clone(),
      fields: None,
      filters: vec![],
      limit: 5,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Bm25,
      bmw_block_size: Some(4),
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: false,
      highlight_field: None,
      aggs: BTreeMap::new(),
    };
    let bm25 = reader.search(&req).unwrap();

    req.execution = ExecutionStrategy::Wand;
    let wand = reader.search(&req).unwrap();
    req.execution = ExecutionStrategy::Bmw;
    let bmw = reader.search(&req).unwrap();

    assert_eq!(bm25.hits.len(), wand.hits.len());
    assert_eq!(bm25.hits.len(), bmw.hits.len());
    for (a, b) in bm25.hits.iter().zip(wand.hits.iter()) {
      assert_eq!(a.doc_id, b.doc_id);
      assert!((a.score - b.score).abs() < 1e-5);
    }
    for (a, b) in bm25.hits.iter().zip(bmw.hits.iter()) {
      assert_eq!(a.doc_id, b.doc_id);
      assert!((a.score - b.score).abs() < 1e-5);
    }
  }
}

#[test]
fn empty_query_returns_no_hits() {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("idx2");
  let idx = Index::create(&path, Schema::default_text_body(), opts(&path)).unwrap();
  let mut rng = StdRng::seed_from_u64(7);
  add_random_docs(&idx, &["rust"], 3, &mut rng);
  let reader = idx.reader().unwrap();
  let req = SearchRequest {
    query: "".to_string(),
    fields: None,
    filters: vec![],
    limit: 5,
    sort: Vec::new(),
    cursor: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    #[cfg(feature = "vectors")]
    vector_query: None,
    return_stored: false,
    highlight_field: None,
    aggs: BTreeMap::new(),
  };
  let resp = reader.search(&req).unwrap();
  assert!(resp.hits.is_empty());
}
