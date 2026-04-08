use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use searchlite_core::api::types::{Document, ExecutionStrategy, IndexOptions, StorageType};
use searchlite_core::api::Index;
use searchlite_core::Schema;
use serde_json::json;

fn make_doc(id: usize) -> Document {
  Document {
    fields: [
      ("_id".to_string(), json!(id.to_string())),
      (
        "body".to_string(),
        json!(format!("word{} testing search engine", id)),
      ),
    ]
    .into_iter()
    .collect(),
  }
}

fn base_search_request(query: &str) -> searchlite_core::api::types::SearchRequest {
  searchlite_core::api::types::SearchRequest {
    query: query.into(),
    fields: None,
    filter: None,
    limit: 10,
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
    return_stored: true,
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

fn create_index(path: &std::path::Path) -> Index {
  let opts = IndexOptions {
    path: path.to_path_buf(),
    create_if_missing: true,
    enable_positions: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::Filesystem,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  Index::create(path, Schema::default_text_body(), opts).unwrap()
}

#[test]
fn concurrent_reads_during_writes() {
  let tmp = tempfile::tempdir().unwrap();
  let idx = create_index(tmp.path());

  // Seed with 1000 initial documents
  {
    let mut writer = idx.writer().unwrap();
    for i in 0..1000 {
      writer.add_document(&make_doc(i)).unwrap();
    }
    writer.commit().unwrap();
  }

  let idx = Arc::new(idx);

  // Spawn 4 reader threads, each searching 100 times.
  // Assert that searches succeed and return hits (not exact counts,
  // since WAND's total_hits_estimate is an approximation).
  let mut handles = Vec::new();
  for reader_id in 0..4 {
    let idx = Arc::clone(&idx);
    handles.push(thread::spawn(move || {
      for iter in 0..100 {
        let reader = idx.reader().unwrap();
        let req = base_search_request("testing");
        let result = reader.search(&req);
        match result {
          Ok(res) => {
            assert!(
              !res.hits.is_empty(),
              "reader {} iter {}: search returned 0 hits",
              reader_id,
              iter,
            );
          }
          Err(e) => {
            panic!("reader {} iter {} got error: {:?}", reader_id, iter, e);
          }
        }
      }
    }));
  }

  // Spawn 1 writer thread: 10 commits of 10 docs each (100 total new docs)
  {
    let idx = Arc::clone(&idx);
    handles.push(thread::spawn(move || {
      for batch in 0..10 {
        let mut writer = idx.writer().unwrap();
        for j in 0..10 {
          let doc_id = 1000 + batch * 10 + j;
          writer.add_document(&make_doc(doc_id)).unwrap();
        }
        writer.commit().unwrap();
      }
    }));
  }

  for h in handles {
    h.join().expect("thread panicked");
  }

  // Final verification with exact count: all 1100 documents present
  let reader = idx.reader().unwrap();
  let mut req = base_search_request("testing");
  req.track_total_hits = Some(true);
  req.limit = 0;
  let result = reader.search(&req).unwrap();
  assert_eq!(
    result.total_hits_estimate, 1100,
    "expected 1100 total docs after concurrent writes, got {}",
    result.total_hits_estimate
  );
}

#[test]
fn rapid_commit_cycles() {
  let tmp = tempfile::tempdir().unwrap();
  let idx = create_index(tmp.path());

  let mut expected_total = 0u64;

  // 100 rapid commit cycles: add 100 docs, commit, verify search works
  for cycle in 0..100 {
    {
      let mut writer = idx.writer().unwrap();
      for j in 0..100 {
        let doc_id = cycle * 100 + j;
        writer.add_document(&make_doc(doc_id)).unwrap();
      }
      writer.commit().unwrap();
    }
    expected_total += 100;

    // Verify search still works after each commit
    let reader = idx.reader().unwrap();
    let req = base_search_request("testing");
    let result = reader.search(&req).unwrap();
    // We should get some hits (at least up to the limit of 10)
    assert!(
      !result.hits.is_empty() || result.total_hits_estimate > 0,
      "cycle {}: search returned no results",
      cycle
    );
  }

  // Verify final state: all 10000 documents present
  let reader = idx.reader().unwrap();
  let mut req = base_search_request("testing");
  req.track_total_hits = Some(true);
  req.limit = 0;
  let result = reader.search(&req).unwrap();
  assert_eq!(
    result.total_hits_estimate, expected_total,
    "expected {} total docs, got {}",
    expected_total, result.total_hits_estimate
  );

  // Verify segment count grew (each commit creates a new segment, no
  // implicit merging should occur)
  let manifest = idx.manifest();
  assert!(
    manifest.segments.len() > 1,
    "expected multiple segments after 100 commits, got {}",
    manifest.segments.len()
  );
}

#[test]
fn readers_survive_compaction() {
  let tmp = tempfile::tempdir().unwrap();
  let idx = create_index(tmp.path());

  // Create 5 segments via 5 commits of 200 docs each
  for seg in 0..5 {
    let mut writer = idx.writer().unwrap();
    for j in 0..200 {
      let doc_id = seg * 200 + j;
      writer.add_document(&make_doc(doc_id)).unwrap();
    }
    writer.commit().unwrap();
  }

  // Confirm 5 segments exist
  let manifest = idx.manifest();
  assert_eq!(
    manifest.segments.len(),
    5,
    "expected 5 segments before compaction, got {}",
    manifest.segments.len()
  );

  let idx = Arc::new(idx);

  // Use a barrier to ensure reader threads have started before compacting.
  // 3 participants: 2 readers + 1 main thread.
  let barrier = Arc::new(Barrier::new(3));
  let reader_running = Arc::new(AtomicBool::new(true));
  let mut reader_handles = Vec::new();

  for reader_id in 0..2 {
    let idx = Arc::clone(&idx);
    let running = Arc::clone(&reader_running);
    let barrier = Arc::clone(&barrier);
    reader_handles.push(thread::spawn(move || {
      // Signal that this reader thread is ready.
      barrier.wait();
      let mut iterations = 0u64;
      while running.load(Ordering::Relaxed) {
        let reader = idx.reader().unwrap();
        let mut req = base_search_request("testing");
        req.track_total_hits = Some(true);
        req.limit = 0;
        let result = reader.search(&req);
        match result {
          Ok(res) => {
            // All 1000 docs must always be visible
            assert_eq!(
              res.total_hits_estimate, 1000,
              "reader {} iter {}: expected 1000 hits, got {}",
              reader_id, iterations, res.total_hits_estimate
            );
          }
          Err(e) => {
            panic!(
              "reader {} iter {} got error: {:?}",
              reader_id, iterations, e
            );
          }
        }
        iterations += 1;
      }
      iterations
    }));
  }

  // Wait until both reader threads are running before compacting.
  barrier.wait();

  // Compact on the main thread
  idx.compact().unwrap();

  // Let readers run a few more iterations post-compaction, then stop.
  thread::sleep(std::time::Duration::from_millis(100));
  reader_running.store(false, Ordering::Relaxed);

  for h in reader_handles {
    let iters = h.join().expect("reader thread panicked");
    assert!(iters > 0, "reader thread did not complete any iterations");
  }

  // Verify post-compaction state: single segment, same data
  let manifest = idx.manifest();
  assert_eq!(
    manifest.segments.len(),
    1,
    "expected 1 segment after compaction, got {}",
    manifest.segments.len()
  );

  let reader = idx.reader().unwrap();
  let mut req = base_search_request("testing");
  req.track_total_hits = Some(true);
  req.limit = 0;
  let result = reader.search(&req).unwrap();
  assert_eq!(
    result.total_hits_estimate, 1000,
    "expected 1000 docs after compaction, got {}",
    result.total_hits_estimate
  );
}
