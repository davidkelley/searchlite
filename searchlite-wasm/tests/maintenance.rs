//! Integration tests for maintenance ops: `compact`, `inspect`, `stats`,
//! `cleanup_orphaned_files`.

#![cfg(target_arch = "wasm32")]

use searchlite_wasm::Searchlite;
use wasm_bindgen_test::*;

mod common;
use common::{text_schema, unique_db};

wasm_bindgen_test_configure!(run_in_browser);

fn schema_json() -> String {
  serde_json::to_string(&text_schema()).unwrap()
}

// ---------- Moved from src/wasm.rs ----------

#[wasm_bindgen_test]
async fn compact_stats_and_inspect_roundtrip() {
  let db = unique_db("searchlite-maintenance");
  let idx = Searchlite::init(db.clone(), schema_json(), None)
    .await
    .unwrap();

  let docs_a = vec![serde_json::json!({ "_id": "doc-1", "body": "alpha" })];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs_a).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  let docs_b = vec![serde_json::json!({ "_id": "doc-2", "body": "beta" })];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs_b).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  let stats_before_js = idx.stats().unwrap();
  let stats_before: serde_json::Value = serde_wasm_bindgen::from_value(stats_before_js).unwrap();
  assert_eq!(stats_before["documents"].as_u64(), Some(2));
  assert_eq!(stats_before["deleted_documents"].as_u64(), Some(0));
  assert_eq!(stats_before["index_name"].as_str(), Some(db.as_str()));
  assert!(stats_before["segments"].as_u64().unwrap() >= 2);

  let compact_js = idx.compact().await.unwrap();
  let compact: serde_json::Value = serde_wasm_bindgen::from_value(compact_js).unwrap();
  assert_eq!(compact["compacted"].as_bool(), Some(true));

  let stats_after_js = idx.stats().unwrap();
  let stats_after: serde_json::Value = serde_wasm_bindgen::from_value(stats_after_js).unwrap();
  assert_eq!(stats_after["documents"].as_u64(), Some(2));
  assert_eq!(stats_after["deleted_documents"].as_u64(), Some(0));
  assert_eq!(stats_after["segments"].as_u64(), Some(1));

  let inspect_js = idx.inspect().unwrap();
  let inspect: serde_json::Value = serde_wasm_bindgen::from_value(inspect_js).unwrap();
  assert!(inspect["manifest"]["write_key"].is_null());
  let segments = inspect["manifest"]["segments"].as_array().unwrap();
  assert_eq!(
    segments.len() as u64,
    stats_after["segments"].as_u64().unwrap()
  );
  for seg in segments {
    assert!(
      seg["write_binding_b64"].is_null(),
      "inspect() must redact write_binding_b64 on every segment"
    );
  }
}

// ---------- New tests ----------

#[wasm_bindgen_test]
async fn stats_reflects_document_count_after_ingest_and_delete() {
  let db = unique_db("searchlite-stats-counts");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();

  // Ingest three docs.
  let docs = vec![
    serde_json::json!({ "_id": "a", "body": "alpha" }),
    serde_json::json!({ "_id": "b", "body": "beta" }),
    serde_json::json!({ "_id": "c", "body": "gamma" }),
  ];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  let stats_after_add_js = idx.stats().unwrap();
  let stats_after_add: serde_json::Value =
    serde_wasm_bindgen::from_value(stats_after_add_js).unwrap();
  assert_eq!(stats_after_add["documents"].as_u64(), Some(3));
  assert_eq!(stats_after_add["deleted_documents"].as_u64(), Some(0));

  // Delete one.
  idx.delete_document("b".to_string()).unwrap();
  idx.commit().await.unwrap();

  let stats_after_delete_js = idx.stats().unwrap();
  let stats_after_delete: serde_json::Value =
    serde_wasm_bindgen::from_value(stats_after_delete_js).unwrap();
  // Active documents drop to 2, tombstones rise to 1 (until compaction).
  assert_eq!(stats_after_delete["documents"].as_u64(), Some(2));
  assert_eq!(stats_after_delete["deleted_documents"].as_u64(), Some(1));
}

#[wasm_bindgen_test]
async fn inspect_redacts_write_key_metadata() {
  // Separate focused assertion: inspect() must not leak write-key material,
  // regardless of whether segments were compacted.
  let db = unique_db("searchlite-inspect-redact");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  idx
    .add_documents(
      serde_wasm_bindgen::to_value(&serde_json::json!([
        { "_id": "doc-1", "body": "x" },
      ]))
      .unwrap(),
    )
    .unwrap();
  idx.commit().await.unwrap();

  let inspect_js = idx.inspect().unwrap();
  let inspect: serde_json::Value = serde_wasm_bindgen::from_value(inspect_js).unwrap();
  assert!(
    inspect["manifest"]["write_key"].is_null(),
    "manifest.write_key must be redacted"
  );
  for seg in inspect["manifest"]["segments"].as_array().unwrap() {
    assert!(
      seg["write_binding_b64"].is_null(),
      "segment.write_binding_b64 must be redacted"
    );
  }
}

#[wasm_bindgen_test]
async fn cleanup_orphaned_files_on_clean_index_returns_zero_orphans() {
  let db = unique_db("searchlite-orphan-clean");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  idx
    .add_documents(
      serde_wasm_bindgen::to_value(&serde_json::json!([
        { "_id": "doc-1", "body": "live" },
      ]))
      .unwrap(),
    )
    .unwrap();
  idx.commit().await.unwrap();

  let result_js = idx.cleanup_orphaned_files(Some(false)).await.unwrap();
  let result: serde_json::Value = serde_wasm_bindgen::from_value(result_js).unwrap();
  assert_eq!(result["orphaned"].as_u64(), Some(0));
  assert_eq!(result["removed"].as_array().map(|arr| arr.len()), Some(0));
  assert_eq!(result["dry_run"].as_bool(), Some(false));

  // Post-cleanup the live doc is still searchable.
  let search = idx.search("live".to_string(), 5, Some(true)).unwrap();
  let search_json: serde_json::Value = serde_wasm_bindgen::from_value(search).unwrap();
  assert_eq!(search_json["hits"].as_array().unwrap().len(), 1);
}

#[wasm_bindgen_test]
async fn cleanup_orphaned_files_dry_run_preserves_flag() {
  let db = unique_db("searchlite-orphan-dry-run");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  idx.commit().await.unwrap();

  let result_js = idx.cleanup_orphaned_files(Some(true)).await.unwrap();
  let result: serde_json::Value = serde_wasm_bindgen::from_value(result_js).unwrap();
  assert_eq!(result["dry_run"].as_bool(), Some(true));
}
