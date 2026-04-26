//! Integration tests for index lifecycle and storage management:
//! `init`, `list_indexes`, `clear_index`, `drop_index`, `storage_usage`,
//! `cleanup_indexes`.

#![cfg(target_arch = "wasm32")]

use searchlite_core::Schema;
use searchlite_wasm::Searchlite;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

mod common;
use common::{text_schema, unique_db, WasmErrorPayload};

wasm_bindgen_test_configure!(run_in_browser);

/// The reserved registry db name — must match `REGISTRY_DB_NAME` in wasm.rs.
const REGISTRY_DB_NAME: &str = "searchlite_registry";

fn schema_json() -> String {
  serde_json::to_string(&text_schema()).unwrap()
}

// ---------- Moved from src/wasm.rs ----------

#[wasm_bindgen_test]
async fn init_reuses_existing_index() {
  let db = unique_db("searchlite-reopen");
  let schema = Schema::default_text_body();
  let schema_json = serde_json::to_string(&schema).unwrap();
  let idx = Searchlite::init(db.clone(), schema_json.clone(), None)
    .await
    .unwrap();
  let docs = vec![serde_json::json!({ "_id": "doc-1", "body": "hello reopen" })];
  let docs_js = serde_wasm_bindgen::to_value(&docs).unwrap();
  idx.add_documents(docs_js).unwrap();
  idx.commit().await.unwrap();
  drop(idx);

  let reopened = Searchlite::init(db, schema_json, None).await.unwrap();
  let result = reopened.search("hello".to_string(), 5, None).unwrap();
  let result_json: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  let hits = result_json["hits"].as_array().unwrap();
  assert_eq!(hits.len(), 1);
}

#[wasm_bindgen_test]
async fn list_indexes_includes_initialized_db() {
  let db = unique_db("searchlite-list-indexes");
  let idx = Searchlite::init(db.clone(), schema_json(), None)
    .await
    .unwrap();
  idx.commit().await.unwrap();

  let indexes_js = Searchlite::list_indexes().await.unwrap();
  let indexes: Vec<serde_json::Value> = serde_wasm_bindgen::from_value(indexes_js).unwrap();
  assert!(indexes
    .iter()
    .any(|entry| entry["db_name"].as_str() == Some(&db)));
}

#[wasm_bindgen_test]
async fn clear_index_resets_contents() {
  let db = unique_db("searchlite-clear-index");
  let idx = Searchlite::init(db.clone(), schema_json(), None)
    .await
    .unwrap();
  let docs = vec![serde_json::json!({ "_id": "doc-1", "body": "clear me" })];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  Searchlite::clear_index(db.clone()).await.unwrap();

  // Verify the index remains discoverable in the registry after clear.
  let indexes_js = Searchlite::list_indexes().await.unwrap();
  let indexes: Vec<serde_json::Value> = serde_wasm_bindgen::from_value(indexes_js).unwrap();
  assert!(
    indexes
      .iter()
      .any(|entry| entry["db_name"].as_str() == Some(&db)),
    "clear_index should preserve the registry entry"
  );

  let reopened = Searchlite::init(db, schema_json(), None).await.unwrap();
  let result = reopened.search("clear".to_string(), 5, None).unwrap();
  let result_json: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  let hits = result_json["hits"].as_array().unwrap();
  assert_eq!(hits.len(), 0);
}

#[wasm_bindgen_test]
async fn drop_index_removes_registry_entry() {
  let db = unique_db("searchlite-drop-index");
  let idx = Searchlite::init(db.clone(), schema_json(), None)
    .await
    .unwrap();
  idx.commit().await.unwrap();

  Searchlite::drop_index(db.clone()).await.unwrap();

  let indexes_js = Searchlite::list_indexes().await.unwrap();
  let indexes: Vec<serde_json::Value> = serde_wasm_bindgen::from_value(indexes_js).unwrap();
  assert!(!indexes
    .iter()
    .any(|entry| entry["db_name"].as_str() == Some(&db)));

  // Can recreate after deletion.
  let recreated = Searchlite::init(db, schema_json(), None).await.unwrap();
  recreated.commit().await.unwrap();
}

#[wasm_bindgen_test]
async fn init_rejects_reserved_registry_name() {
  let err = match Searchlite::init(REGISTRY_DB_NAME.to_string(), schema_json(), None).await {
    Err(err) => err,
    Ok(_) => panic!("expected reserved_name error"),
  };
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "reserved_name");
}

#[wasm_bindgen_test]
async fn storage_usage_returns_supported_or_note() {
  let usage_js = Searchlite::storage_usage().await.unwrap();
  let usage: serde_json::Value = serde_wasm_bindgen::from_value(usage_js).unwrap();
  let supported = usage["supported"].as_bool().unwrap_or(false);
  let has_note = usage["note"].is_string();
  assert!(
    supported || has_note,
    "storage_usage must report supported=true or provide a note"
  );
  if supported {
    assert!(
      !usage["usage_bytes"].is_null()
        || !usage["quota_bytes"].is_null()
        || !usage["persisted"].is_null(),
      "supported storage_usage must include at least one of usage_bytes / quota_bytes / persisted"
    );
  }
}

#[wasm_bindgen_test]
async fn cleanup_indexes_drops_only_stale_entries() {
  // Seed one fresh index and one stale index. We can't backdate the registry
  // entry through the public API, so this test spaces the two `init`s and
  // then uses a threshold that catches only indexes older than the elapsed
  // gap.
  let stale = unique_db("searchlite-cleanup-stale");
  let stale_idx = Searchlite::init(stale.clone(), schema_json(), None)
    .await
    .unwrap();
  stale_idx.commit().await.unwrap();

  // Artificial wait via set_timeout round-trip so the stale entry ages.
  let (tx, rx) = futures::channel::oneshot::channel::<()>();
  let tx = std::rc::Rc::new(std::cell::RefCell::new(Some(tx)));
  common::set_timeout_once(120, move || {
    if let Some(tx) = tx.borrow_mut().take() {
      let _ = tx.send(());
    }
  });
  let _ = rx.await;

  let fresh = unique_db("searchlite-cleanup-fresh");
  let fresh_idx = Searchlite::init(fresh.clone(), schema_json(), None)
    .await
    .unwrap();
  fresh_idx.commit().await.unwrap();

  // Threshold: 60ms — catches the stale index (>120ms old) but not the
  // just-created fresh one.
  let cleanup_js = Searchlite::cleanup_indexes(60.0, Some(false))
    .await
    .unwrap();
  let cleanup: serde_json::Value = serde_wasm_bindgen::from_value(cleanup_js).unwrap();
  let dropped = cleanup["dropped"].as_array().unwrap();
  let kept = cleanup["kept"].as_array().unwrap();
  assert!(dropped.iter().any(|name| name.as_str() == Some(&stale)));
  assert!(kept.iter().any(|name| name.as_str() == Some(&fresh)));

  Searchlite::drop_index(fresh).await.unwrap();
}

// ---------- New tests ----------

#[wasm_bindgen_test]
async fn list_indexes_returns_array() {
  // list_indexes always returns a (possibly empty) array. Verifies the
  // happy path when we haven't just initialised something.
  let indexes_js = Searchlite::list_indexes().await.unwrap();
  let indexes: serde_json::Value = serde_wasm_bindgen::from_value(indexes_js).unwrap();
  assert!(
    indexes.is_array(),
    "list_indexes should always return a JSON array"
  );
}

#[wasm_bindgen_test]
async fn drop_index_rejects_reserved_name() {
  let err = Searchlite::drop_index(REGISTRY_DB_NAME.to_string())
    .await
    .expect_err("expected reserved_name error");
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "reserved_name");
}

#[wasm_bindgen_test]
async fn clear_index_rejects_reserved_name() {
  let err = Searchlite::clear_index(REGISTRY_DB_NAME.to_string())
    .await
    .expect_err("expected reserved_name error");
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "reserved_name");
}

#[wasm_bindgen_test]
async fn drop_index_is_idempotent_for_missing_db() {
  // Dropping a db that was never initialised should succeed silently —
  // `delete_database` is a no-op on a nonexistent database and the registry
  // entry is already absent. This is the contract callers depend on for
  // cleanup scripts.
  let missing = unique_db("searchlite-never-existed");
  Searchlite::drop_index(missing).await.unwrap();
}

#[wasm_bindgen_test]
async fn cleanup_indexes_rejects_negative_duration() {
  let err = Searchlite::cleanup_indexes(-1.0, None)
    .await
    .expect_err("expected invalid_cleanup_request");
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_cleanup_request");
}

#[wasm_bindgen_test]
async fn cleanup_indexes_rejects_nan_duration() {
  let err = Searchlite::cleanup_indexes(f64::NAN, None)
    .await
    .expect_err("expected invalid_cleanup_request");
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_cleanup_request");
}

#[wasm_bindgen_test]
async fn cleanup_indexes_dry_run_does_not_delete() {
  let stale = unique_db("searchlite-dry-run-stale");
  let stale_idx = Searchlite::init(stale.clone(), schema_json(), None)
    .await
    .unwrap();
  stale_idx.commit().await.unwrap();

  let (tx, rx) = futures::channel::oneshot::channel::<()>();
  let tx = std::rc::Rc::new(std::cell::RefCell::new(Some(tx)));
  common::set_timeout_once(120, move || {
    if let Some(tx) = tx.borrow_mut().take() {
      let _ = tx.send(());
    }
  });
  let _ = rx.await;

  let cleanup_js = Searchlite::cleanup_indexes(60.0, Some(true)).await.unwrap();
  let cleanup: serde_json::Value = serde_wasm_bindgen::from_value(cleanup_js).unwrap();
  assert_eq!(cleanup["dry_run"].as_bool(), Some(true));
  let dropped = cleanup["dropped"].as_array().unwrap();
  assert!(dropped.iter().any(|name| name.as_str() == Some(&stale)));

  // Registry still contains the stale entry because dry_run did not delete it.
  let indexes_js = Searchlite::list_indexes().await.unwrap();
  let indexes: Vec<serde_json::Value> = serde_wasm_bindgen::from_value(indexes_js).unwrap();
  assert!(
    indexes
      .iter()
      .any(|entry| entry["db_name"].as_str() == Some(&stale)),
    "dry_run must not remove registry entries"
  );

  Searchlite::drop_index(stale).await.unwrap();
}

// Keep `JsValue` referenced so the cargo check doesn't strip it.
#[allow(dead_code)]
fn _keep_imports_alive() -> JsValue {
  JsValue::NULL
}
