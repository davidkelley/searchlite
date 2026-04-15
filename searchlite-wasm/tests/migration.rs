//! Integration tests for schema migration: `plan_migration` and
//! `migrate_index`. Rollback-on-failure tests stay inline in src/wasm.rs
//! because they need the failure-injection guard.

#![cfg(target_arch = "wasm32")]

use searchlite_core::api::types::KeywordField;
use searchlite_core::Schema;
use searchlite_wasm::Searchlite;
use wasm_bindgen_test::*;

mod common;
use common::{text_schema, unique_db, WasmErrorPayload};

wasm_bindgen_test_configure!(run_in_browser);

fn schema_v1_json() -> String {
  serde_json::to_string(&text_schema()).unwrap()
}

fn schema_v2_json() -> String {
  let mut schema = Schema::default_text_body();
  schema.keyword_fields.push(KeywordField {
    name: "category".to_string(),
    stored: true,
    indexed: true,
    fast: true,
    nullable: false,
  });
  serde_json::to_string(&schema).unwrap()
}

// ---------- Moved from src/wasm.rs ----------

#[wasm_bindgen_test]
async fn plan_migration_reports_compatibility_and_rebuild() {
  let db = unique_db("searchlite-plan-migration");
  let idx = Searchlite::init(db.clone(), schema_v1_json(), None)
    .await
    .unwrap();
  idx.commit().await.unwrap();

  let compatible_js = Searchlite::plan_migration(db.clone(), schema_v1_json())
    .await
    .unwrap();
  let compatible: serde_json::Value = serde_wasm_bindgen::from_value(compatible_js).unwrap();
  assert_eq!(compatible["status"].as_str(), Some("compatible"));
  assert_eq!(compatible["rebuild_required"].as_bool(), Some(false));

  let rebuild_js = Searchlite::plan_migration(db, schema_v2_json())
    .await
    .unwrap();
  let rebuild: serde_json::Value = serde_wasm_bindgen::from_value(rebuild_js).unwrap();
  assert_eq!(rebuild["status"].as_str(), Some("rebuild_required"));
  assert_eq!(rebuild["rebuild_required"].as_bool(), Some(true));
}

#[wasm_bindgen_test]
async fn migrate_index_creates_missing_index() {
  let db = unique_db("searchlite-migrate-create");

  let created_js = Searchlite::migrate_index(db.clone(), schema_v1_json())
    .await
    .unwrap();
  let created: serde_json::Value = serde_wasm_bindgen::from_value(created_js).unwrap();
  assert_eq!(created["status"].as_str(), Some("created"));
  assert_eq!(created["rebuild_performed"].as_bool(), Some(false));

  let idx = Searchlite::init(db, schema_v1_json(), None).await.unwrap();
  let result = idx.search("anything".to_string(), 5, None).unwrap();
  let result_json: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  assert_eq!(result_json["hits"].as_array().unwrap().len(), 0);
}

#[wasm_bindgen_test]
async fn migrate_index_rebuilds_on_schema_change() {
  let db = unique_db("searchlite-migrate-rebuild");
  let idx = Searchlite::init(db.clone(), schema_v1_json(), None)
    .await
    .unwrap();
  let docs = vec![serde_json::json!({ "_id": "doc-1", "body": "keep me if rollback" })];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  let rebuild_js = Searchlite::migrate_index(db.clone(), schema_v2_json())
    .await
    .unwrap();
  let rebuild: serde_json::Value = serde_wasm_bindgen::from_value(rebuild_js).unwrap();
  assert_eq!(rebuild["status"].as_str(), Some("rebuilt"));
  assert_eq!(rebuild["rebuild_performed"].as_bool(), Some(true));

  let reopened = Searchlite::init(db, schema_v2_json(), None).await.unwrap();
  let result = reopened
    .search("rollback".to_string(), 5, Some(true))
    .unwrap();
  let result_json: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  assert_eq!(result_json["hits"].as_array().unwrap().len(), 0);
}

// ---------- New tests ----------

#[wasm_bindgen_test]
async fn plan_migration_for_missing_index_does_not_create_db() {
  // Regression test for the plan_migration_internal read-only fix.
  // Calling plan_migration on an unregistered name must return status
  // "missing" without creating any IndexedDB database or registry entry.
  let missing = unique_db("searchlite-plan-missing");
  let plan_js = Searchlite::plan_migration(missing.clone(), schema_v1_json())
    .await
    .unwrap();
  let plan: serde_json::Value = serde_wasm_bindgen::from_value(plan_js).unwrap();
  assert_eq!(plan["status"].as_str(), Some("missing"));
  assert_eq!(plan["rebuild_required"].as_bool(), Some(false));

  // And the name must NOT appear in the registry.
  let indexes_js = Searchlite::list_indexes().await.unwrap();
  let indexes: Vec<serde_json::Value> = serde_wasm_bindgen::from_value(indexes_js).unwrap();
  assert!(
    !indexes
      .iter()
      .any(|entry| entry["db_name"].as_str() == Some(&missing)),
    "plan_migration must not register the index"
  );
}

#[wasm_bindgen_test]
async fn plan_migration_rejects_invalid_schema_json() {
  let db = unique_db("searchlite-plan-bad-schema");
  let err = Searchlite::plan_migration(db, "not a valid schema".to_string())
    .await
    .expect_err("expected invalid_schema_json error");
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_schema_json");
}

#[wasm_bindgen_test]
async fn migrate_index_rejects_invalid_schema_json() {
  let db = unique_db("searchlite-migrate-bad-schema");
  let err = Searchlite::migrate_index(db, "not a valid schema".to_string())
    .await
    .expect_err("expected invalid_schema_json error");
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_schema_json");
}

#[wasm_bindgen_test]
async fn migrate_index_preserves_registry_entry_on_compatible() {
  // Calling migrate_index with the same schema as the existing index should
  // return status "compatible" and leave the registry unchanged.
  let db = unique_db("searchlite-migrate-compatible");
  let idx = Searchlite::init(db.clone(), schema_v1_json(), None)
    .await
    .unwrap();
  idx.commit().await.unwrap();

  let result_js = Searchlite::migrate_index(db.clone(), schema_v1_json())
    .await
    .unwrap();
  let result: serde_json::Value = serde_wasm_bindgen::from_value(result_js).unwrap();
  assert_eq!(result["status"].as_str(), Some("compatible"));
  assert_eq!(result["rebuild_performed"].as_bool(), Some(false));

  let indexes_js = Searchlite::list_indexes().await.unwrap();
  let indexes: Vec<serde_json::Value> = serde_wasm_bindgen::from_value(indexes_js).unwrap();
  assert!(indexes
    .iter()
    .any(|entry| entry["db_name"].as_str() == Some(&db)));
}
