//! Integration tests for ingest, delete, and update paths:
//! `add_document`, `add_documents`, `commit`, `delete_document`,
//! `delete_documents`, `update_document`.

#![cfg(target_arch = "wasm32")]

use searchlite_wasm::Searchlite;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

mod common;
use common::{text_keyword_schema, text_schema, unique_db, WasmErrorPayload};

wasm_bindgen_test_configure!(run_in_browser);

fn schema_json() -> String {
  serde_json::to_string(&text_schema()).unwrap()
}

// ---------- Moved from src/wasm.rs ----------

#[wasm_bindgen_test]
async fn delete_document_roundtrip() {
  let db = unique_db("searchlite-delete-document");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  let docs = vec![
    serde_json::json!({ "_id": "doc-1", "body": "alpha token" }),
    serde_json::json!({ "_id": "doc-2", "body": "beta token" }),
  ];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  idx.delete_document("doc-1".to_string()).unwrap();
  idx.commit().await.unwrap();

  let deleted = idx.search("alpha".to_string(), 5, Some(true)).unwrap();
  let deleted_json: serde_json::Value = serde_wasm_bindgen::from_value(deleted).unwrap();
  assert_eq!(deleted_json["hits"].as_array().unwrap().len(), 0);

  let retained = idx.search("beta".to_string(), 5, Some(true)).unwrap();
  let retained_json: serde_json::Value = serde_wasm_bindgen::from_value(retained).unwrap();
  assert_eq!(retained_json["hits"].as_array().unwrap().len(), 1);
}

#[wasm_bindgen_test]
async fn delete_documents_roundtrip() {
  let db = unique_db("searchlite-delete-documents");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  let docs = vec![
    serde_json::json!({ "_id": "doc-1", "body": "alpha token" }),
    serde_json::json!({ "_id": "doc-2", "body": "beta token" }),
    serde_json::json!({ "_id": "doc-3", "body": "gamma token" }),
  ];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  let ids = vec!["doc-1", "doc-3"];
  idx
    .delete_documents(serde_wasm_bindgen::to_value(&ids).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  let alpha = idx.search("alpha".to_string(), 5, Some(true)).unwrap();
  let alpha_json: serde_json::Value = serde_wasm_bindgen::from_value(alpha).unwrap();
  assert_eq!(alpha_json["hits"].as_array().unwrap().len(), 0);

  let gamma = idx.search("gamma".to_string(), 5, Some(true)).unwrap();
  let gamma_json: serde_json::Value = serde_wasm_bindgen::from_value(gamma).unwrap();
  assert_eq!(gamma_json["hits"].as_array().unwrap().len(), 0);

  let beta = idx.search("beta".to_string(), 5, Some(true)).unwrap();
  let beta_json: serde_json::Value = serde_wasm_bindgen::from_value(beta).unwrap();
  assert_eq!(beta_json["hits"].as_array().unwrap().len(), 1);
}

#[wasm_bindgen_test]
async fn delete_documents_rejects_non_string_ids() {
  let db = unique_db("searchlite-delete-invalid");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();

  let invalid = serde_json::json!(["doc-1", 42]);
  let err = idx
    .delete_documents(serde_wasm_bindgen::to_value(&invalid).unwrap())
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_doc_id_batch");
}

#[wasm_bindgen_test]
async fn delete_document_rejects_invalid_id() {
  let db = unique_db("searchlite-del-invalid-id");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  let err = idx.delete_document("  ".to_string()).unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_id");
}

#[wasm_bindgen_test]
async fn delete_documents_rejects_invalid_ids() {
  let db = unique_db("searchlite-dels-invalid-id");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  let ids = vec!["valid-doc", "\n"];
  let err = idx
    .delete_documents(serde_wasm_bindgen::to_value(&ids).unwrap())
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_id");
}

#[wasm_bindgen_test]
async fn update_document_set_and_unset_roundtrip() {
  let db = unique_db("searchlite-update-document");
  let schema = text_keyword_schema();
  let schema_json = serde_json::to_string(&schema).unwrap();
  let idx = Searchlite::init(db, schema_json, None).await.unwrap();
  let docs = vec![serde_json::json!({
    "_id": "doc-1",
    "body": "hello patch",
    "category": "guide"
  })];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  let patch = serde_json::json!({
    "id": "doc-1",
    "set": { "body": "updated patch" },
    "unset": ["category"]
  });
  idx
    .update_document(serde_wasm_bindgen::to_value(&patch).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  let result = idx.search("updated".to_string(), 5, Some(true)).unwrap();
  let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  let hits = parsed["hits"].as_array().unwrap();
  assert_eq!(hits.len(), 1);
  assert_eq!(hits[0]["fields"]["body"], "updated patch");
  assert!(hits[0]["fields"]["category"].is_null());
}

#[wasm_bindgen_test]
async fn update_document_rejects_missing_patch() {
  let db = unique_db("searchlite-update-missing-patch");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();

  let patch = serde_json::json!({ "id": "doc-1" });
  let err = idx
    .update_document(serde_wasm_bindgen::to_value(&patch).unwrap())
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "missing_patch");
}

#[wasm_bindgen_test]
async fn update_document_rejects_invalid_id() {
  let db = unique_db("searchlite-update-invalid-id");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();

  let patch = serde_json::json!({
    "id": "   ",
    "set": { "body": "x" }
  });
  let err = idx
    .update_document(serde_wasm_bindgen::to_value(&patch).unwrap())
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_id");
}

#[wasm_bindgen_test]
async fn update_document_reports_not_found() {
  let db = unique_db("searchlite-update-not-found");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();

  let patch = serde_json::json!({
    "id": "missing",
    "set": { "body": "x" }
  });
  let err = idx
    .update_document(serde_wasm_bindgen::to_value(&patch).unwrap())
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "document_not_found");
}

#[wasm_bindgen_test]
async fn update_document_rejects_unknown_field() {
  let db = unique_db("searchlite-update-unknown-field");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  let docs = vec![serde_json::json!({
    "_id": "doc-1",
    "body": "hello patch"
  })];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  let patch = serde_json::json!({
    "id": "doc-1",
    "set": { "unknown": "x" }
  });
  let err = idx
    .update_document(serde_wasm_bindgen::to_value(&patch).unwrap())
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "update_failed");
  assert!(payload.reason.contains("unknown field"));
}

// ---------- New tests ----------

#[wasm_bindgen_test]
async fn add_document_rejects_non_object() {
  let db = unique_db("searchlite-add-non-object");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();

  // A JSON value that parses cleanly but isn't an object.
  let not_an_object =
    serde_wasm_bindgen::to_value(&serde_json::json!(["not", "an", "object"])).unwrap();
  let err = idx.add_document(not_an_object).unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_document");
}

#[wasm_bindgen_test]
async fn add_documents_rejects_non_object_non_array() {
  let db = unique_db("searchlite-addmany-scalar");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();

  // Scalar (number) — not an object, not an array of objects.
  let scalar = serde_wasm_bindgen::to_value(&serde_json::json!(42)).unwrap();
  let err = idx.add_documents(scalar).unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_document_batch");
}

#[wasm_bindgen_test]
async fn add_document_rejects_non_serializable_value() {
  let db = unique_db("searchlite-add-nonserializable");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();

  // `undefined` is not a valid JSON value; `serde_wasm_bindgen::from_value`
  // fails at the JS-to-Rust boundary, producing `invalid_json`.
  let err = idx.add_document(JsValue::UNDEFINED).unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_json");
}

#[wasm_bindgen_test]
async fn update_document_rejects_non_object_request() {
  let db = unique_db("searchlite-update-non-object");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();

  // Request shape must be an object with `id`. An array fails shape parsing.
  let request = serde_wasm_bindgen::to_value(&serde_json::json!(["not", "a", "request"])).unwrap();
  let err = idx.update_document(request).unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_update_request");
}

#[wasm_bindgen_test]
async fn delete_document_with_unknown_id_is_noop_on_commit() {
  // Deleting a doc id that doesn't exist must not fail — commit succeeds and
  // existing docs remain searchable. Callers rely on this for best-effort
  // cleanup paths.
  let db = unique_db("searchlite-delete-unknown");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  idx
    .add_documents(
      serde_wasm_bindgen::to_value(&serde_json::json!([
        { "_id": "doc-1", "body": "existing doc" },
      ]))
      .unwrap(),
    )
    .unwrap();
  idx.commit().await.unwrap();

  idx.delete_document("never-existed".to_string()).unwrap();
  idx.commit().await.unwrap();

  let result = idx.search("existing".to_string(), 5, Some(true)).unwrap();
  let result_json: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  assert_eq!(result_json["hits"].as_array().unwrap().len(), 1);
}
