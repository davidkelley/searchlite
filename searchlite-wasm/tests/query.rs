//! Integration tests for query surface:
//! `search`, `search_request`, `search_request_value`, `mget`, `multi_search`,
//! plus the `*_controlled` and `*_async` variants.

#![cfg(target_arch = "wasm32")]

use searchlite_wasm::Searchlite;
use wasm_bindgen_test::*;

mod common;
use common::{text_schema, unique_db, WasmErrorPayload};

wasm_bindgen_test_configure!(run_in_browser);

fn schema_json() -> String {
  serde_json::to_string(&text_schema()).unwrap()
}

async fn seeded_index(prefix: &str) -> Searchlite {
  let db = unique_db(prefix);
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  let docs = vec![serde_json::json!({ "_id": "doc-1", "body": "hello wasm" })];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
    .unwrap();
  idx.commit().await.unwrap();
  idx
}

// ---------- Moved from src/wasm.rs ----------

#[wasm_bindgen_test]
async fn search_request_roundtrip() {
  let idx = seeded_index("searchlite-search-request").await;
  let request_json = serde_json::json!({
    "query": "hello",
    "limit": 5,
    "return_stored": true,
  })
  .to_string();
  let result = idx.search_request(request_json).unwrap();
  let result_json: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  let hits = result_json["hits"].as_array().unwrap();
  assert_eq!(hits.len(), 1);
}

#[wasm_bindgen_test]
async fn search_defaults_skip_stored_fields() {
  let idx = seeded_index("searchlite-wasm-return-stored-default").await;
  let result = idx.search("hello".to_string(), 5, None).unwrap();
  let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  let hit = &parsed["hits"][0];
  assert!(hit["fields"].is_null());
}

#[wasm_bindgen_test]
async fn search_request_value_respects_return_stored() {
  let idx = seeded_index("searchlite-wasm-return-stored").await;
  let request = serde_json::json!({
    "query": "hello",
    "limit": 5,
    "return_stored": true,
  });
  let request_js = serde_wasm_bindgen::to_value(&request).unwrap();
  let result = idx.search_request_value(request_js).unwrap();
  let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  let hit = &parsed["hits"][0];
  assert!(hit["fields"].is_object());
}

#[wasm_bindgen_test]
async fn search_request_value_controlled_rejects_invalid_timeout() {
  let idx = seeded_index("searchlite-controlled-invalid-timeout").await;
  let req = serde_json::json!({ "query": "hello", "limit": 5, "return_stored": true });
  let err = idx
    .search_request_value_controlled(
      serde_wasm_bindgen::to_value(&req).unwrap(),
      None,
      Some(-1.0),
    )
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_timeout");
}

#[wasm_bindgen_test]
async fn search_request_value_controlled_aborts_with_preaborted_signal() {
  let idx = seeded_index("searchlite-controlled-abort").await;
  let req = serde_json::json!({ "query": "hello", "limit": 5, "return_stored": true });
  let controller = web_sys::AbortController::new().unwrap();
  controller.abort();
  let err = idx
    .search_request_value_controlled(
      serde_wasm_bindgen::to_value(&req).unwrap(),
      Some(controller.signal()),
      None,
    )
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "aborted");
}

#[wasm_bindgen_test]
async fn search_request_value_controlled_times_out() {
  let idx = seeded_index("searchlite-controlled-timeout").await;
  let req = serde_json::json!({ "query": "hello", "limit": 5, "return_stored": true });
  let err = idx
    .search_request_value_controlled(serde_wasm_bindgen::to_value(&req).unwrap(), None, Some(0.0))
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "timeout");
}

#[wasm_bindgen_test]
async fn search_request_value_async_roundtrip() {
  let idx = seeded_index("searchlite-async-search").await;
  let req = serde_json::json!({ "query": "hello", "limit": 5, "return_stored": true });
  let result = idx
    .search_request_value_async(serde_wasm_bindgen::to_value(&req).unwrap(), None, None)
    .await
    .unwrap();
  let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  let hits = parsed["hits"].as_array().unwrap();
  assert_eq!(hits.len(), 1);
}

#[wasm_bindgen_test]
async fn mget_returns_found_missing_and_preserves_order() {
  let db = unique_db("searchlite-mget-order");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  let docs = vec![
    serde_json::json!({ "_id": "doc-1", "body": "alpha" }),
    serde_json::json!({ "_id": "doc-2", "body": "beta" }),
  ];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  let req = serde_json::json!({
    "ids": ["doc-2", "missing", "doc-1"],
    "return_stored": true
  });
  let result = idx
    .mget(serde_wasm_bindgen::to_value(&req).unwrap())
    .unwrap();
  let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  let out = parsed["docs"].as_array().unwrap();
  assert_eq!(out.len(), 3);
  assert_eq!(out[0]["doc_id"], "doc-2");
  assert_eq!(out[1]["doc_id"], "missing");
  assert_eq!(out[2]["doc_id"], "doc-1");
  assert_eq!(out[0]["found"], true);
  assert_eq!(out[1]["found"], false);
  assert_eq!(out[2]["found"], true);
  assert_eq!(out[0]["_source"]["body"], "beta");
  assert!(out[1]["_source"].is_null());
  assert_eq!(out[2]["_source"]["body"], "alpha");
}

#[wasm_bindgen_test]
async fn mget_respects_return_stored_false() {
  let db = unique_db("searchlite-mget-return-stored-false");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  let docs = vec![serde_json::json!({ "_id": "doc-1", "body": "alpha" })];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  let req = serde_json::json!({
    "ids": ["doc-1"],
    "return_stored": false
  });
  let result = idx
    .mget(serde_wasm_bindgen::to_value(&req).unwrap())
    .unwrap();
  let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  let out = parsed["docs"].as_array().unwrap();
  assert_eq!(out.len(), 1);
  assert_eq!(out[0]["found"], true);
  assert!(out[0]["_source"].is_null());
}

#[wasm_bindgen_test]
async fn mget_rejects_invalid_ids() {
  let db = unique_db("searchlite-mget-invalid-id");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  let req = serde_json::json!({
    "ids": ["doc-1", "  "],
    "return_stored": true
  });
  let err = idx
    .mget(serde_wasm_bindgen::to_value(&req).unwrap())
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_id");
}

#[wasm_bindgen_test]
async fn multi_search_returns_ordered_results() {
  let db = unique_db("searchlite-multi-search");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  let docs = vec![
    serde_json::json!({ "_id": "doc-1", "body": "alpha token" }),
    serde_json::json!({ "_id": "doc-2", "body": "beta token" }),
  ];
  idx
    .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
    .unwrap();
  idx.commit().await.unwrap();

  let req = serde_json::json!({
    "searches": [
      { "query": "alpha", "limit": 5, "return_stored": true },
      { "query": "beta", "limit": 5, "return_stored": true }
    ]
  });
  let result = idx
    .multi_search(serde_wasm_bindgen::to_value(&req).unwrap())
    .unwrap();
  let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  let results = parsed["results"].as_array().unwrap();
  assert_eq!(results.len(), 2);
  assert_eq!(results[0]["hits"][0]["fields"]["body"], "alpha token");
  assert_eq!(results[1]["hits"][0]["fields"]["body"], "beta token");
}

#[wasm_bindgen_test]
async fn multi_search_rejects_invalid_request() {
  let db = unique_db("searchlite-multi-search-invalid");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();

  let req = serde_json::json!({ "searches": "not-an-array" });
  let err = idx
    .multi_search(serde_wasm_bindgen::to_value(&req).unwrap())
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_multi_search_request");
}

// ---------- New tests ----------

#[wasm_bindgen_test]
async fn search_controlled_happy_path() {
  // search_controlled mirrors search() but accepts AbortSignal + timeoutMs.
  // Neither fires here — the call should return a normal result.
  let idx = seeded_index("searchlite-controlled-happy").await;
  let result = idx
    .search_controlled(
      "hello".to_string(),
      5,
      Some(true),
      None,
      Some(5_000.0), // generous timeout
    )
    .unwrap();
  let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  assert_eq!(parsed["hits"].as_array().unwrap().len(), 1);
}

#[wasm_bindgen_test]
async fn search_request_controlled_happy_path() {
  let idx = seeded_index("searchlite-request-controlled-happy").await;
  let request_json = serde_json::json!({
    "query": "hello",
    "limit": 5,
    "return_stored": true,
  })
  .to_string();
  let result = idx
    .search_request_controlled(request_json, None, Some(5_000.0))
    .unwrap();
  let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  assert_eq!(parsed["hits"].as_array().unwrap().len(), 1);
}

#[wasm_bindgen_test]
async fn search_request_rejects_invalid_json_string() {
  let idx = seeded_index("searchlite-request-bad-json").await;
  // Malformed JSON — fails before validation as a SearchRequest.
  let err = idx.search_request("not json {".to_string()).unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_search_request");
}

#[wasm_bindgen_test]
async fn search_request_rejects_invalid_search_request_shape() {
  let idx = seeded_index("searchlite-request-bad-shape").await;
  // Valid JSON but wrong shape for SearchRequest (missing `query`, wrong field
  // types).
  let bad = serde_json::json!({ "unexpected": [1, 2, 3] }).to_string();
  let err = idx.search_request(bad).unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_search_request");
}

#[wasm_bindgen_test]
async fn search_request_value_async_propagates_aborted() {
  let idx = seeded_index("searchlite-async-abort").await;
  let req = serde_json::json!({ "query": "hello", "limit": 5, "return_stored": true });
  let controller = web_sys::AbortController::new().unwrap();
  controller.abort();
  let err = idx
    .search_request_value_async(
      serde_wasm_bindgen::to_value(&req).unwrap(),
      Some(controller.signal()),
      None,
    )
    .await
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "aborted");
}

#[wasm_bindgen_test]
async fn search_request_value_async_propagates_timeout() {
  let idx = seeded_index("searchlite-async-timeout").await;
  let req = serde_json::json!({ "query": "hello", "limit": 5, "return_stored": true });
  let err = idx
    .search_request_value_async(serde_wasm_bindgen::to_value(&req).unwrap(), None, Some(0.0))
    .await
    .unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "timeout");
}
