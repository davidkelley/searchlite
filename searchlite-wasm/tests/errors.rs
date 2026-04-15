//! Targeted per-error-type tests.
//!
//! This module exists to give a single scan-able entrypoint for the error
//! surface documented in `docs/wasm-errors.md`. Each test verifies one
//! `err.type` value. Domain-specific files (lifecycle / ingest / query /
//! migration / worker) own the richer end-to-end coverage; this file is the
//! systematic lookup: "which typed error codes have a test?".

#![cfg(target_arch = "wasm32")]

use searchlite_wasm::Searchlite;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::*;

mod common;
use common::{
  call_worker_client_method, demo_worker_assets_available, new_worker_client_instance,
  skip_if_worker_runtime_unavailable, text_schema, unique_db, WasmErrorPayload,
};

wasm_bindgen_test_configure!(run_in_browser);

fn schema_json() -> String {
  serde_json::to_string(&text_schema()).unwrap()
}

/// `invalid_json` — JS-side value cannot be converted to JSON.
/// (Also covered in `ingest_delete_update::add_document_rejects_non_serializable_value`.)
#[wasm_bindgen_test]
async fn invalid_json_on_add_document_with_undefined() {
  let db = unique_db("searchlite-err-invalid-json");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  let err = idx.add_document(JsValue::UNDEFINED).unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_json");
}

/// `invalid_schema_json` — `init` with a malformed schema string.
#[wasm_bindgen_test]
async fn invalid_schema_json_on_init() {
  let db = unique_db("searchlite-err-invalid-schema");
  let err = match Searchlite::init(db, "{ not valid json".to_string(), None).await {
    Err(err) => err,
    Ok(_) => panic!("expected invalid_schema_json"),
  };
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_schema_json");
}

/// `invalid_update_request` — update payload fails shape parsing.
/// (Also covered in `ingest_delete_update::update_document_rejects_non_object_request`.)
#[wasm_bindgen_test]
async fn invalid_update_request_shape() {
  let db = unique_db("searchlite-err-invalid-update");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  let bad = serde_wasm_bindgen::to_value(&serde_json::json!(42)).unwrap();
  let err = idx.update_document(bad).unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_update_request");
}

/// `invalid_mget_request` — mget payload fails shape parsing.
#[wasm_bindgen_test]
async fn invalid_mget_request_shape() {
  let db = unique_db("searchlite-err-invalid-mget");
  let idx = Searchlite::init(db, schema_json(), None).await.unwrap();
  // mget expects `{ ids: string[], return_stored?: bool }`. Passing a scalar
  // or non-object breaks the shape.
  let bad = serde_wasm_bindgen::to_value(&serde_json::json!("not an object")).unwrap();
  let err = idx.mget(bad).unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_mget_request");
}

/// `invalid_argument` — `SearchliteWorkerClient.searchRequest` with a
/// negative `delayMs`. The client pre-validates option shapes and rejects
/// before dispatching to the worker.
#[wasm_bindgen_test]
async fn invalid_argument_on_worker_client_negative_delay() {
  if !demo_worker_assets_available().await {
    return;
  }
  let db = unique_db("searchlite-err-invalid-arg");
  let client = match new_worker_client_instance().await {
    Ok(client) => client,
    Err(err) => {
      if skip_if_worker_runtime_unavailable(&err) {
        return;
      }
      panic!("worker client init failed: {err:?}");
    }
  };

  let init_args = js_sys::Array::new();
  init_args.push(&JsValue::from_str(&db));
  init_args.push(&JsValue::from_str(&schema_json()));
  init_args.push(&JsValue::from_str("indexeddb"));
  let init_res =
    JsFuture::from(call_worker_client_method(&client, "initIndex", &init_args).unwrap()).await;
  if let Err(err) = init_res {
    if skip_if_worker_runtime_unavailable(&err) {
      return;
    }
    panic!("worker client initIndex failed: {err:?}");
  }

  let request = serde_json::json!({ "query": "x", "limit": 5 });
  let options = js_sys::Object::new();
  js_sys::Reflect::set(
    &options,
    &JsValue::from_str("delayMs"),
    &JsValue::from_f64(-1.0),
  )
  .unwrap();
  let search_args = js_sys::Array::new();
  search_args.push(&serde_wasm_bindgen::to_value(&request).unwrap());
  search_args.push(options.as_ref());

  let err = match JsFuture::from(
    call_worker_client_method(&client, "searchRequest", &search_args).unwrap(),
  )
  .await
  {
    Ok(_) => panic!("expected invalid_argument"),
    Err(err) if skip_if_worker_runtime_unavailable(&err) => {
      return;
    }
    Err(err) => err,
  };
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_argument");

  let dispose_args = js_sys::Array::new();
  JsFuture::from(call_worker_client_method(&client, "dispose", &dispose_args).unwrap())
    .await
    .unwrap();
  Searchlite::drop_index(db).await.unwrap();
}
