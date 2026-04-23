//! Integration tests for the worker-first runtime:
//! `searchlite-demo-worker.mjs` and `SearchliteWorkerClient`.
//!
//! Every worker test gates on `demo_worker_assets_available()` so they skip
//! gracefully in environments that can't serve the worker scripts.

#![cfg(target_arch = "wasm32")]

use searchlite_wasm::Searchlite;
use std::cell::Cell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::*;

mod common;
use common::{
  call_worker_client_method, demo_worker_assets_available, new_worker_client_instance,
  set_timeout_once, skip_if_worker_runtime_unavailable, spawn_demo_worker, text_schema, unique_db,
  worker_call, WasmErrorPayload,
};

wasm_bindgen_test_configure!(run_in_browser);

fn schema_json() -> String {
  serde_json::to_string(&text_schema()).unwrap()
}

// ---------- Moved from src/wasm.rs ----------

#[wasm_bindgen_test]
async fn worker_search_request_keeps_main_thread_responsive() {
  if !demo_worker_assets_available().await {
    return;
  }
  let db = unique_db("searchlite-worker-responsive");
  let worker = spawn_demo_worker().unwrap();

  let init_payload = serde_json::json!({
    "dbName": db,
    "schemaJson": schema_json(),
    "storage": "indexeddb",
  });
  let init_result = worker_call(
    &worker,
    1,
    "init_index",
    serde_wasm_bindgen::to_value(&init_payload).unwrap(),
  )
  .await;
  if let Err(err) = init_result {
    if skip_if_worker_runtime_unavailable(&err) {
      worker.terminate();
      return;
    }
    panic!("worker init failed: {err:?}");
  }

  let docs_payload = serde_json::json!({
    "docs": [{ "_id": "doc-1", "body": "hello from worker" }],
  });
  worker_call(
    &worker,
    2,
    "add_documents",
    serde_wasm_bindgen::to_value(&docs_payload).unwrap(),
  )
  .await
  .unwrap();

  let timer_fired = Rc::new(Cell::new(false));
  let timer_fired_ref = Rc::clone(&timer_fired);
  set_timeout_once(20, move || timer_fired_ref.set(true));

  let request_payload = serde_json::json!({
    "request": { "query": "hello", "limit": 5, "return_stored": true },
    "delayMs": 200,
    "timeoutMs": 800,
  });
  let result = match worker_call(
    &worker,
    3,
    "search_request",
    serde_wasm_bindgen::to_value(&request_payload).unwrap(),
  )
  .await
  {
    Ok(result) => result,
    Err(err) if skip_if_worker_runtime_unavailable(&err) => {
      worker.terminate();
      let _ = Searchlite::drop_index(db).await;
      return;
    }
    Err(err) => panic!("worker search failed: {err:?}"),
  };
  let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  assert_eq!(parsed["hits"].as_array().unwrap().len(), 1);
  assert!(
    timer_fired.get(),
    "main thread timer should fire while worker search is in-flight"
  );

  worker.terminate();
  Searchlite::drop_index(db).await.unwrap();
}

#[wasm_bindgen_test]
async fn worker_search_request_timeout_returns_typed_error() {
  if !demo_worker_assets_available().await {
    return;
  }
  let db = unique_db("searchlite-worker-timeout");
  let worker = spawn_demo_worker().unwrap();

  let init_payload = serde_json::json!({
    "dbName": db,
    "schemaJson": schema_json(),
    "storage": "indexeddb",
  });
  let init_result = worker_call(
    &worker,
    1,
    "init_index",
    serde_wasm_bindgen::to_value(&init_payload).unwrap(),
  )
  .await;
  if let Err(err) = init_result {
    if skip_if_worker_runtime_unavailable(&err) {
      worker.terminate();
      return;
    }
    panic!("worker init failed: {err:?}");
  }

  let docs_payload = serde_json::json!({
    "docs": [{ "_id": "doc-1", "body": "timeout path" }],
  });
  worker_call(
    &worker,
    2,
    "add_documents",
    serde_wasm_bindgen::to_value(&docs_payload).unwrap(),
  )
  .await
  .unwrap();

  let request_payload = serde_json::json!({
    "request": { "query": "timeout", "limit": 5, "return_stored": true },
    "delayMs": 200,
    "timeoutMs": 20,
  });
  let err = match worker_call(
    &worker,
    3,
    "search_request",
    serde_wasm_bindgen::to_value(&request_payload).unwrap(),
  )
  .await
  {
    Ok(_) => panic!("expected timeout error from worker search"),
    Err(err) if skip_if_worker_runtime_unavailable(&err) => {
      worker.terminate();
      let _ = Searchlite::drop_index(db).await;
      return;
    }
    Err(err) => err,
  };
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "timeout");

  worker.terminate();
  Searchlite::drop_index(db).await.unwrap();
}

#[wasm_bindgen_test]
async fn worker_client_search_request_abort_returns_typed_error() {
  if !demo_worker_assets_available().await {
    return;
  }
  let db = unique_db("searchlite-worker-client-abort");
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

  let docs = vec![serde_json::json!({ "_id": "doc-1", "body": "abort path" })];
  let add_args = js_sys::Array::new();
  add_args.push(&serde_wasm_bindgen::to_value(&docs).unwrap());
  JsFuture::from(call_worker_client_method(&client, "addDocuments", &add_args).unwrap())
    .await
    .unwrap();

  let controller = web_sys::AbortController::new().unwrap();
  let controller_for_timeout = controller.clone();
  set_timeout_once(20, move || controller_for_timeout.abort());

  let request = serde_json::json!({ "query": "abort", "limit": 5, "return_stored": true });
  let options = js_sys::Object::new();
  js_sys::Reflect::set(
    &options,
    &JsValue::from_str("delayMs"),
    &JsValue::from_f64(200.0),
  )
  .unwrap();
  js_sys::Reflect::set(
    &options,
    &JsValue::from_str("timeoutMs"),
    &JsValue::from_f64(800.0),
  )
  .unwrap();
  js_sys::Reflect::set(
    &options,
    &JsValue::from_str("signal"),
    controller.signal().as_ref(),
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
    Ok(_) => panic!("expected abort error from worker client search"),
    Err(err) if skip_if_worker_runtime_unavailable(&err) => {
      return;
    }
    Err(err) => err,
  };
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "aborted");

  let dispose_args = js_sys::Array::new();
  JsFuture::from(call_worker_client_method(&client, "dispose", &dispose_args).unwrap())
    .await
    .unwrap();
  Searchlite::drop_index(db).await.unwrap();
}

#[wasm_bindgen_test]
async fn worker_client_search_request_rejects_invalid_timeout() {
  if !demo_worker_assets_available().await {
    return;
  }
  let db = unique_db("searchlite-worker-client-invalid-timeout");
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

  let request = serde_json::json!({ "query": "timeout", "limit": 5, "return_stored": true });
  let options = js_sys::Object::new();
  js_sys::Reflect::set(
    &options,
    &JsValue::from_str("timeoutMs"),
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
    Ok(_) => panic!("expected invalid_timeout error from worker client search"),
    Err(err) if skip_if_worker_runtime_unavailable(&err) => {
      return;
    }
    Err(err) => err,
  };
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "invalid_timeout");
  assert!(payload.reason.contains("timeoutMs"));

  let dispose_args = js_sys::Array::new();
  JsFuture::from(call_worker_client_method(&client, "dispose", &dispose_args).unwrap())
    .await
    .unwrap();
  Searchlite::drop_index(db).await.unwrap();
}

#[cfg(not(feature = "threads"))]
#[wasm_bindgen_test]
async fn init_threads_without_feature_returns_typed_error() {
  let db = unique_db("searchlite-threads-disabled");
  let idx = Searchlite::init(db.clone(), schema_json(), None)
    .await
    .unwrap();
  let err = idx.init_threads(None).await.unwrap_err();
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert_eq!(payload.error_type, "threads_feature_disabled");
  Searchlite::drop_index(db).await.unwrap();
}

// ---------- New tests ----------

#[wasm_bindgen_test]
async fn worker_client_add_then_search_roundtrip() {
  if !demo_worker_assets_available().await {
    return;
  }
  let db = unique_db("searchlite-worker-client-roundtrip");
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

  let docs = vec![
    serde_json::json!({ "_id": "doc-1", "body": "alpha via worker" }),
    serde_json::json!({ "_id": "doc-2", "body": "beta via worker" }),
  ];
  let add_args = js_sys::Array::new();
  add_args.push(&serde_wasm_bindgen::to_value(&docs).unwrap());
  JsFuture::from(call_worker_client_method(&client, "addDocuments", &add_args).unwrap())
    .await
    .unwrap();

  let request = serde_json::json!({ "query": "alpha", "limit": 5, "return_stored": true });
  let search_args = js_sys::Array::new();
  search_args.push(&serde_wasm_bindgen::to_value(&request).unwrap());
  search_args.push(&js_sys::Object::new());
  let result =
    JsFuture::from(call_worker_client_method(&client, "searchRequest", &search_args).unwrap())
      .await
      .unwrap();
  let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  assert_eq!(parsed["hits"].as_array().unwrap().len(), 1);
  assert_eq!(parsed["hits"][0]["fields"]["body"], "alpha via worker");

  let dispose_args = js_sys::Array::new();
  JsFuture::from(call_worker_client_method(&client, "dispose", &dispose_args).unwrap())
    .await
    .unwrap();
  Searchlite::drop_index(db).await.unwrap();
}

#[wasm_bindgen_test]
async fn worker_client_reset_index_clears_content() {
  if !demo_worker_assets_available().await {
    return;
  }
  let db = unique_db("searchlite-worker-client-reset");
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
  JsFuture::from(call_worker_client_method(&client, "initIndex", &init_args).unwrap())
    .await
    .unwrap();

  let docs = vec![serde_json::json!({ "_id": "doc-1", "body": "to be reset" })];
  let add_args = js_sys::Array::new();
  add_args.push(&serde_wasm_bindgen::to_value(&docs).unwrap());
  JsFuture::from(call_worker_client_method(&client, "addDocuments", &add_args).unwrap())
    .await
    .unwrap();

  let reset_args = js_sys::Array::new();
  JsFuture::from(call_worker_client_method(&client, "resetIndex", &reset_args).unwrap())
    .await
    .unwrap();

  let request = serde_json::json!({ "query": "reset", "limit": 5, "return_stored": true });
  let search_args = js_sys::Array::new();
  search_args.push(&serde_wasm_bindgen::to_value(&request).unwrap());
  search_args.push(&js_sys::Object::new());
  let result =
    JsFuture::from(call_worker_client_method(&client, "searchRequest", &search_args).unwrap())
      .await
      .unwrap();
  let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  assert_eq!(parsed["hits"].as_array().unwrap().len(), 0);

  let dispose_args = js_sys::Array::new();
  JsFuture::from(call_worker_client_method(&client, "dispose", &dispose_args).unwrap())
    .await
    .unwrap();
  Searchlite::drop_index(db).await.unwrap();
}

#[wasm_bindgen_test]
async fn worker_client_flush_storage_is_idempotent() {
  if !demo_worker_assets_available().await {
    return;
  }
  let db = unique_db("searchlite-worker-client-flush");
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
  JsFuture::from(call_worker_client_method(&client, "initIndex", &init_args).unwrap())
    .await
    .unwrap();

  let empty = js_sys::Array::new();
  // Calling flushStorage twice in a row (with no intervening writes) must
  // not error.
  JsFuture::from(call_worker_client_method(&client, "flushStorage", &empty).unwrap())
    .await
    .unwrap();
  JsFuture::from(call_worker_client_method(&client, "flushStorage", &empty).unwrap())
    .await
    .unwrap();

  let dispose_args = js_sys::Array::new();
  JsFuture::from(call_worker_client_method(&client, "dispose", &dispose_args).unwrap())
    .await
    .unwrap();
  Searchlite::drop_index(db).await.unwrap();
}

#[wasm_bindgen_test]
async fn worker_client_storage_usage_returns_supported_or_note() {
  if !demo_worker_assets_available().await {
    return;
  }
  let client = match new_worker_client_instance().await {
    Ok(client) => client,
    Err(err) => {
      if skip_if_worker_runtime_unavailable(&err) {
        return;
      }
      panic!("worker client init failed: {err:?}");
    }
  };

  let empty = js_sys::Array::new();
  let result = JsFuture::from(call_worker_client_method(&client, "storageUsage", &empty).unwrap())
    .await
    .unwrap();
  let usage: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
  let supported = usage["supported"].as_bool().unwrap_or(false);
  let has_note = usage["note"].is_string();
  assert!(supported || has_note);

  JsFuture::from(call_worker_client_method(&client, "dispose", &empty).unwrap())
    .await
    .unwrap();
}

#[wasm_bindgen_test]
async fn worker_client_dispose_rejects_pending_calls() {
  if !demo_worker_assets_available().await {
    return;
  }
  let db = unique_db("searchlite-worker-client-dispose");
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
  JsFuture::from(call_worker_client_method(&client, "initIndex", &init_args).unwrap())
    .await
    .unwrap();

  let docs = vec![serde_json::json!({ "_id": "doc-1", "body": "pending dispose" })];
  let add_args = js_sys::Array::new();
  add_args.push(&serde_wasm_bindgen::to_value(&docs).unwrap());
  JsFuture::from(call_worker_client_method(&client, "addDocuments", &add_args).unwrap())
    .await
    .unwrap();

  // Kick off a delayed search then dispose quickly. The disposed-then-pending
  // search must reject with a typed error (worker_disposed / worker_crashed /
  // worker_error — whichever the client emits).
  let request = serde_json::json!({ "query": "pending", "limit": 5, "return_stored": true });
  let options = js_sys::Object::new();
  js_sys::Reflect::set(
    &options,
    &JsValue::from_str("delayMs"),
    &JsValue::from_f64(500.0),
  )
  .unwrap();
  let search_args = js_sys::Array::new();
  search_args.push(&serde_wasm_bindgen::to_value(&request).unwrap());
  search_args.push(options.as_ref());
  let pending_promise = call_worker_client_method(&client, "searchRequest", &search_args).unwrap();

  let dispose_args = js_sys::Array::new();
  JsFuture::from(call_worker_client_method(&client, "dispose", &dispose_args).unwrap())
    .await
    .unwrap();

  let err = JsFuture::from(pending_promise)
    .await
    .expect_err("pending search must reject after dispose()");
  let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
  assert!(
    payload.error_type.starts_with("worker_"),
    "expected a worker_* error type after dispose(); got {}",
    payload.error_type
  );

  Searchlite::drop_index(db).await.unwrap();
}
