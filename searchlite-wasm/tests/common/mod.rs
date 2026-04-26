//! Shared fixtures for `searchlite-wasm` integration tests.
//!
//! Each integration test file in `tests/` is compiled as its own crate, and
//! each includes this module via `mod common;`. Not every helper is used by
//! every crate, so `allow(dead_code)` suppresses the usual warnings.

#![cfg(target_arch = "wasm32")]
#![allow(dead_code)]

use futures::channel::oneshot;
use searchlite_core::api::types::KeywordField;
use searchlite_core::Schema;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

/// JS-visible error payload shape. Mirrors the private `WasmErrorPayload`
/// struct in `searchlite-wasm/src/wasm.rs`, reconstructed here so integration
/// tests can decode errors returned by the public API.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct WasmErrorPayload {
  #[serde(rename = "type")]
  pub error_type: String,
  pub reason: String,
}

/// Construct a `{ type, reason }` JS value. Used by the worker harness to
/// build synthetic errors when the underlying JS call fails at a boundary
/// the harness itself enforces (`worker_module_import_error`, etc).
pub fn structured_error(error_type: &str, reason: &str) -> JsValue {
  let payload = WasmErrorPayload {
    error_type: error_type.to_string(),
    reason: reason.to_string(),
  };
  serde_wasm_bindgen::to_value(&payload)
    .unwrap_or_else(|_| JsValue::from_str("failed to serialize wasm error payload"))
}

/// Generate a collision-free DB name scoped to the running test.
pub fn unique_db(name: &str) -> String {
  format!("{name}-{}", js_sys::Date::now() as u64)
}

/// Minimal schema with a single text field named `body`. Default in most tests.
pub fn text_schema() -> Schema {
  Schema::default_text_body()
}

/// Schema used by update/migration tests: text body + nullable keyword `category`.
pub fn text_keyword_schema() -> Schema {
  let mut schema = Schema::default_text_body();
  schema.keyword_fields.push(KeywordField {
    name: "category".to_string(),
    stored: true,
    indexed: true,
    fast: true,
    nullable: true,
  });
  schema
}

/// Schedule a one-shot timer on the main thread. Used to verify that
/// long-running worker searches don't block the main event loop.
pub fn set_timeout_once(ms: i32, callback: impl FnOnce() + 'static) {
  let cb = Closure::once(callback);
  web_sys::window()
    .unwrap()
    .set_timeout_with_callback_and_timeout_and_arguments_0(cb.as_ref().unchecked_ref(), ms)
    .unwrap();
  cb.forget();
}

/// Returns `true` if `fetch(path)` resolves with `response.ok === true`.
pub async fn fetch_ok(path: &str) -> bool {
  let Some(window) = web_sys::window() else {
    return false;
  };
  let response = match JsFuture::from(window.fetch_with_str(path)).await {
    Ok(response) => response,
    Err(_) => return false,
  };
  js_sys::Reflect::get(&response, &JsValue::from_str("ok"))
    .ok()
    .and_then(|value| value.as_bool())
    .unwrap_or(false)
}

/// Worker tests skip gracefully if these assets aren't served by the test harness.
pub async fn demo_worker_assets_available() -> bool {
  fetch_ok("./searchlite-demo-worker.mjs").await
    && fetch_ok("./searchlite-worker-client.mjs").await
    && fetch_ok("./pkg/searchlite_wasm.js").await
}

/// Extract `err.type` from a structured `{ type, reason }` error.
pub fn js_error_type(err: &JsValue) -> Option<String> {
  js_sys::Reflect::get(err, &JsValue::from_str("type"))
    .ok()
    .and_then(|value| value.as_string())
}

/// Match the "worker runtime is not available" error codes. Tests use this to
/// skip (rather than fail) in environments that can't spawn module workers.
pub fn skip_if_worker_runtime_unavailable(err: &JsValue) -> bool {
  matches!(
    js_error_type(err).as_deref(),
    Some("worker_error")
      | Some("worker_spawn_error")
      | Some("worker_module_import_error")
      | Some("worker_client_init_error")
  )
}

/// Dynamically import `searchlite-worker-client.mjs` and construct a
/// `SearchliteWorkerClient` instance.
pub async fn new_worker_client_instance() -> Result<JsValue, JsValue> {
  let module_url = js_sys::eval(
    "new URL('./searchlite-worker-client.mjs', (self.location && self.location.href) || 'http://localhost/').href",
  )
  .ok()
  .and_then(|value| value.as_string())
  .unwrap_or_else(|| "./searchlite-worker-client.mjs".to_string());
  let import_expr = format!(
    "import({})",
    serde_json::to_string(&module_url)
      .unwrap_or_else(|_| "'./searchlite-worker-client.mjs'".into())
  );
  let module_promise = js_sys::eval(&import_expr)
    .map_err(|err| structured_error("worker_module_import_error", &format!("{err:?}")))?
    .dyn_into::<js_sys::Promise>()
    .map_err(|_| {
      structured_error(
        "worker_module_import_error",
        "import did not return a Promise",
      )
    })?;
  let module = JsFuture::from(module_promise)
    .await
    .map_err(|err| structured_error("worker_module_import_error", &format!("{err:?}")))?;
  let ctor = js_sys::Reflect::get(&module, &JsValue::from_str("SearchliteWorkerClient"))?
    .dyn_into::<js_sys::Function>()
    .map_err(|_| {
      structured_error(
        "worker_module_import_error",
        "SearchliteWorkerClient export missing",
      )
    })?;
  js_sys::Reflect::construct(&ctor, &js_sys::Array::new())
    .map_err(|err| structured_error("worker_client_init_error", &format!("{err:?}")))
}

/// Call a method on a `SearchliteWorkerClient` instance reflectively. Returns
/// the Promise so the caller can `JsFuture::from(...)` / `.await` it.
pub fn call_worker_client_method(
  client: &JsValue,
  method: &str,
  args: &js_sys::Array,
) -> Result<js_sys::Promise, JsValue> {
  let method_fn = js_sys::Reflect::get(client, &JsValue::from_str(method))?
    .dyn_into::<js_sys::Function>()
    .map_err(|_| {
      structured_error(
        "worker_client_method_error",
        &format!("missing method {method}"),
      )
    })?;
  let value = method_fn.apply(client, args)?;
  value.dyn_into::<js_sys::Promise>().map_err(|_| {
    structured_error(
      "worker_client_method_error",
      &format!("method {method} did not return a Promise"),
    )
  })
}

/// Spawn the bundled demo worker as a module worker.
pub fn spawn_demo_worker() -> Result<web_sys::Worker, JsValue> {
  let worker_js = js_sys::eval("new Worker('./searchlite-demo-worker.mjs', { type: 'module' })")
    .map_err(|err| structured_error("worker_spawn_error", &format!("{err:?}")))?;
  worker_js
    .dyn_into::<web_sys::Worker>()
    .map_err(|_| structured_error("worker_spawn_error", "failed to cast JS worker"))
}

/// Send a message to the demo worker and await the correlated response.
pub async fn worker_call(
  worker: &web_sys::Worker,
  id: u32,
  action: &str,
  payload: JsValue,
) -> Result<JsValue, JsValue> {
  let (tx, rx) = oneshot::channel::<Result<JsValue, JsValue>>();
  let tx = Rc::new(RefCell::new(Some(tx)));

  let message_tx = tx.clone();
  let message_worker = worker.clone();
  let onmessage =
    Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |event: web_sys::MessageEvent| {
      let data = event.data();
      let msg_id = js_sys::Reflect::get(&data, &JsValue::from_str("id"))
        .ok()
        .and_then(|raw| raw.as_f64())
        .map(|raw| raw as u32);
      if msg_id != Some(id) {
        return;
      }
      let ok = js_sys::Reflect::get(&data, &JsValue::from_str("ok"))
        .ok()
        .and_then(|raw| raw.as_bool())
        .unwrap_or(false);
      let key = if ok { "payload" } else { "error" };
      let value = js_sys::Reflect::get(&data, &JsValue::from_str(key)).unwrap_or(JsValue::NULL);
      if let Some(sender) = message_tx.borrow_mut().take() {
        let _ = sender.send(if ok { Ok(value) } else { Err(value) });
      }
      message_worker.set_onmessage(None);
      message_worker.set_onerror(None);
    });

  let error_tx = tx.clone();
  let error_worker = worker.clone();
  let onerror = Closure::<dyn FnMut(web_sys::Event)>::new(move |event: web_sys::Event| {
    let reason = js_sys::Reflect::get(event.as_ref(), &JsValue::from_str("message"))
      .ok()
      .and_then(|raw| raw.as_string())
      .unwrap_or_else(|| "worker runtime error".to_string());
    if let Some(sender) = error_tx.borrow_mut().take() {
      let _ = sender.send(Err(structured_error("worker_error", &reason)));
    }
    error_worker.set_onmessage(None);
    error_worker.set_onerror(None);
  });

  worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
  worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));
  onmessage.forget();
  onerror.forget();

  let msg = js_sys::Object::new();
  js_sys::Reflect::set(
    &msg,
    &JsValue::from_str("id"),
    &JsValue::from_f64(f64::from(id)),
  )
  .unwrap();
  js_sys::Reflect::set(
    &msg,
    &JsValue::from_str("action"),
    &JsValue::from_str(action),
  )
  .unwrap();
  js_sys::Reflect::set(&msg, &JsValue::from_str("payload"), &payload).unwrap();
  worker.post_message(&msg)?;

  match rx.await {
    Ok(result) => result,
    Err(_) => Err(structured_error(
      "worker_channel_closed",
      "worker response channel closed",
    )),
  }
}
