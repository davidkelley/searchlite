use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use futures::channel::oneshot;
use parking_lot::Mutex;
use parking_lot::RwLock;
use searchlite_core::api::types::{
  Aggregation, ExecutionStrategy, IndexOptions, MgetRequest, MgetResponse, MultiSearchRequest,
  Query, QueryNode, SearchRequest, SortSpec, StorageType,
};
use searchlite_core::api::{Document, IndexReader, IndexWriter, MultiSearchResponse, PatchError};
use searchlite_core::storage::{DynFile, InMemoryStorage, Storage, StorageFile};
use searchlite_core::util::doc_id::validate_doc_id;
use searchlite_core::{Index, Manifest, Schema};
use serde::de::DeserializeOwned;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use wasm_bindgen_futures::JsFuture;
#[cfg(feature = "threads")]
use wasm_bindgen_rayon::init_thread_pool;

const STORE_NAME: &str = "searchlite_files";
const REGISTRY_DB_NAME: &str = "searchlite_registry";
const REGISTRY_STORE_NAME: &str = "indexes";
const META_FILE_NAME: &str = ".searchlite_meta.json";
const SCHEMA_VERSION_V1: u32 = 1;
const IDB_WRITE_BATCH_SIZE: usize = 64;
// BM25 defaults tuned for browser-based search; keep aligned with core defaults.
const BM25_K1: f32 = 0.9;
const BM25_B: f32 = 0.4;
type EventHandler = Rc<RefCell<Option<Closure<dyn FnMut(web_sys::Event)>>>>;

/// Hard ceiling on `from + limit` for any WASM-exported search entrypoint.
/// Browser linear memory is capped at 4 GiB, so an unbounded `limit` from
/// JavaScript can drive the result heap to OOM and abort the
/// WebAssembly.Instance with no recovery path.
///
/// This must stay aligned with `MAX_PAGE_SIZE` in
/// `searchlite-core/src/api/reader.rs` (and `searchlite-http/src/lib.rs`):
/// those layers reject `from + size` above the same cap when `return_hits`
/// is true, so any value the WASM validator lets through still has to fit
/// core's contract. Callers needing more than this should paginate via
/// `cursor` / `search_after`. `candidate_size` is intentionally not
/// capped here — core silently clamps it at `MAX_CANDIDATE_SIZE` and HTTP
/// does the same — so duplicating that as a hard reject would diverge.
pub const WASM_MAX_PAGE_SIZE: usize = 1_000;

thread_local! {
  // Per-thread (per WASM worker) cache of IndexedDB connections.
  // This avoids reconnecting for each persist operation on the same thread.
  static DB_CACHE: RefCell<HashMap<String, web_sys::IdbDatabase>> = RefCell::new(HashMap::new());
}

#[cfg(test)]
thread_local! {
  static MIGRATION_FAIL_AFTER_CLEAR: RefCell<bool> = const { RefCell::new(false) };
  static FORCE_PERSIST_QUOTA_EXCEEDED: RefCell<bool> = const { RefCell::new(false) };
  static PERSIST_BATCH_TX_COUNT: RefCell<u32> = const { RefCell::new(0) };
}

#[cfg(test)]
fn set_migration_fail_after_clear(enabled: bool) {
  MIGRATION_FAIL_AFTER_CLEAR.with(|flag| *flag.borrow_mut() = enabled);
}

#[cfg(test)]
fn migration_fail_after_clear() -> bool {
  MIGRATION_FAIL_AFTER_CLEAR.with(|flag| *flag.borrow())
}

#[cfg(test)]
fn set_force_persist_quota_exceeded(enabled: bool) {
  FORCE_PERSIST_QUOTA_EXCEEDED.with(|flag| *flag.borrow_mut() = enabled);
}

#[cfg(test)]
fn force_persist_quota_exceeded() -> bool {
  FORCE_PERSIST_QUOTA_EXCEEDED.with(|flag| *flag.borrow())
}

#[cfg(test)]
fn reset_persist_batch_tx_count() {
  PERSIST_BATCH_TX_COUNT.with(|count| *count.borrow_mut() = 0);
}

#[cfg(test)]
fn persist_batch_tx_count() -> u32 {
  PERSIST_BATCH_TX_COUNT.with(|count| *count.borrow())
}

#[cfg(test)]
fn bump_persist_batch_tx_count() {
  PERSIST_BATCH_TX_COUNT.with(|count| *count.borrow_mut() += 1);
}

#[cfg(not(test))]
fn migration_fail_after_clear() -> bool {
  false
}

#[cfg(not(test))]
fn force_persist_quota_exceeded() -> bool {
  false
}

#[derive(Clone, Copy)]
enum StorageMode {
  IndexedDb,
  Memory,
}

impl StorageMode {
  fn parse(raw: Option<String>) -> Result<Self, JsValue> {
    match raw.as_deref() {
      None => Ok(Self::IndexedDb),
      Some(value) if value.eq_ignore_ascii_case("indexeddb") => Ok(Self::IndexedDb),
      Some(value) if value.eq_ignore_ascii_case("memory") => Ok(Self::Memory),
      Some(_) => Err(js_error(
        "invalid_argument",
        "storage must be 'indexeddb' or 'memory'",
      )),
    }
  }
}

enum StorageBackend {
  IndexedDb(Arc<JsStorage>),
  Memory,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct WasmErrorPayload {
  #[serde(rename = "type")]
  error_type: String,
  reason: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct IndexRegistryEntry {
  db_name: String,
  schema_version: u32,
  schema_hash: String,
  updated_at_ms: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct IndexMeta {
  schema_version: u32,
  schema_hash: String,
}

fn js_error(error_type: &str, reason: impl Into<String>) -> JsValue {
  let payload = WasmErrorPayload {
    error_type: error_type.to_string(),
    reason: reason.into(),
  };
  serde_wasm_bindgen::to_value(&payload)
    .unwrap_or_else(|_| JsValue::from_str("failed to serialize wasm error payload"))
}

fn to_js_error(err: impl std::fmt::Display) -> JsValue {
  js_error("internal_error", err.to_string())
}

fn typed_js_error(error_type: &str, err: impl std::fmt::Display) -> JsValue {
  js_error(error_type, err.to_string())
}

fn js_error_reason(err: &JsValue) -> String {
  if let Some(reason) = js_sys::Reflect::get(err, &JsValue::from_str("reason"))
    .ok()
    .and_then(|value| value.as_string())
  {
    return reason;
  }
  err
    .as_string()
    .unwrap_or_else(|| format!("non-string js error: {err:?}"))
}

fn map_update_error(err: anyhow::Error) -> JsValue {
  if let Some(patch_err) = err.downcast_ref::<PatchError>() {
    return match patch_err {
      PatchError::DocumentNotFound => js_error("document_not_found", err.to_string()),
      PatchError::VectorFieldsUnsupported => js_error("vector_fields_unsupported", err.to_string()),
    };
  }
  js_error("update_failed", err.to_string())
}

fn is_quota_exceeded_reason(reason: &str) -> bool {
  let lower = reason.to_ascii_lowercase();
  lower.contains("quotaexceedederror") || (lower.contains("quota") && lower.contains("exceed"))
}

fn map_storage_error(err: anyhow::Error, fallback_type: &str, action: &str) -> JsValue {
  let reason = err.to_string();
  if is_quota_exceeded_reason(&reason) {
    return js_error(
      "quota_exceeded",
      format!(
        "indexeddb quota exceeded while {action}; run compact(), remove stale indexes with Searchlite.cleanup_indexes(...), or clear/drop unused indexes before retrying. detail: {reason}"
      ),
    );
  }
  js_error(fallback_type, reason)
}

fn parse_timeout_ms(timeout_ms: Option<f64>) -> Result<Option<f64>, JsValue> {
  match timeout_ms {
    None => Ok(None),
    Some(ms) if ms.is_finite() && ms >= 0.0 => Ok(Some(ms)),
    _ => Err(js_error(
      "invalid_timeout",
      "timeout_ms must be a non-negative finite number",
    )),
  }
}

fn ensure_not_aborted(signal: Option<&web_sys::AbortSignal>) -> Result<(), JsValue> {
  if signal.map(|sig| sig.aborted()).unwrap_or(false) {
    return Err(js_error("aborted", "operation aborted by AbortSignal"));
  }
  Ok(())
}

fn ensure_not_timed_out(started_ms: f64, timeout_ms: Option<f64>) -> Result<(), JsValue> {
  let Some(limit) = timeout_ms else {
    return Ok(());
  };
  let elapsed_ms = now_ms() - started_ms;
  if elapsed_ms >= limit {
    return Err(js_error(
      "timeout",
      format!("operation exceeded timeout_ms={limit} (elapsed={elapsed_ms:.2}ms)"),
    ));
  }
  Ok(())
}

fn parse_request_value<T>(value: JsValue, error_type: &str) -> Result<T, JsValue>
where
  T: DeserializeOwned,
{
  match serde_wasm_bindgen::from_value::<T>(value.clone()) {
    Ok(parsed) => Ok(parsed),
    Err(primary_err) => {
      let fallback_json: serde_json::Value =
        serde_wasm_bindgen::from_value(value).map_err(|json_err| {
          typed_js_error(
            error_type,
            format!("primary decode failed: {primary_err}; json fallback failed: {json_err}"),
          )
        })?;
      serde_json::from_value::<T>(fallback_json).map_err(|json_err| {
        typed_js_error(
          error_type,
          format!("primary decode failed: {primary_err}; json struct decode failed: {json_err}"),
        )
      })
    }
  }
}

impl StorageBackend {
  async fn flush(&self) -> Result<()> {
    match self {
      Self::IndexedDb(storage) => storage.flush().await,
      Self::Memory => Ok(()),
    }
  }
}

fn indexed_db_factory() -> Result<web_sys::IdbFactory> {
  let global = js_sys::global();
  let idb = js_sys::Reflect::get(&global, &JsValue::from_str("indexedDB"))
    .map_err(|_| anyhow!("IndexedDB unavailable"))?;
  if idb.is_null() || idb.is_undefined() {
    return Err(anyhow!("IndexedDB unavailable"));
  }
  idb
    .dyn_into::<web_sys::IdbFactory>()
    .map_err(|_| anyhow!("IndexedDB unavailable"))
}

#[cfg(feature = "threads")]
fn hardware_concurrency() -> u32 {
  let global = js_sys::global();
  let navigator = js_sys::Reflect::get(&global, &JsValue::from_str("navigator"))
    .ok()
    .and_then(|value| value.dyn_into::<web_sys::Navigator>().ok());
  navigator
    .map(|nav| nav.hardware_concurrency() as u32)
    .filter(|&count| count > 0)
    .unwrap_or(1)
}

fn path_key(path: &Path) -> String {
  path.to_string_lossy().to_string()
}

/// Reject WASM search requests whose `limit`, `from`, or `from + limit`
/// exceed [`WASM_MAX_PAGE_SIZE`]. A JS caller that passes `limit = u32::MAX`
/// would otherwise drive the result heap until the WebAssembly linear
/// memory aborts the module with no recovery path.
///
/// The cap is gated on `req.return_hits` to mirror `IndexReader::search()`
/// in `searchlite-core` and `validate_search` in `searchlite-http`: those
/// values only drive the top-k result heap when hits are actually
/// returned, so aggregation-only or metadata queries (which set
/// `return_hits = false`) can legitimately use larger pagination values
/// without WASM rejecting requests that core and HTTP would accept.
/// `candidate_size` is intentionally not validated here — core silently
/// clamps it at `MAX_CANDIDATE_SIZE` and HTTP does not check it either,
/// so duplicating that as a hard reject would diverge from both layers.
fn validate_search_limits(req: &SearchRequest) -> Result<(), JsValue> {
  if !req.return_hits {
    return Ok(());
  }
  if req.limit > WASM_MAX_PAGE_SIZE {
    return Err(js_error(
      "invalid_search_request",
      format!(
        "limit {} exceeds max page size {WASM_MAX_PAGE_SIZE}",
        req.limit
      ),
    ));
  }
  if req.from > WASM_MAX_PAGE_SIZE {
    return Err(js_error(
      "invalid_search_request",
      format!(
        "from {} exceeds max page size {WASM_MAX_PAGE_SIZE}",
        req.from
      ),
    ));
  }
  if req.from.saturating_add(req.limit) > WASM_MAX_PAGE_SIZE {
    return Err(js_error(
      "invalid_search_request",
      format!(
        "from + limit ({}) exceeds max page size {WASM_MAX_PAGE_SIZE}",
        req.from.saturating_add(req.limit),
      ),
    ));
  }
  Ok(())
}

/// Reject a `limit` argument supplied directly by JS to the convenience
/// `search()` entrypoint, which always uses `from = 0`.
fn validate_search_limit_arg(limit: usize) -> Result<(), JsValue> {
  if limit > WASM_MAX_PAGE_SIZE {
    return Err(js_error(
      "invalid_search_request",
      format!("limit {limit} exceeds max page size {WASM_MAX_PAGE_SIZE}"),
    ));
  }
  Ok(())
}

fn value_to_document(value: serde_json::Value) -> Result<Document, JsValue> {
  let obj = value
    .as_object()
    .ok_or_else(|| js_error("invalid_document", "document must be a JSON object"))?;
  let mut fields = BTreeMap::new();
  for (k, v) in obj.iter() {
    fields.insert(k.clone(), v.clone());
  }
  Ok(Document { fields })
}

fn value_to_documents(value: serde_json::Value) -> Result<Vec<Document>, JsValue> {
  match value {
    serde_json::Value::Array(items) => items.into_iter().map(value_to_document).collect(),
    obj @ serde_json::Value::Object(_) => Ok(vec![value_to_document(obj)?]),
    _ => Err(js_error(
      "invalid_document_batch",
      "documents must be an object or array of objects",
    )),
  }
}

fn value_to_doc_ids(value: serde_json::Value) -> Result<Vec<String>, JsValue> {
  match value {
    serde_json::Value::String(id) => Ok(vec![id]),
    serde_json::Value::Array(items) => {
      let mut ids = Vec::with_capacity(items.len());
      for item in items {
        let serde_json::Value::String(id) = item else {
          return Err(js_error(
            "invalid_doc_id_batch",
            "document ids must be a string or array of strings",
          ));
        };
        ids.push(id);
      }
      Ok(ids)
    }
    _ => Err(js_error(
      "invalid_doc_id_batch",
      "document ids must be a string or array of strings",
    )),
  }
}

fn clear_request_handlers(req: &web_sys::IdbRequest, success: &EventHandler, error: &EventHandler) {
  req.set_onsuccess(None);
  req.set_onerror(None);
  success.borrow_mut().take();
  error.borrow_mut().take();
}

fn dom_exception_to_anyhow(context: &str, dom: &web_sys::DomException) -> anyhow::Error {
  anyhow!("{context}: {} ({})", dom.name(), dom.message())
}

fn js_error_to_anyhow(context: &str, err: &JsValue) -> anyhow::Error {
  if let Ok(dom) = err.clone().dyn_into::<web_sys::DomException>() {
    return dom_exception_to_anyhow(context, &dom);
  }
  anyhow!("{context}: {:?}", err)
}

fn request_future(req: &web_sys::IdbRequest) -> impl std::future::Future<Output = Result<JsValue>> {
  let (tx, rx) = oneshot::channel::<Result<JsValue>>();
  let sender = Rc::new(RefCell::new(Some(tx)));
  let success_handler: EventHandler = Rc::new(RefCell::new(None));
  let error_handler: EventHandler = Rc::new(RefCell::new(None));
  let success_req_for_closure = req.clone();
  let success_req_for_handler = req.clone();
  let error_req_for_closure = req.clone();
  let error_req_for_handler = req.clone();

  let success_handler_clone = success_handler.clone();
  let error_handler_clone = error_handler.clone();
  let sender_clone = sender.clone();
  let success = Closure::wrap(Box::new(move |event: web_sys::Event| {
    let result = (|| {
      if let Some(target) = event.target() {
        if let Ok(req) = target.dyn_into::<web_sys::IdbRequest>() {
          if let Ok(result) = req.result() {
            return Ok(result);
          }
        }
      }
      Err(anyhow!("indexeddb request missing result"))
    })();
    if let Some(tx) = sender_clone.borrow_mut().take() {
      let _ = tx.send(result);
    }
    clear_request_handlers(
      &success_req_for_closure,
      &success_handler_clone,
      &error_handler_clone,
    );
  }) as Box<dyn FnMut(_)>);
  *success_handler.borrow_mut() = Some(success);
  success_req_for_handler.set_onsuccess(Some(
    success_handler
      .borrow()
      .as_ref()
      .expect("success handler set")
      .as_ref()
      .unchecked_ref(),
  ));

  let success_handler_clone = success_handler.clone();
  let error_handler_clone = error_handler.clone();
  let sender_clone = sender.clone();
  let error = Closure::wrap(Box::new(move |_event: web_sys::Event| {
    let err = match error_req_for_closure.error() {
      Ok(Some(dom)) => dom_exception_to_anyhow("indexeddb request error", &dom),
      Ok(None) => anyhow!("indexeddb request error"),
      Err(raw) => js_error_to_anyhow("indexeddb request error", &raw),
    };
    if let Some(tx) = sender_clone.borrow_mut().take() {
      let _ = tx.send(Err(err));
    }
    clear_request_handlers(
      &error_req_for_closure,
      &success_handler_clone,
      &error_handler_clone,
    );
  }) as Box<dyn FnMut(_)>);
  *error_handler.borrow_mut() = Some(error);
  error_req_for_handler.set_onerror(Some(
    error_handler
      .borrow()
      .as_ref()
      .expect("error handler set")
      .as_ref()
      .unchecked_ref(),
  ));

  async move {
    match rx.await {
      Ok(result) => result,
      Err(_) => Err(anyhow!("indexeddb request canceled")),
    }
  }
}

fn clear_transaction_handlers(
  tx: &web_sys::IdbTransaction,
  complete: &EventHandler,
  error: &EventHandler,
  abort: &EventHandler,
) {
  tx.set_oncomplete(None);
  tx.set_onerror(None);
  tx.set_onabort(None);
  complete.borrow_mut().take();
  error.borrow_mut().take();
  abort.borrow_mut().take();
}

fn transaction_future(
  tx: &web_sys::IdbTransaction,
) -> impl std::future::Future<Output = Result<()>> {
  let (tx_done, rx) = oneshot::channel::<Result<()>>();
  let sender = Rc::new(RefCell::new(Some(tx_done)));
  let complete_handler: EventHandler = Rc::new(RefCell::new(None));
  let error_handler: EventHandler = Rc::new(RefCell::new(None));
  let abort_handler: EventHandler = Rc::new(RefCell::new(None));

  let tx_complete_for_closure = tx.clone();
  let tx_complete_for_handler = tx.clone();
  let tx_error_for_closure = tx.clone();
  let tx_error_for_handler = tx.clone();
  let tx_abort_for_closure = tx.clone();
  let tx_abort_for_handler = tx.clone();

  let complete_handler_clone = complete_handler.clone();
  let error_handler_clone = error_handler.clone();
  let abort_handler_clone = abort_handler.clone();
  let sender_clone = sender.clone();
  let complete = Closure::wrap(Box::new(move |_event: web_sys::Event| {
    if let Some(done) = sender_clone.borrow_mut().take() {
      let _ = done.send(Ok(()));
    }
    clear_transaction_handlers(
      &tx_complete_for_closure,
      &complete_handler_clone,
      &error_handler_clone,
      &abort_handler_clone,
    );
  }) as Box<dyn FnMut(_)>);
  *complete_handler.borrow_mut() = Some(complete);
  tx_complete_for_handler.set_oncomplete(Some(
    complete_handler
      .borrow()
      .as_ref()
      .expect("transaction complete handler set")
      .as_ref()
      .unchecked_ref(),
  ));

  let complete_handler_clone = complete_handler.clone();
  let error_handler_clone = error_handler.clone();
  let abort_handler_clone = abort_handler.clone();
  let sender_clone = sender.clone();
  let error = Closure::wrap(Box::new(move |_event: web_sys::Event| {
    let err = tx_error_for_closure
      .error()
      .map(|dom| dom_exception_to_anyhow("indexeddb transaction error", &dom))
      .unwrap_or_else(|| anyhow!("indexeddb transaction error"));
    if let Some(done) = sender_clone.borrow_mut().take() {
      let _ = done.send(Err(err));
    }
    clear_transaction_handlers(
      &tx_error_for_closure,
      &complete_handler_clone,
      &error_handler_clone,
      &abort_handler_clone,
    );
  }) as Box<dyn FnMut(_)>);
  *error_handler.borrow_mut() = Some(error);
  tx_error_for_handler.set_onerror(Some(
    error_handler
      .borrow()
      .as_ref()
      .expect("transaction error handler set")
      .as_ref()
      .unchecked_ref(),
  ));

  let complete_handler_clone = complete_handler.clone();
  let error_handler_clone = error_handler.clone();
  let abort_handler_clone = abort_handler.clone();
  let sender_clone = sender.clone();
  let abort = Closure::wrap(Box::new(move |_event: web_sys::Event| {
    let err = tx_abort_for_closure
      .error()
      .map(|dom| dom_exception_to_anyhow("indexeddb transaction aborted", &dom))
      .unwrap_or_else(|| anyhow!("indexeddb transaction aborted"));
    if let Some(done) = sender_clone.borrow_mut().take() {
      let _ = done.send(Err(err));
    }
    clear_transaction_handlers(
      &tx_abort_for_closure,
      &complete_handler_clone,
      &error_handler_clone,
      &abort_handler_clone,
    );
  }) as Box<dyn FnMut(_)>);
  *abort_handler.borrow_mut() = Some(abort);
  tx_abort_for_handler.set_onabort(Some(
    abort_handler
      .borrow()
      .as_ref()
      .expect("transaction abort handler set")
      .as_ref()
      .unchecked_ref(),
  ));

  async move {
    match rx.await {
      Ok(result) => result,
      Err(_) => Err(anyhow!("indexeddb transaction wait canceled")),
    }
  }
}

fn now_ms() -> f64 {
  js_sys::Date::now()
}

fn schema_hash(schema: &Schema) -> Result<String, JsValue> {
  let schema_json =
    serde_json::to_vec(schema).map_err(|err| typed_js_error("schema_serialization_error", err))?;
  // Use a deterministic FNV-1a hash so persisted schema fingerprints remain stable
  // across process restarts and Rust/toolchain upgrades.
  let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
  for byte in schema_json {
    hash ^= byte as u64;
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
  }
  Ok(format!("{hash:016x}"))
}

fn manifest_doc_counts(manifest: &Manifest) -> (u64, u64) {
  let mut total_docs = 0u64;
  let mut deleted_docs = 0u64;
  for seg in manifest.segments.iter() {
    total_docs = total_docs.saturating_add(seg.doc_count as u64);
    deleted_docs = deleted_docs.saturating_add(seg.deleted_docs.len() as u64);
  }
  (total_docs.saturating_sub(deleted_docs), deleted_docs)
}

async fn open_db_with_store(name: &str, store_name: &str) -> Result<web_sys::IdbDatabase> {
  if let Some(db) = DB_CACHE.with(|cache| cache.borrow().get(name).cloned()) {
    return Ok(db);
  }
  let factory = indexed_db_factory()?;
  let request = factory
    .open_with_u32(name, 1)
    .map_err(|e| anyhow!("indexed_db open error: {:?}", e))?;
  let store = store_name.to_string();
  let upgrade = Closure::wrap(Box::new(move |event: web_sys::Event| {
    if let Some(target) = event.target() {
      if let Ok(req) = target.dyn_into::<web_sys::IdbOpenDbRequest>() {
        if let Ok(result) = req.result() {
          if let Ok(db) = result.dyn_into::<web_sys::IdbDatabase>() {
            if let Err(e) = db.create_object_store(&store) {
              web_sys::console::error_1(&JsValue::from_str(&format!(
                "Failed to create IndexedDB object store '{store}': {e:?}"
              )));
            }
          }
        }
      }
    }
  }) as Box<dyn FnMut(_)>);
  request.set_onupgradeneeded(Some(upgrade.as_ref().unchecked_ref()));
  let request_handle: web_sys::IdbRequest = request.clone().into();
  let db_value = request_future(&request_handle).await?;
  request.set_onupgradeneeded(None);
  drop(upgrade);
  let db = db_value
    .dyn_into::<web_sys::IdbDatabase>()
    .map_err(|_| anyhow!("failed to open IndexedDB database"))?;
  DB_CACHE.with(|cache| {
    cache.borrow_mut().insert(name.to_string(), db.clone());
  });
  Ok(db)
}

fn close_cached_db(name: &str) {
  DB_CACHE.with(|cache| {
    if let Some(db) = cache.borrow_mut().remove(name) {
      db.close();
    }
  });
}

async fn open_data_db(name: &str) -> Result<web_sys::IdbDatabase> {
  open_db_with_store(name, STORE_NAME).await
}

async fn open_registry_db() -> Result<web_sys::IdbDatabase> {
  open_db_with_store(REGISTRY_DB_NAME, REGISTRY_STORE_NAME).await
}

async fn delete_database(name: &str) -> Result<()> {
  close_cached_db(name);
  let factory = indexed_db_factory()?;
  let request = factory
    .delete_database(name)
    .map_err(|e| anyhow!("indexed_db delete_database error: {:?}", e))?;
  let request_handle: web_sys::IdbRequest = request.into();
  request_future(&request_handle).await?;
  Ok(())
}

async fn clear_data_store(db_name: &str) -> Result<()> {
  let db = open_data_db(db_name).await?;
  let tx = db
    .transaction_with_str_and_mode(STORE_NAME, web_sys::IdbTransactionMode::Readwrite)
    .map_err(|e| anyhow!("opening read-write transaction for {STORE_NAME}: {:?}", e))?;
  let store = tx
    .object_store(STORE_NAME)
    .map_err(|e| anyhow!("opening object store {STORE_NAME}: {:?}", e))?;
  let tx_done = transaction_future(&tx);
  let req = store
    .clear()
    .map_err(|e| anyhow!("clear failed for {STORE_NAME}: {:?}", e))?;
  request_future(&req).await?;
  tx_done.await
}

async fn upsert_registry_entry(entry: &IndexRegistryEntry) -> Result<()> {
  let db = open_registry_db().await?;
  let tx = db
    .transaction_with_str_and_mode(REGISTRY_STORE_NAME, web_sys::IdbTransactionMode::Readwrite)
    .map_err(|e| {
      anyhow!(
        "opening read-write transaction for {REGISTRY_STORE_NAME}: {:?}",
        e
      )
    })?;
  let store = tx
    .object_store(REGISTRY_STORE_NAME)
    .map_err(|e| anyhow!("opening object store {REGISTRY_STORE_NAME}: {:?}", e))?;
  let key = JsValue::from_str(&entry.db_name);
  let value = serde_wasm_bindgen::to_value(entry)
    .map_err(|e| anyhow!("serializing registry entry failed: {e}"))?;
  let tx_done = transaction_future(&tx);
  let req = store
    .put_with_key(&value, &key)
    .map_err(|e| anyhow!("put registry entry failed: {:?}", e))?;
  request_future(&req).await?;
  tx_done.await
}

async fn remove_registry_entry(db_name: &str) -> Result<()> {
  let db = open_registry_db().await?;
  let tx = db
    .transaction_with_str_and_mode(REGISTRY_STORE_NAME, web_sys::IdbTransactionMode::Readwrite)
    .map_err(|e| {
      anyhow!(
        "opening read-write transaction for {REGISTRY_STORE_NAME}: {:?}",
        e
      )
    })?;
  let tx_done = transaction_future(&tx);
  let store = tx
    .object_store(REGISTRY_STORE_NAME)
    .map_err(|e| anyhow!("opening object store {REGISTRY_STORE_NAME}: {:?}", e))?;
  let key = JsValue::from_str(db_name);
  let req = store
    .delete(&key)
    .map_err(|e| anyhow!("delete registry entry failed: {:?}", e))?;
  request_future(&req).await?;
  tx_done.await
}

async fn get_registry_entry(db_name: &str) -> Result<Option<IndexRegistryEntry>> {
  let db = open_registry_db().await?;
  let tx = db
    .transaction_with_str_and_mode(REGISTRY_STORE_NAME, web_sys::IdbTransactionMode::Readonly)
    .map_err(|e| anyhow!("opening transaction for {REGISTRY_STORE_NAME}: {:?}", e))?;
  let store = tx
    .object_store(REGISTRY_STORE_NAME)
    .map_err(|e| anyhow!("opening object store {REGISTRY_STORE_NAME}: {:?}", e))?;
  let key = JsValue::from_str(db_name);
  let req = store
    .get(&key)
    .map_err(|e| anyhow!("registry get failed: {:?}", e))?;
  let row = request_future(&req).await?;
  if row.is_null() || row.is_undefined() {
    return Ok(None);
  }
  let entry: IndexRegistryEntry =
    serde_wasm_bindgen::from_value(row).map_err(|e| anyhow!("registry decode failed: {e}"))?;
  Ok(Some(entry))
}

async fn restore_registry_entry(db_name: &str, entry: Option<&IndexRegistryEntry>) -> Result<()> {
  match entry {
    Some(existing) => upsert_registry_entry(existing).await,
    None => remove_registry_entry(db_name).await,
  }
}

async fn list_registry_entries() -> Result<Vec<IndexRegistryEntry>> {
  let db = open_registry_db().await?;
  let tx = db
    .transaction_with_str_and_mode(REGISTRY_STORE_NAME, web_sys::IdbTransactionMode::Readonly)
    .map_err(|e| anyhow!("opening transaction for {REGISTRY_STORE_NAME}: {:?}", e))?;
  let store = tx
    .object_store(REGISTRY_STORE_NAME)
    .map_err(|e| anyhow!("opening object store {REGISTRY_STORE_NAME}: {:?}", e))?;
  let req = store
    .get_all()
    .map_err(|e| anyhow!("registry get_all failed: {:?}", e))?;
  let val = request_future(&req).await?;
  let rows: js_sys::Array = val
    .dyn_into()
    .map_err(|_| anyhow!("registry get_all expected an array"))?;
  let mut entries = Vec::with_capacity(rows.length() as usize);
  for row in rows.iter() {
    if row.is_undefined() || row.is_null() {
      continue;
    }
    let entry: IndexRegistryEntry = serde_wasm_bindgen::from_value(row)
      .map_err(|e| anyhow!("registry entry decode failed: {e}"))?;
    entries.push(entry);
  }
  entries.sort_by(|a, b| a.db_name.cmp(&b.db_name));
  Ok(entries)
}

#[derive(Debug)]
enum PersistOperation {
  Put(Vec<u8>),
  Delete,
}

fn chunk_operations(
  operations: Vec<(PathBuf, PersistOperation)>,
  chunk_size: usize,
) -> Vec<Vec<(PathBuf, PersistOperation)>> {
  let mut chunks = Vec::new();
  let mut current = Vec::with_capacity(chunk_size.max(1));
  for operation in operations {
    current.push(operation);
    if current.len() >= chunk_size {
      chunks.push(current);
      current = Vec::with_capacity(chunk_size.max(1));
    }
  }
  if !current.is_empty() {
    chunks.push(current);
  }
  chunks
}

async fn persist_operations_batch(
  db_name: &str,
  operations: &[(PathBuf, PersistOperation)],
) -> Result<()> {
  if operations.is_empty() {
    return Ok(());
  }
  if force_persist_quota_exceeded() {
    return Err(anyhow!(
      "QuotaExceededError: synthetic quota failure for wasm persistence test"
    ));
  }
  #[cfg(test)]
  bump_persist_batch_tx_count();

  let db = open_data_db(db_name).await?;
  let tx = db
    .transaction_with_str_and_mode(STORE_NAME, web_sys::IdbTransactionMode::Readwrite)
    .map_err(|e| anyhow!("opening read-write transaction for {STORE_NAME}: {:?}", e))?;
  let tx_done = transaction_future(&tx);
  let store = tx
    .object_store(STORE_NAME)
    .map_err(|e| anyhow!("opening object store {STORE_NAME}: {:?}", e))?;
  // Queue every request synchronously before yielding. Awaiting each request
  // individually can let Chrome auto-close the transaction between awaits.
  for (path, operation) in operations.iter() {
    let key = JsValue::from_str(&path_key(path));
    match operation {
      PersistOperation::Put(data) => {
        let value: JsValue = js_sys::Uint8Array::from(data.as_slice()).into();
        store
          .put_with_key(&value, &key)
          .map_err(|e| anyhow!("put_with_key failed for {:?}: {:?}", path, e))?;
      }
      PersistOperation::Delete => {
        store
          .delete(&key)
          .map_err(|e| anyhow!("delete failed for {:?}: {:?}", path, e))?;
      }
    };
  }
  tx_done.await
}

async fn load_snapshot(db_name: &str) -> Result<HashMap<PathBuf, Vec<u8>>> {
  let db = open_data_db(db_name).await?;
  let tx = db
    .transaction_with_str_and_mode(STORE_NAME, web_sys::IdbTransactionMode::Readonly)
    .map_err(|e| anyhow!("opening transaction for {STORE_NAME}: {:?}", e))?;
  let tx_done = transaction_future(&tx);
  let store = tx
    .object_store(STORE_NAME)
    .map_err(|e| anyhow!("opening object store {STORE_NAME}: {:?}", e))?;
  let keys_req = store
    .get_all_keys()
    .map_err(|e| anyhow!("get_all_keys failed: {:?}", e))?;
  let values_req = store
    .get_all()
    .map_err(|e| anyhow!("get_all failed: {:?}", e))?;
  let keys_val = request_future(&keys_req).await?;
  let values_val = request_future(&values_req).await?;
  tx_done.await?;

  let keys: js_sys::Array = keys_val
    .dyn_into()
    .map_err(|_| anyhow!("get_all_keys expected an array"))?;
  let values: js_sys::Array = values_val
    .dyn_into()
    .map_err(|_| anyhow!("get_all expected an array"))?;

  let keys_len = keys.length() as usize;
  let values_len = values.length() as usize;
  let paired_len = keys_len.min(values_len);
  if keys_len != values_len {
    web_sys::console::warn_1(&JsValue::from_str(&format!(
      "IndexedDB snapshot load for '{db_name}' returned mismatched key/value counts: keys={keys_len}, values={values_len}",
    )));
  }

  let mut map = HashMap::with_capacity(paired_len);
  for idx in 0..paired_len {
    let key = keys.get(idx as u32);
    let Some(name) = key.as_string() else {
      web_sys::console::warn_1(&JsValue::from_str(&format!(
        "IndexedDB snapshot load for '{db_name}' skipped non-string key: {key:?}",
      )));
      continue;
    };
    let value = values.get(idx as u32);
    let bytes = js_sys::Uint8Array::new(&value).to_vec();
    map.insert(PathBuf::from(name), bytes);
  }
  Ok(map)
}

async fn list_stored_paths(db_name: &str) -> Result<Vec<PathBuf>> {
  let db = open_data_db(db_name).await?;
  let tx = db
    .transaction_with_str_and_mode(STORE_NAME, web_sys::IdbTransactionMode::Readonly)
    .map_err(|e| anyhow!("opening transaction for {STORE_NAME}: {:?}", e))?;
  let tx_done = transaction_future(&tx);
  let store = tx
    .object_store(STORE_NAME)
    .map_err(|e| anyhow!("opening object store {STORE_NAME}: {:?}", e))?;
  let req = store
    .get_all_keys()
    .map_err(|e| anyhow!("get_all_keys failed: {:?}", e))?;
  let keys = request_future(&req).await?;
  tx_done.await?;
  let rows: js_sys::Array = keys
    .dyn_into()
    .map_err(|_| anyhow!("get_all_keys expected an array"))?;
  let mut paths = Vec::with_capacity(rows.length() as usize);
  for row in rows.iter() {
    if let Some(name) = row.as_string() {
      paths.push(PathBuf::from(name));
    }
  }
  Ok(paths)
}

async fn restore_snapshot(db_name: &str, snapshot: &HashMap<PathBuf, Vec<u8>>) -> Result<()> {
  clear_data_store(db_name).await?;
  let mut operations: Vec<_> = snapshot
    .iter()
    .map(|(path, bytes)| (path.clone(), PersistOperation::Put(bytes.clone())))
    .collect();
  operations.sort_by(|(left, _), (right, _)| left.cmp(right));
  for batch in chunk_operations(operations, IDB_WRITE_BATCH_SIZE) {
    persist_operations_batch(db_name, &batch).await?;
  }
  Ok(())
}

#[derive(Clone)]
struct PendingWrites {
  db_name: String,
  pending: Arc<Mutex<Vec<oneshot::Receiver<Result<()>>>>>,
  state: Arc<Mutex<PendingQueueState>>,
}

struct PendingQueueState {
  queue: BTreeMap<PathBuf, PendingEntry>,
  worker_running: bool,
}

struct PendingEntry {
  operation: PersistOperation,
  waiters: Vec<oneshot::Sender<Result<()>>>,
}

impl PendingWrites {
  fn new(db_name: String) -> Self {
    Self {
      db_name,
      pending: Arc::new(Mutex::new(Vec::new())),
      state: Arc::new(Mutex::new(PendingQueueState {
        queue: BTreeMap::new(),
        worker_running: false,
      })),
    }
  }

  fn schedule(&self, path: PathBuf, data: Vec<u8>) {
    self.enqueue(path, PersistOperation::Put(data));
  }

  fn schedule_delete(&self, path: PathBuf) {
    self.enqueue(path, PersistOperation::Delete);
  }

  fn enqueue(&self, path: PathBuf, operation: PersistOperation) {
    let (tx, rx) = oneshot::channel();
    self.pending.lock().push(rx);
    let mut guard = self.state.lock();
    if let Some(entry) = guard.queue.get_mut(&path) {
      entry.operation = operation;
      entry.waiters.push(tx);
    } else {
      guard.queue.insert(
        path,
        PendingEntry {
          operation,
          waiters: vec![tx],
        },
      );
    }
    if guard.worker_running {
      return;
    }
    guard.worker_running = true;
    let db_name = self.db_name.clone();
    let state = self.state.clone();
    spawn_local(async move {
      persist_queue_worker(db_name, state).await;
    });
  }

  async fn flush(&self) -> Result<()> {
    let mut first_error = None;
    // If a previous batch failed and re-queued operations, spawn a new worker
    // to retry before draining receivers. Attach a waiter to the LAST queued
    // entry so we block until the worker either completes the entire queue
    // or fails (in which case all remaining waiters are drained with the
    // error by the worker's failure path). We only make ONE retry attempt
    // per flush() call — a persistent failure returns an error immediately
    // instead of looping.
    {
      let mut guard = self.state.lock();
      if !guard.queue.is_empty() && !guard.worker_running {
        let (tx, rx) = oneshot::channel();
        if let Some(entry) = guard.queue.values_mut().next_back() {
          entry.waiters.push(tx);
          self.pending.lock().push(rx);
        } else {
          drop(tx);
        }
        guard.worker_running = true;
        let db_name = self.db_name.clone();
        let state = self.state.clone();
        spawn_local(async move {
          persist_queue_worker(db_name, state).await;
        });
      }
    }
    // Drain all pending receivers (from enqueues plus any retry waiter).
    let receivers = {
      let mut guard = self.pending.lock();
      std::mem::take(&mut *guard)
    };
    for rx in receivers {
      match rx.await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
          if first_error.is_none() {
            first_error = Some(err);
          }
        }
        Err(_) => {
          if first_error.is_none() {
            first_error = Some(anyhow!("pending persist dropped"));
          }
        }
      }
    }
    if let Some(err) = first_error {
      Err(err)
    } else {
      Ok(())
    }
  }
}

async fn persist_queue_worker(db_name: String, state: Arc<Mutex<PendingQueueState>>) {
  loop {
    let batch_entries = {
      let mut guard = state.lock();
      if guard.queue.is_empty() {
        guard.worker_running = false;
        return;
      }
      let keys: Vec<PathBuf> = guard
        .queue
        .keys()
        .take(IDB_WRITE_BATCH_SIZE)
        .cloned()
        .collect();
      let mut entries = Vec::with_capacity(keys.len());
      for key in keys {
        if let Some(entry) = guard.queue.remove(&key) {
          entries.push((key, entry));
        }
      }
      entries
    };

    let mut operations = Vec::with_capacity(batch_entries.len());
    let mut waiter_sets = Vec::with_capacity(batch_entries.len());
    for (path, entry) in batch_entries {
      operations.push((path, entry.operation));
      waiter_sets.push(entry.waiters);
    }
    let result = persist_operations_batch(&db_name, &operations).await;
    let err_msg = result.as_ref().err().map(|err| err.to_string());
    if let Some(msg) = &err_msg {
      web_sys::console::error_1(&JsValue::from_str(&format!("persist batch error: {msg}")));
      // Re-insert failed operations so a subsequent flush can retry.
      // Do not overwrite entries added concurrently during this batch.
      // Stop the worker after re-queuing to avoid an infinite retry loop;
      // the next explicit flush() will spawn a new worker.
      // Also drain any waiters attached to remaining queued entries (batches
      // that were never attempted) so flush() can return the error without
      // hanging or looping.
      let mut pending_waiters: Vec<oneshot::Sender<Result<()>>> = Vec::new();
      {
        let mut guard = state.lock();
        for (path, op) in operations {
          guard.queue.entry(path).or_insert_with(|| PendingEntry {
            operation: op,
            waiters: Vec::new(),
          });
        }
        for entry in guard.queue.values_mut() {
          pending_waiters.append(&mut entry.waiters);
        }
        guard.worker_running = false;
      }
      for waiters in waiter_sets {
        for tx in waiters {
          let _ = tx.send(Err(anyhow!(msg.clone())));
        }
      }
      for tx in pending_waiters {
        let _ = tx.send(Err(anyhow!(msg.clone())));
      }
      return;
    }
    for waiters in waiter_sets {
      for tx in waiters {
        let send_result = match &err_msg {
          Some(msg) => Err(anyhow!(msg.clone())),
          None => Ok(()),
        };
        let _ = tx.send(send_result);
      }
    }
  }
}

pub struct JsStorage {
  root: PathBuf,
  files: RwLock<HashMap<PathBuf, Arc<RwLock<Vec<u8>>>>>,
  pending: PendingWrites,
}

impl JsStorage {
  pub async fn new(db_name: String, root: PathBuf) -> Result<Self> {
    let snapshot = load_snapshot(&db_name).await?;
    let mut files = HashMap::new();
    for (path, data) in snapshot {
      files.insert(path, Arc::new(RwLock::new(data)));
    }
    Ok(Self {
      root,
      files: RwLock::new(files),
      pending: PendingWrites::new(db_name),
    })
  }

  fn entry(&self, path: &Path) -> Arc<RwLock<Vec<u8>>> {
    let mut map = self.files.write();
    map
      .entry(path.to_path_buf())
      .or_insert_with(|| Arc::new(RwLock::new(Vec::new())))
      .clone()
  }

  fn schedule_persist(&self, path: PathBuf, data: Vec<u8>) {
    self.pending.schedule(path, data);
  }

  fn remove_cached_paths(&self, paths: &[PathBuf]) {
    let mut guard = self.files.write();
    for path in paths {
      guard.remove(path);
    }
  }

  pub async fn flush(&self) -> Result<()> {
    self.pending.flush().await
  }
}

impl Storage for JsStorage {
  fn root(&self) -> &Path {
    &self.root
  }

  fn ensure_dir(&self, _path: &Path) -> Result<()> {
    Ok(())
  }

  fn exists(&self, path: &Path) -> bool {
    self.files.read().contains_key(path)
  }

  fn open_read(&self, path: &Path) -> Result<DynFile> {
    let data = self
      .files
      .read()
      .get(path)
      .cloned()
      .ok_or_else(|| anyhow!("file {:?} missing", path))?;
    Ok(Box::new(JsFile {
      path: path.to_path_buf(),
      data,
      pos: 0,
      pending: self.pending.clone(),
      dirty: false,
    }))
  }

  fn open_write(&self, path: &Path) -> Result<DynFile> {
    self.open_with_mode(path, true, false)
  }

  fn open_append(&self, path: &Path) -> Result<DynFile> {
    self.open_with_mode(path, false, true)
  }

  fn read_to_end(&self, path: &Path) -> Result<Vec<u8>> {
    if let Some(buf) = self.files.read().get(path) {
      return Ok(buf.read().clone());
    }
    Err(anyhow!("file {:?} missing", path))
  }

  fn write_all(&self, path: &Path, data: &[u8]) -> Result<()> {
    let entry = self.entry(path);
    let mut guard = entry.write();
    guard.clear();
    guard.extend_from_slice(data);
    self.schedule_persist(path.to_path_buf(), guard.clone());
    Ok(())
  }

  fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()> {
    self.write_all(path, data)
  }

  fn remove(&self, path: &Path) -> Result<()> {
    self.files.write().remove(path);
    self.pending.schedule_delete(path.to_path_buf());
    Ok(())
  }

  fn remove_dir_all(&self, path: &Path) -> Result<()> {
    let to_remove: Vec<PathBuf> = self
      .files
      .read()
      .keys()
      .filter(|p| p.starts_with(path))
      .cloned()
      .collect();
    let mut guard = self.files.write();
    for p in to_remove {
      guard.remove(&p);
      self.pending.schedule_delete(p);
    }
    Ok(())
  }
}

impl JsStorage {
  fn open_with_mode(&self, path: &Path, truncate: bool, append: bool) -> Result<DynFile> {
    let data = self.entry(path);
    let mut dirty = false;
    if truncate {
      data.write().clear();
      dirty = true;
    }
    let pos = if append { data.read().len() as u64 } else { 0 };
    Ok(Box::new(JsFile {
      path: path.to_path_buf(),
      data,
      pos,
      pending: self.pending.clone(),
      dirty,
    }))
  }
}

struct JsFile {
  path: PathBuf,
  data: Arc<RwLock<Vec<u8>>>,
  pos: u64,
  pending: PendingWrites,
  dirty: bool,
}

impl Drop for JsFile {
  fn drop(&mut self) {
    if self.dirty {
      let data = self.data.read().clone();
      self.pending.schedule(self.path.clone(), data);
    }
  }
}

impl std::io::Read for JsFile {
  fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
    let data = self.data.read();
    if self.pos as usize >= data.len() {
      return Ok(0);
    }
    let available = data.len() - self.pos as usize;
    let len = available.min(buf.len());
    buf[..len].copy_from_slice(&data[self.pos as usize..self.pos as usize + len]);
    self.pos += len as u64;
    Ok(len)
  }
}

impl std::io::Write for JsFile {
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    let mut data = self.data.write();
    let buf_len = buf.len() as u64;
    let max_usize = usize::MAX as u64;
    if self.pos > max_usize || buf_len > max_usize - self.pos {
      return Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "write would overflow address space",
      ));
    }
    let end = (self.pos as usize) + buf.len();
    if end > data.len() {
      data.resize(end, 0);
    }
    data[self.pos as usize..end].copy_from_slice(buf);
    self.pos = end as u64;
    self.dirty = true;
    Ok(buf.len())
  }

  fn flush(&mut self) -> std::io::Result<()> {
    // Flushing schedules an async persist; use `flush_storage` to await completion.
    if self.dirty {
      let data = self.data.read().clone();
      self.pending.schedule(self.path.clone(), data);
      self.dirty = false;
    }
    Ok(())
  }
}

impl std::io::Seek for JsFile {
  fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
    let new = match pos {
      std::io::SeekFrom::Start(off) => off as i64,
      std::io::SeekFrom::End(off) => self.data.read().len() as i64 + off,
      std::io::SeekFrom::Current(off) => self.pos as i64 + off,
    };
    if new < 0 {
      return Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "negative seek",
      ));
    }
    self.pos = new as u64;
    Ok(self.pos)
  }
}

impl StorageFile for JsFile {
  fn set_len(&mut self, len: u64) -> Result<()> {
    let mut data = self.data.write();
    data.resize(len as usize, 0);
    if self.pos > len {
      self.pos = len;
    }
    self.dirty = true;
    Ok(())
  }

  fn sync_all(&mut self) -> Result<()> {
    if self.dirty {
      let data = self.data.read().clone();
      self.pending.schedule(self.path.clone(), data);
      self.dirty = false;
    }
    Ok(())
  }
}

fn meta_path(root: &Path) -> PathBuf {
  root.join(META_FILE_NAME)
}

fn write_index_meta(storage: &dyn Storage, root: &Path, schema: &Schema) -> Result<(), JsValue> {
  let meta = IndexMeta {
    schema_version: SCHEMA_VERSION_V1,
    schema_hash: schema_hash(schema)?,
  };
  let json = serde_json::to_vec(&meta).map_err(|err| typed_js_error("meta_encode_error", err))?;
  storage
    .write_all(&meta_path(root), &json)
    .map_err(|err| typed_js_error("storage_write_error", err))
}

fn read_index_meta(storage: &dyn Storage, root: &Path) -> Result<Option<IndexMeta>, JsValue> {
  let path = meta_path(root);
  if !storage.exists(&path) {
    return Ok(None);
  }
  let bytes = storage
    .read_to_end(&path)
    .map_err(|err| typed_js_error("storage_read_error", err))?;
  let meta = serde_json::from_slice::<IndexMeta>(&bytes)
    .map_err(|err| typed_js_error("meta_decode_error", err))?;
  Ok(Some(meta))
}

fn schema_mismatch_error() -> JsValue {
  js_error(
    "schema_mismatch",
    "schema mismatch for existing index; use Searchlite.plan_migration(...) and Searchlite.migrate_index(...), or clear/drop the index",
  )
}

fn schemas_match(existing: &Schema, requested: &Schema) -> Result<bool, JsValue> {
  serde_json::to_value(existing)
    .and_then(|existing_json| {
      serde_json::to_value(requested).map(|requested_json| existing_json == requested_json)
    })
    .map_err(|err| typed_js_error("schema_serialization_error", err))
}

fn open_opts(path: PathBuf) -> IndexOptions {
  IndexOptions {
    path,
    create_if_missing: false,
    enable_positions: true,
    bm25_k1: BM25_K1,
    bm25_b: BM25_B,
    storage: StorageType::InMemory,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  }
}

fn registry_entry(db_name: &str, schema: &Schema) -> Result<IndexRegistryEntry, JsValue> {
  Ok(IndexRegistryEntry {
    db_name: db_name.to_string(),
    schema_version: SCHEMA_VERSION_V1,
    schema_hash: schema_hash(schema)?,
    updated_at_ms: now_ms(),
  })
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct MigrationPlan {
  db_name: String,
  status: String,
  rebuild_required: bool,
  schema_version: u32,
  existing_schema_hash: Option<String>,
  requested_schema_hash: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct MigrationExecutionResult {
  db_name: String,
  status: String,
  rebuild_performed: bool,
  schema_version: u32,
  existing_schema_hash: Option<String>,
  requested_schema_hash: String,
}

#[derive(Debug, serde::Deserialize)]
struct UpdateRequestPayload {
  id: String,
  #[serde(default)]
  set: BTreeMap<String, serde_json::Value>,
  #[serde(default)]
  unset: Vec<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CompactResponse {
  compacted: bool,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct InspectResponse {
  manifest: Manifest,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StatsResponse {
  documents: u64,
  deleted_documents: u64,
  segments: usize,
  committed_at: String,
  index_uuid: String,
  index_path: String,
  index_name: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StorageUsageResponse {
  supported: bool,
  usage_bytes: Option<u64>,
  quota_bytes: Option<u64>,
  remaining_bytes: Option<u64>,
  persisted: Option<bool>,
  #[serde(skip_serializing_if = "Option::is_none")]
  note: Option<String>,
}

impl StorageUsageResponse {
  fn unsupported(note: impl Into<String>) -> Self {
    Self {
      supported: false,
      usage_bytes: None,
      quota_bytes: None,
      remaining_bytes: None,
      persisted: None,
      note: Some(note.into()),
    }
  }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CleanupIndexesResponse {
  scanned: usize,
  matched: usize,
  dropped: Vec<String>,
  kept: Vec<String>,
  dry_run: bool,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CleanupOrphanedFilesResponse {
  scanned: usize,
  orphaned: usize,
  removed: Vec<String>,
  dry_run: bool,
}

fn f64_to_u64_bytes(value: f64) -> Option<u64> {
  if !value.is_finite() || value < 0.0 {
    return None;
  }
  Some(value.floor() as u64)
}

async fn browser_storage_usage() -> Result<StorageUsageResponse, JsValue> {
  // Use Reflect to access navigator.storage so this works in both Window and
  // Worker contexts (Window exposes Navigator, Workers expose WorkerNavigator,
  // but both provide a StorageManager via navigator.storage).
  let global = js_sys::global();
  let navigator = match js_sys::Reflect::get(&global, &JsValue::from_str("navigator")) {
    Ok(nav) if !nav.is_undefined() && !nav.is_null() => nav,
    _ => {
      return Ok(StorageUsageResponse::unsupported(
        "navigator is unavailable",
      ));
    }
  };
  let storage = match js_sys::Reflect::get(&navigator, &JsValue::from_str("storage")) {
    Ok(s) if !s.is_undefined() && !s.is_null() => s,
    _ => {
      return Ok(StorageUsageResponse::unsupported(
        "navigator.storage is unavailable",
      ));
    }
  };
  let estimate_fn = match js_sys::Reflect::get(&storage, &JsValue::from_str("estimate")) {
    Ok(f) if f.is_function() => f.dyn_into::<js_sys::Function>().unwrap(),
    _ => {
      return Ok(StorageUsageResponse::unsupported(
        "storage.estimate is unavailable",
      ));
    }
  };
  let estimate_promise = match estimate_fn.call0(&storage) {
    Ok(promise) => promise,
    Err(err) => {
      return Ok(StorageUsageResponse::unsupported(format!(
        "storage estimate unavailable: {err:?}"
      )));
    }
  };
  let estimate_value = match JsFuture::from(js_sys::Promise::from(estimate_promise)).await {
    Ok(value) => value,
    Err(err) => {
      return Ok(StorageUsageResponse::unsupported(format!(
        "storage estimate failed: {err:?}"
      )));
    }
  };

  let usage = js_sys::Reflect::get(&estimate_value, &JsValue::from_str("usage"))
    .ok()
    .and_then(|value| value.as_f64())
    .and_then(f64_to_u64_bytes);
  let quota = js_sys::Reflect::get(&estimate_value, &JsValue::from_str("quota"))
    .ok()
    .and_then(|value| value.as_f64())
    .and_then(f64_to_u64_bytes);
  let remaining = match (usage, quota) {
    (Some(used), Some(limit)) => Some(limit.saturating_sub(used)),
    _ => None,
  };
  // Check navigator.storage.persisted() via Reflect as well.
  let persisted = match js_sys::Reflect::get(&storage, &JsValue::from_str("persisted")) {
    Ok(f) if f.is_function() => {
      let func = f.dyn_into::<js_sys::Function>().unwrap();
      match func.call0(&storage) {
        Ok(promise) => match JsFuture::from(js_sys::Promise::from(promise)).await {
          Ok(value) => value.as_bool(),
          Err(_) => None,
        },
        Err(_) => None,
      }
    }
    _ => None,
  };
  Ok(StorageUsageResponse {
    supported: true,
    usage_bytes: usage,
    quota_bytes: quota,
    remaining_bytes: remaining,
    persisted,
    note: None,
  })
}

fn expected_live_paths(root: &Path, manifest: &Manifest) -> (HashSet<PathBuf>, Vec<PathBuf>) {
  let mut exact = HashSet::new();
  exact.insert(Manifest::manifest_path(root));
  exact.insert(meta_path(root));
  exact.insert(root.join("wal.log"));
  #[cfg(feature = "vectors")]
  let mut prefixes = Vec::new();
  #[cfg(not(feature = "vectors"))]
  let prefixes: Vec<PathBuf> = Vec::new();
  for seg in manifest.segments.iter() {
    exact.insert(PathBuf::from(seg.paths.terms.clone()));
    exact.insert(PathBuf::from(seg.paths.postings.clone()));
    exact.insert(PathBuf::from(seg.paths.docstore.clone()));
    exact.insert(PathBuf::from(seg.paths.fast.clone()));
    exact.insert(PathBuf::from(seg.paths.meta.clone()));
    #[cfg(feature = "vectors")]
    if let Some(vector_dir) = seg.paths.vector_dir.as_ref() {
      prefixes.push(PathBuf::from(vector_dir));
    }
  }
  (exact, prefixes)
}

fn path_is_live(path: &Path, live_exact: &HashSet<PathBuf>, live_prefixes: &[PathBuf]) -> bool {
  if live_exact.contains(path) {
    return true;
  }
  live_prefixes.iter().any(|prefix| path.starts_with(prefix))
}

#[wasm_bindgen]
pub struct Searchlite {
  db_name: String,
  schema_hash: String,
  index: Index,
  storage: StorageBackend,
}

#[wasm_bindgen]
impl Searchlite {
  async fn create(
    db_name: String,
    schema_json: String,
    storage_mode: StorageMode,
  ) -> Result<Searchlite, JsValue> {
    if db_name == REGISTRY_DB_NAME {
      return Err(js_error(
        "reserved_name",
        format!(
          "'{REGISTRY_DB_NAME}' is reserved for internal use and cannot be used as an index name"
        ),
      ));
    }
    let schema: Schema = serde_json::from_str(&schema_json)
      .map_err(|err| typed_js_error("invalid_schema_json", err))?;
    let requested_schema_hash = schema_hash(&schema)?;
    let root = PathBuf::from(db_name.clone());
    let (storage, backend) = match storage_mode {
      StorageMode::IndexedDb => {
        let storage = Arc::new(
          JsStorage::new(db_name.clone(), root.clone())
            .await
            .map_err(|err| typed_js_error("storage_open_error", err))?,
        );
        (
          storage.clone() as Arc<dyn Storage>,
          StorageBackend::IndexedDb(storage),
        )
      }
      StorageMode::Memory => (
        Arc::new(InMemoryStorage::new(root.clone())) as Arc<dyn Storage>,
        StorageBackend::Memory,
      ),
    };
    let opts = IndexOptions {
      path: root.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: BM25_K1,
      bm25_b: BM25_B,
      // The wasm Index always uses in-memory storage; JsStorage persists to IndexedDB when enabled.
      // Do not mix storage modes for the same db_name; use a fresh name or clear stored data.
      storage: StorageType::InMemory,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    };
    let manifest_path = root.join("MANIFEST.json");
    let index = if storage.exists(&manifest_path) {
      let open_opts = IndexOptions {
        create_if_missing: false,
        ..opts.clone()
      };
      let index = Index::open_with_storage(open_opts, storage.clone())
        .map_err(|err| typed_js_error("index_open_error", err))?;
      let existing_schema = index.manifest().schema;
      if !schemas_match(&existing_schema, &schema)? {
        return Err(schema_mismatch_error());
      }
      index
    } else {
      Index::create_with_storage(&root, schema.clone(), opts, storage.clone())
        .map_err(|err| typed_js_error("index_create_error", err))?
    };
    write_index_meta(storage.as_ref(), &root, &schema)?;
    if matches!(storage_mode, StorageMode::IndexedDb) {
      let entry = registry_entry(&db_name, &schema)?;
      upsert_registry_entry(&entry)
        .await
        .map_err(|err| typed_js_error("registry_write_error", err))?;
    }
    backend
      .flush()
      .await
      .map_err(|err| map_storage_error(err, "storage_flush_error", "initializing index storage"))?;
    Ok(Searchlite {
      db_name,
      schema_hash: requested_schema_hash,
      index,
      storage: backend,
    })
  }

  /// Public WASM-exported async constructor; delegates to the internal `create` helper.
  /// `db_name` is used for both the IndexedDB database name and the virtual root path.
  /// Pass `"indexeddb"` (default) or `"memory"` to choose the storage backend.
  /// When using `"indexeddb"`, the index itself stays in memory and JsStorage persists snapshots.
  /// Avoid switching storage modes for the same `db_name`; use a new name or clear storage.
  #[wasm_bindgen(js_name = init)]
  pub async fn init(
    db_name: String,
    schema_json: String,
    storage: Option<String>,
  ) -> Result<Searchlite, JsValue> {
    let storage_mode = StorageMode::parse(storage)?;
    Self::create(db_name, schema_json, storage_mode).await
  }

  #[wasm_bindgen(js_name = list_indexes)]
  pub async fn list_indexes() -> Result<JsValue, JsValue> {
    let entries = list_registry_entries()
      .await
      .map_err(|err| typed_js_error("registry_read_error", err))?;
    serde_wasm_bindgen::to_value(&entries).map_err(|err| typed_js_error("serialization_error", err))
  }

  #[wasm_bindgen(js_name = clear_index)]
  pub async fn clear_index(db_name: String) -> Result<(), JsValue> {
    if db_name == REGISTRY_DB_NAME {
      return Err(js_error(
        "reserved_name",
        format!("'{REGISTRY_DB_NAME}' is reserved for internal use"),
      ));
    }
    clear_data_store(&db_name)
      .await
      .map_err(|err| typed_js_error("storage_clear_error", err))?;
    Ok(())
  }

  #[wasm_bindgen(js_name = drop_index)]
  pub async fn drop_index(db_name: String) -> Result<(), JsValue> {
    if db_name == REGISTRY_DB_NAME {
      return Err(js_error(
        "reserved_name",
        format!("'{REGISTRY_DB_NAME}' is reserved for internal use"),
      ));
    }
    delete_database(&db_name)
      .await
      .map_err(|err| typed_js_error("storage_delete_error", err))?;
    remove_registry_entry(&db_name)
      .await
      .map_err(|err| typed_js_error("registry_delete_error", err))?;
    Ok(())
  }

  #[wasm_bindgen(js_name = storage_usage)]
  pub async fn storage_usage() -> Result<JsValue, JsValue> {
    let usage = browser_storage_usage().await?;
    serde_wasm_bindgen::to_value(&usage).map_err(|err| typed_js_error("serialization_error", err))
  }

  #[wasm_bindgen(js_name = cleanup_indexes)]
  pub async fn cleanup_indexes(
    stale_older_than_ms: f64,
    dry_run: Option<bool>,
  ) -> Result<JsValue, JsValue> {
    if !stale_older_than_ms.is_finite() || stale_older_than_ms < 0.0 {
      return Err(js_error(
        "invalid_cleanup_request",
        "stale_older_than_ms must be a non-negative number",
      ));
    }
    let dry_run = dry_run.unwrap_or(false);
    let now = now_ms();
    let entries = list_registry_entries()
      .await
      .map_err(|err| typed_js_error("registry_read_error", err))?;
    let scanned = entries.len();
    let mut matched = 0usize;
    let mut dropped = Vec::new();
    let mut kept = Vec::new();
    for entry in entries {
      let age = if now >= entry.updated_at_ms {
        now - entry.updated_at_ms
      } else {
        0.0
      };
      if age < stale_older_than_ms {
        kept.push(entry.db_name);
        continue;
      }
      matched += 1;
      let db_name = entry.db_name.clone();
      if !dry_run {
        delete_database(&db_name)
          .await
          .map_err(|err| typed_js_error("storage_delete_error", err))?;
        remove_registry_entry(&db_name)
          .await
          .map_err(|err| typed_js_error("registry_delete_error", err))?;
      }
      dropped.push(db_name);
    }
    serde_wasm_bindgen::to_value(&CleanupIndexesResponse {
      scanned,
      matched,
      dropped,
      kept,
      dry_run,
    })
    .map_err(|err| typed_js_error("serialization_error", err))
  }

  #[wasm_bindgen(js_name = cleanup_orphaned_files)]
  pub async fn cleanup_orphaned_files(&self, dry_run: Option<bool>) -> Result<JsValue, JsValue> {
    let dry_run = dry_run.unwrap_or(false);
    let StorageBackend::IndexedDb(storage) = &self.storage else {
      return serde_wasm_bindgen::to_value(&CleanupOrphanedFilesResponse {
        scanned: 0,
        orphaned: 0,
        removed: Vec::new(),
        dry_run,
      })
      .map_err(|err| typed_js_error("serialization_error", err));
    };

    let root = PathBuf::from(self.db_name.clone());
    let manifest = self.index.manifest();
    let (live_exact, live_prefixes) = expected_live_paths(&root, &manifest);
    let mut stored_paths = list_stored_paths(&self.db_name)
      .await
      .map_err(|err| typed_js_error("storage_list_error", err))?;
    stored_paths.sort();
    stored_paths.dedup();
    let scanned = stored_paths.len();
    let orphaned_paths: Vec<PathBuf> = stored_paths
      .into_iter()
      .filter(|path| !path_is_live(path, &live_exact, &live_prefixes))
      .collect();
    if !dry_run && !orphaned_paths.is_empty() {
      let operations: Vec<_> = orphaned_paths
        .iter()
        .cloned()
        .map(|path| (path, PersistOperation::Delete))
        .collect();
      for batch in chunk_operations(operations, IDB_WRITE_BATCH_SIZE) {
        persist_operations_batch(&self.db_name, &batch)
          .await
          .map_err(|err| {
            map_storage_error(err, "storage_cleanup_error", "removing orphaned files")
          })?;
      }
      storage.remove_cached_paths(&orphaned_paths);
    }
    let removed = orphaned_paths
      .iter()
      .map(|path| path_key(path))
      .collect::<Vec<_>>();
    serde_wasm_bindgen::to_value(&CleanupOrphanedFilesResponse {
      scanned,
      orphaned: removed.len(),
      removed,
      dry_run,
    })
    .map_err(|err| typed_js_error("serialization_error", err))
  }

  async fn plan_migration_internal(
    db_name: &str,
    requested_schema: &Schema,
  ) -> Result<MigrationPlan, JsValue> {
    let requested_schema_hash = schema_hash(requested_schema)?;
    // Registry check before opening IndexedDB: avoid creating an empty
    // database/object store just to report "missing". All indexes created
    // by this binding register themselves on init(), so the registry is
    // the source of truth for known indexes.
    let registry_entry = get_registry_entry(db_name)
      .await
      .map_err(|err| typed_js_error("registry_read_error", err))?;
    if registry_entry.is_none() {
      return Ok(MigrationPlan {
        db_name: db_name.to_string(),
        status: "missing".to_string(),
        rebuild_required: false,
        schema_version: SCHEMA_VERSION_V1,
        existing_schema_hash: None,
        requested_schema_hash,
      });
    }
    let root = PathBuf::from(db_name.to_string());
    let storage = Arc::new(
      JsStorage::new(db_name.to_string(), root.clone())
        .await
        .map_err(|err| typed_js_error("storage_open_error", err))?,
    );
    let manifest_path = root.join("MANIFEST.json");
    if !storage.exists(&manifest_path) {
      return Ok(MigrationPlan {
        db_name: db_name.to_string(),
        status: "missing".to_string(),
        rebuild_required: false,
        schema_version: SCHEMA_VERSION_V1,
        existing_schema_hash: None,
        requested_schema_hash,
      });
    }

    let index = Index::open_with_storage(open_opts(root.clone()), storage.clone())
      .map_err(|err| typed_js_error("index_open_error", err))?;
    let existing_schema = index.manifest().schema;
    let existing_schema_hash = match read_index_meta(storage.as_ref(), &root)? {
      Some(meta) => meta.schema_hash,
      None => schema_hash(&existing_schema)?,
    };
    let compatible = schemas_match(&existing_schema, requested_schema)?;
    Ok(MigrationPlan {
      db_name: db_name.to_string(),
      status: if compatible {
        "compatible".to_string()
      } else {
        "rebuild_required".to_string()
      },
      rebuild_required: !compatible,
      schema_version: SCHEMA_VERSION_V1,
      existing_schema_hash: Some(existing_schema_hash),
      requested_schema_hash,
    })
  }

  #[wasm_bindgen(js_name = plan_migration)]
  pub async fn plan_migration(db_name: String, schema_json: String) -> Result<JsValue, JsValue> {
    let requested_schema: Schema = serde_json::from_str(&schema_json)
      .map_err(|err| typed_js_error("invalid_schema_json", err))?;
    let plan = Self::plan_migration_internal(&db_name, &requested_schema).await?;
    serde_wasm_bindgen::to_value(&plan).map_err(|err| typed_js_error("serialization_error", err))
  }

  #[wasm_bindgen(js_name = migrate_index)]
  pub async fn migrate_index(db_name: String, schema_json: String) -> Result<JsValue, JsValue> {
    let requested_schema: Schema = serde_json::from_str(&schema_json)
      .map_err(|err| typed_js_error("invalid_schema_json", err))?;
    let plan = Self::plan_migration_internal(&db_name, &requested_schema).await?;
    let requested_schema_hash = plan.requested_schema_hash.clone();
    let existing_schema_hash = plan.existing_schema_hash.clone();

    if plan.status == "missing" {
      let created = Self::create(db_name.clone(), schema_json, StorageMode::IndexedDb).await?;
      created.commit().await?;
      let result = MigrationExecutionResult {
        db_name,
        status: "created".to_string(),
        rebuild_performed: false,
        schema_version: SCHEMA_VERSION_V1,
        existing_schema_hash: None,
        requested_schema_hash,
      };
      return serde_wasm_bindgen::to_value(&result)
        .map_err(|err| typed_js_error("serialization_error", err));
    }

    if !plan.rebuild_required {
      let result = MigrationExecutionResult {
        db_name,
        status: "compatible".to_string(),
        rebuild_performed: false,
        schema_version: SCHEMA_VERSION_V1,
        existing_schema_hash,
        requested_schema_hash,
      };
      return serde_wasm_bindgen::to_value(&result)
        .map_err(|err| typed_js_error("serialization_error", err));
    }

    let snapshot = load_snapshot(&db_name)
      .await
      .map_err(|err| typed_js_error("storage_snapshot_error", err))?;
    let previous_registry = get_registry_entry(&db_name)
      .await
      .map_err(|err| typed_js_error("registry_read_error", err))?;
    clear_data_store(&db_name)
      .await
      .map_err(|err| typed_js_error("storage_clear_error", err))?;

    let rebuild_attempt = async {
      if migration_fail_after_clear() {
        return Err(js_error(
          "migration_injected_failure",
          "injected migration failure after clear for rollback testing",
        ));
      }
      let migrated =
        Self::create(db_name.clone(), schema_json.clone(), StorageMode::IndexedDb).await?;
      migrated.commit().await?;
      Ok::<(), JsValue>(())
    }
    .await;

    match rebuild_attempt {
      Ok(()) => {
        let result = MigrationExecutionResult {
          db_name,
          status: "rebuilt".to_string(),
          rebuild_performed: true,
          schema_version: SCHEMA_VERSION_V1,
          existing_schema_hash,
          requested_schema_hash,
        };
        serde_wasm_bindgen::to_value(&result)
          .map_err(|err| typed_js_error("serialization_error", err))
      }
      Err(rebuild_err) => {
        let snapshot_restore_err = restore_snapshot(&db_name, &snapshot).await.err();
        let registry_restore_err = restore_registry_entry(&db_name, previous_registry.as_ref())
          .await
          .err();
        if snapshot_restore_err.is_none() && registry_restore_err.is_none() {
          return Err(js_error(
            "migration_rebuild_failed",
            format!(
              "migration rebuild failed but prior snapshot was restored: {}",
              js_error_reason(&rebuild_err)
            ),
          ));
        }
        let mut reason = format!(
          "migration rebuild failed and rollback was incomplete: {}",
          js_error_reason(&rebuild_err)
        );
        if let Some(err) = snapshot_restore_err {
          reason.push_str(&format!("; snapshot restore failed: {err}"));
        }
        if let Some(err) = registry_restore_err {
          reason.push_str(&format!("; registry restore failed: {err}"));
        }
        Err(js_error("migration_rollback_failed", reason))
      }
    }
  }

  async fn touch_registry(&self) -> Result<(), JsValue> {
    if !matches!(self.storage, StorageBackend::IndexedDb(_)) {
      return Ok(());
    }
    let entry = IndexRegistryEntry {
      db_name: self.db_name.clone(),
      schema_version: SCHEMA_VERSION_V1,
      schema_hash: self.schema_hash.clone(),
      updated_at_ms: now_ms(),
    };
    upsert_registry_entry(&entry)
      .await
      .map_err(|err| typed_js_error("registry_write_error", err))
  }

  /// Initialize the rayon pool for threaded execution. COOP/COEP (cross-origin isolation) must
  /// be handled by the embedding app; this helper does not set headers for you.
  #[cfg(feature = "threads")]
  pub async fn init_threads(&self, threads: Option<u32>) -> Result<(), JsValue> {
    let desired = threads.unwrap_or_else(hardware_concurrency);
    JsFuture::from(init_thread_pool(desired as usize))
      .await
      .map(|_| ())
      .map_err(|err| typed_js_error("thread_pool_init_error", format!("{err:?}")))
  }

  /// Threaded mode is disabled unless the `threads` feature is enabled.
  #[cfg(not(feature = "threads"))]
  pub async fn init_threads(&self, _threads: Option<u32>) -> Result<(), JsValue> {
    Err(js_error(
      "threads_feature_disabled",
      "threads feature is disabled; rebuild searchlite-wasm with --features threads and enable wasm atomics/COOP+COEP",
    ))
  }

  fn add_documents_internal(&self, docs: Vec<Document>) -> Result<(), JsValue> {
    let mut writer: IndexWriter = self.index.writer().map_err(to_js_error)?;
    for doc in docs.iter() {
      writer.add_document(doc).map_err(to_js_error)?;
    }
    Ok(())
  }

  fn delete_documents_internal(&self, doc_ids: Vec<String>) -> Result<(), JsValue> {
    let mut writer: IndexWriter = self.index.writer().map_err(to_js_error)?;
    writer.delete_documents(&doc_ids).map_err(to_js_error)?;
    Ok(())
  }

  /// Add a document to the index. Call `commit` to make it searchable and persist it.
  pub fn add_document(&self, doc: JsValue) -> Result<(), JsValue> {
    let value: serde_json::Value =
      serde_wasm_bindgen::from_value(doc).map_err(|err| typed_js_error("invalid_json", err))?;
    self.add_documents_internal(vec![value_to_document(value)?])
  }

  /// Add multiple documents to the index. Call `commit` to persist changes.
  pub fn add_documents(&self, docs: JsValue) -> Result<(), JsValue> {
    let value: serde_json::Value =
      serde_wasm_bindgen::from_value(docs).map_err(|err| typed_js_error("invalid_json", err))?;
    self.add_documents_internal(value_to_documents(value)?)
  }

  /// Queue deletion for a single `_id`/doc id. Call `commit` to persist removal.
  pub fn delete_document(&self, doc_id: String) -> Result<(), JsValue> {
    validate_doc_id(&doc_id)
      .map_err(|err| js_error("invalid_id", format!("invalid document id: {err}")))?;
    self.delete_documents_internal(vec![doc_id])
  }

  /// Queue deletions for one or more doc ids. Accepts a string or array of strings.
  /// Call `commit` to persist removals.
  pub fn delete_documents(&self, doc_ids: JsValue) -> Result<(), JsValue> {
    let value: serde_json::Value = serde_wasm_bindgen::from_value(doc_ids)
      .map_err(|err| typed_js_error("invalid_doc_id_batch", err))?;
    let ids = value_to_doc_ids(value)?;
    for id in ids.iter() {
      validate_doc_id(id)
        .map_err(|err| js_error("invalid_id", format!("invalid document id: {err}")))?;
    }
    self.delete_documents_internal(ids)
  }

  /// Queue a partial document update by id using set/unset patch semantics.
  /// `request` shape: `{ id: string, set?: object, unset?: string[] }`.
  pub fn update_document(&self, request: JsValue) -> Result<(), JsValue> {
    let payload: UpdateRequestPayload = parse_request_value(request, "invalid_update_request")?;
    if payload.set.is_empty() && payload.unset.is_empty() {
      return Err(js_error(
        "missing_patch",
        "update must include at least one set or unset field",
      ));
    }
    validate_doc_id(&payload.id)
      .map_err(|err| js_error("invalid_id", format!("invalid document id: {err}")))?;
    let mut writer = self
      .index
      .writer()
      .map_err(|err| typed_js_error("writer_open_error", err))?;
    writer
      .apply_patch(&payload.id, &payload.set, &payload.unset)
      .map_err(map_update_error)?;
    Ok(())
  }

  /// Commit pending documents and flush the configured storage backend.
  pub async fn commit(&self) -> Result<(), JsValue> {
    let mut writer: IndexWriter = self.index.writer().map_err(to_js_error)?;
    writer.commit().map_err(to_js_error)?;
    self
      .storage
      .flush()
      .await
      .map_err(|err| map_storage_error(err, "storage_flush_error", "committing index data"))?;
    self.touch_registry().await?;
    Ok(())
  }

  /// Compact segments to reduce fragmentation.
  /// Returns `{ compacted: boolean }`, where `false` means no merge was needed.
  pub async fn compact(&self) -> Result<JsValue, JsValue> {
    let before = self.index.manifest().segments.len();
    self
      .index
      .compact()
      .map_err(|err| typed_js_error("compact_failed", err))?;
    self
      .storage
      .flush()
      .await
      .map_err(|err| map_storage_error(err, "storage_flush_error", "compacting index data"))?;
    self.touch_registry().await?;
    let after = self.index.manifest().segments.len();
    serde_wasm_bindgen::to_value(&CompactResponse {
      compacted: after < before,
    })
    .map_err(|err| typed_js_error("serialization_error", err))
  }

  /// Return the current manifest with write-key metadata redacted.
  pub fn inspect(&self) -> Result<JsValue, JsValue> {
    let mut manifest = self.index.manifest();
    manifest.write_key = None;
    for seg in manifest.segments.iter_mut() {
      seg.write_binding_b64 = None;
    }
    serde_wasm_bindgen::to_value(&InspectResponse { manifest })
      .map_err(|err| typed_js_error("serialization_error", err))
  }

  /// Return high-level index statistics.
  pub fn stats(&self) -> Result<JsValue, JsValue> {
    let manifest = self.index.manifest();
    let (live_docs, deleted_docs) = manifest_doc_counts(&manifest);
    serde_wasm_bindgen::to_value(&StatsResponse {
      documents: live_docs,
      deleted_documents: deleted_docs,
      segments: manifest.segments.len(),
      committed_at: manifest.committed_at.clone(),
      index_uuid: manifest.uuid.to_string(),
      index_path: self.db_name.clone(),
      index_name: self.db_name.clone(),
    })
    .map_err(|err| typed_js_error("serialization_error", err))
  }

  /// Fetch documents by id with optional stored fields, preserving request order.
  /// `request` shape: `{ ids: string[], return_stored?: boolean }`.
  pub fn mget(&self, request: JsValue) -> Result<JsValue, JsValue> {
    let req: MgetRequest = parse_request_value(request, "invalid_mget_request")?;
    for id in req.ids.iter() {
      validate_doc_id(id)
        .map_err(|err| js_error("invalid_id", format!("invalid document id: {err}")))?;
    }
    let reader: IndexReader = self
      .index
      .reader()
      .map_err(|err| typed_js_error("reader_open_error", err))?;
    let docs = reader
      .mget(&req.ids, req.return_stored)
      .map_err(|err| typed_js_error("mget_failed", err))?;
    serde_wasm_bindgen::to_value(&MgetResponse { docs })
      .map_err(|err| typed_js_error("serialization_error", err))
  }

  /// Execute multiple search requests in order and return ordered results.
  /// `request` shape: `{ searches: SearchRequest[], parallel?: boolean, max_concurrency?: number }`.
  /// Note: `parallel` and `max_concurrency` are accepted for API compatibility
  /// but have no effect in single-threaded WASM — searches always run serially.
  pub fn multi_search(&self, request: JsValue) -> Result<JsValue, JsValue> {
    let req: MultiSearchRequest = parse_request_value(request, "invalid_multi_search_request")?;
    let reader: IndexReader = self
      .index
      .reader()
      .map_err(|err| typed_js_error("reader_open_error", err))?;
    let results = reader
      .multi_search(&req.searches)
      .map_err(|err| typed_js_error("multi_search_failed", err))?;
    serde_wasm_bindgen::to_value(&MultiSearchResponse { results })
      .map_err(|err| typed_js_error("serialization_error", err))
  }

  pub fn search(
    &self,
    query: String,
    limit: usize,
    return_stored: Option<bool>,
  ) -> Result<JsValue, JsValue> {
    validate_search_limit_arg(limit)?;
    let parsed_query = serde_json::from_str::<QueryNode>(&query)
      .map(Query::Node)
      .unwrap_or(Query::String(query));
    let request = SearchRequest {
      query: parsed_query,
      fields: None,
      filter: None,
      limit,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::<SortSpec>::new(),
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
      return_stored: return_stored.unwrap_or(false),
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs: BTreeMap::<String, Aggregation>::new(),
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    };
    self.run_search(request)
  }

  #[wasm_bindgen(js_name = search_controlled)]
  pub fn search_controlled(
    &self,
    query: String,
    limit: usize,
    return_stored: Option<bool>,
    abort_signal: Option<web_sys::AbortSignal>,
    timeout_ms: Option<f64>,
  ) -> Result<JsValue, JsValue> {
    validate_search_limit_arg(limit)?;
    let parsed_query = serde_json::from_str::<QueryNode>(&query)
      .map(Query::Node)
      .unwrap_or(Query::String(query));
    let request = SearchRequest {
      query: parsed_query,
      fields: None,
      filter: None,
      limit,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::<SortSpec>::new(),
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
      return_stored: return_stored.unwrap_or(false),
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs: BTreeMap::<String, Aggregation>::new(),
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    };
    self.run_search_controlled(request, abort_signal, timeout_ms)
  }

  pub fn search_request(&self, request_json: String) -> Result<JsValue, JsValue> {
    let req: SearchRequest = serde_json::from_str(&request_json)
      .map_err(|err| typed_js_error("invalid_search_request", err))?;
    validate_search_limits(&req)?;
    self.run_search(req)
  }

  #[wasm_bindgen(js_name = search_request_controlled)]
  pub fn search_request_controlled(
    &self,
    request_json: String,
    abort_signal: Option<web_sys::AbortSignal>,
    timeout_ms: Option<f64>,
  ) -> Result<JsValue, JsValue> {
    let req: SearchRequest = serde_json::from_str(&request_json)
      .map_err(|err| typed_js_error("invalid_search_request", err))?;
    validate_search_limits(&req)?;
    self.run_search_controlled(req, abort_signal, timeout_ms)
  }

  pub fn search_request_value(&self, request: JsValue) -> Result<JsValue, JsValue> {
    let req: SearchRequest = parse_request_value(request, "invalid_search_request")?;
    validate_search_limits(&req)?;
    self.run_search(req)
  }

  #[wasm_bindgen(js_name = search_request_value_controlled)]
  pub fn search_request_value_controlled(
    &self,
    request: JsValue,
    abort_signal: Option<web_sys::AbortSignal>,
    timeout_ms: Option<f64>,
  ) -> Result<JsValue, JsValue> {
    let req: SearchRequest = parse_request_value(request, "invalid_search_request")?;
    validate_search_limits(&req)?;
    self.run_search_controlled(req, abort_signal, timeout_ms)
  }

  /// Worker-oriented async search entrypoint with optional timeout/abort checks.
  #[wasm_bindgen(js_name = search_request_value_async)]
  pub async fn search_request_value_async(
    &self,
    request: JsValue,
    abort_signal: Option<web_sys::AbortSignal>,
    timeout_ms: Option<f64>,
  ) -> Result<JsValue, JsValue> {
    self.search_request_value_controlled(request, abort_signal, timeout_ms)
  }

  fn run_search(&self, req: SearchRequest) -> Result<JsValue, JsValue> {
    self.run_search_controlled(req, None, None)
  }

  fn run_search_controlled(
    &self,
    req: SearchRequest,
    abort_signal: Option<web_sys::AbortSignal>,
    timeout_ms: Option<f64>,
  ) -> Result<JsValue, JsValue> {
    let timeout_ms = parse_timeout_ms(timeout_ms)?;
    let started_ms = now_ms();
    ensure_not_aborted(abort_signal.as_ref())?;
    ensure_not_timed_out(started_ms, timeout_ms)?;
    let reader: IndexReader = self.index.reader().map_err(to_js_error)?;
    let result = reader.search(&req).map_err(to_js_error)?;
    ensure_not_aborted(abort_signal.as_ref())?;
    ensure_not_timed_out(started_ms, timeout_ms)?;
    serde_wasm_bindgen::to_value(&result).map_err(|err| typed_js_error("serialization_error", err))
  }

  /// Wait for pending storage writes; `commit` already calls this.
  pub async fn flush_storage(&self) -> Result<(), JsValue> {
    self
      .storage
      .flush()
      .await
      .map_err(|err| map_storage_error(err, "storage_flush_error", "flushing storage"))?;
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use searchlite_core::api::types::ExecutionStrategy;
  use std::io::{Read, Seek, SeekFrom, Write};
  use wasm_bindgen_test::*;

  wasm_bindgen_test_configure!(run_in_browser);

  fn unique_db(name: &str) -> String {
    format!("{name}-{}", js_sys::Date::now() as u64)
  }

  struct MigrationFailureGuard;

  impl MigrationFailureGuard {
    fn enable() -> Self {
      set_migration_fail_after_clear(true);
      Self
    }
  }

  impl Drop for MigrationFailureGuard {
    fn drop(&mut self) {
      set_migration_fail_after_clear(false);
    }
  }

  struct PersistQuotaFailureGuard;

  impl PersistQuotaFailureGuard {
    fn enable() -> Self {
      set_force_persist_quota_exceeded(true);
      Self
    }
  }

  impl Drop for PersistQuotaFailureGuard {
    fn drop(&mut self) {
      set_force_persist_quota_exceeded(false);
    }
  }

  #[wasm_bindgen_test]
  async fn js_storage_persists_entries() {
    let db = unique_db("searchlite-storage");
    let root = PathBuf::from("idx");
    let storage = JsStorage::new(db.clone(), root.clone()).await.unwrap();
    let path = root.join("test.bin");
    storage.write_all(&path, b"hello wasm").unwrap();
    storage.flush().await.unwrap();
    drop(storage);
    let restored = JsStorage::new(db, root.clone()).await.unwrap();
    let contents = restored.read_to_end(&path).unwrap();
    assert_eq!(contents, b"hello wasm");
  }

  #[wasm_bindgen_test]
  fn schema_hash_is_deterministic() {
    let schema = Schema::default_text_body();
    let hash_a = schema_hash(&schema).unwrap();
    let hash_b = schema_hash(&schema).unwrap();
    assert_eq!(hash_a, hash_b);

    let mut schema_v2 = Schema::default_text_body();
    schema_v2
      .keyword_fields
      .push(searchlite_core::api::types::KeywordField {
        name: "category".to_string(),
        stored: true,
        indexed: true,
        fast: true,
        nullable: false,
      });
    let hash_c = schema_hash(&schema_v2).unwrap();
    assert_ne!(hash_a, hash_c);
  }

  #[wasm_bindgen_test]
  async fn js_storage_methods_roundtrip() {
    let db = unique_db("searchlite-storage-methods");
    let root = PathBuf::from("idx-methods");
    let storage = JsStorage::new(db, root.clone()).await.unwrap();
    let path = root.join("notes.txt");

    storage.ensure_dir(&root).unwrap();
    assert!(!storage.exists(&path));

    {
      let mut file = storage.open_write(&path).unwrap();
      file.write_all(b"hello").unwrap();
      file.flush().unwrap();
    }
    storage.flush().await.unwrap();
    assert!(storage.exists(&path));

    {
      let mut file = storage.open_append(&path).unwrap();
      file.write_all(b" world").unwrap();
      file.flush().unwrap();
    }
    storage.flush().await.unwrap();
    let contents = storage.read_to_end(&path).unwrap();
    assert_eq!(contents, b"hello world");

    {
      let mut file = storage.open_read(&path).unwrap();
      let mut buf = Vec::new();
      file.read_to_end(&mut buf).unwrap();
      assert_eq!(buf, b"hello world");
    }

    let atomic_path = root.join("atomic.txt");
    storage.atomic_write(&atomic_path, b"atomic").unwrap();
    storage.flush().await.unwrap();
    let contents = storage.read_to_end(&atomic_path).unwrap();
    assert_eq!(contents, b"atomic");
  }

  #[wasm_bindgen_test]
  async fn js_storage_flush_batches_indexeddb_transactions() {
    let db = unique_db("searchlite-storage-batch");
    let root = PathBuf::from("idx-batch");
    reset_persist_batch_tx_count();
    let storage = JsStorage::new(db.clone(), root.clone()).await.unwrap();
    for idx in 0..12 {
      let path = root.join(format!("file-{idx}.bin"));
      let payload = format!("payload-{idx}");
      storage.write_all(&path, payload.as_bytes()).unwrap();
    }
    storage.flush().await.unwrap();
    assert_eq!(persist_batch_tx_count(), 1);
    Searchlite::drop_index(db).await.unwrap();
  }

  #[wasm_bindgen_test]
  async fn js_storage_flush_waits_for_deletes() {
    let db = unique_db("searchlite-storage-delete-flush");
    let root = PathBuf::from("idx-delete");
    let storage = JsStorage::new(db.clone(), root.clone()).await.unwrap();
    let path = root.join("remove-me.bin");
    storage.write_all(&path, b"delete me").unwrap();
    storage.flush().await.unwrap();
    storage.remove(&path).unwrap();
    storage.flush().await.unwrap();
    drop(storage);
    let restored = JsStorage::new(db.clone(), root.clone()).await.unwrap();
    assert!(!restored.exists(&path));
    Searchlite::drop_index(db).await.unwrap();
  }

  #[wasm_bindgen_test]
  async fn js_file_seek_behaves() {
    let db = unique_db("searchlite-seek");
    let root = PathBuf::from("idx-seek");
    let storage = JsStorage::new(db, root.clone()).await.unwrap();
    let path = root.join("seek.txt");
    let mut file = storage.open_write(&path).unwrap();

    file.write_all(b"abcdef").unwrap();
    file.flush().unwrap();
    file.seek(SeekFrom::Start(2)).unwrap();
    let mut buf = [0u8; 2];
    file.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"cd");
    file.seek(SeekFrom::End(-2)).unwrap();
    let mut tail = [0u8; 2];
    file.read_exact(&mut tail).unwrap();
    assert_eq!(&tail, b"ef");
    file.seek(SeekFrom::Start(0)).unwrap();
    assert!(file.seek(SeekFrom::Current(-1)).is_err());
  }

  #[wasm_bindgen_test]
  async fn indexes_and_searches() {
    let db = unique_db("searchlite-index");
    let root = PathBuf::from("idx2");
    let storage = Arc::new(JsStorage::new(db, root.clone()).await.unwrap());
    let schema = Schema::default_text_body();
    let opts = IndexOptions {
      path: root.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: BM25_K1,
      bm25_b: BM25_B,
      storage: StorageType::InMemory,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    };
    let index = Index::create_with_storage(&root, schema, opts, storage.clone()).unwrap();
    let mut writer: IndexWriter = index.writer().unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!("doc-1")),
          ("body".into(), serde_json::json!("hello wasm")),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    writer.commit().unwrap();
    storage.flush().await.unwrap();
    let reader: IndexReader = index.reader().unwrap();
    let request = SearchRequest {
      query: "hello".into(),
      fields: None,
      filter: None,
      limit: 5,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: vec![],
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
    };
    let result = reader.search(&request).unwrap();
    assert_eq!(result.hits.len(), 1);
  }

  #[wasm_bindgen_test]
  async fn wasm_core_parity_search_mget_multi_search_update_delete() {
    let db = unique_db("searchlite-parity");
    let schema_json = serde_json::to_string(&Schema::default_text_body()).unwrap();
    let idx = Searchlite::init(db, schema_json, None).await.unwrap();
    let docs = vec![
      serde_json::json!({ "_id": "doc-1", "body": "alpha token" }),
      serde_json::json!({ "_id": "doc-2", "body": "beta token" }),
      serde_json::json!({ "_id": "doc-3", "body": "gamma token" }),
    ];
    idx
      .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
      .unwrap();
    idx.commit().await.unwrap();

    let search_req: SearchRequest = serde_json::from_value(serde_json::json!({
      "query": "alpha",
      "limit": 5,
      "return_stored": true
    }))
    .unwrap();
    let wasm_search = idx
      .search_request_value(serde_wasm_bindgen::to_value(&search_req).unwrap())
      .unwrap();
    let wasm_search_json: serde_json::Value = serde_wasm_bindgen::from_value(wasm_search).unwrap();
    let core_search = idx.index.reader().unwrap().search(&search_req).unwrap();
    let core_search_json = serde_json::to_value(core_search).unwrap();
    assert_eq!(wasm_search_json, core_search_json);

    let mget_req = MgetRequest {
      ids: vec![
        "doc-2".to_string(),
        "missing".to_string(),
        "doc-1".to_string(),
      ],
      return_stored: true,
    };
    let wasm_mget = idx
      .mget(serde_wasm_bindgen::to_value(&mget_req).unwrap())
      .unwrap();
    let wasm_mget_json: serde_json::Value = serde_wasm_bindgen::from_value(wasm_mget).unwrap();
    let core_mget_docs = idx
      .index
      .reader()
      .unwrap()
      .mget(&mget_req.ids, mget_req.return_stored)
      .unwrap();
    let core_mget_json = serde_json::to_value(MgetResponse {
      docs: core_mget_docs,
    })
    .unwrap();
    assert_eq!(wasm_mget_json, core_mget_json);

    let multi_req = MultiSearchRequest {
      searches: vec![
        serde_json::from_value(serde_json::json!({
          "query": "alpha",
          "limit": 5,
          "return_stored": true
        }))
        .unwrap(),
        serde_json::from_value(serde_json::json!({
          "query": "beta",
          "limit": 5,
          "return_stored": true
        }))
        .unwrap(),
      ],
      parallel: false,
      max_concurrency: None,
    };
    let wasm_multi = idx
      .multi_search(serde_wasm_bindgen::to_value(&multi_req).unwrap())
      .unwrap();
    let wasm_multi_json: serde_json::Value = serde_wasm_bindgen::from_value(wasm_multi).unwrap();
    let core_multi_results = idx
      .index
      .reader()
      .unwrap()
      .multi_search(&multi_req.searches)
      .unwrap();
    let core_multi_json = serde_json::to_value(MultiSearchResponse {
      results: core_multi_results,
    })
    .unwrap();
    assert_eq!(wasm_multi_json, core_multi_json);

    let patch = serde_json::json!({
      "id": "doc-1",
      "set": { "body": "alpha updated" }
    });
    idx
      .update_document(serde_wasm_bindgen::to_value(&patch).unwrap())
      .unwrap();
    idx.commit().await.unwrap();
    idx.delete_document("doc-2".to_string()).unwrap();
    idx.commit().await.unwrap();

    let verify_req = MgetRequest {
      ids: vec!["doc-1".to_string(), "doc-2".to_string()],
      return_stored: true,
    };
    let wasm_verify = idx
      .mget(serde_wasm_bindgen::to_value(&verify_req).unwrap())
      .unwrap();
    let wasm_verify_json: serde_json::Value = serde_wasm_bindgen::from_value(wasm_verify).unwrap();
    let core_verify_docs = idx
      .index
      .reader()
      .unwrap()
      .mget(&verify_req.ids, verify_req.return_stored)
      .unwrap();
    let core_verify_json = serde_json::to_value(MgetResponse {
      docs: core_verify_docs,
    })
    .unwrap();
    assert_eq!(wasm_verify_json, core_verify_json);
  }

  #[wasm_bindgen_test]
  async fn commit_surfaces_quota_exceeded_error_type() {
    let db = unique_db("searchlite-quota-error");
    let schema_json = serde_json::to_string(&Schema::default_text_body()).unwrap();
    let idx = Searchlite::init(db.clone(), schema_json, None)
      .await
      .unwrap();
    idx
      .add_document(
        serde_wasm_bindgen::to_value(&serde_json::json!({ "_id": "doc-1", "body": "quota" }))
          .unwrap(),
      )
      .unwrap();
    let _guard = PersistQuotaFailureGuard::enable();
    let err = idx.commit().await.unwrap_err();
    let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
    assert_eq!(payload.error_type, "quota_exceeded");
    assert!(payload.reason.contains("compact()"));
    assert!(payload.reason.contains("cleanup_indexes"));
    Searchlite::drop_index(db).await.unwrap();
  }

  #[wasm_bindgen_test]
  async fn cleanup_orphaned_files_removes_only_unknown_paths() {
    let db = unique_db("searchlite-cleanup-orphans");
    let schema_json = serde_json::to_string(&Schema::default_text_body()).unwrap();
    let idx = Searchlite::init(db.clone(), schema_json, None)
      .await
      .unwrap();
    let docs = vec![serde_json::json!({ "_id": "doc-1", "body": "retain me" })];
    idx
      .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
      .unwrap();
    idx.commit().await.unwrap();

    let orphan_path = PathBuf::from(db.clone()).join("orphan.bin");
    persist_operations_batch(
      &db,
      &[(orphan_path.clone(), PersistOperation::Put(vec![1, 2, 3]))],
    )
    .await
    .unwrap();

    let cleanup_js = idx.cleanup_orphaned_files(Some(false)).await.unwrap();
    let cleanup: CleanupOrphanedFilesResponse = serde_wasm_bindgen::from_value(cleanup_js).unwrap();
    assert!(cleanup
      .removed
      .iter()
      .any(|path| path == &path_key(&orphan_path)));

    let result = idx.search("retain".to_string(), 5, Some(true)).unwrap();
    let result_json: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
    let hits = result_json["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    Searchlite::drop_index(db).await.unwrap();
  }

  #[wasm_bindgen_test]
  async fn migrate_index_rolls_back_on_rebuild_failure() {
    let db = unique_db("searchlite-migrate-rollback");
    let schema_v1 = Schema::default_text_body();
    let schema_v1_json = serde_json::to_string(&schema_v1).unwrap();
    let idx = Searchlite::init(db.clone(), schema_v1_json.clone(), None)
      .await
      .unwrap();
    let docs = vec![serde_json::json!({ "_id": "doc-1", "body": "rollback sentinel" })];
    idx
      .add_documents(serde_wasm_bindgen::to_value(&docs).unwrap())
      .unwrap();
    idx.commit().await.unwrap();

    let mut schema_v2 = Schema::default_text_body();
    schema_v2
      .keyword_fields
      .push(searchlite_core::api::types::KeywordField {
        name: "category".to_string(),
        stored: true,
        indexed: true,
        fast: true,
        nullable: false,
      });
    let schema_v2_json = serde_json::to_string(&schema_v2).unwrap();

    let _guard = MigrationFailureGuard::enable();
    let err = Searchlite::migrate_index(db.clone(), schema_v2_json.clone())
      .await
      .unwrap_err();
    let payload: WasmErrorPayload = serde_wasm_bindgen::from_value(err).unwrap();
    assert_eq!(payload.error_type, "migration_rebuild_failed");

    let reopened = Searchlite::init(db.clone(), schema_v1_json, None)
      .await
      .unwrap();
    let result = reopened
      .search("sentinel".to_string(), 5, Some(true))
      .unwrap();
    let result_json: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
    let hits = result_json["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);

    let mismatch = match Searchlite::init(db, schema_v2_json, None).await {
      Ok(_) => panic!("expected schema mismatch after rollback"),
      Err(err) => err,
    };
    let mismatch_payload: WasmErrorPayload = serde_wasm_bindgen::from_value(mismatch).unwrap();
    assert_eq!(mismatch_payload.error_type, "schema_mismatch");
  }
}
