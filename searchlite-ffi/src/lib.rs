use std::any::Any;
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::slice;
use std::str;

#[cfg(test)]
use std::sync::{
  atomic::{AtomicBool, Ordering},
  Mutex,
};

#[cfg(feature = "vectors")]
use searchlite_core::api::types::parse_env_max_vector_candidates;
use searchlite_core::api::types::{
  Aggregation, Document, ExecutionStrategy, IndexOptions, Query, QueryNode, Schema, SearchRequest,
  StorageType,
};
use searchlite_core::api::{Index, WriteKeyError};

/// Classify an error returned by `Index::writer*` into the FFI auth-error
/// code (`-8`) or a generic open-failure code. Callers that distinguish the
/// two must use a typed downcast on the `anyhow::Error` — the previous
/// substring match on `err.to_string()` silently flipped the classification
/// whenever any crate changed the Display form of the underlying error.
///
/// `generic` is the fallback code for non-auth failures (e.g. `-3` for
/// commit paths, `-4` for add paths) so this helper can be shared across
/// every mutating FFI entrypoint without baking the exact integer in.
fn classify_writer_err(err: &anyhow::Error, generic: c_int) -> c_int {
  if err.downcast_ref::<WriteKeyError>().is_some() {
    -8
  } else {
    generic
  }
}

#[repr(C)]
pub struct IndexHandle {
  index: Index,
}

/// Returned when a Rust panic was caught inside an FFI entrypoint. The in-flight
/// operation is aborted; callers may retry. After a panic from a mutating call,
/// reopening the index handle is the safest way to ensure on-disk consistency.
const ERR_PANIC: c_int = -100;
// All extern "C" functions use catch_unwind to prevent unwinding across the C
// boundary. Panics abort the current operation; state is left as-consistent-as-
// possible, but reopen after panics from mutating calls to be conservative.

#[inline]
fn catch_unwind_default<T>(api: &str, default: T, f: impl FnOnce() -> T) -> T {
  match panic::catch_unwind(AssertUnwindSafe(f)) {
    Ok(v) => v,
    Err(payload) => {
      log_panic(api, &payload);
      default
    }
  }
}

fn log_panic(api: &str, payload: &Box<dyn Any + Send + 'static>) {
  let msg = payload
    .downcast_ref::<&str>()
    .copied()
    .or_else(|| payload.downcast_ref::<String>().map(|s| s.as_str()))
    .unwrap_or("<non-string panic payload>");
  eprintln!("searchlite FFI: panic in {api}: {msg}");
}

#[cfg(test)]
static NEXT_CALL_SHOULD_PANIC: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
static TEST_PANIC_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
fn request_panic_for_next_call() {
  NEXT_CALL_SHOULD_PANIC.store(true, Ordering::SeqCst);
}

#[cfg(test)]
fn maybe_panic_for_tests() {
  if NEXT_CALL_SHOULD_PANIC.swap(false, Ordering::SeqCst) {
    panic!("ffi test panic");
  }
}

#[cfg(test)]
fn test_guard() -> std::sync::MutexGuard<'static, ()> {
  TEST_PANIC_LOCK.lock().unwrap()
}

unsafe fn read_json_str(ptr: *const c_char, len: usize) -> Option<String> {
  if ptr.is_null() {
    return None;
  }
  if len > 0 {
    let bytes = slice::from_raw_parts(ptr as *const u8, len);
    return str::from_utf8(bytes).ok().map(|s| s.to_string());
  }
  CStr::from_ptr(ptr).to_str().ok().map(|s| s.to_string())
}

fn value_to_document(value: serde_json::Value) -> Result<Document, ()> {
  let obj = value.as_object().ok_or(())?;
  let mut fields = BTreeMap::new();
  for (k, v) in obj.iter() {
    fields.insert(k.clone(), v.clone());
  }
  Ok(Document { fields })
}

fn optional_cstr(ptr: *const c_char) -> Result<Option<String>, std::str::Utf8Error> {
  if ptr.is_null() {
    return Ok(None);
  }
  let c_str = unsafe { CStr::from_ptr(ptr) };
  c_str.to_str().map(|s| Some(s.to_string()))
}

fn open_index(
  path: *const c_char,
  create_if_missing: bool,
  write_key: Option<String>,
) -> *mut IndexHandle {
  if path.is_null() {
    return std::ptr::null_mut();
  }
  let c_str = unsafe { CStr::from_ptr(path) };
  let path_buf = PathBuf::from(c_str.to_string_lossy().to_string());
  let opts = IndexOptions {
    path: path_buf.clone(),
    create_if_missing,
    enable_positions: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::Filesystem,
    checksum_policy: Default::default(),
    checksum_audit_failure_hook: None,
    read_only: false,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  };
  let manifest_path = path_buf.join("MANIFEST.json");
  let index_res = if create_if_missing && !manifest_path.exists() {
    Index::create_with_write_key(
      &path_buf,
      Schema::default_text_body(),
      opts,
      write_key.as_deref(),
    )
  } else {
    Index::open(opts)
  };
  match index_res {
    Ok(index) => Box::into_raw(Box::new(IndexHandle { index })),
    Err(_) => std::ptr::null_mut(),
  }
}

fn value_to_documents(value: serde_json::Value) -> Result<Vec<Document>, ()> {
  match value {
    serde_json::Value::Array(items) => items.into_iter().map(value_to_document).collect(),
    obj @ serde_json::Value::Object(_) => Ok(vec![value_to_document(obj)?]),
    _ => Err(()),
  }
}

/// Copy a JSON payload into a caller-provided C buffer and return the number of
/// bytes written (not counting the NUL terminator).
///
/// Return value convention:
/// - `0` — the output pointer was null. Nothing was written and the caller has
///   no way to receive a required-size signal.
/// - `N` where `0 < N <= buf_cap - 1` — success; `N` bytes of JSON followed by a
///   NUL byte were written into `out_json_buf`.
/// - `N` where `N > buf_cap` — the buffer was too small. No JSON was written
///   (when `buf_cap >= 1` a single NUL byte is written so the buffer is a valid
///   empty C string). `N` is the required buffer size including the NUL
///   terminator; the caller should allocate a buffer of that size and retry.
///
/// Note: `N == buf_cap` can never be returned — success always leaves at least
/// one byte for the NUL terminator, so the success case satisfies
/// `N <= buf_cap - 1`. This makes `N > buf_cap` an unambiguous "buffer too
/// small" signal even when `buf_cap == 0`.
fn write_json_to_buffer(json: String, out_json_buf: *mut c_char, buf_cap: usize) -> usize {
  if out_json_buf.is_null() {
    return 0;
  }
  let bytes = json.as_bytes();
  // Required capacity includes the trailing NUL. `saturating_add` defends
  // against a pathological `bytes.len() == usize::MAX`, which cannot occur in
  // practice but lets the helper be total.
  let required = bytes.len().saturating_add(1);
  if required > buf_cap {
    // NUL-terminate what we can so callers that ignore the return value see
    // an empty C string rather than stale or partial data.
    if buf_cap >= 1 {
      unsafe {
        *out_json_buf = 0;
      }
    }
    return required;
  }
  unsafe {
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_json_buf as *mut u8, bytes.len());
    *out_json_buf.add(bytes.len()) = 0;
  }
  bytes.len()
}

fn run_search(
  index: &Index,
  req: SearchRequest,
  out_json_buf: *mut c_char,
  buf_cap: usize,
) -> usize {
  let reader = match index.reader() {
    Ok(r) => r,
    Err(_) => return 0,
  };
  let res = match reader.search(&req) {
    Ok(r) => r,
    Err(_) => return 0,
  };
  match serde_json::to_string(&res) {
    Ok(json) => write_json_to_buffer(json, out_json_buf, buf_cap),
    Err(_) => 0,
  }
}

/// # Safety
/// `path` must be a valid, non-null C string pointer that remains valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn searchlite_index_open(
  path: *const c_char,
  create_if_missing: bool,
) -> *mut IndexHandle {
  catch_unwind_default("searchlite_index_open", std::ptr::null_mut(), || {
    #[cfg(test)]
    maybe_panic_for_tests();

    open_index(path, create_if_missing, None)
  })
}

/// # Safety
/// Same as `searchlite_index_open`, but accepts an optional write key that is used when creating a new
/// key-protected index. The key is not required to open an existing index, but is still enforced on write calls.
#[no_mangle]
pub unsafe extern "C" fn searchlite_index_open_with_write_key(
  path: *const c_char,
  create_if_missing: bool,
  write_key: *const c_char,
) -> *mut IndexHandle {
  catch_unwind_default(
    "searchlite_index_open_with_write_key",
    std::ptr::null_mut(),
    || {
      #[cfg(test)]
      maybe_panic_for_tests();

      let write_key = match optional_cstr(write_key) {
        Ok(v) => v,
        Err(_) => return std::ptr::null_mut(),
      };
      open_index(path, create_if_missing, write_key)
    },
  )
}

/// # Safety
/// `handle` must be a pointer returned by `searchlite_index_open` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn searchlite_index_close(handle: *mut IndexHandle) {
  catch_unwind_default("searchlite_index_close", (), || {
    #[cfg(test)]
    maybe_panic_for_tests();

    if handle.is_null() {
      return;
    }
    drop(Box::from_raw(handle));
  });
}

/// # Safety
/// `handle` must be a valid pointer from `searchlite_index_open`, and `json` must point to a valid UTF-8 string.
/// Documents are queued; call `searchlite_commit` to make them searchable and durable.
#[no_mangle]
pub unsafe extern "C" fn searchlite_add_json(
  handle: *mut IndexHandle,
  json: *const c_char,
  len: usize,
) -> c_int {
  catch_unwind_default("searchlite_add_json", ERR_PANIC, || {
    #[cfg(test)]
    maybe_panic_for_tests();

    if handle.is_null() || json.is_null() {
      return -1;
    }
    let h = &mut *handle;
    let Some(json_str) = read_json_str(json, len) else {
      return -5;
    };
    let val = match serde_json::from_str::<serde_json::Value>(&json_str) {
      Ok(val) => val,
      Err(_) => return -5,
    };
    let doc = match value_to_document(val) {
      Ok(doc) => doc,
      Err(_) => return -6,
    };
    let mut writer = match h.index.writer() {
      Ok(w) => w,
      Err(err) => return classify_writer_err(&err, -4),
    };
    match writer.add_document(&doc) {
      Ok(res) => res as c_int,
      Err(_) => -2,
    }
  })
}

/// # Safety
/// Same as `searchlite_add_json` but accepts an optional write key (UTF-8 C string).
#[no_mangle]
pub unsafe extern "C" fn searchlite_add_json_with_write_key(
  handle: *mut IndexHandle,
  json: *const c_char,
  len: usize,
  write_key: *const c_char,
) -> c_int {
  catch_unwind_default("searchlite_add_json_with_write_key", ERR_PANIC, || {
    #[cfg(test)]
    maybe_panic_for_tests();

    if handle.is_null() || json.is_null() {
      return -1;
    }
    let h = &mut *handle;
    let write_key = match optional_cstr(write_key) {
      Ok(v) => v,
      Err(_) => return -8,
    };
    let Some(json_str) = read_json_str(json, len) else {
      return -5;
    };
    let val = match serde_json::from_str::<serde_json::Value>(&json_str) {
      Ok(val) => val,
      Err(_) => return -5,
    };
    let doc = match value_to_document(val) {
      Ok(doc) => doc,
      Err(_) => return -6,
    };
    let mut writer = match h.index.writer_with_key(write_key.as_deref()) {
      Ok(w) => w,
      // A caller that passed a key explicitly is asserting "I want auth"; any
      // failure at writer-open time on that path is an auth failure for
      // classification purposes (previous behaviour). For callers that did
      // not pass a key, only typed WriteKeyError variants are auth failures.
      Err(err) => {
        return if write_key.is_some() {
          -8
        } else {
          classify_writer_err(&err, -4)
        };
      }
    };
    match writer.add_document(&doc) {
      Ok(res) => res as c_int,
      Err(_) => -2,
    }
  })
}

/// # Safety
/// Adds one or more documents encoded as a JSON object or array of objects. Call `searchlite_commit` to make changes visible.
/// `handle` must be valid and `json` must point to UTF-8 data of length `len` (or be null-terminated when `len == 0`).
/// Returns the number of documents queued on success, or a negative error code on failure.
#[no_mangle]
pub unsafe extern "C" fn searchlite_add_json_batch(
  handle: *mut IndexHandle,
  json: *const c_char,
  len: usize,
) -> c_int {
  catch_unwind_default("searchlite_add_json_batch", ERR_PANIC, || {
    #[cfg(test)]
    maybe_panic_for_tests();

    if handle.is_null() || json.is_null() {
      return -1;
    }
    let h = &mut *handle;
    let Some(json_str) = read_json_str(json, len) else {
      return -5;
    };
    let val = match serde_json::from_str::<serde_json::Value>(&json_str) {
      Ok(val) => val,
      Err(_) => return -5,
    };
    let docs = match value_to_documents(val) {
      Ok(docs) => docs,
      Err(_) => return -6,
    };
    let mut writer = match h.index.writer() {
      Ok(w) => w,
      Err(err) => return classify_writer_err(&err, -4),
    };
    match writer.add_documents_batch(&docs) {
      Ok(count) => count as c_int,
      Err(_) => -2,
    }
  })
}

/// # Safety
/// Batch ingest with an optional write key.
#[no_mangle]
pub unsafe extern "C" fn searchlite_add_json_batch_with_write_key(
  handle: *mut IndexHandle,
  json: *const c_char,
  len: usize,
  write_key: *const c_char,
) -> c_int {
  catch_unwind_default(
    "searchlite_add_json_batch_with_write_key",
    ERR_PANIC,
    || {
      #[cfg(test)]
      maybe_panic_for_tests();

      if handle.is_null() || json.is_null() {
        return -1;
      }
      let h = &mut *handle;
      let write_key = match optional_cstr(write_key) {
        Ok(v) => v,
        Err(_) => return -8,
      };
      let Some(json_str) = read_json_str(json, len) else {
        return -5;
      };
      let val = match serde_json::from_str::<serde_json::Value>(&json_str) {
        Ok(val) => val,
        Err(_) => return -5,
      };
      let docs = match value_to_documents(val) {
        Ok(docs) => docs,
        Err(_) => return -6,
      };
      let mut writer = match h.index.writer_with_key(write_key.as_deref()) {
        Ok(w) => w,
        Err(err) => {
          return if write_key.is_some() {
            -8
          } else {
            classify_writer_err(&err, -4)
          };
        }
      };
      match writer.add_documents_batch(&docs) {
        Ok(count) => count as c_int,
        Err(_) => -2,
      }
    },
  )
}

/// # Safety
/// `handle` must be a valid pointer returned by `searchlite_index_open` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn searchlite_commit(handle: *mut IndexHandle) -> c_int {
  catch_unwind_default("searchlite_commit", ERR_PANIC, || {
    #[cfg(test)]
    maybe_panic_for_tests();

    if handle.is_null() {
      return -1;
    }
    let h = &mut *handle;
    match h.index.writer() {
      Ok(mut w) => match w.commit() {
        Ok(_) => 0,
        Err(_) => -2,
      },
      Err(err) => classify_writer_err(&err, -3),
    }
  })
}

/// # Safety
/// Commit pending writes with an optional write key (required when the index enforces one).
#[no_mangle]
pub unsafe extern "C" fn searchlite_commit_with_write_key(
  handle: *mut IndexHandle,
  write_key: *const c_char,
) -> c_int {
  catch_unwind_default("searchlite_commit_with_write_key", ERR_PANIC, || {
    #[cfg(test)]
    maybe_panic_for_tests();

    if handle.is_null() {
      return -1;
    }
    let h = &mut *handle;
    let write_key = match optional_cstr(write_key) {
      Ok(v) => v,
      Err(_) => return -8,
    };
    match h.index.writer_with_key(write_key.as_deref()) {
      Ok(mut w) => match w.commit() {
        Ok(_) => 0,
        Err(_) => -2,
      },
      Err(err) => {
        if write_key.is_some() {
          -8
        } else {
          classify_writer_err(&err, -3)
        }
      }
    }
  })
}

/// # Safety
/// `handle` must be a valid pointer from `searchlite_index_open`; `query` must be a valid C string; `cursor`, when provided,
/// must be a valid C string produced by a previous response; `aggs_json`, when provided, must point to `aggs_len` bytes of JSON;
/// `out_json_buf` must be a writable buffer of at least `buf_cap` bytes. Stored fields are omitted by default; set
/// `return_stored` via `searchlite_search_request` when needed.
#[no_mangle]
pub unsafe extern "C" fn searchlite_search(
  handle: *mut IndexHandle,
  query: *const c_char,
  limit: usize,
  cursor: *const c_char,
  aggs_json: *const c_char,
  aggs_len: usize,
  out_json_buf: *mut c_char,
  buf_cap: usize,
) -> isize {
  catch_unwind_default("searchlite_search", ERR_PANIC as isize, || {
    #[cfg(test)]
    maybe_panic_for_tests();

    if handle.is_null() || query.is_null() {
      return 0;
    }
    let h = &mut *handle;
    let query_str = CStr::from_ptr(query).to_string_lossy().to_string();
    let query_node: Query = serde_json::from_str::<QueryNode>(&query_str)
      .map(Query::Node)
      .unwrap_or_else(|_| query_str.clone().into());
    let cursor = if cursor.is_null() {
      None
    } else {
      Some(CStr::from_ptr(cursor).to_string_lossy().to_string())
    };
    let aggs_map: BTreeMap<String, Aggregation> = if !aggs_json.is_null() && aggs_len > 0 {
      let raw = slice::from_raw_parts(aggs_json as *const u8, aggs_len);
      let body = String::from_utf8_lossy(raw).to_string();
      match serde_json::from_str(&body) {
        Ok(map) => map,
        Err(err) => {
          eprintln!("searchlite_search: failed to parse aggregation JSON: {err}");
          return 0;
        }
      }
    } else {
      BTreeMap::new()
    };
    #[cfg(feature = "vectors")]
    let env_max_vec = parse_env_max_vector_candidates();
    let req = SearchRequest {
      query: query_node,
      fields: None,
      filter: None,
      limit,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: env_max_vec,
      sort: Vec::new(),
      cursor,
      search_after: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      track_total_hits: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      #[cfg(feature = "vectors")]
      vector_filter: None,
      return_stored: false,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs: aggs_map,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    };
    run_search(&h.index, req, out_json_buf, buf_cap) as isize
  })
}

/// # Safety
/// Executes a full `SearchRequest` encoded as JSON. The `request_json` pointer must reference UTF-8 data of length `request_len`
/// (or be null-terminated when `request_len == 0`). Results are written to `out_json_buf` as a JSON string. `return_stored`
/// defaults to `false` when omitted.
#[no_mangle]
pub unsafe extern "C" fn searchlite_search_request(
  handle: *mut IndexHandle,
  request_json: *const c_char,
  request_len: usize,
  out_json_buf: *mut c_char,
  buf_cap: usize,
) -> usize {
  catch_unwind_default("searchlite_search_request", 0, || {
    #[cfg(test)]
    maybe_panic_for_tests();

    if handle.is_null() || request_json.is_null() {
      return 0;
    }
    let h = &mut *handle;
    let Some(req_str) = read_json_str(request_json, request_len) else {
      return 0;
    };
    let req: SearchRequest = match serde_json::from_str(&req_str) {
      Ok(req) => req,
      Err(err) => {
        eprintln!("searchlite_search_request: invalid search request JSON: {err}");
        return 0;
      }
    };
    run_search(&h.index, req, out_json_buf, buf_cap)
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use serde_json::json;
  use std::ffi::{CStr, CString};
  use tempfile::tempdir;

  #[test]
  fn ffi_roundtrip_search() {
    let _guard = test_guard();
    let dir = tempdir().unwrap();
    let path = CString::new(dir.path().to_string_lossy().to_string()).unwrap();
    let handle = unsafe { searchlite_index_open(path.as_ptr(), true) };
    assert!(!handle.is_null());

    let doc = CString::new(r#"{"_id":"ffi-1","body":"hello from ffi"}"#).unwrap();
    let added = unsafe { searchlite_add_json(handle, doc.as_ptr(), doc.as_bytes().len()) };
    assert!(added >= 0);
    assert_eq!(unsafe { searchlite_commit(handle) }, 0);

    let mut buf = vec![0 as c_char; 1024];
    let query = CString::new("hello").unwrap();
    let written = unsafe {
      searchlite_search(
        handle,
        query.as_ptr(),
        5,
        std::ptr::null(),
        std::ptr::null(),
        0,
        buf.as_mut_ptr(),
        buf.len(),
      )
    };
    assert!(written > 0);
    unsafe { searchlite_index_close(handle) };
  }

  #[test]
  fn ffi_search_invalid_aggs_json_returns_error() {
    let _guard = test_guard();
    let dir = tempdir().unwrap();
    let path = CString::new(dir.path().to_string_lossy().to_string()).unwrap();
    let handle = unsafe { searchlite_index_open(path.as_ptr(), true) };
    assert!(!handle.is_null());

    let doc = CString::new(r#"{"_id":"ffi-1","body":"hello from ffi"}"#).unwrap();
    let added = unsafe { searchlite_add_json(handle, doc.as_ptr(), doc.as_bytes().len()) };
    assert!(added >= 0);
    assert_eq!(unsafe { searchlite_commit(handle) }, 0);

    let mut buf = vec![0 as c_char; 1024];
    let query = CString::new("hello").unwrap();
    let bad_aggs = CString::new("not valid json").unwrap();
    let written = unsafe {
      searchlite_search(
        handle,
        query.as_ptr(),
        5,
        std::ptr::null(),
        bad_aggs.as_ptr(),
        bad_aggs.as_bytes().len(),
        buf.as_mut_ptr(),
        buf.len(),
      )
    };
    assert_eq!(written, 0);

    unsafe { searchlite_index_close(handle) };
  }

  #[test]
  fn ffi_add_requires_commit() {
    let _guard = test_guard();
    let dir = tempdir().unwrap();
    let path = CString::new(dir.path().to_string_lossy().to_string()).unwrap();
    let handle = unsafe { searchlite_index_open(path.as_ptr(), true) };
    assert!(!handle.is_null());

    let doc = CString::new(r#"{"_id":"ffi-commit-1","body":"needs commit"}"#).unwrap();
    assert!(unsafe { searchlite_add_json(handle, doc.as_ptr(), doc.as_bytes().len()) } >= 0);

    let mut buf = vec![0 as c_char; 2048];
    let request = json!({
      "query": "commit",
      "limit": 5
    });
    let request_c = CString::new(request.to_string()).unwrap();
    let before = unsafe {
      searchlite_search_request(
        handle,
        request_c.as_ptr(),
        request_c.as_bytes().len(),
        buf.as_mut_ptr(),
        buf.len(),
      )
    };
    assert!(before > 0);
    let json_before = unsafe { CStr::from_ptr(buf.as_ptr()) }
      .to_string_lossy()
      .to_string();
    let parsed_before: serde_json::Value = serde_json::from_str(&json_before).unwrap();
    assert_eq!(parsed_before["hits"].as_array().unwrap().len(), 0);

    assert_eq!(unsafe { searchlite_commit(handle) }, 0);

    let after = unsafe {
      searchlite_search_request(
        handle,
        request_c.as_ptr(),
        request_c.as_bytes().len(),
        buf.as_mut_ptr(),
        buf.len(),
      )
    };
    assert!(after > 0);
    let json_after = unsafe { CStr::from_ptr(buf.as_ptr()) }
      .to_string_lossy()
      .to_string();
    let parsed_after: serde_json::Value = serde_json::from_str(&json_after).unwrap();
    assert_eq!(parsed_after["hits"].as_array().unwrap().len(), 1);

    unsafe { searchlite_index_close(handle) };
  }

  #[test]
  fn ffi_respects_return_stored_flag() {
    let _guard = test_guard();
    let dir = tempdir().unwrap();
    let path = CString::new(dir.path().to_string_lossy().to_string()).unwrap();
    let handle = unsafe { searchlite_index_open(path.as_ptr(), true) };
    assert!(!handle.is_null());

    let doc = CString::new(r#"{"_id":"ffi-store-1","body":"stored flag"}"#).unwrap();
    assert!(unsafe { searchlite_add_json(handle, doc.as_ptr(), doc.as_bytes().len()) } >= 0);
    assert_eq!(unsafe { searchlite_commit(handle) }, 0);

    let mut buf = vec![0 as c_char; 2048];
    let request_no_fields = json!({
      "query": "stored",
      "limit": 5
    });
    let request_no_fields_c = CString::new(request_no_fields.to_string()).unwrap();
    let written_no_fields = unsafe {
      searchlite_search_request(
        handle,
        request_no_fields_c.as_ptr(),
        request_no_fields_c.as_bytes().len(),
        buf.as_mut_ptr(),
        buf.len(),
      )
    };
    assert!(written_no_fields > 0);
    let json_no_fields = unsafe { CStr::from_ptr(buf.as_ptr()) }
      .to_string_lossy()
      .to_string();
    let parsed_no_fields: serde_json::Value = serde_json::from_str(&json_no_fields).unwrap();
    let hit_no_fields = &parsed_no_fields["hits"][0];
    assert!(hit_no_fields["fields"].is_null());

    let request_with_fields = json!({
      "query": "stored",
      "limit": 5,
      "return_stored": true
    });
    let request_with_fields_c = CString::new(request_with_fields.to_string()).unwrap();
    let written_with_fields = unsafe {
      searchlite_search_request(
        handle,
        request_with_fields_c.as_ptr(),
        request_with_fields_c.as_bytes().len(),
        buf.as_mut_ptr(),
        buf.len(),
      )
    };
    assert!(written_with_fields > 0);
    let json_with_fields = unsafe { CStr::from_ptr(buf.as_ptr()) }
      .to_string_lossy()
      .to_string();
    let parsed_with_fields: serde_json::Value = serde_json::from_str(&json_with_fields).unwrap();
    let hit_with_fields = &parsed_with_fields["hits"][0];
    assert!(hit_with_fields["fields"].is_object());

    unsafe { searchlite_index_close(handle) };
  }

  #[test]
  fn ffi_add_json_batch_adds_multiple_documents() {
    let _guard = test_guard();
    let dir = tempdir().unwrap();
    let path = CString::new(dir.path().to_string_lossy().to_string()).unwrap();
    let handle = unsafe { searchlite_index_open(path.as_ptr(), true) };
    assert!(!handle.is_null());

    let docs = CString::new(
      r#"[{"_id":"batch-1","body":"shared term"},{"_id":"batch-2","body":"shared term"}]"#,
    )
    .unwrap();
    let added = unsafe { searchlite_add_json_batch(handle, docs.as_ptr(), docs.as_bytes().len()) };
    assert_eq!(added, 2);
    assert_eq!(unsafe { searchlite_commit(handle) }, 0);

    let mut buf = vec![0 as c_char; 2048];
    let request = json!({
      "query": "shared",
      "limit": 10
    });
    let request_c = CString::new(request.to_string()).unwrap();
    let written = unsafe {
      searchlite_search_request(
        handle,
        request_c.as_ptr(),
        request_c.as_bytes().len(),
        buf.as_mut_ptr(),
        buf.len(),
      )
    };
    assert!(written > 0);
    let json = unsafe { CStr::from_ptr(buf.as_ptr()) }
      .to_string_lossy()
      .to_string();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["hits"].as_array().unwrap().len(), 2);

    unsafe { searchlite_index_close(handle) };
  }

  #[test]
  fn ffi_panic_is_contained_and_returns_error_code() {
    let _guard = test_guard();

    let dir = tempdir().unwrap();
    let path = CString::new(dir.path().to_string_lossy().to_string()).unwrap();
    let handle = unsafe { searchlite_index_open(path.as_ptr(), true) };
    assert!(!handle.is_null());

    request_panic_for_next_call();
    let doc = CString::new(r#"{"_id":"panic","body":"boom"}"#).unwrap();
    let res = unsafe { searchlite_add_json(handle, doc.as_ptr(), doc.as_bytes().len()) };
    assert_eq!(res, ERR_PANIC);

    let doc_ok = CString::new(r#"{"_id":"ok","body":"still works"}"#).unwrap();
    let added = unsafe { searchlite_add_json(handle, doc_ok.as_ptr(), doc_ok.as_bytes().len()) };
    assert!(added >= 0);
    assert_eq!(unsafe { searchlite_commit(handle) }, 0);

    let mut buf = vec![0 as c_char; 512];
    let query = CString::new("still").unwrap();
    let written = unsafe {
      searchlite_search(
        handle,
        query.as_ptr(),
        5,
        std::ptr::null(),
        std::ptr::null(),
        0,
        buf.as_mut_ptr(),
        buf.len(),
      )
    };
    assert!(written > 0);

    unsafe { searchlite_index_close(handle) };
  }

  #[test]
  fn ffi_write_key_protected_index_requires_key() {
    let _guard = test_guard();
    let dir = tempdir().unwrap();
    let path = CString::new(dir.path().to_string_lossy().to_string()).unwrap();
    let key = CString::new("test-write-key-123").unwrap();
    let handle = unsafe { searchlite_index_open_with_write_key(path.as_ptr(), true, key.as_ptr()) };
    assert!(!handle.is_null());

    // Missing key should fail to add.
    let doc = CString::new(r#"{"_id":"1","body":"secured"}"#).unwrap();
    let added = unsafe { searchlite_add_json(handle, doc.as_ptr(), doc.as_bytes().len()) };
    assert!(added < 0);

    // Providing the correct key should succeed.
    let added_with_key = unsafe {
      searchlite_add_json_with_write_key(handle, doc.as_ptr(), doc.as_bytes().len(), key.as_ptr())
    };
    assert!(added_with_key >= 0);
    assert_eq!(
      unsafe { searchlite_commit_with_write_key(handle, key.as_ptr()) },
      0
    );

    unsafe { searchlite_index_close(handle) };
  }

  // Regression tests for BUG-029: callers must be able to distinguish a
  // successful write from an undersized buffer rather than receiving a
  // silently-truncated JSON payload.
  mod write_json_to_buffer {
    use super::super::write_json_to_buffer;
    use std::ffi::CStr;
    use std::os::raw::c_char;

    #[test]
    fn success_returns_bytes_written_and_nul_terminates() {
      let json = "{\"a\":1}".to_string();
      let expected_len = json.len();
      let mut buf = vec![0xAAu8 as c_char; 64];
      let written = write_json_to_buffer(json.clone(), buf.as_mut_ptr(), buf.len());
      assert_eq!(written, expected_len);
      assert!(written < buf.len(), "success return must be < buf_cap");
      let c_str = unsafe { CStr::from_ptr(buf.as_ptr()) };
      assert_eq!(c_str.to_str().unwrap(), json);
    }

    #[test]
    fn undersized_buffer_returns_required_size_not_truncated_payload() {
      // Payload is 7 bytes, so required = 8 (including NUL). Give the caller
      // only 4 bytes.
      let json = "{\"a\":1}".to_string();
      let required = json.len() + 1;
      let mut buf = vec![0xAAu8 as c_char; 4];
      let written = write_json_to_buffer(json, buf.as_mut_ptr(), buf.len());
      // The return value must exceed buf_cap so callers can detect the
      // too-small-buffer condition, and it must equal the required size.
      assert_eq!(written, required);
      assert!(written > buf.len(), "required size must exceed buf_cap");
      // The buffer must be a valid empty C string rather than a truncated
      // JSON fragment (no silent data loss).
      assert_eq!(buf[0], 0);
      let c_str = unsafe { CStr::from_ptr(buf.as_ptr()) };
      assert_eq!(c_str.to_bytes(), b"");
    }

    #[test]
    fn zero_cap_buffer_still_reports_required_size() {
      // buf_cap == 0 used to be indistinguishable from other errors because
      // the helper returned 0. Callers must now learn the required size.
      let json = "{}".to_string();
      let required = json.len() + 1;
      let mut sentinel = 0u8 as c_char;
      let written = write_json_to_buffer(json, &mut sentinel as *mut c_char, 0);
      assert_eq!(written, required);
      assert!(written > 0);
    }

    #[test]
    fn exact_fit_buffer_is_success_not_truncation() {
      // buf_cap == required_size (payload + NUL) must be treated as success,
      // not as "buffer too small". The return must be bytes_written, which is
      // strictly less than buf_cap so callers see a value < buf_cap.
      let json = "hi".to_string();
      let payload_len = json.len();
      let mut buf = vec![0xAAu8 as c_char; payload_len + 1];
      let buf_cap = buf.len();
      let written = write_json_to_buffer(json, buf.as_mut_ptr(), buf_cap);
      assert_eq!(written, payload_len);
      assert!(written < buf_cap, "success return must be < buf_cap");
      let c_str = unsafe { CStr::from_ptr(buf.as_ptr()) };
      assert_eq!(c_str.to_str().unwrap(), "hi");
    }

    #[test]
    fn null_pointer_returns_zero_for_error() {
      // With a null pointer we have no way to signal a required size, so
      // fall back to the generic "0 means error" convention.
      let json = "{}".to_string();
      let written = write_json_to_buffer(json, std::ptr::null_mut(), 0);
      assert_eq!(written, 0);
    }
  }

  #[test]
  fn ffi_search_request_signals_buffer_too_small_without_truncation() {
    let _guard = test_guard();
    let dir = tempdir().unwrap();
    let path = CString::new(dir.path().to_string_lossy().to_string()).unwrap();
    let handle = unsafe { searchlite_index_open(path.as_ptr(), true) };
    assert!(!handle.is_null());

    let doc = CString::new(r#"{"_id":"small-buf","body":"buffer too small regression"}"#).unwrap();
    assert!(unsafe { searchlite_add_json(handle, doc.as_ptr(), doc.as_bytes().len()) } >= 0);
    assert_eq!(unsafe { searchlite_commit(handle) }, 0);

    let request = json!({
      "query": "buffer",
      "limit": 5,
      "return_stored": true
    });
    let request_c = CString::new(request.to_string()).unwrap();

    // First: deliberately undersized buffer. The old behaviour silently
    // truncated the JSON; the new contract returns the required size.
    let small_cap = 8usize;
    let mut small_buf = vec![0xAAu8 as c_char; small_cap];
    let small_ret = unsafe {
      searchlite_search_request(
        handle,
        request_c.as_ptr(),
        request_c.as_bytes().len(),
        small_buf.as_mut_ptr(),
        small_cap,
      )
    };
    assert!(
      small_ret > small_cap,
      "buffer-too-small must be signalled as return > buf_cap, got {small_ret} for cap {small_cap}"
    );
    // Buffer must not contain a truncated JSON fragment.
    let small_bytes = unsafe { CStr::from_ptr(small_buf.as_ptr()) }.to_bytes();
    assert!(
      small_bytes.is_empty(),
      "undersized buffer must not contain truncated payload, got {:?}",
      String::from_utf8_lossy(small_bytes)
    );

    // Second: allocate the required size returned by the first call and
    // retry. The new contract guarantees this retry succeeds.
    let mut big_buf = vec![0u8 as c_char; small_ret];
    let big_ret = unsafe {
      searchlite_search_request(
        handle,
        request_c.as_ptr(),
        request_c.as_bytes().len(),
        big_buf.as_mut_ptr(),
        big_buf.len(),
      )
    };
    assert!(big_ret > 0);
    assert!(
      big_ret < big_buf.len(),
      "success return must be < buf_cap (got {big_ret} / {})",
      big_buf.len()
    );
    let parsed: serde_json::Value =
      serde_json::from_slice(unsafe { CStr::from_ptr(big_buf.as_ptr()) }.to_bytes())
        .expect("retry result must be valid JSON");
    assert_eq!(parsed["hits"].as_array().unwrap().len(), 1);

    unsafe { searchlite_index_close(handle) };
  }

  #[test]
  fn ffi_search_signals_buffer_too_small_via_positive_required_size() {
    let _guard = test_guard();
    let dir = tempdir().unwrap();
    let path = CString::new(dir.path().to_string_lossy().to_string()).unwrap();
    let handle = unsafe { searchlite_index_open(path.as_ptr(), true) };
    assert!(!handle.is_null());

    let doc = CString::new(r#"{"_id":"s1","body":"search signal regression"}"#).unwrap();
    assert!(unsafe { searchlite_add_json(handle, doc.as_ptr(), doc.as_bytes().len()) } >= 0);
    assert_eq!(unsafe { searchlite_commit(handle) }, 0);

    let query = CString::new("search").unwrap();
    let small_cap = 8usize;
    let mut small_buf = vec![0xAAu8 as c_char; small_cap];
    let ret = unsafe {
      searchlite_search(
        handle,
        query.as_ptr(),
        5,
        std::ptr::null(),
        std::ptr::null(),
        0,
        small_buf.as_mut_ptr(),
        small_cap,
      )
    };
    // Positive and greater than buf_cap signals "required size".
    assert!(
      ret > small_cap as isize,
      "expected required size > buf_cap, got {ret}"
    );
    let bytes = unsafe { CStr::from_ptr(small_buf.as_ptr()) }.to_bytes();
    assert!(bytes.is_empty(), "buffer must not contain truncated data");

    unsafe { searchlite_index_close(handle) };
  }

  // Regression tests for BUG-020: the FFI classifies write-key (auth)
  // failures via a typed `anyhow::Error::downcast_ref::<WriteKeyError>()`
  // rather than sniffing the error's Display form. These tests pin the new
  // contract so a future rename of the error message cannot silently flip
  // the returned error code from -8 to -4/-3.
  mod classify_writer_err {
    use super::super::{
      classify_writer_err, searchlite_add_json, searchlite_add_json_batch, searchlite_commit,
      searchlite_index_close, searchlite_index_open_with_write_key, test_guard,
    };
    use searchlite_core::api::WriteKeyError;
    use std::ffi::CString;
    use tempfile::tempdir;

    #[test]
    fn classifies_typed_write_key_error_as_auth_failure() {
      let err = anyhow::Error::from(WriteKeyError::Required);
      assert_eq!(classify_writer_err(&err, -4), -8);

      let err = anyhow::Error::from(WriteKeyError::Mismatch(
        "WAL binding; index may be tampered",
      ));
      assert_eq!(classify_writer_err(&err, -4), -8);

      let err = anyhow::Error::from(WriteKeyError::MetadataTampered);
      assert_eq!(classify_writer_err(&err, -4), -8);

      let err = anyhow::Error::from(WriteKeyError::FeatureDisabled);
      assert_eq!(classify_writer_err(&err, -4), -8);
    }

    #[test]
    fn classifies_typed_write_key_error_even_when_context_is_added() {
      // anyhow::Context wraps the inner error; downcast_ref must still find
      // the WriteKeyError through the wrapper. This is the core property
      // that BUG-020's substring-match approach broke whenever a caller
      // added `.context("...")` that shadowed the "write key" substring.
      let err = anyhow::Error::from(WriteKeyError::Required)
        .context("while opening writer for background compaction")
        .context("completely unrelated wrapper that hides the original message");
      assert_eq!(classify_writer_err(&err, -4), -8);
    }

    #[test]
    fn does_not_classify_unrelated_errors_as_auth_failure() {
      let err = anyhow::anyhow!("some generic failure unrelated to auth");
      assert_eq!(classify_writer_err(&err, -4), -4);
      assert_eq!(classify_writer_err(&err, -3), -3);
      assert_eq!(classify_writer_err(&err, -2), -2);
    }

    #[test]
    fn does_not_auth_classify_message_that_merely_contains_write_key_substring() {
      // Previous substring-matching logic returned -8 for any error whose
      // Display happened to include "write key" — e.g. a future feature
      // error such as "a write key index cannot be opened in this mode".
      // With typed classification, only the typed variant triggers -8.
      let err = anyhow::anyhow!("index contains 'write key' marker but this is a storage error");
      assert_eq!(classify_writer_err(&err, -4), -4);
    }

    // End-to-end: a protected index without a provided key must still
    // classify as -8 through the full FFI pipeline. This was already the
    // behaviour before the fix, but it is re-verified to guard against
    // regression in the downcast-driven path.
    #[test]
    fn protected_index_without_key_returns_minus_eight_via_ffi() {
      let _guard = test_guard();
      let dir = tempdir().unwrap();
      let path = CString::new(dir.path().to_string_lossy().to_string()).unwrap();
      let key = CString::new("key-for-bug-020").unwrap();
      let handle =
        unsafe { searchlite_index_open_with_write_key(path.as_ptr(), true, key.as_ptr()) };
      assert!(!handle.is_null());

      // Missing key for add_json: FFI must see the typed error and return -8.
      let doc = CString::new(r#"{"_id":"x","body":"auth"}"#).unwrap();
      let added = unsafe { searchlite_add_json(handle, doc.as_ptr(), doc.as_bytes().len()) };
      assert_eq!(added, -8, "missing write key must return -8, got {added}");

      // Same for add_json_batch.
      let docs = CString::new(r#"[{"_id":"y","body":"batch-auth"}]"#).unwrap();
      let batched =
        unsafe { searchlite_add_json_batch(handle, docs.as_ptr(), docs.as_bytes().len()) };
      assert_eq!(
        batched, -8,
        "missing write key must return -8, got {batched}"
      );

      // And for commit.
      let committed = unsafe { searchlite_commit(handle) };
      assert_eq!(
        committed, -8,
        "missing write key must return -8, got {committed}"
      );

      unsafe { searchlite_index_close(handle) };
    }
  }
}
