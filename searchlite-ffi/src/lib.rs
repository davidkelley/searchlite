use std::any::Any;
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;

#[cfg(test)]
use std::sync::{
  atomic::{AtomicBool, Ordering},
  Mutex,
};

use searchlite_core::api::types::{
  Aggregation, Document, ExecutionStrategy, IndexOptions, Query, QueryNode, SearchRequest,
  StorageType,
};
use searchlite_core::api::Index;

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

    if path.is_null() {
      return std::ptr::null_mut();
    }
    let c_str = CStr::from_ptr(path);
    let path_buf = PathBuf::from(c_str.to_string_lossy().to_string());
    let opts = IndexOptions {
      path: path_buf,
      create_if_missing,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    };
    match Index::open(opts) {
      Ok(index) => Box::into_raw(Box::new(IndexHandle { index })),
      Err(_) => std::ptr::null_mut(),
    }
  })
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
/// `handle` must be a valid pointer returned by `searchlite_index_open`; `json` must be a valid, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn searchlite_add_json(
  handle: *mut IndexHandle,
  json: *const c_char,
  _len: usize,
) -> c_int {
  catch_unwind_default("searchlite_add_json", ERR_PANIC, || {
    #[cfg(test)]
    maybe_panic_for_tests();

    if handle.is_null() || json.is_null() {
      return -1;
    }
    let h = &mut *handle;
    let json_str = CStr::from_ptr(json).to_string_lossy().to_string();
    match serde_json::from_str::<serde_json::Value>(&json_str) {
      Ok(val) => {
        let mut fields = BTreeMap::new();
        if let Some(map) = val.as_object() {
          for (k, v) in map.iter() {
            fields.insert(k.clone(), v.clone());
          }
        }
        let doc = Document { fields };
        if let Ok(mut writer) = h.index.writer() {
          let res = writer.add_document(&doc);
          if res.is_err() {
            return -2;
          }
          if writer.commit().is_err() {
            return -3;
          }
          return res.unwrap() as c_int;
        }
        -4
      }
      Err(_) => -5,
    }
  })
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
      Err(_) => -3,
    }
  })
}

/// # Safety
/// `handle` must be a valid pointer from `searchlite_index_open`; `query` must be a valid C string; `cursor`, when provided,
/// must be a valid C string produced by a previous response; `aggs_json`, when provided, must point to `aggs_len` bytes of JSON;
/// `out_json_buf` must be a writable buffer of at least `buf_cap` bytes. Returns the number of bytes written, or a negative error
/// code. On panic the function returns `ERR_PANIC` and the operation is aborted; reopen the index after panics from mutating calls
/// to guarantee consistency before retrying.
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
    let reader = match h.index.reader() {
      Ok(r) => r,
      Err(_) => return 0,
    };
    let cursor = if cursor.is_null() {
      None
    } else {
      Some(CStr::from_ptr(cursor).to_string_lossy().to_string())
    };
    #[cfg(feature = "vectors")]
    let env_max_vec = searchlite_core::api::types::parse_env_max_vector_candidates();
    let aggs_map: BTreeMap<String, Aggregation> = if !aggs_json.is_null() && aggs_len > 0 {
      let raw = std::slice::from_raw_parts(aggs_json as *const u8, aggs_len);
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
    let req = SearchRequest {
      query: query_node,
      fields: None,
      filter: None,
      limit,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: env_max_vec,
      sort: Vec::new(),
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      return_stored: true,
      highlight_field: None,
      highlight: None,
      collapse: None,
      cursor,
      aggs: aggs_map,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
      #[cfg(feature = "vectors")]
      vector_query: None,

      #[cfg(feature = "vectors")]
      vector_filter: None,
    };
    let res = match reader.search(&req) {
      Ok(r) => r,
      Err(_) => return 0,
    };
    if out_json_buf.is_null() || buf_cap == 0 {
      return 0;
    }
    let encoded = serde_json::to_string(&res).unwrap_or_else(|_| "{}".to_string());
    let bytes = encoded.as_bytes();
    let len = bytes.len().min(buf_cap.saturating_sub(1));
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_json_buf as *mut u8, len);
    *out_json_buf.add(len) = 0;
    len as isize
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::ffi::CString;
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
}
