use std::collections::BTreeMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::slice;
use std::str;

use searchlite_core::api::types::{
  Aggregation, Document, ExecutionStrategy, IndexOptions, Query, QueryNode, SearchRequest,
  StorageType,
};
use searchlite_core::api::Index;

#[repr(C)]
pub struct IndexHandle {
  index: Index,
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

fn value_to_documents(value: serde_json::Value) -> Result<Vec<Document>, ()> {
  match value {
    serde_json::Value::Array(items) => items.into_iter().map(value_to_document).collect(),
    obj @ serde_json::Value::Object(_) => Ok(vec![value_to_document(obj)?]),
    _ => Err(()),
  }
}

fn write_json_to_buffer(json: String, out_json_buf: *mut c_char, buf_cap: usize) -> usize {
  if out_json_buf.is_null() || buf_cap == 0 {
    return 0;
  }
  let bytes = json.as_bytes();
  let len = bytes.len().min(buf_cap.saturating_sub(1));
  unsafe {
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_json_buf as *mut u8, len);
    *out_json_buf.add(len) = 0;
  }
  len
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
}

/// # Safety
/// `handle` must be a pointer returned by `searchlite_index_open` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn searchlite_index_close(handle: *mut IndexHandle) {
  if handle.is_null() {
    return;
  }
  drop(Box::from_raw(handle));
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
    Err(_) => return -4,
  };
  match writer.add_document(&doc) {
    Ok(res) => res as c_int,
    Err(_) => -2,
  }
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
    Err(_) => return -4,
  };
  let mut added: c_int = 0;
  for doc in docs.iter() {
    match writer.add_document(doc) {
      Ok(_) => added += 1,
      Err(_) => return -2,
    }
  }
  added
}

/// # Safety
/// `handle` must be a valid pointer returned by `searchlite_index_open` and not already freed.
#[no_mangle]
pub unsafe extern "C" fn searchlite_commit(handle: *mut IndexHandle) -> c_int {
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
) -> usize {
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
  let req = SearchRequest {
    query: query_node,
    fields: None,
    filter: None,
    limit,
    return_hits: true,
    candidate_size: None,
    sort: Vec::new(),
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    fuzzy: None,
    return_stored: false,
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
  run_search(&h.index, req, out_json_buf, buf_cap)
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
}

#[cfg(test)]
mod tests {
  use super::*;
  use serde_json::json;
  use std::ffi::{CStr, CString};
  use tempfile::tempdir;

  #[test]
  fn ffi_roundtrip_search() {
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
}
