use std::collections::BTreeMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;

use searchlite_core::api::types::{
  Aggregation, Document, ExecutionStrategy, IndexOptions, SearchRequest, StorageType,
};
use searchlite_core::api::Index;

#[repr(C)]
pub struct IndexHandle {
  index: Index,
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
/// `handle` must be a valid pointer from `searchlite_index_open`, and `json` must point to a valid, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn searchlite_add_json(
  handle: *mut IndexHandle,
  json: *const c_char,
  _len: usize,
) -> c_int {
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
/// `out_json_buf` must be a writable buffer of at least `buf_cap` bytes.
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
  let reader = match h.index.reader() {
    Ok(r) => r,
    Err(_) => return 0,
  };
  let cursor = if cursor.is_null() {
    None
  } else {
    Some(CStr::from_ptr(cursor).to_string_lossy().to_string())
  };
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
    query: query_str,
    fields: None,
    filters: vec![],
    limit,
    sort: Vec::new(),
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    return_stored: true,
    highlight_field: None,
    cursor,
    aggs: aggs_map,
    #[cfg(feature = "vectors")]
    vector_query: None,
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
  len
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::ffi::CString;
  use tempfile::tempdir;

  #[test]
  fn ffi_roundtrip_search() {
    let dir = tempdir().unwrap();
    let path = CString::new(dir.path().to_string_lossy().to_string()).unwrap();
    let handle = unsafe { searchlite_index_open(path.as_ptr(), true) };
    assert!(!handle.is_null());

    let doc = CString::new(r#"{"body":"hello from ffi"}"#).unwrap();
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

    let doc = CString::new(r#"{"body":"hello from ffi"}"#).unwrap();
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
}
