use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use futures::channel::oneshot;
use parking_lot::Mutex;
use parking_lot::RwLock;
use searchlite_core::api::types::{
  Aggregation, ExecutionStrategy, Filter, IndexOptions, SearchRequest, SortSpec, StorageType,
};
use searchlite_core::api::{Document, IndexReader, IndexWriter};
use searchlite_core::storage::{DynFile, Storage, StorageFile};
use searchlite_core::{Index, Schema};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{spawn_local, JsFuture};
#[cfg(feature = "threads")]
use wasm_bindgen_rayon::init_thread_pool;

const STORE_NAME: &str = "searchlite_files";

fn path_key(path: &Path) -> String {
  path.to_string_lossy().to_string()
}

fn to_js_error(err: impl std::fmt::Display) -> JsValue {
  JsValue::from_str(&err.to_string())
}

fn value_to_document(value: serde_json::Value) -> Result<Document, JsValue> {
  let obj = value
    .as_object()
    .ok_or_else(|| JsValue::from_str("document must be a JSON object"))?;
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
    _ => Err(JsValue::from_str(
      "documents must be an object or array of objects",
    )),
  }
}

fn request_future(req: &web_sys::IdbRequest) -> impl std::future::Future<Output = Result<JsValue>> {
  let success_req = req.clone();
  let error_req = req.clone();
  let promise = js_sys::Promise::new(&mut |resolve, reject| {
    let success = Closure::wrap(Box::new(move |event: web_sys::Event| {
      if let Some(target) = event.target() {
        if let Ok(req) = target.dyn_into::<web_sys::IdbRequest>() {
          if let Ok(result) = req.result() {
            let _ = resolve.call1(&JsValue::UNDEFINED, &result);
            return;
          }
        }
      }
      let _ = resolve.call0(&JsValue::UNDEFINED);
    }) as Box<dyn FnMut(_)>);
    let err_req = error_req.clone();
    let error = Closure::wrap(Box::new(move |_event: web_sys::Event| {
      let err_val = match err_req.error() {
        Ok(e) => e.into(),
        Err(_) => JsValue::from_str("indexeddb request error"),
      };
      let _ = reject.call1(&JsValue::UNDEFINED, &err_val);
    }) as Box<dyn FnMut(_)>);
    success_req.set_onsuccess(Some(success.as_ref().unchecked_ref()));
    error_req.set_onerror(Some(error.as_ref().unchecked_ref()));
    success.forget();
    error.forget();
  });
  async move {
    JsFuture::from(promise)
      .await
      .map_err(|err| anyhow!("indexeddb request failed: {:?}", err))
  }
}

async fn open_db(name: &str) -> Result<web_sys::IdbDatabase> {
  let window = web_sys::window().ok_or_else(|| anyhow!("window missing"))?;
  let factory = window
    .indexed_db()
    .map_err(|e| anyhow!("indexed_db error: {:?}", e))?
    .ok_or_else(|| anyhow!("IndexedDB unavailable"))?;
  let request = factory
    .open_with_u32(name, 1)
    .map_err(|e| anyhow!("indexed_db open error: {:?}", e))?;
  {
    let store = STORE_NAME.to_string();
    let upgrade = Closure::wrap(Box::new(move |event: web_sys::Event| {
      if let Some(target) = event.target() {
        if let Ok(req) = target.dyn_into::<web_sys::IdbOpenDbRequest>() {
          if let Ok(result) = req.result() {
            if let Ok(db) = result.dyn_into::<web_sys::IdbDatabase>() {
              let _ = db.create_object_store(&store);
            }
          }
        }
      }
    }) as Box<dyn FnMut(_)>);
    request.set_onupgradeneeded(Some(upgrade.as_ref().unchecked_ref()));
    upgrade.forget();
  }
  let request: web_sys::IdbRequest = request.into();
  let db_value = request_future(&request).await?;
  db_value
    .dyn_into::<web_sys::IdbDatabase>()
    .map_err(|_| anyhow!("failed to open IndexedDB database"))
}

async fn load_snapshot(db_name: &str) -> Result<HashMap<PathBuf, Vec<u8>>> {
  let db = open_db(db_name).await?;
  let tx = db
    .transaction_with_str_and_mode(STORE_NAME, web_sys::IdbTransactionMode::Readonly)
    .map_err(|e| anyhow!("opening transaction for {STORE_NAME}: {:?}", e))?;
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
  let keys: js_sys::Array = keys_val.dyn_into().unwrap_or_else(|_| js_sys::Array::new());
  let values: js_sys::Array = values_val
    .dyn_into()
    .unwrap_or_else(|_| js_sys::Array::new());
  let mut map = HashMap::new();
  for (key, value) in keys.iter().zip(values.iter()) {
    if let Some(name) = key.as_string() {
      let bytes = js_sys::Uint8Array::new(&value).to_vec();
      map.insert(PathBuf::from(name), bytes);
    }
  }
  Ok(map)
}

async fn persist_file(db_name: &str, path: &Path, data: Vec<u8>) -> Result<()> {
  let db = open_db(db_name).await?;
  let tx = db
    .transaction_with_str_and_mode(STORE_NAME, web_sys::IdbTransactionMode::Readwrite)
    .map_err(|e| anyhow!("opening rw transaction for {STORE_NAME}: {:?}", e))?;
  let store = tx
    .object_store(STORE_NAME)
    .map_err(|e| anyhow!("opening object store {STORE_NAME}: {:?}", e))?;
  let key = JsValue::from_str(&path_key(path));
  let value: JsValue = js_sys::Uint8Array::from(data.as_slice()).into();
  let req = store
    .put_with_key(&value, &key)
    .map_err(|e| anyhow!("put_with_key failed: {:?}", e))?;
  request_future(&req).await?;
  Ok(())
}

#[derive(Clone)]
struct PendingWrites {
  db_name: String,
  pending: Arc<Mutex<Vec<oneshot::Receiver<()>>>>,
}

impl PendingWrites {
  fn new(db_name: String) -> Self {
    Self {
      db_name,
      pending: Arc::new(Mutex::new(Vec::new())),
    }
  }

  fn schedule(&self, path: PathBuf, data: Vec<u8>) {
    let (tx, rx) = oneshot::channel();
    let db = self.db_name.clone();
    spawn_local(async move {
      if let Err(err) = persist_file(&db, &path, data).await {
        web_sys::console::error_1(&JsValue::from_str(&format!(
          "persist error for {:?}: {err}",
          path
        )));
      }
      let _ = tx.send(());
    });
    self.pending.lock().push(rx);
  }

  async fn flush(&self) {
    let receivers = {
      let mut guard = self.pending.lock();
      std::mem::take(&mut *guard)
    };
    for rx in receivers {
      let _ = rx.await;
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

  pub async fn flush(&self) {
    self.pending.flush().await;
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
    if !self.exists(path) {
      return Err(anyhow!("file {:?} missing", path));
    }
    self.open_with_mode(path, false, false)
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
}

impl JsStorage {
  fn open_with_mode(&self, path: &Path, truncate: bool, append: bool) -> Result<DynFile> {
    let data = self.entry(path);
    if truncate {
      data.write().clear();
    }
    let pos = if append { data.read().len() as u64 } else { 0 };
    Ok(Box::new(JsFile {
      path: path.to_path_buf(),
      data,
      pos,
      pending: self.pending.clone(),
    }))
  }
}

struct JsFile {
  path: PathBuf,
  data: Arc<RwLock<Vec<u8>>>,
  pos: u64,
  pending: PendingWrites,
}

impl Drop for JsFile {
  fn drop(&mut self) {
    let data = self.data.read().clone();
    self.pending.schedule(self.path.clone(), data);
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
    let end = (self.pos as usize).saturating_add(buf.len());
    if end > data.len() {
      data.resize(end, 0);
    }
    data[self.pos as usize..end].copy_from_slice(buf);
    self.pos = end as u64;
    Ok(buf.len())
  }

  fn flush(&mut self) -> std::io::Result<()> {
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
    self.pending.schedule(self.path.clone(), data.clone());
    Ok(())
  }

  fn sync_all(&mut self) -> Result<()> {
    let data = self.data.read().clone();
    self.pending.schedule(self.path.clone(), data);
    Ok(())
  }
}

#[wasm_bindgen]
pub struct Searchlite {
  index: Index,
  storage: Arc<JsStorage>,
}

#[wasm_bindgen]
impl Searchlite {
  async fn create(db_name: String, schema_json: String) -> Result<Searchlite, JsValue> {
    let schema: Schema =
      serde_json::from_str(&schema_json).map_err(|err| JsValue::from_str(&err.to_string()))?;
    let root = PathBuf::from(db_name.clone());
    let storage = Arc::new(
      JsStorage::new(db_name.clone(), root.clone())
        .await
        .map_err(to_js_error)?,
    );
    let opts = IndexOptions {
      path: root.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::InMemory,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    };
    let index =
      Index::create_with_storage(&root, schema, opts, storage.clone()).map_err(to_js_error)?;
    storage.flush().await;
    Ok(Searchlite { index, storage })
  }

  /// Preferred async constructor for wasm (avoids async ctor warnings in bindings).
  #[wasm_bindgen(js_name = init)]
  pub async fn init(db_name: String, schema_json: String) -> Result<Searchlite, JsValue> {
    Self::create(db_name, schema_json).await
  }

  /// Initialize the rayon pool for threaded execution. COOP/COEP (cross-origin isolation) must
  /// be handled by the embedding app; this helper does not set headers for you.
  #[cfg(feature = "threads")]
  pub async fn init_threads(&self, threads: Option<u32>) -> Result<(), JsValue> {
    let desired = threads.unwrap_or_else(|| {
      web_sys::window()
        .map(|w| w.navigator().hardware_concurrency())
        .map(|n| n as u32)
        .filter(|&n| n > 0)
        .unwrap_or(1)
    });
    JsFuture::from(init_thread_pool(desired as usize))
      .await
      .map(|_| ())
      .map_err(|err| JsValue::from_str(&format!("{err:?}")))
  }

  /// Threaded mode is disabled unless the `threads` feature is enabled.
  #[cfg(not(feature = "threads"))]
  pub async fn init_threads(&self, _threads: Option<u32>) -> Result<(), JsValue> {
    Err(JsValue::from_str(
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

  pub async fn add_document(&self, doc: JsValue) -> Result<(), JsValue> {
    let value: serde_json::Value =
      serde_wasm_bindgen::from_value(doc).map_err(|err| JsValue::from_str(&err.to_string()))?;
    self.add_documents_internal(vec![value_to_document(value)?])
  }

  pub async fn add_documents(&self, docs: JsValue) -> Result<(), JsValue> {
    let value: serde_json::Value =
      serde_wasm_bindgen::from_value(docs).map_err(|err| JsValue::from_str(&err.to_string()))?;
    self.add_documents_internal(value_to_documents(value)?)
  }

  pub async fn commit(&self) -> Result<(), JsValue> {
    let mut writer: IndexWriter = self.index.writer().map_err(to_js_error)?;
    writer.commit().map_err(to_js_error)?;
    self.storage.flush().await;
    Ok(())
  }

  pub async fn search(&self, query: String, limit: usize) -> Result<JsValue, JsValue> {
    let request = SearchRequest {
      query,
      fields: None,
      filters: Vec::<Filter>::new(),
      limit,
      sort: Vec::<SortSpec>::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: true,
      highlight_field: None,
      aggs: BTreeMap::<String, Aggregation>::new(),
    };
    self.run_search(request)
  }

  pub async fn search_request(&self, request_json: String) -> Result<JsValue, JsValue> {
    let req: SearchRequest = serde_json::from_str(&request_json)
      .map_err(|err| JsValue::from_str(&format!("invalid search request: {err}")))?;
    self.run_search(req)
  }

  fn run_search(&self, req: SearchRequest) -> Result<JsValue, JsValue> {
    let reader: IndexReader = self.index.reader().map_err(to_js_error)?;
    let result = reader.search(&req).map_err(to_js_error)?;
    serde_wasm_bindgen::to_value(&result).map_err(|err| JsValue::from_str(&err.to_string()))
  }

  pub async fn flush_storage(&self) -> Result<(), JsValue> {
    self.storage.flush().await;
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use searchlite_core::api::types::ExecutionStrategy;
  use wasm_bindgen_test::*;

  wasm_bindgen_test_configure!(run_in_browser);

  fn unique_db(name: &str) -> String {
    format!("{name}-{}", js_sys::Date::now() as u64)
  }

  #[wasm_bindgen_test]
  async fn js_storage_persists_entries() {
    let db = unique_db("searchlite-storage");
    let root = PathBuf::from("idx");
    let storage = JsStorage::new(db.clone(), root.clone()).await.unwrap();
    let path = root.join("test.bin");
    storage.write_all(&path, b"hello wasm").unwrap();
    storage.flush().await;
    drop(storage);
    let restored = JsStorage::new(db, root.clone()).await.unwrap();
    let contents = restored.read_to_end(&path).unwrap();
    assert_eq!(contents, b"hello wasm");
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
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::InMemory,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    };
    let index = Index::create_with_storage(&root, schema, opts, storage.clone()).unwrap();
    let mut writer: IndexWriter = index.writer().unwrap();
    writer
      .add_document(&Document {
        fields: [("body".into(), serde_json::json!("hello wasm"))]
          .into_iter()
          .collect(),
      })
      .unwrap();
    writer.commit().unwrap();
    storage.flush().await;
    let reader: IndexReader = index.reader().unwrap();
    let request = SearchRequest {
      query: "hello".to_string(),
      fields: None,
      filters: vec![],
      limit: 5,
      sort: vec![],
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      return_stored: true,
      highlight_field: None,
      aggs: BTreeMap::new(),
    };
    let result = reader.search(request).unwrap();
    assert_eq!(result.hits.len(), 1);
  }
}
