use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use futures::channel::oneshot;
use parking_lot::Mutex;
use parking_lot::RwLock;
use searchlite_core::api::types::{
  Aggregation, ExecutionStrategy, Filter, IndexOptions, Query, QueryNode, SearchRequest, SortSpec,
  StorageType,
};
use searchlite_core::api::{Document, IndexReader, IndexWriter};
use searchlite_core::storage::{DynFile, InMemoryStorage, Storage, StorageFile};
use searchlite_core::{Index, Schema};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
#[cfg(feature = "threads")]
use wasm_bindgen_futures::JsFuture;
#[cfg(feature = "threads")]
use wasm_bindgen_rayon::init_thread_pool;

const STORE_NAME: &str = "searchlite_files";
// BM25 defaults tuned for browser-based search; keep aligned with core defaults.
const BM25_K1: f32 = 0.9;
const BM25_B: f32 = 0.4;

thread_local! {
  // Per-thread (per WASM worker) cache of IndexedDB connections.
  // This avoids reconnecting for each persist operation on the same thread.
  static DB_CACHE: RefCell<HashMap<String, web_sys::IdbDatabase>> = RefCell::new(HashMap::new());
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
      Some(_) => Err(JsValue::from_str("storage must be 'indexeddb' or 'memory'")),
    }
  }
}

enum StorageBackend {
  IndexedDb(Arc<JsStorage>),
  Memory,
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

fn clear_request_handlers(
  req: &web_sys::IdbRequest,
  success: &Rc<RefCell<Option<Closure<dyn FnMut(web_sys::Event)>>>>,
  error: &Rc<RefCell<Option<Closure<dyn FnMut(web_sys::Event)>>>>,
) {
  req.set_onsuccess(None);
  req.set_onerror(None);
  success.borrow_mut().take();
  error.borrow_mut().take();
}

fn request_future(req: &web_sys::IdbRequest) -> impl std::future::Future<Output = Result<JsValue>> {
  let (tx, rx) = oneshot::channel::<Result<JsValue>>();
  let sender = Rc::new(RefCell::new(Some(tx)));
  let success_handler: Rc<RefCell<Option<Closure<dyn FnMut(web_sys::Event)>>>> =
    Rc::new(RefCell::new(None));
  let error_handler: Rc<RefCell<Option<Closure<dyn FnMut(web_sys::Event)>>>> =
    Rc::new(RefCell::new(None));
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
    let err_val = match error_req_for_closure.error() {
      Ok(e) => e.into(),
      Err(_) => JsValue::from_str("indexeddb request error"),
    };
    if let Some(tx) = sender_clone.borrow_mut().take() {
      let _ = tx.send(Err(anyhow!("indexeddb request error: {:?}", err_val)));
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

async fn open_db(name: &str) -> Result<web_sys::IdbDatabase> {
  if let Some(db) = DB_CACHE.with(|cache| cache.borrow().get(name).cloned()) {
    return Ok(db);
  }
  let factory = indexed_db_factory()?;
  let request = factory
    .open_with_u32(name, 1)
    .map_err(|e| anyhow!("indexed_db open error: {:?}", e))?;
  let store = STORE_NAME.to_string();
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
  let keys: js_sys::Array = match keys_val.dyn_into() {
    Ok(array) => array,
    Err(_) => {
      web_sys::console::warn_1(&JsValue::from_str(&format!(
        "IndexedDB snapshot load for '{db_name}' expected array keys; skipping stored files"
      )));
      js_sys::Array::new()
    }
  };
  let values: js_sys::Array = match values_val.dyn_into() {
    Ok(array) => array,
    Err(_) => {
      web_sys::console::warn_1(&JsValue::from_str(&format!(
        "IndexedDB snapshot load for '{db_name}' expected array values; skipping stored files"
      )));
      js_sys::Array::new()
    }
  };
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
    .map_err(|e| anyhow!("opening read-write transaction for {STORE_NAME}: {:?}", e))?;
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
  pending: Arc<Mutex<Vec<oneshot::Receiver<Result<()>>>>>,
  queue: Arc<Mutex<HashMap<PathBuf, PendingEntry>>>,
}

struct PendingEntry {
  pending: Option<Vec<u8>>,
  waiters: Vec<oneshot::Sender<Result<()>>>,
  inflight: bool,
}

impl PendingWrites {
  fn new(db_name: String) -> Self {
    Self {
      db_name,
      pending: Arc::new(Mutex::new(Vec::new())),
      queue: Arc::new(Mutex::new(HashMap::new())),
    }
  }

  fn schedule(&self, path: PathBuf, data: Vec<u8>) {
    let (tx, rx) = oneshot::channel();
    self.pending.lock().push(rx);
    let mut guard = self.queue.lock();
    let entry = guard.entry(path.clone()).or_insert(PendingEntry {
      pending: None,
      waiters: Vec::new(),
      inflight: false,
    });
    entry.pending = Some(data);
    entry.waiters.push(tx);
    if entry.inflight {
      return;
    }
    entry.inflight = true;
    drop(guard);
    let db = self.db_name.clone();
    let queue = self.queue.clone();
    spawn_local(async move {
      persist_queue(db, path, queue).await;
    });
  }

  async fn flush(&self) -> Result<()> {
    let receivers = {
      let mut guard = self.pending.lock();
      std::mem::take(&mut *guard)
    };
    let mut first_error = None;
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

async fn persist_queue(
  db_name: String,
  path: PathBuf,
  queue: Arc<Mutex<HashMap<PathBuf, PendingEntry>>>,
) {
  loop {
    let (data, waiters) = {
      let mut guard = queue.lock();
      let entry = match guard.get_mut(&path) {
        Some(entry) => entry,
        None => return,
      };
      let data = match entry.pending.take() {
        Some(data) => data,
        None => {
          entry.inflight = false;
          if entry.waiters.is_empty() {
            guard.remove(&path);
          }
          return;
        }
      };
      let waiters = std::mem::take(&mut entry.waiters);
      (data, waiters)
    };
    let result = persist_file(&db_name, &path, data).await;
    let err_msg = result.as_ref().err().map(|err| err.to_string());
    if let Some(msg) = &err_msg {
      web_sys::console::error_1(&JsValue::from_str(&format!(
        "persist error for {:?}: {}",
        path, msg
      )));
    }
    for tx in waiters {
      let send_result = match &err_msg {
        Some(msg) => Err(anyhow!(msg.clone())),
        None => Ok(()),
      };
      let _ = tx.send(send_result);
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

#[wasm_bindgen]
pub struct Searchlite {
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
    let schema: Schema =
      serde_json::from_str(&schema_json).map_err(|err| JsValue::from_str(&err.to_string()))?;
    let root = PathBuf::from(db_name.clone());
    let (storage, backend) = match storage_mode {
      StorageMode::IndexedDb => {
        let storage = Arc::new(
          JsStorage::new(db_name.clone(), root.clone())
            .await
            .map_err(to_js_error)?,
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
      let index = Index::open_with_storage(open_opts, storage).map_err(to_js_error)?;
      let existing_schema = index.manifest().schema;
      let existing = serde_json::to_value(&existing_schema).map_err(to_js_error)?;
      let requested = serde_json::to_value(&schema).map_err(to_js_error)?;
      if existing != requested {
        return Err(JsValue::from_str(
          "schema mismatch for existing index; use a new db_name or delete the stored index",
        ));
      }
      index
    } else {
      Index::create_with_storage(&root, schema, opts, storage).map_err(to_js_error)?
    };
    backend.flush().await.map_err(to_js_error)?;
    Ok(Searchlite {
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

  /// Initialize the rayon pool for threaded execution. COOP/COEP (cross-origin isolation) must
  /// be handled by the embedding app; this helper does not set headers for you.
  #[cfg(feature = "threads")]
  pub async fn init_threads(&self, threads: Option<u32>) -> Result<(), JsValue> {
    let desired = threads.unwrap_or_else(hardware_concurrency);
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

  /// Add a document to the index. Call `commit` to make it searchable and persist it.
  pub fn add_document(&self, doc: JsValue) -> Result<(), JsValue> {
    let value: serde_json::Value =
      serde_wasm_bindgen::from_value(doc).map_err(|err| JsValue::from_str(&err.to_string()))?;
    self.add_documents_internal(vec![value_to_document(value)?])
  }

  /// Add multiple documents to the index. Call `commit` to persist changes.
  pub fn add_documents(&self, docs: JsValue) -> Result<(), JsValue> {
    let value: serde_json::Value =
      serde_wasm_bindgen::from_value(docs).map_err(|err| JsValue::from_str(&err.to_string()))?;
    self.add_documents_internal(value_to_documents(value)?)
  }

  /// Commit pending documents and flush the configured storage backend.
  pub async fn commit(&self) -> Result<(), JsValue> {
    let mut writer: IndexWriter = self.index.writer().map_err(to_js_error)?;
    writer.commit().map_err(to_js_error)?;
    self.storage.flush().await.map_err(to_js_error)?;
    Ok(())
  }

  pub fn search(&self, query: String, limit: usize) -> Result<JsValue, JsValue> {
    let parsed_query = serde_json::from_str::<QueryNode>(&query)
      .map(Query::Node)
      .unwrap_or(Query::String(query));
    let request = SearchRequest {
      query: parsed_query,
      fields: None,
      filter: None,
      filters: Vec::<Filter>::new(),
      limit,
      sort: Vec::<SortSpec>::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,

      #[cfg(feature = "vectors")]
      vector_filter: None,
      return_stored: true,
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

  pub fn search_request(&self, request_json: String) -> Result<JsValue, JsValue> {
    let req: SearchRequest = serde_json::from_str(&request_json)
      .map_err(|err| JsValue::from_str(&format!("invalid search request: {err}")))?;
    self.run_search(req)
  }

  fn run_search(&self, req: SearchRequest) -> Result<JsValue, JsValue> {
    let reader: IndexReader = self.index.reader().map_err(to_js_error)?;
    let result = reader.search(&req).map_err(to_js_error)?;
    serde_wasm_bindgen::to_value(&result).map_err(|err| JsValue::from_str(&err.to_string()))
  }

  /// Wait for pending storage writes; `commit` already calls this.
  pub async fn flush_storage(&self) -> Result<(), JsValue> {
    self.storage.flush().await.map_err(to_js_error)?;
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
      filters: vec![],
      limit: 5,
      candidate_size: None,
      sort: vec![],
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
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
  async fn search_request_roundtrip() {
    let db = unique_db("searchlite-search-request");
    let schema = Schema::default_text_body();
    let schema_json = serde_json::to_string(&schema).unwrap();
    let idx = Searchlite::init(db, schema_json, None).await.unwrap();
    let docs = vec![serde_json::json!({ "_id": "doc-1", "body": "hello wasm" })];
    let docs_js = serde_wasm_bindgen::to_value(&docs).unwrap();
    idx.add_documents(docs_js).unwrap();
    idx.commit().await.unwrap();

    let request = SearchRequest {
      query: "hello".into(),
      fields: None,
      filter: None,
      filters: vec![],
      limit: 5,
      candidate_size: None,
      sort: vec![],
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
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
    let request_json = serde_json::to_string(&request).unwrap();
    let result = idx.search_request(request_json).unwrap();
    let result_json: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
    let hits = result_json["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
  }

  #[wasm_bindgen_test]
  async fn init_reuses_existing_index() {
    let db = unique_db("searchlite-reopen");
    let schema = Schema::default_text_body();
    let schema_json = serde_json::to_string(&schema).unwrap();
    let idx = Searchlite::init(db.clone(), schema_json.clone(), None)
      .await
      .unwrap();
    let docs = vec![serde_json::json!({ "_id": "doc-1", "body": "hello reopen" })];
    let docs_js = serde_wasm_bindgen::to_value(&docs).unwrap();
    idx.add_documents(docs_js).unwrap();
    idx.commit().await.unwrap();
    drop(idx);

    let reopened = Searchlite::init(db, schema_json, None).await.unwrap();
    let result = reopened.search("hello".to_string(), 5).unwrap();
    let result_json: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
    let hits = result_json["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
  }
}
