use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::body::Body;
use axum::error_handling::HandleErrorLayer;
use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use futures_util::stream::FuturesUnordered;
use futures_util::StreamExt;
use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Document, IndexOptions, MgetRequest, MgetResponse, MultiSearchRequest, SearchRequest, StorageType,
};
use searchlite_core::api::PatchError;
use searchlite_core::api::{MultiSearchResponse, SearchResult};
use searchlite_core::util::doc_id::validate_doc_id;
use searchlite_core::{Index, Manifest, Schema};
use thiserror::Error;
use tokio::io::AsyncBufReadExt;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio::sync::Semaphore;
use tokio_util::io::StreamReader;
use tower::limit::ConcurrencyLimitLayer;
use tower::timeout::TimeoutLayer;
use tower::{BoxError, ServiceBuilder};
use tower_http::limit::RequestBodyLimitLayer;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, EnvFilter};

const DEFAULT_K1: f32 = 0.9;
const DEFAULT_B: f32 = 0.4;
#[cfg(feature = "vectors")]
const DEFAULT_MAX_VECTOR_GLOBAL_CANDIDATES: usize = 20_000;

#[derive(Debug, Clone)]
pub struct IndexSpec {
  pub name: String,
  pub path: PathBuf,
  pub auto_commit_interval_secs: Option<u64>,
  pub auto_refresh_interval_secs: Option<u64>,
}

pub fn parse_index_spec(raw: &str) -> Result<IndexSpec, String> {
  let Some((name, path_and_opts)) = raw.split_once([':', '=']) else {
    return Err("expected NAME:PATH".into());
  };
  if name.trim().is_empty() {
    return Err("index name cannot be empty".into());
  }
  let mut path_parts = path_and_opts.split(',');
  let Some(path) = path_parts.next() else {
    return Err("index path cannot be empty".into());
  };
  if path.trim().is_empty() {
    return Err("index path cannot be empty".into());
  }
  let mut spec = IndexSpec {
    name: name.trim().to_string(),
    path: PathBuf::from(path.trim()),
    auto_commit_interval_secs: None,
    auto_refresh_interval_secs: None,
  };

  for raw_option in path_parts {
    let option = raw_option.trim();
    if option.is_empty() {
      return Err("index option cannot be empty".into());
    }
    let Some((raw_key, raw_value)) = option.split_once('=') else {
      return Err(format!("index option `{option}` must be KEY=VALUE"));
    };
    let key = raw_key.trim();
    if key.is_empty() {
      return Err("index option key cannot be empty".into());
    }
    let value = raw_value.trim();
    if value.is_empty() {
      return Err(format!(
        "index option `{key}` must be a non-negative integer"
      ));
    }
    let parsed = value
      .parse::<u64>()
      .map_err(|_| format!("index option `{key}` must be a non-negative integer"))?;
    match key {
      "auto_commit" => {
        if spec.auto_commit_interval_secs.is_some() {
          return Err("duplicate index option `auto_commit`".into());
        }
        spec.auto_commit_interval_secs = Some(parsed);
      }
      "auto_refresh" => {
        if spec.auto_refresh_interval_secs.is_some() {
          return Err("duplicate index option `auto_refresh`".into());
        }
        spec.auto_refresh_interval_secs = Some(parsed);
      }
      _ => return Err(format!("unsupported index option `{key}`")),
    }
  }

  Ok(spec)
}

#[derive(Debug, Clone)]
pub struct AliasSpec {
  pub alias: String,
  pub target: String,
}

pub fn parse_alias_spec(raw: &str) -> Result<AliasSpec, String> {
  let Some((alias, target)) = raw.split_once([':', '=']) else {
    return Err("expected ALIAS:TARGET".into());
  };
  if alias.trim().is_empty() || target.trim().is_empty() {
    return Err("alias and target must be non-empty".into());
  }
  Ok(AliasSpec {
    alias: alias.trim().to_string(),
    target: target.trim().to_string(),
  })
}

#[derive(Parser, Debug, Clone)]
#[command(
  name = "searchlite-http",
  version,
  about = "HTTP API for multiple searchlite indexes"
)]
pub struct ServeArgs {
  /// Mount one or more indexes as NAME:PATH pairs; repeat for multiple mounts.
  /// When using SEARCHLITE_INDEX_MAP env var, separate entries with `;`.
  #[arg(
    short = 'I',
    long = "index",
    value_name = "NAME:PATH",
    value_parser = parse_index_spec,
    env = "SEARCHLITE_INDEX_MAP",
    value_delimiter = ';',
    required = true
  )]
  pub indexes: Vec<IndexSpec>,

  /// Optional aliases in the form ALIAS:TARGET (TARGET must be a mounted index name).
  /// When using SEARCHLITE_INDEX_ALIASES env var, separate entries with `;`.
  #[arg(
    long = "alias",
    value_name = "ALIAS:TARGET",
    value_parser = parse_alias_spec,
    env = "SEARCHLITE_INDEX_ALIASES",
    value_delimiter = ';'
  )]
  pub aliases: Vec<AliasSpec>,

  /// Bind address for the HTTP server.
  /// WARNING: Binding to 0.0.0.0 or any non-localhost address exposes this
  /// unauthenticated service to the network; front it with a proxy or firewall.
  #[arg(long, env = "SEARCHLITE_BIND_ADDR", default_value = "127.0.0.1:8080")]
  pub bind: SocketAddr,

  /// Require each index to already exist on disk at startup.
  #[arg(
    long,
    env = "SEARCHLITE_REQUIRE_EXISTING_INDEX",
    default_value_t = false
  )]
  pub require_existing_index: bool,

  /// Maximum allowed request body size in bytes.
  #[arg(long, env = "SEARCHLITE_MAX_BODY_BYTES", default_value_t = 50 * 1024 * 1024)]
  pub max_body_bytes: u64,

  /// Maximum number of in-flight requests.
  #[arg(long, env = "SEARCHLITE_MAX_CONCURRENCY", default_value_t = 64)]
  pub max_concurrency: usize,

  /// Per-request timeout in seconds.
  #[arg(long, env = "SEARCHLITE_REQUEST_TIMEOUT_SECS", default_value_t = 30)]
  pub request_timeout_secs: u64,

  /// Grace period in seconds before forcing shutdown after a signal.
  #[arg(long, env = "SEARCHLITE_GRACEFUL_SHUTDOWN_SECS", default_value_t = 5)]
  pub shutdown_grace_secs: u64,

  /// If set, commit also triggers a reader refresh to surface changes immediately.
  #[arg(long, env = "SEARCHLITE_REFRESH_ON_COMMIT", default_value_t = false)]
  pub refresh_on_commit: bool,

  /// Default auto-commit interval for all indexes in seconds (0 disables).
  #[arg(
    long = "auto-commit-interval-secs",
    env = "SEARCHLITE_AUTO_COMMIT_INTERVAL_SECS",
    default_value_t = 0
  )]
  pub auto_commit_interval_secs: u64,

  /// Default auto-refresh interval for all indexes in seconds (0 disables).
  #[arg(
    long = "auto-refresh-interval-secs",
    env = "SEARCHLITE_AUTO_REFRESH_INTERVAL_SECS",
    default_value_t = 0
  )]
  pub auto_refresh_interval_secs: u64,

  /// Global cap for combined vector candidates across clauses (when feature `vectors` is enabled).
  #[cfg(feature = "vectors")]
  #[arg(
    long = "max-vector-candidates",
    env = "SEARCHLITE_MAX_VECTOR_CANDIDATES",
    default_value_t = DEFAULT_MAX_VECTOR_GLOBAL_CANDIDATES
  )]
  pub max_vector_candidates: usize,
}

#[derive(Clone)]
struct ManagedIndex {
  name: String,
  path: PathBuf,
  require_existing_index: bool,
  refresh_on_commit: bool,
  auto_commit_interval_secs: u64,
  auto_refresh_interval_secs: u64,
  #[cfg(feature = "vectors")]
  max_vector_candidates: usize,
  index: Arc<tokio::sync::RwLock<Option<Arc<Index>>>>,
  writer_lock: Arc<tokio::sync::Mutex<()>>,
  auto_commit_enabled: Arc<AtomicBool>,
}

impl ManagedIndex {
  fn new(
    spec: &IndexSpec,
    require_existing_index: bool,
    refresh_on_commit: bool,
    auto_commit_interval_secs: u64,
    auto_refresh_interval_secs: u64,
    #[cfg(feature = "vectors")] max_vector_candidates: usize,
  ) -> Self {
    Self {
      name: spec.name.clone(),
      path: spec.path.clone(),
      require_existing_index,
      refresh_on_commit,
      auto_commit_interval_secs,
      auto_refresh_interval_secs,
      #[cfg(feature = "vectors")]
      max_vector_candidates,
      index: Arc::new(tokio::sync::RwLock::new(None)),
      writer_lock: Arc::new(tokio::sync::Mutex::new(())),
      auto_commit_enabled: Arc::new(AtomicBool::new(true)),
    }
  }

  async fn bootstrap(&self) -> anyhow::Result<()> {
    if !self.manifest_exists() {
      if self.require_existing_index {
        anyhow::bail!(
          "index `{}` does not exist at {:?}",
          self.name,
          Manifest::manifest_path(&self.path)
        );
      }
      return Ok(());
    }
    let idx = Index::open(self.index_options(false)).with_context(|| {
      format!(
        "failed to open existing index `{}` during startup",
        self.name
      )
    })?;
    let arc = Arc::new(idx);
    let mut guard = self.index.write().await;
    *guard = Some(arc);
    Ok(())
  }

  fn manifest_exists(&self) -> bool {
    Manifest::manifest_path(&self.path).exists()
  }

  fn index_options(&self, create_if_missing: bool) -> IndexOptions {
    IndexOptions {
      path: self.path.clone(),
      create_if_missing,
      enable_positions: true,
      bm25_k1: DEFAULT_K1,
      bm25_b: DEFAULT_B,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    }
  }

  async fn set_index(&self, index: Index) -> Arc<Index> {
    let arc = Arc::new(index);
    let mut guard = self.index.write().await;
    *guard = Some(arc.clone());
    arc
  }

  async fn require_index(&self) -> ApiResult<Arc<Index>> {
    if let Some(existing) = self.index.read().await.as_ref() {
      return Ok(existing.clone());
    }
    if !self.manifest_exists() {
      return Err(HttpError::not_found(
        "index_missing",
        format!(
          "index `{}` is not initialized; call /indexes/{}/init first",
          self.name, self.name
        ),
      ));
    }
    let idx = Index::open(self.index_options(false))
      .map_err(|e| HttpError::from_anyhow("open_index", StatusCode::SERVICE_UNAVAILABLE, e))?;
    Ok(self.set_index(idx).await)
  }

  async fn describe(&self) -> ApiResult<IndexDescriptor> {
    let exists = self.manifest_exists();
    let mut committed_at = None;
    let mut doc_count = None;

    if exists {
      if let Some(index) = self.index.read().await.as_ref().cloned() {
        let manifest = index.manifest();
        committed_at = Some(manifest.committed_at.clone());
        let (live_docs, _) = manifest_doc_counts(&manifest);
        doc_count = Some(live_docs);
      } else {
        let path = self.path.clone();
        let (loaded_committed_at, loaded_doc_count) = tokio::task::spawn_blocking(move || {
          let manifest_path = Manifest::manifest_path(&path);
          let bytes = std::fs::read(&manifest_path)
            .with_context(|| format!("reading manifest metadata at {manifest_path:?}"))?;
          let manifest: Manifest = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing manifest metadata at {manifest_path:?}"))?;
          let (live_docs, _) = manifest_doc_counts(&manifest);
          Ok::<(String, u64), anyhow::Error>((manifest.committed_at.clone(), live_docs))
        })
        .await
        .map_err(|err| {
          HttpError::from_anyhow(
            "indexes_join",
            StatusCode::INTERNAL_SERVER_ERROR,
            anyhow::anyhow!(err.to_string()),
          )
        })?
        .map_err(|err| {
          HttpError::from_anyhow("indexes_failed", StatusCode::INTERNAL_SERVER_ERROR, err)
        })?;
        committed_at = Some(loaded_committed_at);
        doc_count = Some(loaded_doc_count);
      }
    }

    Ok(IndexDescriptor {
      name: self.name.clone(),
      path: self.path.display().to_string(),
      exists,
      committed_at,
      doc_count,
      auto_commit_secs: self.auto_commit_interval_secs,
      auto_refresh_secs: self.auto_refresh_interval_secs,
      refresh_on_commit: self.refresh_on_commit,
    })
  }

  async fn auto_commit_once(&self) -> anyhow::Result<bool> {
    if !self.auto_commit_enabled.load(Ordering::Relaxed) {
      return Ok(false);
    }
    let index = match self.require_index().await {
      Ok(index) => index,
      Err(err) if err.kind == "index_missing" => return Ok(false),
      Err(err) => anyhow::bail!("{}: {}", err.kind, err.reason),
    };
    let manifest = index.manifest();
    if write_key_required(&manifest) {
      self.auto_commit_enabled.store(false, Ordering::Relaxed);
      anyhow::bail!(
        "auto-commit disabled for index `{}` because it requires a write key",
        self.name
      );
    }
    let writer_lock = self.writer_lock.clone();
    let refresh_on_commit = self.refresh_on_commit;
    tokio::task::spawn_blocking(move || -> anyhow::Result<bool> {
      let _guard = writer_lock.blocking_lock();
      let committed_before = index.manifest().committed_at;
      let mut writer = index.writer()?;
      writer.commit()?;
      let committed_after = index.manifest().committed_at;
      let did_commit = committed_after != committed_before;
      if did_commit && refresh_on_commit {
        trigger_reader_refresh(&index)?;
      }
      Ok(did_commit)
    })
    .await
    .map_err(|err| anyhow::anyhow!(err.to_string()))?
  }

  async fn committed_at_marker(&self) -> anyhow::Result<Option<String>> {
    let index = match self.require_index().await {
      Ok(index) => index,
      Err(err) if err.kind == "index_missing" => return Ok(None),
      Err(err) => anyhow::bail!("{}: {}", err.kind, err.reason),
    };
    Ok(Some(index.manifest().committed_at))
  }

  async fn auto_refresh_once(
    &self,
    last_refreshed_committed_at: &mut Option<String>,
  ) -> anyhow::Result<bool> {
    let index = match self.require_index().await {
      Ok(index) => index,
      Err(err) if err.kind == "index_missing" => return Ok(false),
      Err(err) => anyhow::bail!("{}: {}", err.kind, err.reason),
    };
    let current_committed_at = index.manifest().committed_at;
    if !should_refresh(
      last_refreshed_committed_at.as_deref(),
      current_committed_at.as_str(),
    ) {
      return Ok(false);
    }
    tokio::task::spawn_blocking(move || trigger_reader_refresh(&index))
      .await
      .map_err(|err| anyhow::anyhow!(err.to_string()))??;
    *last_refreshed_committed_at = Some(current_committed_at);
    Ok(true)
  }
}

#[derive(Clone)]
struct IndexRegistry {
  indexes: HashMap<String, Arc<ManagedIndex>>,
  aliases: HashMap<String, String>,
}

impl IndexRegistry {
  fn from_args(args: &ServeArgs) -> anyhow::Result<Self> {
    if args.indexes.is_empty() {
      anyhow::bail!("at least one index must be configured via --index or SEARCHLITE_INDEX_MAP");
    }

    let mut indexes = HashMap::new();
    for spec in args.indexes.iter() {
      if indexes.contains_key(&spec.name) {
        anyhow::bail!("duplicate index name provided: {}", spec.name);
      }
      let auto_commit_interval_secs = spec
        .auto_commit_interval_secs
        .unwrap_or(args.auto_commit_interval_secs);
      let auto_refresh_interval_secs = spec
        .auto_refresh_interval_secs
        .unwrap_or(args.auto_refresh_interval_secs);
      let managed = ManagedIndex::new(
        spec,
        args.require_existing_index,
        args.refresh_on_commit,
        auto_commit_interval_secs,
        auto_refresh_interval_secs,
        #[cfg(feature = "vectors")]
        args.max_vector_candidates,
      );
      indexes.insert(spec.name.clone(), Arc::new(managed));
    }

    let mut aliases = HashMap::new();
    for alias_spec in args.aliases.iter() {
      if indexes.contains_key(&alias_spec.alias) {
        anyhow::bail!(
          "alias `{}` conflicts with existing index name",
          alias_spec.alias
        );
      }
      if aliases.contains_key(&alias_spec.alias) {
        anyhow::bail!("duplicate alias name provided: {}", alias_spec.alias);
      }
      aliases.insert(alias_spec.alias.clone(), alias_spec.target.clone());
    }

    Self::validate_aliases(&indexes, &aliases)?;

    Ok(Self { indexes, aliases })
  }

  async fn bootstrap_all(&self) -> anyhow::Result<()> {
    for managed in self.indexes.values() {
      managed.bootstrap().await?;
    }
    Ok(())
  }

  fn resolve(&self, name: &str) -> ApiResult<Arc<ManagedIndex>> {
    let mut cursor = name;
    let mut visited = HashSet::new();
    while let Some(target) = self.aliases.get(cursor) {
      if !visited.insert(cursor.to_string()) {
        return Err(HttpError::from_anyhow(
          "alias_cycle",
          StatusCode::INTERNAL_SERVER_ERROR,
          anyhow::anyhow!("alias cycle detected for {name}"),
        ));
      }
      cursor = target;
    }
    self.indexes.get(cursor).cloned().ok_or_else(|| {
      HttpError::not_found("unknown_index", format!("index `{name}` not registered"))
    })
  }

  fn validate_aliases(
    indexes: &HashMap<String, Arc<ManagedIndex>>,
    aliases: &HashMap<String, String>,
  ) -> anyhow::Result<()> {
    for (alias_name, initial_target) in aliases.iter() {
      let mut cursor: &str = alias_name;
      let mut visited = HashSet::new();
      loop {
        if indexes.contains_key(cursor) {
          break;
        }
        if !visited.insert(cursor.to_string()) {
          anyhow::bail!(
            "alias cycle detected involving `{cursor}` while validating alias `{alias_name}`"
          );
        }
        match aliases.get(cursor) {
          Some(next) => cursor = next,
          None => {
            anyhow::bail!("alias `{alias_name}` targets unknown index or alias `{initial_target}`");
          }
        }
      }
    }
    Ok(())
  }

  async fn list_indexes(&self) -> ApiResult<Vec<IndexDescriptor>> {
    let mut tasks = FuturesUnordered::new();
    for idx in self.indexes.values() {
      let managed = idx.clone();
      tasks.push(async move { managed.describe().await });
    }

    let mut items = Vec::with_capacity(self.indexes.len());
    while let Some(item) = tasks.next().await {
      items.push(item?);
    }
    items.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(items)
  }

  fn list_aliases(&self) -> Vec<AliasDescriptor> {
    let mut items: Vec<_> = self
      .aliases
      .iter()
      .map(|(alias, target)| AliasDescriptor {
        alias: alias.clone(),
        target: target.clone(),
      })
      .collect();
    items.sort_by(|a, b| a.alias.cmp(&b.alias));
    items
  }
}

#[derive(Clone)]
struct AppState {
  registry: Arc<IndexRegistry>,
  _maintenance: Arc<MaintenanceRuntime>,
}

struct MaintenanceRuntime {
  shutdown_tx: watch::Sender<bool>,
  handles: Vec<tokio::task::JoinHandle<()>>,
}

impl MaintenanceRuntime {
  fn spawn(registry: Arc<IndexRegistry>) -> anyhow::Result<Arc<Self>> {
    let (shutdown_tx, _) = watch::channel(false);
    let mut handles = Vec::new();
    for managed in registry.indexes.values() {
      if managed.auto_commit_interval_secs > 0 {
        Self::validate_auto_commit_support(managed)?;
        let managed = managed.clone();
        let interval_secs = managed.auto_commit_interval_secs;
        let mut shutdown_rx = shutdown_tx.subscribe();
        let handle = tokio::spawn(async move {
          let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
          // Skip the immediate first tick so the first run happens after one full interval.
          ticker.tick().await;
          loop {
            tokio::select! {
              changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                  break;
                }
              }
              _ = ticker.tick() => {
                if let Err(err) = managed.auto_commit_once().await {
                  error!(index = %managed.name, error = ?err, "auto-commit tick failed");
                  if !managed.auto_commit_enabled.load(Ordering::Relaxed) {
                    break;
                  }
                  tokio::time::sleep(Duration::from_secs(1)).await;
                }
              }
            }
          }
        });
        handles.push(handle);
      }

      if managed.auto_refresh_interval_secs > 0 {
        let managed = managed.clone();
        let interval_secs = managed.auto_refresh_interval_secs;
        let mut shutdown_rx = shutdown_tx.subscribe();
        let handle = tokio::spawn(async move {
          let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
          let mut last_refreshed_committed_at = match managed.committed_at_marker().await {
            Ok(marker) => marker,
            Err(err) => {
              error!(index = %managed.name, error = ?err, "auto-refresh initialization failed");
              None
            }
          };
          // Skip the immediate first tick so the first run happens after one full interval.
          ticker.tick().await;
          loop {
            tokio::select! {
              changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                  break;
                }
              }
              _ = ticker.tick() => {
                if let Err(err) = managed.auto_refresh_once(&mut last_refreshed_committed_at).await {
                  error!(index = %managed.name, error = ?err, "auto-refresh tick failed");
                  tokio::time::sleep(Duration::from_secs(1)).await;
                }
              }
            }
          }
        });
        handles.push(handle);
      }
    }
    Ok(Arc::new(Self {
      shutdown_tx,
      handles,
    }))
  }

  fn validate_auto_commit_support(managed: &ManagedIndex) -> anyhow::Result<()> {
    if !managed.manifest_exists() {
      return Ok(());
    }
    let manifest_path = Manifest::manifest_path(&managed.path);
    let bytes = std::fs::read(&manifest_path)
      .with_context(|| format!("reading manifest metadata at {manifest_path:?}"))?;
    let manifest: Manifest = serde_json::from_slice(&bytes)
      .with_context(|| format!("parsing manifest metadata at {manifest_path:?}"))?;
    if write_key_required(&manifest) {
      anyhow::bail!(
        "index `{}` requires a write key; disable auto-commit for this index or remove write-key protection",
        managed.name
      );
    }
    Ok(())
  }
}

impl Drop for MaintenanceRuntime {
  fn drop(&mut self) {
    let _ = self.shutdown_tx.send(true);
    for handle in self.handles.iter() {
      handle.abort();
    }
  }
}

impl AppState {
  fn new(registry: Arc<IndexRegistry>, maintenance: Arc<MaintenanceRuntime>) -> Self {
    Self {
      registry,
      _maintenance: maintenance,
    }
  }

  fn registry(&self) -> Arc<IndexRegistry> {
    self.registry.clone()
  }
}

fn build_app_state(registry: Arc<IndexRegistry>) -> anyhow::Result<Arc<AppState>> {
  let maintenance = MaintenanceRuntime::spawn(registry.clone())?;
  Ok(Arc::new(AppState::new(registry, maintenance)))
}

/// Number of NDJSON batches buffered between reader and writer; small to bound memory while
/// allowing a little headroom to hide writer latency.
const INGEST_CHANNEL_BUFFER_BATCHES: usize = 4;
/// Max documents per NDJSON batch.
const NDJSON_BATCH_SIZE: usize = 1000;
const MAX_PAGE_SIZE: usize = 1_000;
const MAX_MGET_IDS: usize = 1_024;
const DEFAULT_MULTI_SEARCH_MAX_CONCURRENCY: usize = 4;
const HARD_MULTI_SEARCH_MAX_CONCURRENCY: usize = 16;

#[derive(Debug)]
enum IngestMsg {
  Batch(Vec<Document>),
  Commit,
  Abort,
}

#[derive(Debug, Error)]
#[error("{reason}")]
struct HttpError {
  status: StatusCode,
  kind: &'static str,
  reason: String,
}

type ApiResult<T> = Result<T, HttpError>;

// Returned to every caller whose request body fails JSON deserialization. The
// underlying `JsonRejection::to_string()` contains the target-type field path
// of the failing value (e.g. `text_fields: invalid type: integer ...`), which
// lets an unauthenticated probe enumerate the server's internal request
// types. Log the detail for operators and return a constant message. See
// BUG-016.
const MALFORMED_REQUEST_BODY_MESSAGE: &str = "malformed request body";

fn parse_json<T>(payload: Result<Json<T>, JsonRejection>) -> ApiResult<T> {
  payload.map(|Json(inner)| inner).map_err(|err| {
    warn!(error = %err, "malformed request body");
    HttpError::bad_request("invalid_request", MALFORMED_REQUEST_BODY_MESSAGE)
  })
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ErrorResponseBody {
  // `r#type` is a Rust raw identifier; the serialized JSON field name is "type".
  r#type: String,
  reason: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ErrorResponse {
  error: ErrorResponseBody,
}

fn extract_write_key(headers: &HeaderMap) -> Option<String> {
  headers
    .get("x-write-key")
    .and_then(|v| v.to_str().ok())
    .map(|s| s.to_string())
}

fn write_key_required(manifest: &Manifest) -> bool {
  manifest.write_key.is_some()
    || manifest
      .segments
      .iter()
      .any(|s| s.write_binding_b64.is_some())
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct InitResponse {
  created: bool,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct IngestResponse {
  queued: usize,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DeleteResponse {
  queued: usize,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CommitResponse {
  committed: bool,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct RefreshResponse {
  refreshed: bool,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CompactResponse {
  compacted: bool,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct InspectResponse {
  manifest: Manifest,
}

// Note: `index_path` was removed from this response as a Security fix (BUG-015).
// The on-disk filesystem path of an index is an operator-side implementation
// detail and must not be exposed to unauthenticated callers of `/stats`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StatsResponse {
  documents: u64,
  deleted_documents: u64,
  segments: usize,
  committed_at: String,
  index_uuid: String,
  index_name: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct HealthResponse {
  status: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct IndexDescriptor {
  name: String,
  path: String,
  exists: bool,
  committed_at: Option<String>,
  doc_count: Option<u64>,
  auto_commit_secs: u64,
  auto_refresh_secs: u64,
  refresh_on_commit: bool,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct AliasDescriptor {
  alias: String,
  target: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct IndexListResponse {
  indexes: Vec<IndexDescriptor>,
  aliases: Vec<AliasDescriptor>,
}

#[derive(Debug, serde::Deserialize)]
struct BulkRequest {
  docs: Vec<serde_json::Value>,
}

#[derive(Debug, serde::Deserialize)]
struct DeleteRequest {
  ids: Vec<String>,
}

#[derive(Debug, serde::Deserialize)]
struct UpdateRequest {
  id: String,
  #[serde(default)]
  set: BTreeMap<String, serde_json::Value>,
  #[serde(default)]
  unset: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
struct UpdateResponse {
  accepted: bool,
}

#[derive(Debug, serde::Deserialize)]
struct BulkUpdateAction {
  update: BulkUpdateActionMeta,
}

#[derive(Debug, serde::Deserialize)]
struct BulkUpdateActionMeta {
  #[serde(rename = "_id")]
  id: String,
}

#[derive(Debug, serde::Deserialize)]
struct BulkUpdatePatch {
  #[serde(default)]
  set: BTreeMap<String, serde_json::Value>,
  #[serde(default)]
  unset: Vec<String>,
}

#[derive(Debug)]
enum BulkUpdateMsg {
  Batch(Vec<(String, BulkUpdatePatch)>),
  Commit,
  Abort,
}

#[derive(Debug, serde::Serialize)]
struct BulkUpdateItem {
  id: String,
  status: u16,
  #[serde(skip_serializing_if = "Option::is_none")]
  error: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct BulkUpdateResponse {
  updated: u64,
  failed: u64,
  items: Vec<BulkUpdateItem>,
}

impl HttpError {
  fn bad_request(kind: &'static str, reason: impl Into<String>) -> Self {
    Self {
      status: StatusCode::BAD_REQUEST,
      kind,
      reason: reason.into(),
    }
  }

  fn not_found(kind: &'static str, reason: impl Into<String>) -> Self {
    Self {
      status: StatusCode::NOT_FOUND,
      kind,
      reason: reason.into(),
    }
  }

  fn conflict(kind: &'static str, reason: impl Into<String>) -> Self {
    Self {
      status: StatusCode::CONFLICT,
      kind,
      reason: reason.into(),
    }
  }

  fn from_anyhow(kind: &'static str, status: StatusCode, err: anyhow::Error) -> Self {
    Self {
      status,
      kind,
      reason: err.to_string(),
    }
  }
}

impl IntoResponse for HttpError {
  fn into_response(self) -> Response {
    let body = Json(ErrorResponse {
      error: ErrorResponseBody {
        r#type: self.kind.to_string(),
        reason: self.reason,
      },
    });
    (self.status, body).into_response()
  }
}

pub async fn run(args: ServeArgs) -> anyhow::Result<()> {
  let registry = Arc::new(IndexRegistry::from_args(&args)?);
  registry.bootstrap_all().await?;
  let state = build_app_state(registry.clone())?;
  let listener = TcpListener::bind(args.bind)
    .await
    .with_context(|| format!("binding to {}", args.bind))?;
  let local_addr = listener
    .local_addr()
    .context("reading local listening address")?;
  info!(address = ?local_addr, "searchlite HTTP server listening");
  let app = router(state.clone(), &args);
  axum::serve(listener, app)
    .with_graceful_shutdown(shutdown_signal(args.shutdown_grace_secs))
    .await
    .context("running HTTP server")
}

fn router(state: Arc<AppState>, args: &ServeArgs) -> Router {
  let max_body = args
    .max_body_bytes
    .try_into()
    .expect("configured max_body_bytes does not fit into usize");
  let middleware = ServiceBuilder::new()
    .layer(HandleErrorLayer::new(handle_middleware_error))
    .layer(TimeoutLayer::new(Duration::from_secs(
      args.request_timeout_secs,
    )))
    .layer(ConcurrencyLimitLayer::new(args.max_concurrency))
    .layer(RequestBodyLimitLayer::new(max_body));

  Router::new()
    .route("/healthz", get(health))
    .route("/indexes", get(list_indexes))
    .route("/indexes/:name/init", post(init_index))
    .route("/indexes/:name/add", post(add_ndjson))
    .route("/indexes/:name/bulk", post(bulk_ingest))
    .route("/indexes/:name/update", post(update_document))
    .route("/indexes/:name/_bulk_update", post(bulk_update))
    .route("/indexes/:name/delete", post(delete_documents))
    .route("/indexes/:name/commit", post(commit))
    .route("/indexes/:name/refresh", post(refresh))
    .route("/indexes/:name/compact", post(compact))
    .route("/indexes/:name/search", post(search))
    .route("/indexes/:name/mget", post(mget))
    .route("/indexes/:name/multi_search", post(multi_search))
    .route("/indexes/:name/inspect", get(inspect))
    .route("/indexes/:name/stats", get(stats))
    .with_state(state)
    .layer(middleware)
    .layer(middleware::from_fn(move |req, next| {
      map_413(max_body, req, next)
    }))
}

async fn map_413(max_body: usize, req: Request, next: Next) -> Response {
  let mut res = next.run(req).await;
  if res.status() == StatusCode::PAYLOAD_TOO_LARGE {
    res = HttpError::from_anyhow(
      "body_too_large",
      StatusCode::PAYLOAD_TOO_LARGE,
      anyhow::anyhow!(format!(
        "request body exceeded configured limit of {max_body} bytes"
      )),
    )
    .into_response();
  }
  res
}

async fn handle_middleware_error(err: BoxError) -> Response {
  if err.is::<tower::timeout::error::Elapsed>() {
    return HttpError::from_anyhow(
      "timeout",
      StatusCode::GATEWAY_TIMEOUT,
      anyhow::anyhow!("request timed out"),
    )
    .into_response();
  }
  // The `Debug` form of the underlying `BoxError` can include panic messages,
  // tokio runtime state, and source-chain stringifications — information that
  // belongs in operator logs, not in the response body. See BUG-016.
  error!(error = ?err, "middleware error");
  HttpError::from_anyhow(
    "middleware_error",
    StatusCode::INTERNAL_SERVER_ERROR,
    anyhow::anyhow!("internal server error"),
  )
  .into_response()
}

async fn health() -> impl IntoResponse {
  (
    StatusCode::OK,
    Json(HealthResponse {
      status: "ok".into(),
    }),
  )
}

async fn list_indexes(State(state): State<Arc<AppState>>) -> ApiResult<Json<IndexListResponse>> {
  let registry = state.registry();
  Ok(Json(IndexListResponse {
    indexes: registry.list_indexes().await?,
    aliases: registry.list_aliases(),
  }))
}

async fn init_index(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
  headers: HeaderMap,
  payload: Result<Json<Schema>, JsonRejection>,
) -> ApiResult<Json<InitResponse>> {
  let schema = parse_json(payload)?;
  let managed = state.registry().resolve(&index_name)?;
  if managed.manifest_exists() {
    return Err(HttpError::conflict(
      "index_exists",
      "index already exists at this path",
    ));
  }
  let path = managed.path.clone();
  let opts = managed.index_options(true);
  let write_key = extract_write_key(&headers);
  let created = tokio::task::spawn_blocking(move || {
    IndexBuilder::create_with_write_key(&path, schema, opts, write_key.as_deref())
  })
  .await
  .map_err(|err| {
    HttpError::from_anyhow(
      "init_join",
      StatusCode::INTERNAL_SERVER_ERROR,
      anyhow::anyhow!(err.to_string()),
    )
  })?
  .map_err(|err| HttpError::from_anyhow("init_failed", StatusCode::BAD_REQUEST, err))?;
  managed.set_index(created).await;
  Ok(Json(InitResponse { created: true }))
}

// Await the ingest writer task, surfacing its returned `HttpError` on success
// and converting a `JoinError` (i.e. task panic) into a generic 500 while
// recording the panic detail via `tracing`. `JoinError::to_string()` leaks the
// panic message and source location, so it must never reach the response body.
// See BUG-016.
async fn await_writer_or_default(
  writer_task: &mut Option<tokio::task::JoinHandle<Result<usize, HttpError>>>,
  default: HttpError,
) -> HttpError {
  if let Some(handle) = writer_task.take() {
    match handle.await {
      Ok(Err(e)) => e,
      Ok(Ok(_)) => default,
      Err(join_err) => {
        error!(error = %join_err, "ingest writer task failed to join");
        HttpError::from_anyhow(
          "add_join",
          StatusCode::INTERNAL_SERVER_ERROR,
          anyhow::anyhow!("internal server error"),
        )
      }
    }
  } else {
    default
  }
}

async fn add_ndjson(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
  headers: HeaderMap,
  body: Body,
) -> ApiResult<Json<IngestResponse>> {
  let managed = state.registry().resolve(&index_name)?;
  let index = managed.require_index().await?;
  let write_key = extract_write_key(&headers);
  let manifest = index.manifest();
  if write_key.is_none() && write_key_required(&manifest) {
    return Err(HttpError::from_anyhow(
      "write_key_required",
      StatusCode::UNAUTHORIZED,
      anyhow::anyhow!("write key required"),
    ));
  }
  let mapped_stream = body
    .into_data_stream()
    .map(|chunk| chunk.map_err(io::Error::other));
  let mut reader = StreamReader::new(mapped_stream);
  let mut buf = String::new();

  let (tx, rx) = mpsc::channel::<IngestMsg>(INGEST_CHANNEL_BUFFER_BATCHES);
  let mut rx_slot = Some(rx);
  let mut writer_task: Option<tokio::task::JoinHandle<Result<usize, HttpError>>> = None;
  let write_key_clone = write_key.clone();

  let ensure_writer_task =
    |rx_slot: &mut Option<mpsc::Receiver<IngestMsg>>,
     writer_task: &mut Option<tokio::task::JoinHandle<Result<usize, HttpError>>>| {
      if writer_task.is_some() {
        return;
      }
      let Some(rx) = rx_slot.take() else {
        return;
      };
      let writer_lock = managed.writer_lock.clone();
      let index_ref = index.clone();
      let write_key = write_key_clone.clone();
      let handle = tokio::task::spawn_blocking(move || -> Result<usize, HttpError> {
        let _writer_guard = writer_lock.blocking_lock();
        let mut writer = index_ref
          .writer_with_key(write_key.as_deref())
          .map_err(|e| {
            let msg = e.to_string().to_lowercase();
            let status = if msg.contains("write key") || msg.contains("unauthorized") {
              if write_key.is_some() {
                StatusCode::FORBIDDEN
              } else {
                StatusCode::UNAUTHORIZED
              }
            } else {
              StatusCode::INTERNAL_SERVER_ERROR
            };
            HttpError::from_anyhow("writer_open", status, e)
          })?;
        let mut total = 0usize;
        let mut rx = rx;
        while let Some(msg) = rx.blocking_recv() {
          match msg {
            IngestMsg::Batch(batch) => {
              for doc in batch.iter() {
                if let Err(err) = writer.add_document(doc) {
                  if let Err(rollback_err) = writer.rollback() {
                    error!(
                      error = ?rollback_err,
                      "failed to rollback writer after NDJSON add failure"
                    );
                  }
                  return Err(HttpError::bad_request("add_failed", err.to_string()));
                }
              }
              total += batch.len();
            }
            IngestMsg::Commit => return Ok(total),
            IngestMsg::Abort => {
              if let Err(rollback_err) = writer.rollback() {
                error!(
                  error = ?rollback_err,
                  "failed to rollback writer after NDJSON add failure"
                );
              }
              return Ok(total);
            }
          }
        }
        // Channel closed without explicit commit; rollback to avoid partial ingest.
        if let Err(rollback_err) = writer.rollback() {
          error!(
            error = ?rollback_err,
            "failed to rollback writer after NDJSON channel closed unexpectedly"
          );
        }
        Ok(total)
      });
      *writer_task = Some(handle);
    };

  let mut docs = Vec::with_capacity(NDJSON_BATCH_SIZE);
  let mut line_number = 0usize;

  let ingest_result: ApiResult<()> = async {
    loop {
      buf.clear();
      let read = reader
        .read_line(&mut buf)
        .await
        .map_err(|e| HttpError::from_anyhow("read_body", StatusCode::BAD_REQUEST, e.into()))?;

      if read == 0 {
        break;
      }

      line_number += 1;
      let trimmed = buf.trim();
      if trimmed.is_empty() {
        continue;
      }

      let value: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
        HttpError::bad_request(
          "invalid_document",
          format!("invalid JSON document on NDJSON line {line_number}: {e}"),
        )
      })?;

      let doc = value_to_document(value)?;
      docs.push(doc);

      if docs.len() >= NDJSON_BATCH_SIZE {
        ensure_writer_task(&mut rx_slot, &mut writer_task);
        let batch = std::mem::replace(&mut docs, Vec::with_capacity(NDJSON_BATCH_SIZE));
        if tx.send(IngestMsg::Batch(batch)).await.is_err() {
          let writer_err = await_writer_or_default(
            &mut writer_task,
            HttpError::from_anyhow(
              "ingest_worker_closed",
              StatusCode::INTERNAL_SERVER_ERROR,
              anyhow::anyhow!("ingest worker terminated early"),
            ),
          )
          .await;
          return Err(writer_err);
        }
      }
    }

    if !docs.is_empty() {
      ensure_writer_task(&mut rx_slot, &mut writer_task);
      if tx.send(IngestMsg::Batch(docs)).await.is_err() {
        let writer_err = await_writer_or_default(
          &mut writer_task,
          HttpError::from_anyhow(
            "ingest_worker_closed",
            StatusCode::INTERNAL_SERVER_ERROR,
            anyhow::anyhow!("ingest worker terminated early"),
          ),
        )
        .await;
        return Err(writer_err);
      }
    }

    Ok(())
  }
  .await;

  if let Err(err) = ingest_result {
    // Signal abort to rollback partial batches; ignore send errors if worker is gone.
    if writer_task.is_some() {
      let _ = tx.send(IngestMsg::Abort).await;
    }
    drop(tx);
    if let Some(handle) = writer_task.take() {
      let _ = handle.await;
    }
    return Err(err);
  }

  if writer_task.is_none() {
    drop(tx);
    return Ok(Json(IngestResponse { queued: 0 }));
  }

  // Signal commit; ignore send error if worker already exited.
  ensure_writer_task(&mut rx_slot, &mut writer_task);
  let _ = tx.send(IngestMsg::Commit).await;
  drop(tx);

  let total_queued = match writer_task.take() {
    Some(handle) => match handle.await {
      Ok(Ok(total)) => total,
      Ok(Err(e)) => return Err(e),
      Err(join_err) => {
        return Err(HttpError::from_anyhow(
          "add_join",
          StatusCode::INTERNAL_SERVER_ERROR,
          anyhow::anyhow!(join_err.to_string()),
        ))
      }
    },
    None => {
      return Ok(Json(IngestResponse { queued: 0 }));
    }
  };

  Ok(Json(IngestResponse {
    queued: total_queued,
  }))
}

async fn bulk_ingest(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
  headers: HeaderMap,
  payload: Result<Json<BulkRequest>, JsonRejection>,
) -> ApiResult<Json<IngestResponse>> {
  let body = parse_json(payload)?;
  if body.docs.is_empty() {
    return Err(HttpError::bad_request(
      "missing_documents",
      "docs array must contain at least one document",
    ));
  }
  let docs: Vec<Document> = body
    .docs
    .into_iter()
    .map(value_to_document)
    .collect::<ApiResult<_>>()?;
  let managed = state.registry().resolve(&index_name)?;
  let index = managed.require_index().await?;
  let manifest = index.manifest();
  let write_key = extract_write_key(&headers);
  if write_key.is_none() && write_key_required(&manifest) {
    return Err(HttpError::from_anyhow(
      "write_key_required",
      StatusCode::UNAUTHORIZED,
      anyhow::anyhow!("write key required"),
    ));
  }

  let writer_lock = managed.writer_lock.clone();
  tokio::task::spawn_blocking(move || {
    let _writer_guard = writer_lock.blocking_lock();
    let mut writer = index.writer_with_key(write_key.as_deref()).map_err(|e| {
      let msg = e.to_string().to_lowercase();
      let status = if msg.contains("write key") || msg.contains("unauthorized") {
        if write_key.is_some() {
          StatusCode::FORBIDDEN
        } else {
          StatusCode::UNAUTHORIZED
        }
      } else {
        StatusCode::INTERNAL_SERVER_ERROR
      };
      HttpError::from_anyhow("writer_open", status, e)
    })?;
    for doc in docs.iter() {
      if let Err(err) = writer.add_document(doc) {
        if let Err(rollback_err) = writer.rollback() {
          error!(
            error = ?rollback_err,
            "failed to rollback writer after bulk add failure"
          );
        }
        return Err(HttpError::bad_request("add_failed", err.to_string()));
      }
    }
    Ok(Json(IngestResponse { queued: docs.len() }))
  })
  .await
  .map_err(|err| {
    HttpError::from_anyhow(
      "add_join",
      StatusCode::INTERNAL_SERVER_ERROR,
      anyhow::anyhow!(err.to_string()),
    )
  })?
}

async fn delete_documents(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
  headers: HeaderMap,
  payload: Result<Json<DeleteRequest>, JsonRejection>,
) -> ApiResult<Json<DeleteResponse>> {
  let body = parse_json(payload)?;
  if body.ids.is_empty() {
    return Err(HttpError::bad_request(
      "missing_ids",
      "ids array must contain at least one document id",
    ));
  }
  validate_ids(&body.ids)?;
  let managed = state.registry().resolve(&index_name)?;
  let index = managed.require_index().await?;
  let manifest = index.manifest();
  let write_key = extract_write_key(&headers);
  if write_key.is_none() && write_key_required(&manifest) {
    return Err(HttpError::from_anyhow(
      "write_key_required",
      StatusCode::UNAUTHORIZED,
      anyhow::anyhow!("write key required"),
    ));
  }
  let _writer_guard = managed.writer_lock.lock().await;
  let mut writer = index.writer_with_key(write_key.as_deref()).map_err(|e| {
    let msg = e.to_string().to_lowercase();
    let status = if msg.contains("write key") || msg.contains("unauthorized") {
      if write_key.is_some() {
        StatusCode::FORBIDDEN
      } else {
        StatusCode::UNAUTHORIZED
      }
    } else {
      StatusCode::INTERNAL_SERVER_ERROR
    };
    HttpError::from_anyhow("writer_open", status, e)
  })?;
  writer
    .delete_documents(&body.ids)
    .map_err(|e| HttpError::bad_request("delete_failed", e.to_string()))?;
  Ok(Json(DeleteResponse {
    queued: body.ids.len(),
  }))
}

async fn update_document(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
  headers: HeaderMap,
  payload: Result<Json<UpdateRequest>, JsonRejection>,
) -> ApiResult<Json<UpdateResponse>> {
  let body = parse_json(payload)?;
  if body.set.is_empty() && body.unset.is_empty() {
    return Err(HttpError::bad_request(
      "missing_patch",
      "update must include at least one set or unset field",
    ));
  }
  if let Err(err) = validate_doc_id(&body.id) {
    return Err(HttpError::bad_request(
      "invalid_id",
      format!("invalid document id: {err}"),
    ));
  }
  let managed = state.registry().resolve(&index_name)?;
  let index = managed.require_index().await?;
  let manifest = index.manifest();
  let write_key = extract_write_key(&headers);
  if write_key.is_none() && write_key_required(&manifest) {
    return Err(HttpError::from_anyhow(
      "write_key_required",
      StatusCode::UNAUTHORIZED,
      anyhow::anyhow!("write key required"),
    ));
  }

  let writer_lock = managed.writer_lock.clone();
  tokio::task::spawn_blocking(move || {
    let _writer_guard = writer_lock.blocking_lock();
    let mut writer = index.writer_with_key(write_key.as_deref()).map_err(|e| {
      let msg = e.to_string().to_lowercase();
      let status = if msg.contains("write key") || msg.contains("unauthorized") {
        if write_key.is_some() {
          StatusCode::FORBIDDEN
        } else {
          StatusCode::UNAUTHORIZED
        }
      } else {
        StatusCode::INTERNAL_SERVER_ERROR
      };
      HttpError::from_anyhow("writer_open", status, e)
    })?;
    if let Err(err) = writer.apply_patch(&body.id, &body.set, &body.unset) {
      if let Some(PatchError::DocumentNotFound) = err.downcast_ref::<PatchError>() {
        return Err(HttpError::not_found("document_not_found", err.to_string()));
      }
      return Err(HttpError::bad_request("update_failed", err.to_string()));
    }
    Ok(Json(UpdateResponse { accepted: true }))
  })
  .await
  .map_err(|err| {
    HttpError::from_anyhow(
      "update_join",
      StatusCode::INTERNAL_SERVER_ERROR,
      anyhow::anyhow!(err.to_string()),
    )
  })?
}

async fn bulk_update(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
  headers: HeaderMap,
  body: Body,
) -> ApiResult<Json<BulkUpdateResponse>> {
  let managed = state.registry().resolve(&index_name)?;
  let index = managed.require_index().await?;
  let manifest = index.manifest();
  let write_key = extract_write_key(&headers);
  if write_key.is_none() && write_key_required(&manifest) {
    return Err(HttpError::from_anyhow(
      "write_key_required",
      StatusCode::UNAUTHORIZED,
      anyhow::anyhow!("write key required"),
    ));
  }

  let mapped_stream = body
    .into_data_stream()
    .map(|chunk| chunk.map_err(io::Error::other));
  let mut reader = StreamReader::new(mapped_stream);
  let mut buf = String::new();
  let mut line_number = 0usize;
  let mut pending_action: Option<String> = None;

  async fn await_writer_or_default(
    writer_task: &mut Option<tokio::task::JoinHandle<Result<BulkUpdateResponse, HttpError>>>,
    default: HttpError,
  ) -> HttpError {
    if let Some(handle) = writer_task.take() {
      match handle.await {
        Ok(Err(e)) => e,
        Ok(Ok(_)) => default,
        Err(join_err) => HttpError::from_anyhow(
          "bulk_update_join",
          StatusCode::INTERNAL_SERVER_ERROR,
          anyhow::anyhow!(join_err.to_string()),
        ),
      }
    } else {
      default
    }
  }

  let (tx, rx) = mpsc::channel::<BulkUpdateMsg>(INGEST_CHANNEL_BUFFER_BATCHES);
  let mut rx_slot = Some(rx);
  let mut writer_task: Option<tokio::task::JoinHandle<Result<BulkUpdateResponse, HttpError>>> =
    None;
  let write_key_clone = write_key.clone();

  let ensure_writer_task =
    |rx_slot: &mut Option<mpsc::Receiver<BulkUpdateMsg>>,
     writer_task: &mut Option<tokio::task::JoinHandle<Result<BulkUpdateResponse, HttpError>>>| {
      if writer_task.is_some() {
        return;
      }
      let Some(rx) = rx_slot.take() else {
        return;
      };
      let writer_lock = managed.writer_lock.clone();
      let index_ref = index.clone();
      let write_key = write_key_clone.clone();
      let handle = tokio::task::spawn_blocking(move || -> Result<BulkUpdateResponse, HttpError> {
        let _writer_guard = writer_lock.blocking_lock();
        let mut writer = index_ref
          .writer_with_key(write_key.as_deref())
          .map_err(|e| {
            let msg = e.to_string().to_lowercase();
            let status = if msg.contains("write key") || msg.contains("unauthorized") {
              if write_key.is_some() {
                StatusCode::FORBIDDEN
              } else {
                StatusCode::UNAUTHORIZED
              }
            } else {
              StatusCode::INTERNAL_SERVER_ERROR
            };
            HttpError::from_anyhow("writer_open", status, e)
          })?;
        let checkpoint = writer.checkpoint().map_err(|e| {
          HttpError::from_anyhow(
            "bulk_update_checkpoint",
            StatusCode::INTERNAL_SERVER_ERROR,
            e,
          )
        })?;
        let mut updated = 0u64;
        let mut failed = 0u64;
        let mut items: Vec<BulkUpdateItem> = Vec::new();
        let mut rx = rx;
        while let Some(msg) = rx.blocking_recv() {
          match msg {
            BulkUpdateMsg::Batch(batch) => {
              for (id, patch) in batch {
                match writer.apply_patch(&id, &patch.set, &patch.unset) {
                  Ok(()) => {
                    updated = updated.saturating_add(1);
                    items.push(BulkUpdateItem {
                      id,
                      status: StatusCode::OK.as_u16(),
                      error: None,
                    });
                  }
                  Err(err) => {
                    failed = failed.saturating_add(1);
                    let status = if matches!(
                      err.downcast_ref::<PatchError>(),
                      Some(PatchError::DocumentNotFound)
                    ) {
                      StatusCode::NOT_FOUND.as_u16()
                    } else {
                      StatusCode::BAD_REQUEST.as_u16()
                    };
                    items.push(BulkUpdateItem {
                      id,
                      status,
                      error: Some(err.to_string()),
                    });
                  }
                }
              }
            }
            BulkUpdateMsg::Commit => {
              return Ok(BulkUpdateResponse {
                updated,
                failed,
                items,
              });
            }
            BulkUpdateMsg::Abort => {
              if let Err(rollback_err) = writer.rollback_to(checkpoint) {
                error!(
                  error = ?rollback_err,
                  "failed to rollback bulk update request scope"
                );
              }
              return Ok(BulkUpdateResponse {
                updated,
                failed,
                items,
              });
            }
          }
        }
        if let Err(rollback_err) = writer.rollback_to(checkpoint) {
          error!(
            error = ?rollback_err,
            "failed to rollback bulk update request scope after channel closure"
          );
        }
        Ok(BulkUpdateResponse {
          updated,
          failed,
          items,
        })
      });
      *writer_task = Some(handle);
    };

  let mut updates = Vec::with_capacity(NDJSON_BATCH_SIZE);

  let ingest_result: ApiResult<()> = async {
    loop {
      buf.clear();
      let read = reader
        .read_line(&mut buf)
        .await
        .map_err(|e| HttpError::from_anyhow("read_body", StatusCode::BAD_REQUEST, e.into()))?;
      if read == 0 {
        break;
      }
      line_number += 1;
      let trimmed = buf.trim();
      if trimmed.is_empty() {
        continue;
      }

      if pending_action.is_none() {
        let action: BulkUpdateAction = serde_json::from_str(trimmed).map_err(|e| {
          HttpError::bad_request(
            "invalid_bulk_update",
            format!("invalid update action on NDJSON line {line_number}: {e}"),
          )
        })?;
        if let Err(err) = validate_doc_id(&action.update.id) {
          return Err(HttpError::bad_request(
            "invalid_bulk_update",
            format!(
              "invalid update action on NDJSON line {line_number}: invalid document id: {err}"
            ),
          ));
        }
        pending_action = Some(action.update.id);
      } else {
        let patch: BulkUpdatePatch = serde_json::from_str(trimmed).map_err(|e| {
          HttpError::bad_request(
            "invalid_bulk_update",
            format!("invalid update body on NDJSON line {line_number}: {e}"),
          )
        })?;
        let id = pending_action
          .take()
          .expect("pending action must exist for update body");
        updates.push((id, patch));

        if updates.len() >= NDJSON_BATCH_SIZE {
          ensure_writer_task(&mut rx_slot, &mut writer_task);
          let batch = std::mem::replace(&mut updates, Vec::with_capacity(NDJSON_BATCH_SIZE));
          if tx.send(BulkUpdateMsg::Batch(batch)).await.is_err() {
            let writer_err = await_writer_or_default(
              &mut writer_task,
              HttpError::from_anyhow(
                "bulk_update_worker_closed",
                StatusCode::INTERNAL_SERVER_ERROR,
                anyhow::anyhow!("bulk update worker terminated early"),
              ),
            )
            .await;
            return Err(writer_err);
          }
        }
      }
    }
    Ok(())
  }
  .await;

  if let Err(err) = ingest_result {
    if writer_task.is_some() {
      let _ = tx.send(BulkUpdateMsg::Abort).await;
    }
    drop(tx);
    if let Some(handle) = writer_task.take() {
      let _ = handle.await;
    }
    return Err(err);
  }

  if pending_action.is_some() {
    if writer_task.is_some() {
      let _ = tx.send(BulkUpdateMsg::Abort).await;
    }
    drop(tx);
    if let Some(handle) = writer_task.take() {
      let _ = handle.await;
    }
    return Err(HttpError::bad_request(
      "invalid_bulk_update",
      "update action missing corresponding body line",
    ));
  }

  if !updates.is_empty() {
    ensure_writer_task(&mut rx_slot, &mut writer_task);
    if tx.send(BulkUpdateMsg::Batch(updates)).await.is_err() {
      let writer_err = await_writer_or_default(
        &mut writer_task,
        HttpError::from_anyhow(
          "bulk_update_worker_closed",
          StatusCode::INTERNAL_SERVER_ERROR,
          anyhow::anyhow!("bulk update worker terminated early"),
        ),
      )
      .await;
      return Err(writer_err);
    }
  }

  if writer_task.is_none() {
    drop(tx);
    return Ok(Json(BulkUpdateResponse {
      updated: 0,
      failed: 0,
      items: Vec::new(),
    }));
  }

  if tx.send(BulkUpdateMsg::Commit).await.is_err() {
    let writer_err = await_writer_or_default(
      &mut writer_task,
      HttpError::from_anyhow(
        "bulk_update_worker_closed",
        StatusCode::INTERNAL_SERVER_ERROR,
        anyhow::anyhow!("bulk update worker terminated early"),
      ),
    )
    .await;
    return Err(writer_err);
  }

  drop(tx);
  let response = match writer_task.take() {
    Some(handle) => match handle.await {
      Ok(Ok(resp)) => resp,
      Ok(Err(err)) => return Err(err),
      Err(err) => {
        return Err(HttpError::from_anyhow(
          "bulk_update_join",
          StatusCode::INTERNAL_SERVER_ERROR,
          anyhow::anyhow!(err.to_string()),
        ))
      }
    },
    None => BulkUpdateResponse {
      updated: 0,
      failed: 0,
      items: Vec::new(),
    },
  };

  Ok(Json(response))
}

fn trigger_reader_refresh(index: &Index) -> anyhow::Result<()> {
  // Opening a reader reloads searchers; the returned reader can be dropped
  // immediately when only a refresh side effect is desired.
  index.reader().map(|_| ())
}

fn should_refresh(last_refreshed_committed_at: Option<&str>, current_committed_at: &str) -> bool {
  !matches!(
    last_refreshed_committed_at,
    Some(previous) if previous == current_committed_at
  )
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

async fn commit(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
  headers: HeaderMap,
) -> ApiResult<Json<CommitResponse>> {
  let managed = state.registry().resolve(&index_name)?;
  let index = managed.require_index().await?;
  let manifest = index.manifest();
  let write_key = extract_write_key(&headers);
  if write_key.is_none() && write_key_required(&manifest) {
    return Err(HttpError::from_anyhow(
      "write_key_required",
      StatusCode::UNAUTHORIZED,
      anyhow::anyhow!("write key required"),
    ));
  }
  let refresh = managed.refresh_on_commit;
  let writer_lock = managed.writer_lock.clone();
  let write_key_clone = write_key.clone();
  tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
    let _guard = writer_lock.blocking_lock();
    let mut writer = index.writer_with_key(write_key_clone.as_deref())?;
    writer.commit()?;
    if refresh {
      trigger_reader_refresh(&index)?;
    }
    Ok(())
  })
  .await
  .map_err(|err| {
    HttpError::from_anyhow(
      "commit_join",
      StatusCode::INTERNAL_SERVER_ERROR,
      anyhow::anyhow!(err.to_string()),
    )
  })?
  .map_err(|err| {
    let msg = err.to_string();
    let status = if msg.to_lowercase().contains("write key") {
      if write_key.is_some() {
        StatusCode::FORBIDDEN
      } else {
        StatusCode::UNAUTHORIZED
      }
    } else {
      StatusCode::INTERNAL_SERVER_ERROR
    };
    HttpError::from_anyhow("commit_failed", status, err)
  })?;
  Ok(Json(CommitResponse { committed: true }))
}

async fn refresh(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
) -> ApiResult<Json<RefreshResponse>> {
  let managed = state.registry().resolve(&index_name)?;
  let index = managed.require_index().await?;
  tokio::task::spawn_blocking(move || trigger_reader_refresh(&index))
    .await
    .map_err(|err| {
      HttpError::from_anyhow(
        "refresh_join",
        StatusCode::INTERNAL_SERVER_ERROR,
        anyhow::anyhow!(err.to_string()),
      )
    })?
    .map_err(|err| {
      HttpError::from_anyhow("refresh_failed", StatusCode::INTERNAL_SERVER_ERROR, err)
    })?;
  Ok(Json(RefreshResponse { refreshed: true }))
}

async fn compact(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
  headers: HeaderMap,
) -> ApiResult<Json<CompactResponse>> {
  let managed = state.registry().resolve(&index_name)?;
  let index = managed.require_index().await?;
  let manifest = index.manifest();
  let write_key = extract_write_key(&headers);
  if write_key.is_none() && write_key_required(&manifest) {
    return Err(HttpError::from_anyhow(
      "write_key_required",
      StatusCode::UNAUTHORIZED,
      anyhow::anyhow!("write key required"),
    ));
  }
  let writer_lock = managed.writer_lock.clone();
  let write_key_clone = write_key.clone();
  tokio::task::spawn_blocking(move || {
    let _guard = writer_lock.blocking_lock();
    index.compact_with_key(write_key_clone.as_deref())
  })
  .await
  .map_err(|err| {
    HttpError::from_anyhow(
      "compact_join",
      StatusCode::INTERNAL_SERVER_ERROR,
      anyhow::anyhow!(err.to_string()),
    )
  })?
  .map_err(|err| {
    let msg = err.to_string();
    let status = if msg.to_lowercase().contains("write key") {
      if write_key.is_some() {
        StatusCode::FORBIDDEN
      } else {
        StatusCode::UNAUTHORIZED
      }
    } else {
      StatusCode::INTERNAL_SERVER_ERROR
    };
    HttpError::from_anyhow("compact_failed", status, err)
  })?;
  Ok(Json(CompactResponse { compacted: true }))
}

async fn search(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
  payload: Result<Json<SearchRequest>, JsonRejection>,
) -> ApiResult<Json<SearchResult>> {
  let mut request = parse_json(payload)?;
  if request.limit == 0 {
    if request.cursor.is_some() {
      return Err(HttpError::bad_request(
        "invalid_cursor",
        "cursor is not supported when limit is 0",
      ));
    }
    if request.from > 0 {
      return Err(HttpError::bad_request(
        "invalid_pagination",
        "from is not supported when limit is 0",
      ));
    }
  }
  if request.cursor.is_some() && request.search_after.is_some() {
    return Err(HttpError::bad_request(
      "invalid_pagination",
      "cursor cannot be combined with search_after; supply only one pagination token",
    ));
  }
  if request.search_after.is_some() && request.from > 0 {
    return Err(HttpError::bad_request(
      "invalid_pagination",
      "search_after cannot be combined with from; choose one pagination method",
    ));
  }
  if request.cursor.is_some() && request.from > 0 {
    return Err(HttpError::bad_request(
      "invalid_pagination",
      "from must be 0 when using cursor pagination; remove from or use offset pagination alone",
    ));
  }
  if request.cursor.is_some() {
    request.search_after = None;
    request.from = 0;
  }
  let page_cap = request.from.saturating_add(request.limit);
  if request.return_hits && page_cap > MAX_PAGE_SIZE {
    return Err(HttpError::bad_request(
      "page_too_large",
      format!("from + size exceeds max page size {MAX_PAGE_SIZE}"),
    ));
  }
  let managed = state.registry().resolve(&index_name)?;
  #[cfg(feature = "vectors")]
  {
    // Precedence: per-request JSON overrides the server default (CLI/env); otherwise use the server default.
    request
      .max_global_vector_candidates
      .get_or_insert(managed.max_vector_candidates);
  }
  let index = managed.require_index().await?;
  let result = tokio::task::spawn_blocking(move || -> anyhow::Result<SearchResult> {
    let reader = index.reader()?;
    reader.search(&request)
  })
  .await
  .map_err(|err| {
    HttpError::from_anyhow(
      "search_join",
      StatusCode::INTERNAL_SERVER_ERROR,
      anyhow::anyhow!(err.to_string()),
    )
  })?
  .map_err(|err| HttpError::from_anyhow("search_failed", StatusCode::BAD_REQUEST, err))?;
  Ok(Json(result))
}

async fn mget(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
  payload: Result<Json<MgetRequest>, JsonRejection>,
) -> ApiResult<Json<MgetResponse>> {
  let body = parse_json(payload)?;
  if body.ids.is_empty() {
    return Err(HttpError::bad_request(
      "missing_ids",
      "ids array must contain at least one document id",
    ));
  }
  if body.ids.len() > MAX_MGET_IDS {
    return Err(HttpError::bad_request(
      "too_many_ids",
      format!("mget supports up to {MAX_MGET_IDS} ids per request"),
    ));
  }
  validate_ids(&body.ids)?;
  let managed = state.registry().resolve(&index_name)?;
  let index = managed.require_index().await?;
  let return_stored = body.return_stored;
  let ids = body.ids.clone();
  let resp = tokio::task::spawn_blocking(move || -> anyhow::Result<MgetResponse> {
    let reader = index.reader()?;
    let docs = reader.mget(&ids, return_stored)?;
    Ok(MgetResponse { docs })
  })
  .await
  .map_err(|err| {
    HttpError::from_anyhow(
      "mget_join",
      StatusCode::INTERNAL_SERVER_ERROR,
      anyhow::anyhow!(err.to_string()),
    )
  })?
  .map_err(|err| HttpError::from_anyhow("mget_failed", StatusCode::BAD_REQUEST, err))?;
  Ok(Json(resp))
}

async fn multi_search(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
  payload: Result<Json<MultiSearchRequest>, JsonRejection>,
) -> ApiResult<Json<MultiSearchResponse>> {
  let MultiSearchRequest {
    searches,
    parallel,
    max_concurrency,
  } = parse_json(payload)?;
  if searches.is_empty() {
    return Err(HttpError::bad_request(
      "missing_searches",
      "searches array must contain at least one search request",
    ));
  }
  // Validate each sub-request to mirror /search pagination and page-size rules.
  let validate_search = |req: &SearchRequest| -> Result<(), HttpError> {
    if req.limit == 0 {
      if req.cursor.is_some() {
        return Err(HttpError::bad_request(
          "invalid_cursor",
          "cursor is not supported when limit is 0",
        ));
      }
      if req.from > 0 {
        return Err(HttpError::bad_request(
          "invalid_pagination",
          "from is not supported when limit is 0",
        ));
      }
    }
    let has_cursor = req.cursor.is_some();
    let tmp_from = if has_cursor { 0 } else { req.from };
    if has_cursor && req.search_after.is_some() {
      return Err(HttpError::bad_request(
        "invalid_pagination",
        "cursor cannot be combined with search_after; choose one pagination method",
      ));
    }
    if has_cursor && req.from > 0 {
      return Err(HttpError::bad_request(
        "invalid_pagination",
        "from must be 0 when using cursor pagination",
      ));
    }
    if req.search_after.is_some() && req.from > 0 {
      return Err(HttpError::bad_request(
        "invalid_pagination",
        "search_after cannot be combined with from; choose one pagination method",
      ));
    }
    let cap = tmp_from.saturating_add(req.limit);
    if req.return_hits && cap > MAX_PAGE_SIZE {
      return Err(HttpError::bad_request(
        "page_too_large",
        format!("from + size exceeds max page size {MAX_PAGE_SIZE}"),
      ));
    }
    Ok(())
  };
  for req in searches.iter() {
    validate_search(req)?;
  }
  #[cfg(feature = "vectors")]
  let mut searches = searches;
  #[cfg(not(feature = "vectors"))]
  let searches = searches;
  let managed = state.registry().resolve(&index_name)?;
  #[cfg(feature = "vectors")]
  {
    for req in searches.iter_mut() {
      req
        .max_global_vector_candidates
        .get_or_insert(managed.max_vector_candidates);
    }
  }
  let index = managed.require_index().await?;
  let max_concurrency = max_concurrency
    .unwrap_or(DEFAULT_MULTI_SEARCH_MAX_CONCURRENCY)
    .clamp(1, HARD_MULTI_SEARCH_MAX_CONCURRENCY);

  if !parallel {
    let resp = tokio::task::spawn_blocking(move || -> anyhow::Result<MultiSearchResponse> {
      let reader = index.reader()?;
      let mut results = Vec::with_capacity(searches.len());
      for mut req in searches.into_iter() {
        if req.cursor.is_some() {
          req.search_after = None;
          req.from = 0;
        }
        results.push(reader.search(&req)?);
      }
      Ok(MultiSearchResponse { results })
    })
    .await
    .map_err(|err| {
      HttpError::from_anyhow(
        "multi_search_join",
        StatusCode::INTERNAL_SERVER_ERROR,
        anyhow::anyhow!(err.to_string()),
      )
    })?
    .map_err(|err| HttpError::from_anyhow("multi_search_failed", StatusCode::BAD_REQUEST, err))?;
    return Ok(Json(resp));
  }

  let idx = index.clone();
  let semaphore = Arc::new(Semaphore::new(max_concurrency));
  let mut tasks: FuturesUnordered<_> = FuturesUnordered::new();
  for (search_idx, mut req) in searches.into_iter().enumerate() {
    if req.cursor.is_some() {
      req.search_after = None;
      req.from = 0;
    }
    let semaphore_clone = semaphore.clone();
    let index_clone = idx.clone();
    tasks.push(async move {
      let permit = semaphore_clone.acquire_owned().await.map_err(|err| {
        HttpError::from_anyhow(
          "multi_search_cancelled",
          StatusCode::INTERNAL_SERVER_ERROR,
          anyhow::anyhow!(err.to_string()),
        )
      })?;
      let handle = tokio::task::spawn_blocking(move || -> anyhow::Result<SearchResult> {
        let _permit = permit;
        let reader = index_clone.reader()?;
        reader.search(&req)
      });
      let joined = handle.await.map_err(|err: tokio::task::JoinError| {
        HttpError::from_anyhow(
          "multi_search_join",
          StatusCode::INTERNAL_SERVER_ERROR,
          anyhow::anyhow!(err.to_string()),
        )
      })?;
      let search_res = joined.map_err(|err| {
        HttpError::from_anyhow("multi_search_failed", StatusCode::BAD_REQUEST, err)
      })?;
      Ok::<(usize, SearchResult), HttpError>((search_idx, search_res))
    });
  }
  let mut results: Vec<Option<SearchResult>> = vec![None; tasks.len()];
  while let Some(res) = tasks.next().await {
    let (idx_search, search_res) = res?;
    if let Some(slot) = results.get_mut(idx_search) {
      *slot = Some(search_res);
    }
  }
  let results: Vec<SearchResult> = results
    .into_iter()
    .map(|r| r.expect("multi_search invariant violated: missing SearchResult for a task"))
    .collect();
  Ok(Json(MultiSearchResponse { results }))
}

async fn inspect(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
) -> ApiResult<Json<InspectResponse>> {
  let managed = state.registry().resolve(&index_name)?;
  let index = managed.require_index().await?;
  let mut manifest = tokio::task::spawn_blocking(move || Ok::<_, anyhow::Error>(index.manifest()))
    .await
    .map_err(|err| {
      HttpError::from_anyhow(
        "inspect_join",
        StatusCode::INTERNAL_SERVER_ERROR,
        anyhow::anyhow!(err.to_string()),
      )
    })?
    .map_err(|err| {
      HttpError::from_anyhow("inspect_failed", StatusCode::INTERNAL_SERVER_ERROR, err)
    })?;
  // Redact write-key metadata to avoid leaking hash/salt material in public responses.
  manifest.write_key = None;
  for seg in manifest.segments.iter_mut() {
    seg.write_binding_b64 = None;
  }
  Ok(Json(InspectResponse { manifest }))
}

async fn stats(
  State(state): State<Arc<AppState>>,
  Path(index_name): Path<String>,
) -> ApiResult<Json<StatsResponse>> {
  let managed = state.registry().resolve(&index_name)?;
  let index = managed.require_index().await?;
  let manifest = tokio::task::spawn_blocking(move || Ok::<_, anyhow::Error>(index.manifest()))
    .await
    .map_err(|err| {
      HttpError::from_anyhow(
        "stats_join",
        StatusCode::INTERNAL_SERVER_ERROR,
        anyhow::anyhow!(err.to_string()),
      )
    })?
    .map_err(|err| {
      HttpError::from_anyhow("stats_failed", StatusCode::INTERNAL_SERVER_ERROR, err)
    })?;
  let (live_docs, deleted_docs) = manifest_doc_counts(&manifest);
  Ok(Json(StatsResponse {
    documents: live_docs,
    deleted_documents: deleted_docs,
    segments: manifest.segments.len(),
    committed_at: manifest.committed_at.clone(),
    index_uuid: manifest.uuid.to_string(),
    index_name: managed.name.clone(),
  }))
}

fn value_to_document(value: serde_json::Value) -> ApiResult<Document> {
  let Some(obj) = value.as_object() else {
    return Err(HttpError::bad_request(
      "invalid_document",
      "document must be a JSON object with fields at the top level",
    ));
  };
  let mut fields = BTreeMap::new();
  for (k, v) in obj.iter() {
    fields.insert(k.clone(), v.clone());
  }
  Ok(Document { fields })
}

fn validate_ids(ids: &[String]) -> ApiResult<()> {
  for (idx, id) in ids.iter().enumerate() {
    if let Err(err) = validate_doc_id(id) {
      return Err(HttpError::bad_request(
        "invalid_id",
        format!("id at position {idx} is invalid: {err}"),
      ));
    }
  }
  Ok(())
}

async fn shutdown_signal(grace_secs: u64) {
  let ctrl_c = async {
    if let Err(err) = tokio::signal::ctrl_c().await {
      error!(error = ?err, "failed to install ctrl+c handler");
    }
  };
  #[cfg(unix)]
  let terminate = async {
    use tokio::signal::unix::{signal, SignalKind};
    if let Ok(mut sig) = signal(SignalKind::terminate()) {
      sig.recv().await;
    }
  };
  #[cfg(not(unix))]
  let terminate = std::future::pending::<()>();

  tokio::select! {
    _ = ctrl_c => {},
    _ = terminate => {},
  }

  info!("shutdown signal received, draining in-flight requests");
  if grace_secs > 0 {
    tokio::time::sleep(Duration::from_secs(grace_secs)).await;
  }
}

pub fn init_tracing() {
  let env_filter =
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,tower_http=info"));
  fmt()
    .with_target(false)
    .with_env_filter(env_filter)
    .json()
    .try_init()
    .ok();
}

#[cfg(test)]
mod tests {
  use super::*;
  use reqwest::{Client, StatusCode as HttpStatus};
  #[cfg(feature = "vectors")]
  use searchlite_core::api::types::VectorQuery;
  use searchlite_core::api::types::{
    Aggregation, AggregationResponse, CollapseRequest, ExecutionStrategy, HighlightField,
    HighlightRequest, IndexOptions, MgetResponse, MultiSearchRequest, Query, StorageType,
    SuggestRequest,
  };
  use searchlite_core::api::MultiSearchResponse;
  #[cfg(feature = "vectors")]
  use searchlite_core::api::QueryNode;
  use serde_json::json;
  use std::collections::BTreeMap;
  use std::path::PathBuf;
  use tempfile::tempdir;
  use tokio::task::JoinHandle;

  const INDEX_NAME: &str = "primary";

  async fn spawn_server(
    args: ServeArgs,
    state: Arc<AppState>,
  ) -> anyhow::Result<(SocketAddr, JoinHandle<anyhow::Result<()>>)> {
    let listener = TcpListener::bind(args.bind).await?;
    let addr = listener.local_addr()?;
    let app = router(state, &args);
    let handle = tokio::spawn(async move {
      axum::serve(listener, app)
        .with_graceful_shutdown(async {
          // Tests stop the server by dropping the handle.
          std::future::pending::<()>().await;
        })
        .await
        .context("serve test app")
    });
    Ok((addr, handle))
  }

  async fn setup_server(
    index: PathBuf,
  ) -> (
    Client,
    String,
    String,
    JoinHandle<anyhow::Result<()>>,
    Arc<AppState>,
    ServeArgs,
  ) {
    let args = default_args(index);
    let registry = Arc::new(IndexRegistry::from_args(&args).unwrap());
    registry.bootstrap_all().await.unwrap();
    let state = build_app_state(registry).unwrap();
    let (addr, handle) = spawn_server(args.clone(), state.clone()).await.unwrap();
    let client = Client::new();
    let base = format!("http://{addr}");
    let index_base = format!("{base}/indexes/{INDEX_NAME}");
    (client, base, index_base, handle, state, args)
  }

  fn default_args(index: PathBuf) -> ServeArgs {
    ServeArgs {
      indexes: vec![IndexSpec {
        name: INDEX_NAME.into(),
        path: index,
        auto_commit_interval_secs: None,
        auto_refresh_interval_secs: None,
      }],
      aliases: vec![],
      bind: "127.0.0.1:0".parse().unwrap(),
      require_existing_index: false,
      max_body_bytes: 10 * 1024 * 1024,
      max_concurrency: 8,
      request_timeout_secs: 10,
      shutdown_grace_secs: 0,
      refresh_on_commit: false,
      auto_commit_interval_secs: 0,
      auto_refresh_interval_secs: 0,
      #[cfg(feature = "vectors")]
      max_vector_candidates: DEFAULT_MAX_VECTOR_GLOBAL_CANDIDATES,
    }
  }

  #[tokio::test]
  async fn http_flow_covers_search_lifecycle() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx");
    let (client, base, index_base, handle, _state, _args) = setup_server(index_path.clone()).await;

    // init
    let schema = Schema::default_text_body();
    let res = client
      .post(format!("{index_base}/init"))
      .json(&schema)
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());
    // add docs
    let ndjson =
      "{\"_id\":\"1\",\"body\":\"Rust search\"}\n{\"_id\":\"2\",\"body\":\"Another doc\"}\n";
    let res = client
      .post(format!("{index_base}/add"))
      .body(ndjson.to_string())
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());

    // commit
    let res = client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());

    // search
    let req = SearchRequest {
      query: Query::String("rust".into()),
      fields: None,
      filter: None,
      limit: 5,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
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
      aggs: Default::default(),
      suggest: Default::default(),
      rescore: None,
      explain: false,
      profile: false,
    };
    let res = client
      .post(format!("{index_base}/search"))
      .json(&req)
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());
    let body: SearchResult = res.json().await.unwrap();
    assert_eq!(body.hits.len(), 1);
    assert_eq!(body.hits[0].doc_id, "1");

    // inspect
    let res = client
      .get(format!("{index_base}/inspect"))
      .send()
      .await
      .unwrap()
      .json::<InspectResponse>()
      .await
      .unwrap();
    assert_eq!(res.manifest.segments.len(), 1);

    // stats
    let stats = client
      .get(format!("{index_base}/stats"))
      .send()
      .await
      .unwrap()
      .json::<StatsResponse>()
      .await
      .unwrap();
    assert_eq!(stats.documents, 2);
    assert_eq!(stats.index_name, INDEX_NAME);

    // list indexes
    let list = client
      .get(format!("{base}/indexes"))
      .send()
      .await
      .unwrap()
      .json::<IndexListResponse>()
      .await
      .unwrap();
    assert_eq!(list.indexes.len(), 1);

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn list_indexes_exposes_runtime_metadata() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-list-meta");
    let (client, base, index_base, handle, _state, _args) = setup_server(index_path).await;

    let before: serde_json::Value = client
      .get(format!("{base}/indexes"))
      .send()
      .await
      .unwrap()
      .json()
      .await
      .unwrap();
    let first = &before["indexes"][0];
    assert_eq!(first["name"], INDEX_NAME);
    assert_eq!(first["exists"], false);
    assert!(first["committed_at"].is_null());
    assert!(first["doc_count"].is_null());
    assert_eq!(first["auto_commit_secs"], 0);
    assert_eq!(first["auto_refresh_secs"], 0);
    assert_eq!(first["refresh_on_commit"], false);

    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    client
      .post(format!("{index_base}/add"))
      .body("{\"_id\":\"1\",\"body\":\"doc\"}\n")
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let after: serde_json::Value = client
      .get(format!("{base}/indexes"))
      .send()
      .await
      .unwrap()
      .json()
      .await
      .unwrap();
    let first_after = &after["indexes"][0];
    assert_eq!(first_after["exists"], true);
    assert!(first_after["committed_at"].as_str().is_some());
    assert_eq!(first_after["doc_count"], 1);

    handle.abort();
    let _ = handle.await;
  }

  // Regression test for BUG-015: the public `/stats` endpoint must not leak
  // the on-disk filesystem path of the index. Operator-only details are
  // confined to server logs / admin tooling.
  #[tokio::test]
  async fn stats_response_does_not_expose_filesystem_path() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-stats-no-path");
    let (client, _base, index_base, handle, _state, _args) = setup_server(index_path.clone()).await;

    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/add"))
      .body("{\"_id\":\"1\",\"body\":\"doc\"}\n")
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let stats: serde_json::Value = client
      .get(format!("{index_base}/stats"))
      .send()
      .await
      .unwrap()
      .json()
      .await
      .unwrap();

    // Path-leaking fields must not appear in the response under any alias.
    // Walk the parsed JSON and check string values directly so we are not
    // tricked by JSON escaping (e.g. Windows backslashes are doubled in the
    // raw serialized form, which would let a regression slip past a naive
    // substring check on the wire bytes).
    fn json_contains_string_value(value: &serde_json::Value, target: &str) -> bool {
      match value {
        serde_json::Value::String(s) => s == target,
        serde_json::Value::Array(values) => values
          .iter()
          .any(|value| json_contains_string_value(value, target)),
        serde_json::Value::Object(map) => map
          .values()
          .any(|value| json_contains_string_value(value, target)),
        _ => false,
      }
    }

    let stats_obj = stats.as_object().expect("stats body is a JSON object");
    assert!(
      !stats_obj.contains_key("index_path"),
      "stats response must not include `index_path` (leaks FS layout): {stats_obj:?}"
    );
    let fs_path_str = index_path.display().to_string();
    assert!(
      !json_contains_string_value(&stats, fs_path_str.as_str()),
      "stats response must not contain the raw index filesystem path: {stats:?}"
    );

    // Sanity: the public fields are still present.
    assert_eq!(stats["index_name"], INDEX_NAME);
    assert!(stats["index_uuid"].as_str().is_some());
    assert_eq!(stats["documents"], 1);

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn auto_commit_persists_pending_writes() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-auto-commit");
    let mut args = default_args(index_path);
    args.indexes[0].auto_commit_interval_secs = Some(1);
    let registry = Arc::new(IndexRegistry::from_args(&args).unwrap());
    registry.bootstrap_all().await.unwrap();
    let state = build_app_state(registry).unwrap();
    let (addr, handle) = spawn_server(args.clone(), state).await.unwrap();
    let client = Client::new();
    let base = format!("http://{addr}");
    let index_base = format!("{base}/indexes/{INDEX_NAME}");

    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/add"))
      .body("{\"_id\":\"1\",\"body\":\"scheduled commit\"}\n")
      .send()
      .await
      .unwrap();

    let mut committed_docs = 0u64;
    for _ in 0..20 {
      tokio::time::sleep(Duration::from_millis(150)).await;
      let stats: StatsResponse = client
        .get(format!("{index_base}/stats"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
      committed_docs = stats.documents;
      if committed_docs > 0 {
        break;
      }
    }
    assert_eq!(committed_docs, 1);

    handle.abort();
    let _ = handle.await;
  }

  #[test]
  fn refresh_guard_skips_unchanged_commit_marker() {
    assert!(!should_refresh(
      Some("2026-03-03T00:00:00Z"),
      "2026-03-03T00:00:00Z"
    ));
    assert!(should_refresh(
      Some("2026-03-03T00:00:00Z"),
      "2026-03-03T00:00:01Z"
    ));
    assert!(should_refresh(None, "2026-03-03T00:00:00Z"));
  }

  #[tokio::test]
  async fn auto_refresh_only_index_starts_maintenance_task() {
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-auto-refresh-only");
    let mut args = default_args(index_path);
    args.indexes[0].auto_commit_interval_secs = Some(0);
    args.indexes[0].auto_refresh_interval_secs = Some(1);

    let registry = Arc::new(IndexRegistry::from_args(&args).unwrap());
    registry.bootstrap_all().await.unwrap();
    let state = build_app_state(registry).unwrap();
    assert_eq!(state._maintenance.handles.len(), 1);
  }

  #[tokio::test]
  async fn auto_commit_rejects_write_key_protected_existing_index() {
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-write-key");
    let opts = IndexOptions {
      path: index_path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: DEFAULT_K1,
      bm25_b: DEFAULT_B,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    };
    IndexBuilder::create_with_write_key(
      &index_path,
      Schema::default_text_body(),
      opts,
      Some("server-secret"),
    )
    .unwrap();

    let mut args = default_args(index_path);
    args.indexes[0].auto_commit_interval_secs = Some(1);
    let registry = Arc::new(IndexRegistry::from_args(&args).unwrap());
    registry.bootstrap_all().await.unwrap();
    match build_app_state(registry) {
      Ok(_) => panic!("expected startup error for write-key protected auto-commit index"),
      Err(err) => assert!(err.to_string().contains("requires a write key")),
    }
  }

  #[tokio::test]
  async fn http_supports_aggs_suggest_and_highlight() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-aggs");
    let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

    let schema: Schema = serde_json::from_value(json!({
      "type": "object",
      "properties": {
        "body": { "type": "string" },
        "lang": { "type": "string", "searchlite:kind": "keyword" }
      }
    }))
    .unwrap();
    client
      .post(format!("{index_base}/init"))
      .json(&schema)
      .send()
      .await
      .unwrap();

    let ndjson = "{\"_id\":\"1\",\"body\":\"Rust search\",\"lang\":\"en\"}\n\
                  {\"_id\":\"2\",\"body\":\"Rustaceans write Rust\",\"lang\":\"en\"}\n\
                  {\"_id\":\"3\",\"body\":\"Recherche en français\",\"lang\":\"fr\"}\n";
    client
      .post(format!("{index_base}/add"))
      .body(ndjson.to_string())
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let aggs: BTreeMap<String, Aggregation> = serde_json::from_value(json!({
      "langs": { "type": "terms", "field": "lang", "size": 5 }
    }))
    .unwrap();
    let mut suggest = BTreeMap::new();
    suggest.insert(
      "complete".into(),
      SuggestRequest::Completion {
        field: "body".into(),
        prefix: "ru".into(),
        size: 3,
        fuzzy: None,
      },
    );
    let mut highlight_fields = BTreeMap::new();
    highlight_fields.insert(
      "body".into(),
      HighlightField {
        pre_tag: "<em>".into(),
        post_tag: "</em>".into(),
        fragment_size: 64,
        number_of_fragments: 1,
      },
    );
    let request = SearchRequest {
      query: Query::String("rust".into()),
      fields: None,
      filter: None,
      limit: 5,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
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
      highlight_field: Some("body".into()),
      highlight: Some(HighlightRequest {
        fields: highlight_fields,
      }),
      collapse: Some(CollapseRequest {
        field: "lang".into(),
        inner_hits: None,
      }),
      aggs,
      suggest,
      rescore: None,
      explain: false,
      profile: false,
    };
    let res = client
      .post(format!("{index_base}/search"))
      .json(&request)
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());
    let body: SearchResult = res.json().await.unwrap();
    assert!(!body.hits.is_empty());
    assert!(body.hits.iter().any(|h| h
      .highlights
      .as_ref()
      .map(|m| m.contains_key("body"))
      .unwrap_or(false)));
    let langs = body.aggregations.get("langs").expect("langs aggregation");
    match langs {
      AggregationResponse::Terms { buckets, .. } => {
        assert!(buckets.iter().any(|b| b.key == json!("en")));
      }
      _ => panic!("expected terms aggregation"),
    }
    let suggestions = body.suggest.get("complete").expect("suggest results");
    assert!(!suggestions.options.is_empty());

    let compact = client
      .post(format!("{index_base}/compact"))
      .send()
      .await
      .unwrap();
    assert!(compact.status().is_success());

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn http_supports_nested_aggregations() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-nested-aggs");
    let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

    let schema: Schema = serde_json::from_value(json!({
      "type": "object",
      "properties": {
        "body": { "type": "string" },
        "images": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "illustrator": { "type": "string", "searchlite:kind": "keyword" }
            }
          }
        }
      }
    }))
    .unwrap();
    client
      .post(format!("{index_base}/init"))
      .json(&schema)
      .send()
      .await
      .unwrap();

    let ndjson = "{\"_id\":\"1\",\"body\":\"Rust search\",\"images\":[{\"illustrator\":\"alice\"},{\"illustrator\":\"bob\"}]}\n\
                  {\"_id\":\"2\",\"body\":\"Rust faceting\",\"images\":[{\"illustrator\":\"alice\"}]}\n";
    client
      .post(format!("{index_base}/add"))
      .body(ndjson.to_string())
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let aggs: BTreeMap<String, Aggregation> = serde_json::from_value(json!({
      "illustrators": {
        "type": "nested",
        "path": "images",
        "aggs": {
          "names": { "type": "terms", "field": "images.illustrator", "size": 5 }
        }
      }
    }))
    .unwrap();
    let request = SearchRequest {
      query: Query::String("rust".into()),
      fields: None,
      filter: None,
      limit: 0,
      from: 0,
      return_hits: false,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
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
      return_stored: false,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs,
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    };
    let res = client
      .post(format!("{index_base}/search"))
      .json(&request)
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());
    let body: SearchResult = res.json().await.unwrap();
    let nested = body
      .aggregations
      .get("illustrators")
      .expect("nested aggregation");
    match nested {
      AggregationResponse::Nested {
        doc_count,
        aggregations,
        ..
      } => {
        assert_eq!(*doc_count, 3);
        if let Some(AggregationResponse::Terms { buckets, .. }) = aggregations.get("names") {
          assert_eq!(buckets[0].key, json!("alice"));
          assert_eq!(buckets[0].doc_count, 2);
        } else {
          panic!("expected nested names terms aggregation");
        }
      }
      _ => panic!("expected nested aggregation response"),
    }

    handle.abort();
    let _ = handle.await;
  }

  #[cfg(feature = "vectors")]
  #[tokio::test]
  async fn http_supports_vector_search() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-vector");
    let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

    let schema: Schema = serde_json::from_value(json!({
      "type": "object",
      "properties": {
        "body": { "type": "string" },
        "embedding": {
          "type": "array",
          "items": { "type": "number" },
          "searchlite:vector": { "dim": 2, "metric": "Cosine" }
        }
      }
    }))
    .unwrap();
    client
      .post(format!("{index_base}/init"))
      .json(&schema)
      .send()
      .await
      .unwrap();

    let bulk = json!({
      "docs": [
        { "_id": "vec-1", "body": "rust search", "embedding": [1.0, 0.0] },
        { "_id": "vec-2", "body": "other doc", "embedding": [0.0, 1.0] },
        { "_id": "no-vector", "body": "no embedding here" }
      ]
    });
    client
      .post(format!("{index_base}/bulk"))
      .json(&bulk)
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let request = SearchRequest {
      query: Query::Node(QueryNode::Vector(VectorQuery {
        field: "embedding".into(),
        vector: vec![1.0, 0.0],
        k: Some(3),
        alpha: Some(0.0),
        ef_search: None,
        candidate_size: Some(3),
        boost: None,
      })),
      fields: None,
      filter: None,
      limit: 2,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
      cursor: None,
      search_after: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      track_total_hits: None,
      vector_query: None,
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
    let res = client
      .post(format!("{index_base}/search"))
      .json(&request)
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());
    let body: SearchResult = res.json().await.unwrap();
    assert!(!body.hits.is_empty());
    assert_eq!(body.hits[0].doc_id, "vec-1");
    assert!(body.hits[0].vector_score.is_some());
    assert!(body.hits.iter().all(|h| h.doc_id != "no-vector"));

    handle.abort();
    let _ = handle.await;
  }

  #[cfg(feature = "vectors")]
  #[tokio::test]
  async fn multi_search_applies_server_vector_cap() {
    init_tracing();
    let dir = tempdir().unwrap();
    let mut args = default_args(dir.path().join("idx-multi-vector-cap"));
    args.max_vector_candidates = 1;
    let registry = Arc::new(IndexRegistry::from_args(&args).unwrap());
    registry.bootstrap_all().await.unwrap();
    let state = build_app_state(registry).unwrap();
    let (addr, handle) = spawn_server(args.clone(), state.clone()).await.unwrap();
    let client = Client::new();
    let base = format!("http://{addr}");
    let index_base = format!("{base}/indexes/{INDEX_NAME}");

    let schema: Schema = serde_json::from_value(json!({
      "type": "object",
      "properties": {
        "body": { "type": "string" },
        "embedding": {
          "type": "array",
          "items": { "type": "number" },
          "searchlite:vector": { "dim": 2, "metric": "Cosine" }
        }
      }
    }))
    .unwrap();
    client
      .post(format!("{index_base}/init"))
      .json(&schema)
      .send()
      .await
      .unwrap();

    let bulk = json!({
      "docs": [
        { "_id": "vec-1", "body": "rust search", "embedding": [1.0, 0.0] },
        { "_id": "vec-2", "body": "other doc", "embedding": [0.0, 1.0] }
      ]
    });
    client
      .post(format!("{index_base}/bulk"))
      .json(&bulk)
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let vector_a = QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![1.0, 0.0],
      k: Some(2),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: None,
      boost: None,
    });
    let vector_b = QueryNode::Vector(VectorQuery {
      field: "embedding".into(),
      vector: vec![0.0, 1.0],
      k: Some(2),
      alpha: Some(0.0),
      ef_search: None,
      candidate_size: None,
      boost: None,
    });
    let request = SearchRequest {
      query: Query::Node(QueryNode::Bool {
        must: vec![],
        should: vec![vector_a, vector_b],
        must_not: vec![],
        filter: vec![],
        minimum_should_match: Some(1),
        boost: None,
      }),
      fields: None,
      filter: None,
      limit: 2,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
      cursor: None,
      search_after: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      track_total_hits: None,
      vector_query: None,
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
    let req = MultiSearchRequest {
      searches: vec![request],
      parallel: false,
      max_concurrency: None,
    };
    let res = client
      .post(format!("{index_base}/multi_search"))
      .json(&req)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::BAD_REQUEST);
    let err: ErrorResponse = res.json().await.unwrap();
    assert_eq!(err.error.r#type, "multi_search_failed");
    assert!(err.error.reason.contains("max_global_vector_candidates"));

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn http_supports_mget_and_missing_order() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-mget");
    let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    let ndjson = "{\"_id\":\"1\",\"body\":\"Rust search\"}\n{\"_id\":\"2\",\"body\":\"Another\"}\n";
    client
      .post(format!("{index_base}/add"))
      .body(ndjson.to_string())
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let req = serde_json::json!({ "ids": ["1", "missing", "2", "1"], "return_stored": true });
    let res = client
      .post(format!("{index_base}/mget"))
      .json(&req)
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());
    let body: MgetResponse = res.json().await.unwrap();
    assert_eq!(body.docs.len(), 4);
    assert!(body.docs[0].found);
    assert!(!body.docs[1].found);
    assert!(body.docs[2].found);
    assert!(body.docs[3].found);

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn http_supports_update_document() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-update");
    let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

    let schema = Schema::default_text_body();
    client
      .post(format!("{index_base}/init"))
      .json(&schema)
      .send()
      .await
      .unwrap();

    let bulk = serde_json::json!({
      "docs": [ { "_id": "doc-1", "body": "hello" } ]
    });
    client
      .post(format!("{index_base}/bulk"))
      .json(&bulk)
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let update = serde_json::json!({
      "id": "doc-1",
      "set": { "body": "updated" },
      "unset": []
    });
    let res = client
      .post(format!("{index_base}/update"))
      .json(&update)
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());
    let body: serde_json::Value = res.json().await.unwrap();
    assert_eq!(body["accepted"], true);
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let mget = serde_json::json!({ "ids": ["doc-1"], "return_stored": true });
    let res = client
      .post(format!("{index_base}/mget"))
      .json(&mget)
      .send()
      .await
      .unwrap();
    let body: serde_json::Value = res.json().await.unwrap();
    assert_eq!(body["docs"][0]["_source"]["body"], "updated");

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn http_supports_bulk_update_best_effort() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-bulk-update");
    let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

    let schema = Schema::default_text_body();
    client
      .post(format!("{index_base}/init"))
      .json(&schema)
      .send()
      .await
      .unwrap();

    let bulk = serde_json::json!({
      "docs": [ { "_id": "doc-1", "body": "hello" } ]
    });
    client
      .post(format!("{index_base}/bulk"))
      .json(&bulk)
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let ndjson = [
      r#"{"update":{"_id":"doc-1"}}"#,
      r#"{"set":{"body":"updated"}}"#,
      r#"{"update":{"_id":"missing"}}"#,
      r#"{"set":{"body":"nope"}}"#,
      "",
    ]
    .join("\n");

    let res = client
      .post(format!("{index_base}/_bulk_update"))
      .body(ndjson)
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());
    let body: serde_json::Value = res.json().await.unwrap();
    assert_eq!(body["updated"], 1);
    assert_eq!(body["failed"], 1);

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn bulk_update_abort_preserves_preexisting_pending_writes() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-bulk-update-abort-preserve");
    let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

    let schema = Schema::default_text_body();
    client
      .post(format!("{index_base}/init"))
      .json(&schema)
      .send()
      .await
      .unwrap();

    // Queue a write from another request and leave it pending (no commit yet).
    let bulk = serde_json::json!({
      "docs": [ { "_id": "seed", "body": "seed" } ]
    });
    let res = client
      .post(format!("{index_base}/bulk"))
      .json(&bulk)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::OK);

    // Force writer task startup (first full batch), then trigger abort with a trailing action line.
    let mut ndjson = String::new();
    for i in 0..NDJSON_BATCH_SIZE {
      ndjson.push_str(r#"{"update":{"_id":"seed"}}"#);
      ndjson.push('\n');
      ndjson.push_str(&format!(r#"{{"set":{{"body":"bulk-{i}"}}}}"#));
      ndjson.push('\n');
    }
    ndjson.push_str(r#"{"update":{"_id":"seed"}}"#);
    ndjson.push('\n');

    let res = client
      .post(format!("{index_base}/_bulk_update"))
      .body(ndjson)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::BAD_REQUEST);
    let err: ErrorResponse = res.json().await.unwrap();
    assert_eq!(err.error.r#type, "invalid_bulk_update");

    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let mget = serde_json::json!({ "ids": ["seed"], "return_stored": true });
    let res = client
      .post(format!("{index_base}/mget"))
      .json(&mget)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::OK);
    let body: MgetResponse = res.json().await.unwrap();
    assert_eq!(body.docs.len(), 1);
    assert!(body.docs[0].found);
    assert_eq!(body.docs[0]._source.as_ref().unwrap()["body"], "seed");

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn http_bulk_update_rejects_invalid_id() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-bulk-update-invalid-id");
    let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

    let schema = Schema::default_text_body();
    client
      .post(format!("{index_base}/init"))
      .json(&schema)
      .send()
      .await
      .unwrap();

    let ndjson = "{\"update\":{\"_id\":\"bad\\u0001\"}}\n{\"set\":{\"body\":\"updated\"}}\n";
    let res = client
      .post(format!("{index_base}/_bulk_update"))
      .body(ndjson)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: serde_json::Value = res.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_bulk_update");

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn http_supports_from_and_search_after() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-page");
    let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

    let schema: Schema = serde_json::from_value(serde_json::json!({
      "type": "object",
      "properties": {
        "body": { "type": "string" },
        "rank": { "type": "integer", "searchlite:stored": true }
      }
    }))
    .unwrap();
    client
      .post(format!("{index_base}/init"))
      .json(&schema)
      .send()
      .await
      .unwrap();

    let ndjson = "{\"_id\":\"a\",\"body\":\"rust one\",\"rank\":1}\n\
                  {\"_id\":\"b\",\"body\":\"rust two\",\"rank\":2}\n\
                  {\"_id\":\"c\",\"body\":\"rust three\",\"rank\":3}\n";
    client
      .post(format!("{index_base}/add"))
      .body(ndjson.to_string())
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let first_req = serde_json::json!({
      "query": "rust",
      "limit": 1,
      "return_stored": true,
      "sort": [{ "field": "rank", "order": "asc" }]
    });
    let first_res = client
      .post(format!("{index_base}/search"))
      .json(&first_req)
      .send()
      .await
      .unwrap();
    assert!(first_res.status().is_success());
    let first_body: SearchResult = first_res.json().await.unwrap();
    assert_eq!(first_body.hits.len(), 1);
    let first_id = &first_body.hits[0].doc_id;
    assert_eq!(first_id, "a");
    let token = first_body
      .next_search_after
      .clone()
      .expect("next_search_after present");

    let second_req = serde_json::json!({
      "query": "rust",
      "limit": 1,
      "search_after": token,
      "return_stored": true,
      "sort": [{ "field": "rank", "order": "asc" }]
    });
    let second_res = client
      .post(format!("{index_base}/search"))
      .json(&second_req)
      .send()
      .await
      .unwrap();
    let second_body: SearchResult = second_res.json().await.unwrap();
    assert_eq!(second_body.hits.len(), 1);
    assert_ne!(second_body.hits[0].doc_id, *first_id);
    assert_eq!(second_body.hits[0].doc_id, "b");

    let from_req = serde_json::json!({
      "query": "rust",
      "limit": 1,
      "from": 1,
      "return_stored": true,
      "sort": [{ "field": "rank", "order": "asc" }]
    });
    let from_res = client
      .post(format!("{index_base}/search"))
      .json(&from_req)
      .send()
      .await
      .unwrap();
    let from_body: SearchResult = from_res.json().await.unwrap();
    assert_eq!(from_body.hits.len(), 1);
    assert_eq!(from_body.hits[0].doc_id, "b");

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn http_supports_multi_search() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-multi");
    let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let ndjson = "{\"_id\":\"1\",\"body\":\"rust\"}\n{\"_id\":\"2\",\"body\":\"go\"}\n";
    client
      .post(format!("{index_base}/add"))
      .body(ndjson.to_string())
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let req = MultiSearchRequest {
      searches: vec![
        SearchRequest {
          query: Query::String("rust".into()),
          fields: None,
          filter: None,
          limit: 1,
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
          return_stored: false,
          highlight_field: None,
          highlight: None,
          collapse: None,
          aggs: BTreeMap::new(),
          suggest: BTreeMap::new(),
          rescore: None,
          explain: false,
          profile: false,
        },
        SearchRequest {
          query: Query::String("go".into()),
          fields: None,
          filter: None,
          limit: 1,
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
          return_stored: false,
          highlight_field: None,
          highlight: None,
          collapse: None,
          aggs: BTreeMap::new(),
          suggest: BTreeMap::new(),
          rescore: None,
          explain: false,
          profile: false,
        },
      ],
      parallel: true,
      max_concurrency: Some(2),
    };
    let res = client
      .post(format!("{index_base}/multi_search"))
      .json(&req)
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());
    let body: MultiSearchResponse = res.json().await.unwrap();
    assert_eq!(body.results.len(), 2);
    assert_eq!(body.results[0].hits.first().unwrap().doc_id, "1");
    assert_eq!(body.results[1].hits.first().unwrap().doc_id, "2");

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn multi_search_allows_zero_limit() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-multi-zero-limit");
    let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let ndjson = "{\"_id\":\"1\",\"body\":\"rust\"}\n{\"_id\":\"2\",\"body\":\"go\"}\n";
    client
      .post(format!("{index_base}/add"))
      .body(ndjson.to_string())
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let req = json!({
      "searches": [
        { "query": "rust", "limit": 0, "return_stored": true },
        { "query": "go", "limit": 1, "return_stored": true }
      ],
      "parallel": true,
      "max_concurrency": 2
    });
    let res = client
      .post(format!("{index_base}/multi_search"))
      .json(&req)
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());
    let body: MultiSearchResponse = res.json().await.unwrap();
    assert_eq!(body.results.len(), 2);
    assert!(body.results[0].hits.is_empty());
    assert_eq!(body.results[1].hits.first().unwrap().doc_id, "2");

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn require_existing_index_blocks_startup() {
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("missing");
    let mut args = default_args(index_path);
    args.require_existing_index = true;
    let err = IndexRegistry::from_args(&args)
      .unwrap()
      .bootstrap_all()
      .await
      .unwrap_err();
    assert!(err.to_string().contains("does not exist"));
  }

  #[tokio::test]
  async fn invalid_schema_returns_error() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (_client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-invalid")).await;
    let bad_schema: Schema = serde_json::from_value(json!({
      "type": "object",
      "properties": {},
      "searchlite:docIdField": "a.b"
    }))
    .unwrap();
    let res = _client
      .post(format!("{index_base}/init"))
      .json(&bad_schema)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::BAD_REQUEST);
    let body: ErrorResponse = res.json().await.unwrap();
    assert_eq!(body.error.r#type, "init_failed");
    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn invalid_search_request_returns_structured_error() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-invalid-search")).await;
    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let invalid = json!({
      "query": { "type": "query_string", "query": "rust" },
      "limit": 0,
      "cursor": "010203",
      "return_stored": true,
      "execution": "wand"
    });
    let res = client
      .post(format!("{index_base}/search"))
      .json(&invalid)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::BAD_REQUEST);
    let body: ErrorResponse = res.json().await.unwrap();
    assert_eq!(body.error.r#type, "invalid_cursor");
    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn invalid_multi_match_fuzziness_returns_bad_request() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-invalid-fuzziness")).await;
    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let invalid = json!({
      "query": {
        "type": "multi_match",
        "query": "rust",
        "fields": ["body"],
        "fuzziness": 3
      },
      "return_stored": true
    });
    let res = client
      .post(format!("{index_base}/search"))
      .json(&invalid)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::BAD_REQUEST);
    let body: ErrorResponse = res.json().await.unwrap();
    assert_eq!(body.error.r#type, "invalid_request");
    assert!(!body.error.reason.is_empty());
    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn search_allows_zero_limit() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-zero-limit")).await;

    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let ndjson =
      "{\"_id\":\"1\",\"body\":\"Rust search\"}\n{\"_id\":\"2\",\"body\":\"Another doc\"}\n";
    client
      .post(format!("{index_base}/add"))
      .body(ndjson.to_string())
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let req = json!({
      "query": "rust",
      "limit": 0,
      "return_stored": true
    });
    let res = client
      .post(format!("{index_base}/search"))
      .json(&req)
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());
    let body: SearchResult = res.json().await.unwrap();
    assert!(body.hits.is_empty());
    assert!(body.total_hits_estimate > 0);

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn body_limit_rejects_large_payloads() {
    init_tracing();
    let dir = tempdir().unwrap();
    let mut args = default_args(dir.path().join("idx-limit"));
    args.max_body_bytes = 512;
    let registry = Arc::new(IndexRegistry::from_args(&args).unwrap());
    registry.bootstrap_all().await.unwrap();
    let state = build_app_state(registry).unwrap();
    let (addr, handle) = spawn_server(args.clone(), state.clone()).await.unwrap();
    let client = Client::new();
    let base = format!("http://{addr}");
    let index_base = format!("{base}/indexes/{INDEX_NAME}");

    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    let long_line = format!("{{\"_id\":\"1\",\"body\":\"{}\"}}\n", "a".repeat(400));
    let body = long_line.repeat(3);
    let res = client
      .post(format!("{index_base}/add"))
      .body(body)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::PAYLOAD_TOO_LARGE);
    let err: ErrorResponse = res.json().await.unwrap();
    assert_eq!(err.error.r#type, "body_too_large");

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn init_conflict_returns_409() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-conflict")).await;
    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let res = client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::CONFLICT);
    let err: ErrorResponse = res.json().await.unwrap();
    assert_eq!(err.error.r#type, "index_exists");
    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn missing_index_requests_return_404() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-missing-req")).await;

    let res = client
      .post(format!("{index_base}/add"))
      .body("{\"_id\":\"1\"}\n")
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::NOT_FOUND);
    let err: ErrorResponse = res.json().await.unwrap();
    assert_eq!(err.error.r#type, "index_missing");

    let search_res = client
      .post(format!("{index_base}/search"))
      .json(&serde_json::json!({
        "query": "rust",
        "limit": 1,
        "return_stored": true
      }))
      .send()
      .await
      .unwrap();
    assert_eq!(search_res.status(), HttpStatus::NOT_FOUND);
    let err: ErrorResponse = search_res.json().await.unwrap();
    assert_eq!(err.error.r#type, "index_missing");

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn invalid_ndjson_returns_bad_request() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-bad-ndjson")).await;
    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let res = client
      .post(format!("{index_base}/add"))
      .body("{\"_id\":\"1\"}\nnot-json\n")
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::BAD_REQUEST);
    let err: ErrorResponse = res.json().await.unwrap();
    assert_eq!(err.error.r#type, "invalid_document");
    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn bulk_requires_docs_and_delete_requires_ids() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-empty-bulk")).await;
    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    let bulk_res = client
      .post(format!("{index_base}/bulk"))
      .json(&serde_json::json!({ "docs": [] }))
      .send()
      .await
      .unwrap();
    assert_eq!(bulk_res.status(), HttpStatus::BAD_REQUEST);
    let bulk_err: ErrorResponse = bulk_res.json().await.unwrap();
    assert_eq!(bulk_err.error.r#type, "missing_documents");

    let delete_res = client
      .post(format!("{index_base}/delete"))
      .json(&serde_json::json!({ "ids": [] }))
      .send()
      .await
      .unwrap();
    assert_eq!(delete_res.status(), HttpStatus::BAD_REQUEST);
    let delete_err: ErrorResponse = delete_res.json().await.unwrap();
    assert_eq!(delete_err.error.r#type, "missing_ids");

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn delete_rejects_control_character_ids() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-control-ids")).await;
    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let res = client
      .post(format!("{index_base}/delete"))
      .json(&serde_json::json!({ "ids": ["ok", "bad\tid"] }))
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::BAD_REQUEST);
    let err: ErrorResponse = res.json().await.unwrap();
    assert_eq!(err.error.r#type, "invalid_id");

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn delete_rejects_whitespace_only_ids() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-whitespace-ids")).await;
    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let res = client
      .post(format!("{index_base}/delete"))
      .json(&serde_json::json!({ "ids": ["  ", "ok"] }))
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::BAD_REQUEST);
    let err: ErrorResponse = res.json().await.unwrap();
    assert_eq!(err.error.r#type, "invalid_id");

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn delete_allows_whitespace_padded_ids() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-whitespace-delete")).await;
    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    let padded_id = "  padded-http  ";
    let ndjson = format!("{{\"_id\":\"{padded_id}\",\"body\":\"spaced\"}}\n");
    client
      .post(format!("{index_base}/add"))
      .body(ndjson)
      .send()
      .await
      .unwrap();
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let delete_res = client
      .post(format!("{index_base}/delete"))
      .json(&serde_json::json!({ "ids": [padded_id] }))
      .send()
      .await
      .unwrap();
    assert_eq!(delete_res.status(), HttpStatus::OK);
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    let request = SearchRequest {
      query: Query::String("spaced".into()),
      fields: None,
      filter: None,
      limit: 5,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
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
      return_stored: false,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs: Default::default(),
      suggest: Default::default(),
      rescore: None,
      explain: false,
      profile: false,
    };
    let search_res = client
      .post(format!("{index_base}/search"))
      .json(&request)
      .send()
      .await
      .unwrap();
    assert_eq!(search_res.status(), HttpStatus::OK);
    let body: SearchResult = search_res.json().await.unwrap();
    assert!(
      body.hits.is_empty(),
      "padded id document should be deletable via HTTP"
    );

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn health_endpoint_returns_ok() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, base, _index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-healthz")).await;
    let res = client.get(format!("{base}/healthz")).send().await.unwrap();
    assert_eq!(res.status(), HttpStatus::OK);
    let body: HealthResponse = res.json().await.unwrap();
    assert_eq!(body.status, "ok");

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn refresh_endpoint_returns_ok() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-refresh")).await;
    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    let res = client
      .post(format!("{index_base}/refresh"))
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::OK);
    let body: RefreshResponse = res.json().await.unwrap();
    assert!(body.refreshed);

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn ingest_ndjson_batches_correctly() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-batches")).await;

    // Init
    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    // Create 2500 documents (2 full batches + 1 partial)
    let mut ndjson = String::new();
    for i in 0..2500 {
      ndjson.push_str(&format!("{{\"_id\":\"{i}\",\"body\":\"doc {i}\"}}\n"));
    }

    let res = client
      .post(format!("{index_base}/add"))
      .body(ndjson)
      .send()
      .await
      .unwrap();

    assert_eq!(res.status(), HttpStatus::OK);
    let body: IngestResponse = res.json().await.unwrap();
    assert_eq!(body.queued, 2500);

    // Commit
    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();

    // Verify stats
    let stats_res = client
      .get(format!("{index_base}/stats"))
      .send()
      .await
      .unwrap();
    let stats: StatsResponse = stats_res.json().await.unwrap();
    assert_eq!(stats.documents, 2500);

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn ingest_ndjson_exact_batch_size() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-batches-boundary")).await;

    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    let mut ndjson = String::new();
    for i in 0..NDJSON_BATCH_SIZE {
      ndjson.push_str(&format!("{{\"_id\":\"{i}\",\"body\":\"doc {i}\"}}\n"));
    }

    let res = client
      .post(format!("{index_base}/add"))
      .body(ndjson)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::OK);
    let body: IngestResponse = res.json().await.unwrap();
    assert_eq!(body.queued as u64, NDJSON_BATCH_SIZE as u64);

    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();
    let stats: StatsResponse = client
      .get(format!("{index_base}/stats"))
      .send()
      .await
      .unwrap()
      .json()
      .await
      .unwrap();
    assert_eq!(stats.documents, NDJSON_BATCH_SIZE as u64);

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn ingest_ndjson_writer_failure_mid_batch() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-batches-writer-fail")).await;

    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    let ndjson =
      "{\"_id\":\"ok\",\"body\":\"doc\"}\n{\"body\":\"no id\"}\n{\"_id\":\"later\",\"body\":\"doc\"}\n";
    let res = client
      .post(format!("{index_base}/add"))
      .body(ndjson)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::BAD_REQUEST);
    let err: ErrorResponse = res.json().await.unwrap();
    assert_eq!(err.error.r#type, "add_failed");

    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();
    let stats: StatsResponse = client
      .get(format!("{index_base}/stats"))
      .send()
      .await
      .unwrap()
      .json()
      .await
      .unwrap();
    assert_eq!(stats.documents, 0);

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn ingest_ndjson_aborts_on_parse_error_without_writer() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-batches-parse-fail")).await;

    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    let res = client
      .post(format!("{index_base}/add"))
      .body("{\"_id\":\"ok\",\"body\":\"doc\"}\nnot-json\n")
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::BAD_REQUEST);
    let err: ErrorResponse = res.json().await.unwrap();
    assert_eq!(err.error.r#type, "invalid_document");

    client
      .post(format!("{index_base}/commit"))
      .send()
      .await
      .unwrap();
    let stats: StatsResponse = client
      .get(format!("{index_base}/stats"))
      .send()
      .await
      .unwrap()
      .json()
      .await
      .unwrap();
    assert_eq!(stats.documents, 0);

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn unknown_index_returns_404() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (_client, base, _index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-unknown")).await;
    let res = _client
      .get(format!("{base}/indexes/not-there/stats"))
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::NOT_FOUND);
    let err: ErrorResponse = res.json().await.unwrap();
    assert_eq!(err.error.r#type, "unknown_index");
    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn alias_resolves_to_target() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-alias");
    let mut args = default_args(index_path.clone());
    args.aliases = vec![AliasSpec {
      alias: "alias".into(),
      target: INDEX_NAME.into(),
    }];
    let registry = Arc::new(IndexRegistry::from_args(&args).unwrap());
    registry.bootstrap_all().await.unwrap();
    let state = build_app_state(registry).unwrap();
    let (addr, handle) = spawn_server(args.clone(), state.clone()).await.unwrap();
    let client = Client::new();
    let base = format!("http://{addr}");
    let index_base = format!("{base}/indexes/alias");

    client
      .post(format!("{index_base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    let stats = client
      .get(format!("{index_base}/stats"))
      .send()
      .await
      .unwrap();
    assert_eq!(stats.status(), HttpStatus::OK);

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn alias_cycle_returns_error() {
    let dir = tempdir().unwrap();
    let base = dir.path().to_path_buf();
    let mut args = ServeArgs {
      indexes: vec![
        IndexSpec {
          name: "idx1".into(),
          path: base.join("a"),
          auto_commit_interval_secs: None,
          auto_refresh_interval_secs: None,
        },
        IndexSpec {
          name: "idx2".into(),
          path: base.join("b"),
          auto_commit_interval_secs: None,
          auto_refresh_interval_secs: None,
        },
        IndexSpec {
          name: "idx3".into(),
          path: base.join("c"),
          auto_commit_interval_secs: None,
          auto_refresh_interval_secs: None,
        },
      ],
      aliases: vec![],
      ..default_args(base.join("primary"))
    };
    args.aliases = vec![
      AliasSpec {
        alias: "a".into(),
        target: "b".into(),
      },
      AliasSpec {
        alias: "b".into(),
        target: "c".into(),
      },
      AliasSpec {
        alias: "c".into(),
        target: "a".into(),
      },
    ];
    match IndexRegistry::from_args(&args) {
      Ok(_) => panic!("expected alias cycle detection"),
      Err(err) => assert!(err.to_string().contains("alias cycle detected")),
    }
  }

  #[test]
  fn parse_index_spec_accepts_runtime_overrides() {
    let spec = parse_index_spec("items:/data/items,auto_commit=30,auto_refresh=10").unwrap();
    assert_eq!(spec.name, "items");
    assert_eq!(spec.path, PathBuf::from("/data/items"));
    assert_eq!(spec.auto_commit_interval_secs, Some(30));
    assert_eq!(spec.auto_refresh_interval_secs, Some(10));
  }

  #[test]
  fn parse_index_spec_rejects_unknown_runtime_override_key() {
    let err = parse_index_spec("items:/data/items,unknown=30").unwrap_err();
    assert!(err.contains("unsupported index option"));
  }

  #[test]
  fn parse_index_spec_rejects_invalid_runtime_override_value() {
    let err = parse_index_spec("items:/data/items,auto_commit=abc").unwrap_err();
    assert!(err.contains("must be a non-negative integer"));
  }

  #[test]
  fn parse_index_spec_rejects_empty_runtime_override_key() {
    let err = parse_index_spec("items:/data/items,=10").unwrap_err();
    assert!(err.contains("index option key cannot be empty"));
  }

  #[test]
  fn duplicate_index_name_rejected() {
    let mut args = default_args(PathBuf::from("/tmp/idx-one"));
    args.indexes.push(IndexSpec {
      name: INDEX_NAME.into(),
      path: PathBuf::from("/tmp/idx-two"),
      auto_commit_interval_secs: None,
      auto_refresh_interval_secs: None,
    });
    match IndexRegistry::from_args(&args) {
      Ok(_) => panic!("expected duplicate index name error"),
      Err(err) => assert!(err.to_string().contains("duplicate index name provided")),
    }
  }

  // Regression tests for BUG-016: error responses must not leak internal
  // deserialization, middleware, or task-panic detail to clients.

  #[tokio::test]
  async fn malformed_json_body_returns_generic_reason() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, _base, index_base, handle, _state, _args) =
      setup_server(dir.path().join("idx-malformed-json")).await;
    // `text_fields` is an array of strings on `Schema`. Supplying an integer
    // triggers a `JsonRejection` whose `Display` includes the field path and
    // expected type — the exact detail the bug says must not be echoed.
    let invalid = json!({ "text_fields": 1 });
    let res = client
      .post(format!("{index_base}/init"))
      .json(&invalid)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::BAD_REQUEST);
    let body: ErrorResponse = res.json().await.unwrap();
    assert_eq!(body.error.r#type, "invalid_request");
    assert_eq!(body.error.reason, MALFORMED_REQUEST_BODY_MESSAGE);
    // The leaked serde detail must not appear in the client-visible body.
    assert!(
      !body.error.reason.contains("text_fields"),
      "reason leaked serde field path: {}",
      body.error.reason
    );
    assert!(
      !body.error.reason.contains("expected"),
      "reason leaked serde expected-type detail: {}",
      body.error.reason
    );
    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn handle_middleware_error_returns_generic_reason() {
    init_tracing();
    // A synthetic `BoxError` whose Debug form would leak a panic-like message.
    #[derive(Debug)]
    struct NoisyError;
    impl std::fmt::Display for NoisyError {
      fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
          f,
          "task panicked at src/secret.rs:42: internal state violated"
        )
      }
    }
    impl std::error::Error for NoisyError {}

    let err: BoxError = Box::new(NoisyError);
    let response = handle_middleware_error(err).await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
      .await
      .unwrap();
    let body: ErrorResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body.error.r#type, "middleware_error");
    assert_eq!(body.error.reason, "internal server error");
    // Neither the noisy Display nor its Debug form should be reflected.
    assert!(!body.error.reason.contains("src/secret.rs"));
    assert!(!body.error.reason.contains("panicked"));
    assert!(!body.error.reason.contains("NoisyError"));
  }

  #[tokio::test]
  async fn await_writer_or_default_masks_task_panic() {
    init_tracing();
    // Spawn a task that panics with a message that would otherwise leak into
    // the response via `JoinError::to_string()`.
    let handle: tokio::task::JoinHandle<Result<usize, HttpError>> = tokio::spawn(async {
      panic!("index out of bounds: the len is 0 but the index is 4");
    });
    let mut slot = Some(handle);
    let default = HttpError::bad_request("should_not_be_used", "default");
    let err = await_writer_or_default(&mut slot, default).await;
    assert_eq!(err.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(err.kind, "add_join");
    assert_eq!(err.reason, "internal server error");
    assert!(!err.reason.contains("index out of bounds"));
    assert!(!err.reason.contains("panicked"));
  }
}
