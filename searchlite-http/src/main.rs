use std::collections::BTreeMap;
use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::body::Body;
use axum::error_handling::HandleErrorLayer;
use axum::extract::rejection::JsonRejection;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use futures_util::StreamExt;
use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{Document, IndexOptions, SearchRequest, StorageType};
use searchlite_core::api::SearchResult;
use searchlite_core::{Index, Manifest, Schema};
use thiserror::Error;
use tokio::io::AsyncBufReadExt;
use tokio::net::TcpListener;
use tokio_util::io::StreamReader;
use tower::limit::ConcurrencyLimitLayer;
use tower::timeout::TimeoutLayer;
use tower::{BoxError, ServiceBuilder};
use tower_http::limit::RequestBodyLimitLayer;
use tracing::{error, info};
use tracing_subscriber::{fmt, EnvFilter};

const DEFAULT_K1: f32 = 0.9;
const DEFAULT_B: f32 = 0.4;

#[derive(Parser, Debug, Clone)]
#[command(
  name = "searchlite-http",
  version,
  about = "HTTP API for a single searchlite index"
)]
struct ServeArgs {
  /// Path to the index directory on disk.
  #[arg(long, env = "SEARCHLITE_INDEX_PATH")]
  index: PathBuf,

  /// Bind address for the HTTP server.
  /// WARNING: Binding to 0.0.0.0 or any non-localhost address exposes this
  /// unauthenticated service to the network; front it with a proxy or firewall.
  #[arg(long, env = "SEARCHLITE_BIND_ADDR", default_value = "127.0.0.1:8080")]
  bind: SocketAddr,

  /// Require the index to already exist on disk at startup.
  #[arg(
    long,
    env = "SEARCHLITE_REQUIRE_EXISTING_INDEX",
    default_value_t = false
  )]
  require_existing_index: bool,

  /// Maximum allowed request body size in bytes.
  #[arg(long, env = "SEARCHLITE_MAX_BODY_BYTES", default_value_t = 50 * 1024 * 1024)]
  max_body_bytes: u64,

  /// Maximum number of in-flight requests.
  #[arg(long, env = "SEARCHLITE_MAX_CONCURRENCY", default_value_t = 64)]
  max_concurrency: usize,

  /// Per-request timeout in seconds.
  #[arg(long, env = "SEARCHLITE_REQUEST_TIMEOUT_SECS", default_value_t = 30)]
  request_timeout_secs: u64,

  /// Grace period in seconds before forcing shutdown after a signal.
  #[arg(long, env = "SEARCHLITE_GRACEFUL_SHUTDOWN_SECS", default_value_t = 5)]
  shutdown_grace_secs: u64,

  /// If set, commit also triggers a reader refresh to surface changes immediately.
  #[arg(long, env = "SEARCHLITE_REFRESH_ON_COMMIT", default_value_t = false)]
  refresh_on_commit: bool,
}

#[derive(Clone)]
struct AppState {
  index_path: PathBuf,
  require_existing_index: bool,
  refresh_on_commit: bool,
  index: Arc<tokio::sync::RwLock<Option<Arc<Index>>>>,
  // Serialize writer access across handlers to avoid concurrent writers.
  writer_lock: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Debug, Error)]
#[error("{reason}")]
struct HttpError {
  status: StatusCode,
  kind: &'static str,
  reason: String,
}

type ApiResult<T> = Result<T, HttpError>;

fn parse_json<T>(payload: Result<Json<T>, JsonRejection>) -> ApiResult<T> {
  payload
    .map(|Json(inner)| inner)
    .map_err(|err| HttpError::bad_request("invalid_request", err.to_string()))
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

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StatsResponse {
  documents: u64,
  deleted_documents: u64,
  segments: usize,
  committed_at: String,
  index_uuid: String,
  index_path: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct HealthResponse {
  status: String,
}

#[derive(Debug, serde::Deserialize)]
struct BulkRequest {
  docs: Vec<serde_json::Value>,
}

#[derive(Debug, serde::Deserialize)]
struct DeleteRequest {
  ids: Vec<String>,
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

impl AppState {
  fn new(args: &ServeArgs) -> Self {
    Self {
      index_path: args.index.clone(),
      require_existing_index: args.require_existing_index,
      refresh_on_commit: args.refresh_on_commit,
      index: Arc::new(tokio::sync::RwLock::new(None)),
      writer_lock: Arc::new(tokio::sync::Mutex::new(())),
    }
  }

  async fn bootstrap(&self) -> anyhow::Result<()> {
    if !self.manifest_exists() {
      if self.require_existing_index {
        anyhow::bail!(
          "index does not exist at {:?}",
          Manifest::manifest_path(&self.index_path)
        );
      }
      return Ok(());
    }
    let idx = Index::open(self.index_options(false))
      .with_context(|| "failed to open existing index during startup".to_string())?;
    let arc = Arc::new(idx);
    let mut guard = self.index.write().await;
    *guard = Some(arc);
    Ok(())
  }

  fn manifest_exists(&self) -> bool {
    Manifest::manifest_path(&self.index_path).exists()
  }

  fn index_options(&self, create_if_missing: bool) -> IndexOptions {
    IndexOptions {
      path: self.index_path.clone(),
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
        "index is not initialized; call /init first",
      ));
    }
    let idx = Index::open(self.index_options(false))
      .map_err(|e| HttpError::from_anyhow("open_index", StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(self.set_index(idx).await)
  }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  init_tracing();
  let args = ServeArgs::parse();
  if let Err(err) = run(args.clone()).await {
    error!("{err:?}");
    std::process::exit(1);
  }
  Ok(())
}

async fn run(args: ServeArgs) -> anyhow::Result<()> {
  let state = Arc::new(AppState::new(&args));
  state.bootstrap().await?;
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
    .route("/init", post(init_index))
    .route("/add", post(add_ndjson))
    .route("/bulk", post(bulk_ingest))
    .route("/delete", post(delete_documents))
    .route("/commit", post(commit))
    .route("/refresh", post(refresh))
    .route("/compact", post(compact))
    .route("/search", post(search))
    .route("/inspect", get(inspect))
    .route("/stats", get(stats))
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
        "request body exceeded configured limit of {} bytes",
        max_body
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
  error!(error = ?err, "middleware error");
  HttpError::from_anyhow(
    "middleware_error",
    StatusCode::INTERNAL_SERVER_ERROR,
    anyhow::anyhow!(format!("{err:?}")),
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

async fn init_index(
  State(state): State<Arc<AppState>>,
  payload: Result<Json<Schema>, JsonRejection>,
) -> ApiResult<Json<InitResponse>> {
  let schema = parse_json(payload)?;
  if state.manifest_exists() {
    return Err(HttpError::conflict(
      "index_exists",
      "index already exists at this path",
    ));
  }
  let path = state.index_path.clone();
  let opts = state.index_options(true);
  let created = tokio::task::spawn_blocking(move || IndexBuilder::create(&path, schema, opts))
    .await
    .map_err(|err| {
      HttpError::from_anyhow(
        "init_join",
        StatusCode::INTERNAL_SERVER_ERROR,
        anyhow::anyhow!(err.to_string()),
      )
    })?
    .map_err(|err| HttpError::from_anyhow("init_failed", StatusCode::BAD_REQUEST, err))?;
  // IndexBuilder::create must either return a fully initialized, ready-to-use index
  // or fail with an error. At this point the index is safe for subsequent writer/reader
  // creation, otherwise the call above would have erred.
  state.set_index(created).await;
  Ok(Json(InitResponse { created: true }))
}

async fn add_ndjson(
  State(state): State<Arc<AppState>>,
  body: Body,
) -> ApiResult<Json<IngestResponse>> {
  let index = state.require_index().await?;
  let mapped_stream = body
    .into_data_stream()
    .map(|chunk| chunk.map_err(io::Error::other));
  let mut reader = StreamReader::new(mapped_stream);
  let mut buf = String::new();
  let mut docs = Vec::new();
  let mut line_number = 0usize;
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
        format!("invalid JSON document on NDJSON line {}: {e}", line_number),
      )
    })?;
    let doc = value_to_document(value)?;
    docs.push(doc);
  }
  if docs.is_empty() {
    return Ok(Json(IngestResponse { queued: 0 }));
  }
  let _writer_guard = state.writer_lock.lock().await;
  let mut writer = index
    .writer()
    .map_err(|e| HttpError::from_anyhow("writer_open", StatusCode::INTERNAL_SERVER_ERROR, e))?;
  for doc in docs.iter() {
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
  Ok(Json(IngestResponse { queued: docs.len() }))
}

async fn bulk_ingest(
  State(state): State<Arc<AppState>>,
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
  let index = state.require_index().await?;
  let _writer_guard = state.writer_lock.lock().await;
  let mut writer = index
    .writer()
    .map_err(|e| HttpError::from_anyhow("writer_open", StatusCode::INTERNAL_SERVER_ERROR, e))?;
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
}

async fn delete_documents(
  State(state): State<Arc<AppState>>,
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
  let index = state.require_index().await?;
  let _writer_guard = state.writer_lock.lock().await;
  let mut writer = index
    .writer()
    .map_err(|e| HttpError::from_anyhow("writer_open", StatusCode::INTERNAL_SERVER_ERROR, e))?;
  writer
    .delete_documents(&body.ids)
    .map_err(|e| HttpError::bad_request("delete_failed", e.to_string()))?;
  Ok(Json(DeleteResponse {
    queued: body.ids.len(),
  }))
}

fn trigger_reader_refresh(index: &Index) -> anyhow::Result<()> {
  // Opening a reader reloads searchers; the returned reader can be dropped
  // immediately when only a refresh side effect is desired.
  index.reader().map(|_| ())
}

async fn commit(State(state): State<Arc<AppState>>) -> ApiResult<Json<CommitResponse>> {
  let index = state.require_index().await?;
  let refresh = state.refresh_on_commit;
  let writer_lock = state.writer_lock.clone();
  tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
    let _guard = writer_lock.blocking_lock();
    let mut writer = index.writer()?;
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
  .map_err(|err| HttpError::from_anyhow("commit_failed", StatusCode::INTERNAL_SERVER_ERROR, err))?;
  Ok(Json(CommitResponse { committed: true }))
}

async fn refresh(State(state): State<Arc<AppState>>) -> ApiResult<Json<RefreshResponse>> {
  let index = state.require_index().await?;
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

async fn compact(State(state): State<Arc<AppState>>) -> ApiResult<Json<CompactResponse>> {
  let index = state.require_index().await?;
  let writer_lock = state.writer_lock.clone();
  tokio::task::spawn_blocking(move || {
    let _guard = writer_lock.blocking_lock();
    index.compact()
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
    HttpError::from_anyhow("compact_failed", StatusCode::INTERNAL_SERVER_ERROR, err)
  })?;
  Ok(Json(CompactResponse { compacted: true }))
}

async fn search(
  State(state): State<Arc<AppState>>,
  payload: Result<Json<SearchRequest>, JsonRejection>,
) -> ApiResult<Json<SearchResult>> {
  let request = parse_json(payload)?;
  let index = state.require_index().await?;
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

async fn inspect(State(state): State<Arc<AppState>>) -> ApiResult<Json<InspectResponse>> {
  let index = state.require_index().await?;
  let manifest = tokio::task::spawn_blocking(move || Ok::<_, anyhow::Error>(index.manifest()))
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
  Ok(Json(InspectResponse { manifest }))
}

async fn stats(State(state): State<Arc<AppState>>) -> ApiResult<Json<StatsResponse>> {
  let index = state.require_index().await?;
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
  let mut total_docs = 0u64;
  let mut deleted_docs = 0u64;
  for seg in manifest.segments.iter() {
    total_docs = total_docs.saturating_add(seg.doc_count as u64);
    deleted_docs = deleted_docs.saturating_add(seg.deleted_docs.len() as u64);
  }
  Ok(Json(StatsResponse {
    documents: total_docs.saturating_sub(deleted_docs),
    deleted_documents: deleted_docs,
    segments: manifest.segments.len(),
    committed_at: manifest.committed_at.clone(),
    index_uuid: manifest.uuid.to_string(),
    index_path: state.index_path.display().to_string(),
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
  // Avoid control characters to prevent invisible ids or log injection.
  for (idx, id) in ids.iter().enumerate() {
    let trimmed = id.trim();
    if trimmed.is_empty() {
      return Err(HttpError::bad_request(
        "invalid_id",
        format!("id at position {} is empty", idx),
      ));
    }
    if trimmed.len() != id.len() {
      return Err(HttpError::bad_request(
        "invalid_id",
        format!("id at position {} has leading or trailing whitespace", idx),
      ));
    }
    if id.chars().any(|c| c.is_control()) {
      return Err(HttpError::bad_request(
        "invalid_id",
        format!("id at position {} contains control characters", idx),
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

fn init_tracing() {
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
    HighlightRequest, Query, QueryNode, SuggestRequest,
  };
  use serde_json::json;
  use std::collections::BTreeMap;
  use std::path::PathBuf;
  use tempfile::tempdir;
  use tokio::task::JoinHandle;

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
    JoinHandle<anyhow::Result<()>>,
    Arc<AppState>,
    ServeArgs,
  ) {
    let args = default_args(index);
    let state = Arc::new(AppState::new(&args));
    state.bootstrap().await.unwrap();
    let (addr, handle) = spawn_server(args.clone(), state.clone()).await.unwrap();
    let client = Client::new();
    let base = format!("http://{}", addr);
    (client, base, handle, state, args)
  }

  fn default_args(index: PathBuf) -> ServeArgs {
    ServeArgs {
      index,
      bind: "127.0.0.1:0".parse().unwrap(),
      require_existing_index: false,
      max_body_bytes: 10 * 1024 * 1024,
      max_concurrency: 8,
      request_timeout_secs: 10,
      shutdown_grace_secs: 0,
      refresh_on_commit: false,
    }
  }

  #[tokio::test]
  async fn http_flow_covers_search_lifecycle() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx");
    let (client, base, handle, _state, _args) = setup_server(index_path.clone()).await;

    // init
    let schema = Schema::default_text_body();
    let res = client
      .post(format!("{base}/init"))
      .json(&schema)
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());

    // add docs
    let ndjson =
      "{\"_id\":\"1\",\"body\":\"Rust search\"}\n{\"_id\":\"2\",\"body\":\"Another doc\"}\n";
    let res = client
      .post(format!("{base}/add"))
      .body(ndjson.to_string())
      .send()
      .await
      .unwrap();
    assert!(res.status().is_success());

    // commit
    let res = client.post(format!("{base}/commit")).send().await.unwrap();
    assert!(res.status().is_success());

    // search
    let req = SearchRequest {
      query: Query::String("rust".into()),
      fields: None,
      filter: None,
      filters: Vec::new(),
      limit: 5,
      candidate_size: None,
      sort: Vec::new(),
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
      aggs: Default::default(),
      suggest: Default::default(),
      rescore: None,
      explain: false,
      profile: false,
    };
    let res = client
      .post(format!("{base}/search"))
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
      .get(format!("{base}/inspect"))
      .send()
      .await
      .unwrap()
      .json::<InspectResponse>()
      .await
      .unwrap();
    assert_eq!(res.manifest.segments.len(), 1);

    // stats
    let stats = client
      .get(format!("{base}/stats"))
      .send()
      .await
      .unwrap()
      .json::<StatsResponse>()
      .await
      .unwrap();
    assert_eq!(stats.documents, 2);

    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn http_supports_aggs_suggest_and_highlight() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-aggs");
    let (client, base, handle, _state, _args) = setup_server(index_path).await;

    let schema: Schema = serde_json::from_value(json!({
      "doc_id_field": "_id",
      "text_fields": [
        { "name": "body", "analyzer": "default", "stored": true, "indexed": true, "nullable": false }
      ],
      "keyword_fields": [
        { "name": "lang", "stored": true, "indexed": true, "fast": true, "nullable": false }
      ],
      "numeric_fields": [],
      "nested_fields": [],
      "vector_fields": []
    }))
    .unwrap();
    client
      .post(format!("{base}/init"))
      .json(&schema)
      .send()
      .await
      .unwrap();

    let ndjson = "{\"_id\":\"1\",\"body\":\"Rust search\",\"lang\":\"en\"}\n\
                  {\"_id\":\"2\",\"body\":\"Rustaceans write Rust\",\"lang\":\"en\"}\n\
                  {\"_id\":\"3\",\"body\":\"Recherche en français\",\"lang\":\"fr\"}\n";
    client
      .post(format!("{base}/add"))
      .body(ndjson.to_string())
      .send()
      .await
      .unwrap();
    client.post(format!("{base}/commit")).send().await.unwrap();

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
      filters: Vec::new(),
      limit: 5,
      candidate_size: None,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
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
      .post(format!("{base}/search"))
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

    let compact = client.post(format!("{base}/compact")).send().await.unwrap();
    assert!(compact.status().is_success());

    handle.abort();
    let _ = handle.await;
  }

  #[cfg(feature = "vectors")]
  #[tokio::test]
  async fn http_supports_vector_search() {
    init_tracing();
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("idx-vector");
    let (client, base, handle, _state, _args) = setup_server(index_path).await;

    let schema: Schema = serde_json::from_value(json!({
      "doc_id_field": "_id",
      "text_fields": [
        { "name": "body", "analyzer": "default", "stored": true, "indexed": true, "nullable": false }
      ],
      "keyword_fields": [],
      "numeric_fields": [],
      "nested_fields": [],
      "vector_fields": [
        { "name": "embedding", "dim": 2, "metric": "Cosine" }
      ]
    }))
    .unwrap();
    client
      .post(format!("{base}/init"))
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
      .post(format!("{base}/bulk"))
      .json(&bulk)
      .send()
      .await
      .unwrap();
    client.post(format!("{base}/commit")).send().await.unwrap();

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
      filters: Vec::new(),
      limit: 2,
      candidate_size: None,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
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
      .post(format!("{base}/search"))
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

  #[tokio::test]
  async fn require_existing_index_blocks_startup() {
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("missing");
    let args = ServeArgs {
      require_existing_index: true,
      ..default_args(index_path)
    };
    let state = Arc::new(AppState::new(&args));
    let err = state.bootstrap().await.unwrap_err();
    assert!(err.to_string().contains("does not exist"));
  }

  #[tokio::test]
  async fn invalid_schema_returns_error() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, base, handle, _state, _args) = setup_server(dir.path().join("idx-invalid")).await;
    let bad_schema: Schema = serde_json::from_value(json!({
      "doc_id_field": "a.b",
      "text_fields": [],
      "keyword_fields": [],
      "numeric_fields": [],
      "nested_fields": [],
      "vector_fields": []
    }))
    .unwrap();
    let res = client
      .post(format!("{base}/init"))
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
    let (client, base, handle, _state, _args) =
      setup_server(dir.path().join("idx-invalid-search")).await;
    client
      .post(format!("{base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let invalid = json!({
      "query": { "type": "query_string", "query": "rust" },
      "filter": { "KeywordEq": { "field": "lang", "value": "en" }},
      "filters": [{ "KeywordEq": { "field": "lang", "value": "en" }}],
      "limit": 1,
      "return_stored": true,
      "execution": "wand"
    });
    let res = client
      .post(format!("{base}/search"))
      .json(&invalid)
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::BAD_REQUEST);
    let body: ErrorResponse = res.json().await.unwrap();
    assert_eq!(body.error.r#type, "invalid_request");
    handle.abort();
    let _ = handle.await;
  }

  #[tokio::test]
  async fn body_limit_rejects_large_payloads() {
    init_tracing();
    let dir = tempdir().unwrap();
    let mut args = default_args(dir.path().join("idx-limit"));
    args.max_body_bytes = 512;
    let state = Arc::new(AppState::new(&args));
    state.bootstrap().await.unwrap();
    let (addr, handle) = spawn_server(args.clone(), state.clone()).await.unwrap();
    let client = Client::new();
    let base = format!("http://{}", addr);

    client
      .post(format!("{base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    let long_line = format!("{{\"_id\":\"1\",\"body\":\"{}\"}}\n", "a".repeat(400));
    let body = long_line.repeat(3);
    let res = client
      .post(format!("{base}/add"))
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
    let (client, base, handle, _state, _args) = setup_server(dir.path().join("idx-conflict")).await;
    client
      .post(format!("{base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let res = client
      .post(format!("{base}/init"))
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
    let (client, base, handle, _state, _args) =
      setup_server(dir.path().join("idx-missing-req")).await;

    let res = client
      .post(format!("{base}/add"))
      .body("{\"_id\":\"1\"}\n")
      .send()
      .await
      .unwrap();
    assert_eq!(res.status(), HttpStatus::NOT_FOUND);
    let err: ErrorResponse = res.json().await.unwrap();
    assert_eq!(err.error.r#type, "index_missing");

    let search_res = client
      .post(format!("{base}/search"))
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
    let (client, base, handle, _state, _args) =
      setup_server(dir.path().join("idx-bad-ndjson")).await;
    client
      .post(format!("{base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let res = client
      .post(format!("{base}/add"))
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
    let (client, base, handle, _state, _args) =
      setup_server(dir.path().join("idx-empty-bulk")).await;
    client
      .post(format!("{base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    let bulk_res = client
      .post(format!("{base}/bulk"))
      .json(&serde_json::json!({ "docs": [] }))
      .send()
      .await
      .unwrap();
    assert_eq!(bulk_res.status(), HttpStatus::BAD_REQUEST);
    let bulk_err: ErrorResponse = bulk_res.json().await.unwrap();
    assert_eq!(bulk_err.error.r#type, "missing_documents");

    let delete_res = client
      .post(format!("{base}/delete"))
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
    let (client, base, handle, _state, _args) =
      setup_server(dir.path().join("idx-control-ids")).await;
    client
      .post(format!("{base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let res = client
      .post(format!("{base}/delete"))
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
    let (client, base, handle, _state, _args) =
      setup_server(dir.path().join("idx-whitespace-ids")).await;
    client
      .post(format!("{base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();
    let res = client
      .post(format!("{base}/delete"))
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
  async fn health_endpoint_returns_ok() {
    init_tracing();
    let dir = tempdir().unwrap();
    let (client, base, handle, _state, _args) = setup_server(dir.path().join("idx-healthz")).await;
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
    let (client, base, handle, _state, _args) = setup_server(dir.path().join("idx-refresh")).await;
    client
      .post(format!("{base}/init"))
      .json(&Schema::default_text_body())
      .send()
      .await
      .unwrap();

    let res = client.post(format!("{base}/refresh")).send().await.unwrap();
    assert_eq!(res.status(), HttpStatus::OK);
    let body: RefreshResponse = res.json().await.unwrap();
    assert!(body.refreshed);

    handle.abort();
    let _ = handle.await;
  }
}
