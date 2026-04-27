use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::error_handling::HandleErrorLayer;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::Router;
use clap::{Parser, ValueEnum};
use tokio::net::TcpListener;
use tower::limit::ConcurrencyLimitLayer;
use tower::timeout::TimeoutLayer;
use tower::{BoxError, ServiceBuilder};
use tower_http::limit::RequestBodyLimitLayer;
use tracing::{error, info};
use tracing_subscriber::{fmt, EnvFilter};

pub mod client;
pub mod compat;
pub mod error;
pub mod routes;
pub mod state;
pub mod translate;

pub use client::SearchliteClient;
pub use error::ESError;
pub use state::AppState;

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedPolicy {
  /// Reject the request with HTTP 400 when an unsupported feature is encountered.
  Reject,
  /// Log a warning and translate as best as possible (may produce empty/partial results).
  Warn,
}

#[derive(Parser, Debug, Clone)]
#[command(
  name = "searchlite-elastic",
  version,
  about = "Elasticsearch-compatible HTTP adapter for searchlite-http"
)]
pub struct AdapterArgs {
  /// Bind address for the Elasticsearch-compatible HTTP server.
  /// WARNING: Binding to 0.0.0.0 or any non-localhost address exposes this
  /// unauthenticated service to the network; front it with a proxy or firewall.
  #[arg(long, env = "SEARCHLITE_ELASTIC_BIND_ADDR", default_value = "127.0.0.1:9200")]
  pub bind: SocketAddr,

  /// Base URL of the upstream searchlite-http instance.
  #[arg(
    long,
    env = "SEARCHLITE_ELASTIC_UPSTREAM_URL",
    default_value = "http://127.0.0.1:8080"
  )]
  pub upstream_url: String,

  /// Optional write key forwarded to upstream as `x-searchlite-write-key`.
  #[arg(long, env = "SEARCHLITE_ELASTIC_WRITE_KEY")]
  pub write_key: Option<String>,

  /// Version string returned in the cluster banner (mimic Elasticsearch major version).
  #[arg(
    long,
    env = "SEARCHLITE_ELASTIC_VERSION_BANNER",
    default_value = "8.11.0"
  )]
  pub version_banner: String,

  /// Policy when a request uses an Elasticsearch feature SearchLite cannot model.
  #[arg(
    long,
    value_enum,
    env = "SEARCHLITE_ELASTIC_ON_UNSUPPORTED",
    default_value = "reject"
  )]
  pub on_unsupported: UnsupportedPolicy,

  /// Per-request timeout in seconds for upstream calls.
  #[arg(long, env = "SEARCHLITE_ELASTIC_REQUEST_TIMEOUT_SECS", default_value_t = 30)]
  pub request_timeout_secs: u64,

  /// Maximum allowed request body size in bytes.
  #[arg(long, env = "SEARCHLITE_ELASTIC_MAX_BODY_BYTES", default_value_t = 100 * 1024 * 1024)]
  pub max_body_bytes: u64,

  /// Maximum number of in-flight requests.
  #[arg(long, env = "SEARCHLITE_ELASTIC_MAX_CONCURRENCY", default_value_t = 64)]
  pub max_concurrency: usize,

  /// Grace period in seconds before forcing shutdown after a signal.
  #[arg(
    long,
    env = "SEARCHLITE_ELASTIC_GRACEFUL_SHUTDOWN_SECS",
    default_value_t = 5
  )]
  pub shutdown_grace_secs: u64,
}

pub async fn run(args: AdapterArgs) -> anyhow::Result<()> {
  let client = Arc::new(SearchliteClient::new(&args)?);
  let state = Arc::new(AppState::new(client, args.clone()));

  let listener = TcpListener::bind(args.bind)
    .await
    .with_context(|| format!("binding to {}", args.bind))?;
  let local_addr = listener
    .local_addr()
    .context("reading local listening address")?;
  info!(
    address = ?local_addr,
    upstream = %args.upstream_url,
    "searchlite elasticsearch adapter listening"
  );

  let app = router(state, &args);
  axum::serve(listener, app)
    .with_graceful_shutdown(shutdown_signal(args.shutdown_grace_secs))
    .await
    .context("running HTTP server")
}

pub fn router(state: Arc<AppState>, args: &AdapterArgs) -> Router {
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

  routes::router(state)
    .layer(middleware)
    .layer(middleware::from_fn(move |req, next| {
      map_413(max_body, req, next)
    }))
}

async fn map_413(max_body: usize, req: Request, next: Next) -> Response {
  let mut res = next.run(req).await;
  if res.status() == StatusCode::PAYLOAD_TOO_LARGE {
    res = ESError::bad_request(
      "request_entity_too_large_exception",
      format!("request body exceeded configured limit of {max_body} bytes"),
    )
    .with_status(StatusCode::PAYLOAD_TOO_LARGE)
    .into_response();
  }
  res
}

async fn handle_middleware_error(err: BoxError) -> Response {
  if err.is::<tower::timeout::error::Elapsed>() {
    return ESError::bad_request("timeout_exception", "request timed out")
      .with_status(StatusCode::GATEWAY_TIMEOUT)
      .into_response();
  }
  ESError::internal("internal_server_error", err.to_string()).into_response()
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
