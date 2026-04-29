use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use axum::http::StatusCode;
use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use reqwest::Url;
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;
use tokio::sync::Mutex;

use crate::error::ESError;
use crate::AdapterArgs;

const WRITE_KEY_HEADER: &str = "x-searchlite-write-key";

/// TTL for the cached `list_indexes` response. Aliases and index membership
/// rarely change at high frequency, but every alias-resolving request would
/// otherwise hit upstream — Kibana / SDKs probe these endpoints repeatedly.
/// 5 seconds is short enough that a deleted index still surfaces quickly.
const LIST_INDEXES_TTL: Duration = Duration::from_secs(5);

/// Characters that must be percent-encoded inside a URL path segment.
/// `/`, `?`, `#` are structural; `%` needs encoding to avoid double-decode;
/// space and `<`, `>`, `"`, `\\`, `^`, `\`` are unsafe in path syntax. Also
/// percent-encode all controls. Most legitimate index name characters
/// (alphanumerics, `-`, `_`, `.`, `:`) pass through unchanged.
const PATH_SEGMENT: &AsciiSet = &CONTROLS
  .add(b' ')
  .add(b'/')
  .add(b'?')
  .add(b'#')
  .add(b'%')
  .add(b'<')
  .add(b'>')
  .add(b'"')
  .add(b'\\')
  .add(b'^')
  .add(b'`')
  .add(b'{')
  .add(b'}')
  .add(b'|');

/// Build a relative path under `indexes/<index>/<suffix>` with the index name
/// percent-encoded as a single path segment. Without this, an index name
/// containing `/`, `?`, `#`, or whitespace would be parsed structurally by
/// `Url::join` and route to the wrong endpoint upstream.
pub(crate) fn upstream_path(index: &str, suffix: &str) -> String {
  let encoded = utf8_percent_encode(index, PATH_SEGMENT);
  format!("indexes/{encoded}/{suffix}")
}

pub struct SearchliteClient {
  http: reqwest::Client,
  base: Url,
  write_key: Option<String>,
  // Cached upstream `list_indexes` response. Every alias-resolving handler
  // (`_settings`, `_mapping`, `_aliases`, HEAD `/{index}`, `GET /{index}`,
  // and the alias resolution before search/mget/msearch) hits this once
  // per request — without the cache that's one upstream round-trip per
  // call. The cache is read-mostly and protected by a Mutex; on miss we
  // refetch under the lock.
  list_cache: Arc<Mutex<Option<(Instant, Value)>>>,
}

impl SearchliteClient {
  pub fn new(args: &AdapterArgs) -> Result<Self> {
    // RFC 3986 `Url::join` drops the last path segment when the base URL has
    // no trailing slash — so `http://host/prefix` + "indexes/foo" becomes
    // `http://host/indexes/foo`, silently breaking deployments behind a path
    // prefix. Force the base to end with `/` so prefixes are preserved.
    let mut raw = args.upstream_url.clone();
    if !raw.ends_with('/') {
      raw.push('/');
    }
    let base =
      Url::parse(&raw).with_context(|| format!("parsing upstream URL `{}`", args.upstream_url))?;
    // Reject non-HTTP schemes at startup so misconfiguration surfaces as a
    // clear error instead of a cryptic reqwest failure on the first request.
    match base.scheme() {
      "http" | "https" => {}
      other => {
        return Err(anyhow!(
          "upstream URL scheme must be http or https, got `{other}` from `{}`",
          args.upstream_url
        ))
      }
    }
    let http = reqwest::Client::builder()
      .timeout(Duration::from_secs(args.request_timeout_secs))
      .build()
      .context("building reqwest client for upstream searchlite-http")?;
    Ok(Self {
      http,
      base,
      write_key: args.write_key.clone(),
      list_cache: Arc::new(Mutex::new(None)),
    })
  }

  pub async fn search(&self, index: &str, body: &Value) -> Result<Value, ClientError> {
    self.post_json(&upstream_path(index, "search"), body).await
  }

  pub async fn mget(&self, index: &str, body: &Value) -> Result<Value, ClientError> {
    self.post_json(&upstream_path(index, "mget"), body).await
  }

  pub async fn multi_search(&self, index: &str, body: &Value) -> Result<Value, ClientError> {
    self
      .post_json(&upstream_path(index, "multi_search"), body)
      .await
  }

  pub async fn list_indexes(&self) -> Result<Value, ClientError> {
    let mut guard = self.list_cache.lock().await;
    if let Some((stamped_at, value)) = guard.as_ref() {
      if stamped_at.elapsed() < LIST_INDEXES_TTL {
        return Ok(value.clone());
      }
    }
    let fresh = self.get_json("indexes").await?;
    *guard = Some((Instant::now(), fresh.clone()));
    Ok(fresh)
  }

  /// Test/debug helper: drop any cached `list_indexes` response so the next
  /// call goes upstream. Not used in production code paths.
  #[cfg(test)]
  pub async fn invalidate_list_indexes_cache(&self) {
    *self.list_cache.lock().await = None;
  }

  pub async fn inspect(&self, index: &str) -> Result<Value, ClientError> {
    self.get_json(&upstream_path(index, "inspect")).await
  }

  pub async fn healthz(&self) -> Result<Value, ClientError> {
    self.get_json("healthz").await
  }

  async fn post_json(&self, path: &str, body: &Value) -> Result<Value, ClientError> {
    let url = self.url(path)?;
    let mut req = self.http.post(url).json(body);
    if let Some(key) = &self.write_key {
      req = req.header(WRITE_KEY_HEADER, key);
    }
    let resp = req.send().await.map_err(ClientError::from_reqwest)?;
    parse_response(resp).await
  }

  async fn get_json(&self, path: &str) -> Result<Value, ClientError> {
    let url = self.url(path)?;
    let mut req = self.http.get(url);
    if let Some(key) = &self.write_key {
      req = req.header(WRITE_KEY_HEADER, key);
    }
    let resp = req.send().await.map_err(ClientError::from_reqwest)?;
    parse_response(resp).await
  }

  fn url(&self, path: &str) -> Result<Url, ClientError> {
    self
      .base
      .join(path)
      .map_err(|err| ClientError::InvalidUrl(err.to_string()))
  }
}

#[derive(Debug, Error)]
pub enum ClientError {
  #[error("upstream returned {status}: {kind} — {reason}")]
  Upstream {
    status: StatusCode,
    kind: String,
    reason: String,
  },
  #[error("upstream connection failed: {0}")]
  Connection(String),
  #[error("upstream timeout: {0}")]
  Timeout(String),
  #[error("invalid upstream URL: {0}")]
  InvalidUrl(String),
  #[error("upstream returned non-JSON body (status {status}): {body}")]
  Decode { status: StatusCode, body: String },
}

impl ClientError {
  fn from_reqwest(err: reqwest::Error) -> Self {
    if err.is_timeout() {
      ClientError::Timeout(err.to_string())
    } else {
      ClientError::Connection(err.to_string())
    }
  }
}

#[derive(Deserialize)]
struct UpstreamErrorEnvelope {
  error: UpstreamErrorBody,
}

#[derive(Deserialize)]
struct UpstreamErrorBody {
  #[serde(rename = "type")]
  kind: String,
  reason: String,
}

async fn parse_response(resp: reqwest::Response) -> Result<Value, ClientError> {
  let status = resp.status();
  let bytes = resp
    .bytes()
    .await
    .map_err(|err| ClientError::Connection(err.to_string()))?;

  let status_code =
    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

  if status.is_success() {
    if bytes.is_empty() {
      return Ok(Value::Null);
    }
    return serde_json::from_slice::<Value>(&bytes).map_err(|_err| ClientError::Decode {
      status: status_code,
      body: String::from_utf8_lossy(&bytes).to_string(),
    });
  }

  if let Ok(envelope) = serde_json::from_slice::<UpstreamErrorEnvelope>(&bytes) {
    return Err(ClientError::Upstream {
      status: status_code,
      kind: envelope.error.kind,
      reason: envelope.error.reason,
    });
  }

  Err(ClientError::Upstream {
    status: status_code,
    kind: "unknown_upstream_error".to_string(),
    reason: String::from_utf8_lossy(&bytes).to_string(),
  })
}

impl From<ClientError> for ESError {
  fn from(err: ClientError) -> ESError {
    match err {
      ClientError::Upstream {
        status,
        kind,
        reason,
      } => {
        if status == StatusCode::NOT_FOUND || kind == "unknown_index" {
          ESError::not_found("index_not_found_exception", reason)
        } else if status == StatusCode::BAD_REQUEST {
          ESError::bad_request("x_content_parse_exception", reason)
        } else if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
          ESError::new(status, "security_exception", reason)
        } else {
          // Preserve the upstream's `kind` as the ESError type for 5xx so
          // callers branching on `error.type` keep their discriminator
          // (e.g. `index_locked_exception`, `vector_dimension_mismatch`).
          // Previously these collapsed into a generic
          // `internal_server_error` and the kind was buried in `reason`.
          ESError::new(status, kind, reason)
        }
      }
      ClientError::Connection(reason) => ESError::bad_gateway("connection_exception", reason),
      ClientError::Timeout(reason) => ESError::gateway_timeout("timeout_exception", reason),
      ClientError::InvalidUrl(reason) => ESError::internal("internal_server_error", reason),
      ClientError::Decode { status, body } => ESError::new(
        if status.is_success() {
          StatusCode::BAD_GATEWAY
        } else {
          status
        },
        "internal_server_error",
        format!("could not decode upstream response: {body}"),
      ),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::AdapterArgs;
  use clap::Parser;

  fn args(upstream_url: &str) -> AdapterArgs {
    // Construct AdapterArgs by parsing only the URL flag; clap fills the rest
    // with defaults. We only care about `upstream_url` here.
    AdapterArgs::parse_from(["searchlite-elastic", "--upstream-url", upstream_url])
  }

  #[test]
  fn upstream_path_passes_through_simple_index_name() {
    assert_eq!(
      upstream_path("simple-name.v1", "search"),
      "indexes/simple-name.v1/search"
    );
  }

  #[test]
  fn upstream_path_encodes_slash_in_index_name() {
    // Without encoding, `format!("indexes/{index}/search")` with index=`a/b`
    // would create a 4-segment path that Url::join parses structurally,
    // routing to the wrong endpoint upstream.
    assert_eq!(upstream_path("a/b", "search"), "indexes/a%2Fb/search");
  }

  #[test]
  fn upstream_path_encodes_query_and_fragment_chars() {
    assert_eq!(upstream_path("a?b", "search"), "indexes/a%3Fb/search");
    assert_eq!(upstream_path("a#b", "search"), "indexes/a%23b/search");
  }

  #[test]
  fn upstream_path_encodes_whitespace_and_percent() {
    assert_eq!(upstream_path("a b", "search"), "indexes/a%20b/search");
    assert_eq!(upstream_path("a%b", "search"), "indexes/a%25b/search");
  }

  #[test]
  fn client_new_rejects_file_scheme_url() {
    // Regression: file:// (or any non-HTTP scheme) used to be accepted at
    // startup and only fail later at request time with a cryptic reqwest
    // error. Reject loudly instead.
    let result = SearchliteClient::new(&args("file:///etc/passwd"));
    assert!(
      result.is_err(),
      "expected file:// to be rejected at startup"
    );
    let msg = result.err().unwrap().to_string();
    assert!(msg.contains("scheme"), "got: {msg}");
    assert!(msg.contains("file"), "got: {msg}");
  }

  #[test]
  fn client_new_accepts_http_and_https() {
    assert!(SearchliteClient::new(&args("http://127.0.0.1:8080")).is_ok());
    assert!(SearchliteClient::new(&args("https://example.com")).is_ok());
  }

  #[test]
  fn client_new_normalizes_missing_trailing_slash_for_path_prefix() {
    // Prefix base URLs without a trailing slash would have `Url::join` drop
    // the prefix segment. Verify we patch the base before parsing.
    let client =
      SearchliteClient::new(&args("http://example.com/prefix")).expect("parse with prefix");
    let url = client
      .url(&upstream_path("foo", "search"))
      .expect("build url");
    assert_eq!(url.path(), "/prefix/indexes/foo/search");
  }

  #[test]
  fn client_url_construction_with_encoded_index_name_produces_3_path_segments() {
    let client = SearchliteClient::new(&args("http://example.com")).expect("parse simple base");
    let url = client
      .url(&upstream_path("a/b", "search"))
      .expect("build url");
    // Index `a/b` should be a single segment, encoded — total path is
    // /indexes/a%2Fb/search (3 segments), not /indexes/a/b/search (4).
    assert_eq!(url.path(), "/indexes/a%2Fb/search");
  }
}
