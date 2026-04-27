use std::time::Duration;

use anyhow::{Context, Result};
use axum::http::StatusCode;
use reqwest::Url;
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::error::ESError;
use crate::AdapterArgs;

const WRITE_KEY_HEADER: &str = "x-searchlite-write-key";

pub struct SearchliteClient {
  http: reqwest::Client,
  base: Url,
  write_key: Option<String>,
}

impl SearchliteClient {
  pub fn new(args: &AdapterArgs) -> Result<Self> {
    let base = Url::parse(&args.upstream_url)
      .with_context(|| format!("parsing upstream URL `{}`", args.upstream_url))?;
    let http = reqwest::Client::builder()
      .timeout(Duration::from_secs(args.request_timeout_secs))
      .build()
      .context("building reqwest client for upstream searchlite-http")?;
    Ok(Self {
      http,
      base,
      write_key: args.write_key.clone(),
    })
  }

  pub async fn search(&self, index: &str, body: &Value) -> Result<Value, ClientError> {
    self
      .post_json(&format!("indexes/{index}/search"), body)
      .await
  }

  pub async fn mget(&self, index: &str, body: &Value) -> Result<Value, ClientError> {
    self
      .post_json(&format!("indexes/{index}/mget"), body)
      .await
  }

  pub async fn multi_search(&self, index: &str, body: &Value) -> Result<Value, ClientError> {
    self
      .post_json(&format!("indexes/{index}/multi_search"), body)
      .await
  }

  pub async fn list_indexes(&self) -> Result<Value, ClientError> {
    self.get_json("indexes").await
  }

  pub async fn inspect(&self, index: &str) -> Result<Value, ClientError> {
    self.get_json(&format!("indexes/{index}/inspect")).await
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

  let status_code = StatusCode::from_u16(status.as_u16())
    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

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
          ESError::new(
            status,
            "internal_server_error",
            format!("upstream {kind}: {reason}"),
          )
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
