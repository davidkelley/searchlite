use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("{reason}")]
pub struct ESError {
  pub status: StatusCode,
  pub error_type: String,
  pub reason: String,
  pub root_cause: Vec<RootCause>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RootCause {
  #[serde(rename = "type")]
  pub error_type: String,
  pub reason: String,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
  error: ErrorPayload<'a>,
  status: u16,
}

#[derive(Serialize)]
struct ErrorPayload<'a> {
  #[serde(rename = "type")]
  error_type: &'a str,
  reason: &'a str,
  root_cause: &'a [RootCause],
}

impl ESError {
  pub fn new(status: StatusCode, error_type: impl Into<String>, reason: impl Into<String>) -> Self {
    let error_type = error_type.into();
    let reason = reason.into();
    let root_cause = vec![RootCause {
      error_type: error_type.clone(),
      reason: reason.clone(),
    }];
    Self {
      status,
      error_type,
      reason,
      root_cause,
    }
  }

  pub fn bad_request(error_type: impl Into<String>, reason: impl Into<String>) -> Self {
    Self::new(StatusCode::BAD_REQUEST, error_type, reason)
  }

  pub fn not_found(error_type: impl Into<String>, reason: impl Into<String>) -> Self {
    Self::new(StatusCode::NOT_FOUND, error_type, reason)
  }

  pub fn internal(error_type: impl Into<String>, reason: impl Into<String>) -> Self {
    Self::new(StatusCode::INTERNAL_SERVER_ERROR, error_type, reason)
  }

  pub fn bad_gateway(error_type: impl Into<String>, reason: impl Into<String>) -> Self {
    Self::new(StatusCode::BAD_GATEWAY, error_type, reason)
  }

  pub fn gateway_timeout(error_type: impl Into<String>, reason: impl Into<String>) -> Self {
    Self::new(StatusCode::GATEWAY_TIMEOUT, error_type, reason)
  }

  pub fn unsupported(feature: impl Into<String>) -> Self {
    let feature = feature.into();
    Self::bad_request(
      "x_content_parse_exception",
      format!("feature `{feature}` not supported by searchlite adapter"),
    )
  }

  /// Override the HTTP status returned to the client. The body's `status`
  /// field is rebuilt from `self.status` at render time
  /// (see `IntoResponse for ESError`), so callers always get the new status
  /// reflected in both the HTTP envelope and the JSON body. The error
  /// `type`/`reason` strings are NOT rebuilt — set those via the constructor
  /// before calling `with_status` to keep the body coherent.
  pub fn with_status(mut self, status: StatusCode) -> Self {
    self.status = status;
    self
  }
}

impl IntoResponse for ESError {
  fn into_response(self) -> Response {
    let body = ErrorBody {
      error: ErrorPayload {
        error_type: &self.error_type,
        reason: &self.reason,
        root_cause: &self.root_cause,
      },
      status: self.status.as_u16(),
    };
    (self.status, Json(body)).into_response()
  }
}

pub type ESResult<T> = Result<T, ESError>;
