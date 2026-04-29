use axum::http::StatusCode;
use axum::response::IntoResponse;

use crate::error::ESError;

pub async fn write_not_supported() -> impl IntoResponse {
  ESError::new(
    StatusCode::BAD_REQUEST,
    "not_supported_in_v1",
    "write/DDL operations are not supported by the searchlite elasticsearch adapter (read-only v1)",
  )
}

pub async fn doc_get_not_supported() -> impl IntoResponse {
  ESError::new(
    StatusCode::BAD_REQUEST,
    "not_supported_in_v1",
    "GET /_doc/{id} is not supported in v1; use POST /_search with a term query",
  )
}

pub async fn cross_index_search() -> impl IntoResponse {
  ESError::new(
    StatusCode::BAD_REQUEST,
    "illegal_argument_exception",
    "cross-index search is not supported by the searchlite elasticsearch adapter; specify exactly one index in the path",
  )
}
