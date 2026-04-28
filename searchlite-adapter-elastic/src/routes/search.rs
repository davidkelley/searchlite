use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, Query, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::error::{ESError, ESResult};
use crate::state::AppState;
use crate::translate::{translate_search_body, translate_search_response};

#[derive(Debug, Deserialize, Default)]
pub struct SearchParams {
  q: Option<String>,
  from: Option<usize>,
  size: Option<usize>,
  sort: Option<String>,
  #[serde(rename = "df")]
  default_field: Option<String>,
  #[serde(rename = "_source")]
  source: Option<String>,
  #[serde(rename = "track_total_hits")]
  track_total_hits: Option<bool>,
}

pub async fn search(
  State(state): State<Arc<AppState>>,
  Path(index): Path<String>,
  Query(params): Query<SearchParams>,
  body: Option<Json<Value>>,
) -> ESResult<Json<Value>> {
  let merged = merge_query_params_into_body(body.map(|Json(v)| v), &params)?;
  let sl_body = translate_search_body(&merged)?;
  let started = Instant::now();
  let sl_response = state.client().search(&index, &sl_body).await?;
  let took_ms = started.elapsed().as_millis() as u64;
  Ok(Json(translate_search_response(
    &index,
    &sl_response,
    took_ms,
  )))
}

pub async fn count(
  State(state): State<Arc<AppState>>,
  Path(index): Path<String>,
  Query(params): Query<SearchParams>,
  body: Option<Json<Value>>,
) -> ESResult<Json<Value>> {
  let mut merged = merge_query_params_into_body(body.map(|Json(v)| v), &params)?;
  if let Some(map) = merged.as_object_mut() {
    map.insert("size".into(), Value::from(0u64));
    map.insert("track_total_hits".into(), Value::Bool(true));
  }
  let sl_body = translate_search_body(&merged)?;
  let sl_response = state.client().search(&index, &sl_body).await?;
  let total = sl_response
    .get("total_hits_estimate")
    .and_then(Value::as_u64)
    .unwrap_or(0);
  Ok(Json(json!({
    "count": total,
    "_shards": { "total": 1, "successful": 1, "skipped": 0, "failed": 0 },
  })))
}

fn merge_query_params_into_body(body: Option<Value>, params: &SearchParams) -> ESResult<Value> {
  let mut map = match body {
    Some(Value::Object(map)) => map,
    Some(Value::Null) | None => Map::new(),
    Some(_) => {
      return Err(ESError::bad_request(
        "x_content_parse_exception",
        "request body must be a JSON object",
      ));
    }
  };

  if let Some(q) = &params.q {
    let mut qs = Map::new();
    qs.insert("query".into(), Value::String(q.clone()));
    if let Some(df) = &params.default_field {
      qs.insert("default_field".into(), Value::String(df.clone()));
    }
    map.insert("query".into(), json!({ "query_string": Value::Object(qs) }));
  }
  if let Some(from) = params.from {
    map
      .entry("from".to_string())
      .or_insert(Value::from(from as u64));
  }
  if let Some(size) = params.size {
    map
      .entry("size".to_string())
      .or_insert(Value::from(size as u64));
  }
  if let Some(sort) = &params.sort {
    let parts: Vec<Value> = sort
      .split(',')
      .filter(|s| !s.is_empty())
      .map(|chunk| {
        if let Some((field, order)) = chunk.split_once(':') {
          json!({ field: { "order": order } })
        } else {
          Value::String(chunk.to_string())
        }
      })
      .collect();
    map.entry("sort".to_string()).or_insert(Value::Array(parts));
  }
  if let Some(track) = params.track_total_hits {
    map
      .entry("track_total_hits".to_string())
      .or_insert(Value::Bool(track));
  }
  if let Some(source) = &params.source {
    let trimmed = source.trim();
    // Match Elasticsearch's URL semantics for ?_source: bare booleans
    // turn the source on/off entirely; otherwise treat as a comma-separated
    // includes list. Without this, `?_source=false` would be interpreted as
    // requesting a field literally called `false`.
    match trimmed.to_ascii_lowercase().as_str() {
      "true" => {
        map
          .entry("_source".to_string())
          .or_insert(Value::Bool(true));
      }
      "false" => {
        map
          .entry("_source".to_string())
          .or_insert(Value::Bool(false));
      }
      _ => {
        let parts: Vec<Value> = trimmed
          .split(',')
          .map(str::trim)
          .filter(|s| !s.is_empty())
          .map(|s| Value::String(s.to_string()))
          .collect();
        if !parts.is_empty() {
          map
            .entry("_source".to_string())
            .or_insert(Value::Array(parts));
        }
      }
    }
  }
  Ok(Value::Object(map))
}
