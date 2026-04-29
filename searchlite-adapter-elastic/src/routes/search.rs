use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, Query, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::error::{ESError, ESResult};
use crate::routes::indices::resolve_index_or_alias;
use crate::state::AppState;
use crate::translate::{
  extract_agg_meta, inject_agg_meta, translate_search_body, translate_search_response,
};

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
  // String rather than bool so we accept ES's integer-cap form
  // (e.g. `?track_total_hits=10000`) alongside `true`/`false`. Parsing into
  // a JSON value happens in `merge_query_params_into_body`; downstream
  // pagination already handles bool vs integer.
  #[serde(rename = "track_total_hits")]
  track_total_hits: Option<String>,
}

pub async fn search(
  State(state): State<Arc<AppState>>,
  Path(index): Path<String>,
  Query(params): Query<SearchParams>,
  body: Option<Json<Value>>,
) -> ESResult<Json<Value>> {
  // Resolve aliases up front so hits are stamped with the concrete target
  // index name, matching ES. Without this, `_index` on each hit echoes the
  // alias the caller used — and ES SDKs round-trip `_index` from search
  // hits back into write requests, so the wrong-index drift is real.
  let resolved = resolve_index_or_alias(&state, &index)
    .await?
    .ok_or_else(|| {
      ESError::not_found(
        "index_not_found_exception",
        format!("no such index [{index}]"),
      )
    })?;
  let merged = merge_query_params_into_body(body.map(|Json(v)| v), &params)?;
  // Capture the caller's track_total_hits intent before translation so the
  // response can report exact-vs-approximate semantics correctly.
  let track = extract_track_total_hits(&merged);
  // Capture each top-level agg's `meta` blob before translation so we can
  // re-inject it into the response — SearchLite has no `meta` plumbing.
  let agg_meta = merged
    .get("aggs")
    .or_else(|| merged.get("aggregations"))
    .and_then(Value::as_object)
    .map(extract_agg_meta)
    .unwrap_or_default();
  let sl_body = translate_search_body(&merged)?;
  let started = Instant::now();
  let sl_response = state.client().search(&resolved, &sl_body).await?;
  let took_ms = started.elapsed().as_millis() as u64;
  let mut response = translate_search_response(&resolved, &sl_response, took_ms, track);
  inject_agg_meta(&mut response, &agg_meta);
  Ok(Json(response))
}

pub async fn count(
  State(state): State<Arc<AppState>>,
  Path(index): Path<String>,
  Query(params): Query<SearchParams>,
  body: Option<Json<Value>>,
) -> ESResult<Json<Value>> {
  let resolved = resolve_index_or_alias(&state, &index)
    .await?
    .ok_or_else(|| {
      ESError::not_found(
        "index_not_found_exception",
        format!("no such index [{index}]"),
      )
    })?;
  let mut merged = merge_query_params_into_body(body.map(|Json(v)| v), &params)?;
  if let Some(map) = merged.as_object_mut() {
    map.insert("size".into(), Value::from(0u64));
    map.insert("track_total_hits".into(), Value::Bool(true));
  }
  let sl_body = translate_search_body(&merged)?;
  let sl_response = state.client().search(&resolved, &sl_body).await?;
  let total = sl_response
    .get("total_hits_estimate")
    .and_then(Value::as_u64)
    .unwrap_or(0);
  Ok(Json(json!({
    "count": total,
    "_shards": { "total": 1, "successful": 1, "skipped": 0, "failed": 0 },
  })))
}

/// Resolve the request's `track_total_hits` to the Some/None contract that
/// `translate_search_response` expects:
/// - boolean → that bool
/// - integer N: > 0 → `Some(true)` (exact totals); 0 → `Some(false)`
/// - missing or non-numeric/bool → `None` (default lower-bound semantics)
fn extract_track_total_hits(body: &Value) -> Option<bool> {
  let value = body.as_object()?.get("track_total_hits")?;
  match value {
    Value::Bool(b) => Some(*b),
    Value::Number(n) => n.as_u64().map(|cap| cap > 0),
    _ => None,
  }
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
    // Trim each comma-separated chunk, then trim around the optional `:`
    // separator. Without this, common forms like `?sort=foo:desc, _score`
    // produce a field literally named " _score" (leading space) and
    // `?sort=foo : desc` carries trailing/leading whitespace into both
    // field and order — both round-trip incorrectly to upstream.
    let parts: Vec<Value> = sort
      .split(',')
      .map(str::trim)
      .filter(|s| !s.is_empty())
      .map(|chunk| {
        if let Some((field, order)) = chunk.split_once(':') {
          let field = field.trim();
          let order = order.trim();
          let mut obj = serde_json::Map::new();
          obj.insert(field.to_string(), json!({ "order": order }));
          Value::Object(obj)
        } else {
          Value::String(chunk.to_string())
        }
      })
      .collect();
    map.entry("sort".to_string()).or_insert(Value::Array(parts));
  }
  if let Some(track) = &params.track_total_hits {
    let trimmed = track.trim();
    let value = match trimmed.to_ascii_lowercase().as_str() {
      "true" => Value::Bool(true),
      "false" => Value::Bool(false),
      other => match other.parse::<u64>() {
        Ok(n) => Value::from(n),
        Err(_) => {
          return Err(ESError::bad_request(
            "x_content_parse_exception",
            format!(
              "track_total_hits must be `true`, `false`, or a non-negative integer, got `{trimmed}`"
            ),
          ));
        }
      },
    };
    map.entry("track_total_hits".to_string()).or_insert(value);
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
