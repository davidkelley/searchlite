use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::{json, Value};

use crate::error::{ESError, ESResult};
use crate::state::AppState;
use crate::translate::{translate_search_body, translate_search_response};

/// `POST /_msearch` (and `POST /{index}/_msearch`) — NDJSON of alternating
/// header + body lines. Header may include `index` (string or array);
/// requests without an explicit index inherit the path index, if any.
pub async fn msearch(
  State(state): State<Arc<AppState>>,
  index: Option<Path<String>>,
  body: Bytes,
) -> ESResult<Json<Value>> {
  let default_index = index.map(|Path(i)| i);
  let entries = parse_msearch_ndjson(&body, default_index.as_deref())?;
  if entries.is_empty() {
    return Ok(Json(json!({ "took": 0, "responses": [] })));
  }

  // Group by index, preserving the original request order.
  let mut grouped: BTreeMap<String, Vec<(usize, Value)>> = BTreeMap::new();
  for (idx, entry) in entries.iter().enumerate() {
    grouped
      .entry(entry.index.clone())
      .or_default()
      .push((idx, entry.body.clone()));
  }

  let started = Instant::now();
  let mut by_position: BTreeMap<usize, Value> = BTreeMap::new();
  for (index, group) in grouped {
    let searches: Vec<Value> = group
      .iter()
      .map(|(_, body)| translate_search_body(body))
      .collect::<Result<Vec<_>, _>>()?;
    let upstream_request = json!({
      "searches": searches,
      "parallel": true,
    });
    let upstream = state
      .client()
      .multi_search(&index, &upstream_request)
      .await?;
    let results = upstream
      .get("results")
      .and_then(Value::as_array)
      .cloned()
      .unwrap_or_default();
    if results.len() != group.len() {
      return Err(ESError::bad_gateway(
        "internal_server_error",
        format!(
          "upstream multi_search returned {} results for {} requested searches",
          results.len(),
          group.len()
        ),
      ));
    }
    for ((position, _), result) in group.into_iter().zip(results.into_iter()) {
      let translated = translate_search_response(&index, &result, 0);
      by_position.insert(position, translated);
    }
  }

  let took_ms = started.elapsed().as_millis() as u64;
  let responses: Vec<Value> = by_position.into_values().collect();
  Ok(Json(json!({
    "took": took_ms,
    "responses": responses,
  })))
}

struct MSearchEntry {
  index: String,
  body: Value,
}

fn parse_msearch_ndjson(body: &[u8], default_index: Option<&str>) -> ESResult<Vec<MSearchEntry>> {
  let text = std::str::from_utf8(body)
    .map_err(|_| ESError::bad_request("x_content_parse_exception", "msearch body must be UTF-8"))?;
  // `str::lines()` strips both `\n` and `\r\n` per line, so CRLF-delimited
  // NDJSON (common from Windows clients and some HTTP proxies) parses
  // correctly. Plain `split('\n')` would leave a trailing `\r` that breaks
  // serde_json::from_str on every line.
  let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
  if !lines.len().is_multiple_of(2) {
    return Err(ESError::bad_request(
      "x_content_parse_exception",
      "msearch ndjson must contain an even number of non-empty lines (header + body pairs)",
    ));
  }

  let mut entries = Vec::with_capacity(lines.len() / 2);
  for pair in lines.chunks(2) {
    let header_value: Value = serde_json::from_str(pair[0]).map_err(|err| {
      ESError::bad_request(
        "x_content_parse_exception",
        format!("invalid msearch header JSON: {err}"),
      )
    })?;
    let body_value: Value = serde_json::from_str(pair[1]).map_err(|err| {
      ESError::bad_request(
        "x_content_parse_exception",
        format!("invalid msearch body JSON: {err}"),
      )
    })?;
    let index = pick_index(&header_value, default_index)?;
    entries.push(MSearchEntry {
      index,
      body: body_value,
    });
  }
  Ok(entries)
}

fn pick_index(header: &Value, default_index: Option<&str>) -> ESResult<String> {
  if let Some(map) = header.as_object() {
    if let Some(value) = map.get("index") {
      match value {
        Value::String(s) => return Ok(s.clone()),
        Value::Array(items) => {
          let names: Vec<&str> = items.iter().filter_map(Value::as_str).collect();
          if names.len() == 1 {
            return Ok(names[0].to_string());
          }
          if names.len() > 1 {
            return Err(ESError::bad_request(
              "illegal_argument_exception",
              "multi-index msearch entries are not supported (single index per request)",
            ));
          }
        }
        _ => {}
      }
    }
  }
  default_index.map(str::to_string).ok_or_else(|| {
    ESError::bad_request("x_content_parse_exception", "msearch entry missing `index`")
  })
}
