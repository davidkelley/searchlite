use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::{json, Value};

use crate::error::{ESError, ESResult};
use crate::routes::indices::resolve_index_or_alias;
use crate::state::AppState;
use crate::translate::{
  extract_agg_meta, inject_agg_meta, translate_search_body, translate_search_response,
};

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

  // Group by RESOLVED index name. Each entry's `index` may be an alias —
  // resolving it here means downstream calls (and the `_index` stamping in
  // the response via translate_search_response) use the concrete target.
  let mut grouped: BTreeMap<String, Vec<(usize, Value)>> = BTreeMap::new();
  for (idx, entry) in entries.iter().enumerate() {
    let resolved = resolve_index_or_alias(&state, &entry.index)
      .await?
      .ok_or_else(|| {
        ESError::not_found(
          "index_not_found_exception",
          format!("no such index [{}]", entry.index),
        )
      })?;
    grouped
      .entry(resolved)
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
    for ((position, body), result) in group.into_iter().zip(results.into_iter()) {
      // Each msearch entry can carry its own track_total_hits and per-agg
      // meta. Extract both before translation and re-inject meta after, so
      // per-response semantics match what the caller asked for.
      let track = extract_track_total_hits(&body);
      let agg_meta = body
        .get("aggs")
        .or_else(|| body.get("aggregations"))
        .and_then(Value::as_object)
        .map(extract_agg_meta)
        .unwrap_or_default();
      let mut translated = translate_search_response(&index, &result, 0, track);
      inject_agg_meta(&mut translated, &agg_meta);
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

/// Per-entry equivalent of the search route's helper. Maps the ES request's
/// `track_total_hits` flag to the Some/None contract used by the response
/// translator.
fn extract_track_total_hits(body: &Value) -> Option<bool> {
  let value = body.as_object()?.get("track_total_hits")?;
  match value {
    Value::Bool(b) => Some(*b),
    Value::Number(n) => n.as_u64().map(|cap| cap > 0),
    _ => None,
  }
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
          // Validate every element. Silently dropping non-strings via
          // filter_map could turn a malformed `["demo", 42]` into a
          // single-index ("demo") request and route queries to an
          // unintended index — fail loudly instead.
          let mut names = Vec::with_capacity(items.len());
          for item in items {
            match item.as_str() {
              Some(s) => names.push(s.to_string()),
              None => {
                return Err(ESError::bad_request(
                  "x_content_parse_exception",
                  format!("msearch entry `index` array contains non-string element: {item}"),
                ));
              }
            }
          }
          if names.len() == 1 {
            return Ok(names.into_iter().next().unwrap());
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
