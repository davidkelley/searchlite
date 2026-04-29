use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use serde_json::{json, Map, Value};

use crate::error::{ESError, ESResult};
use crate::state::AppState;

/// `POST /{index}/_mget` — accepts `{ ids: [...] }` or `{ docs: [{_id, _source?}] }`.
pub async fn mget(
  State(state): State<Arc<AppState>>,
  Path(index): Path<String>,
  Json(body): Json<Value>,
) -> ESResult<Json<Value>> {
  let ids = collect_ids(&body, &index)?;
  if ids.is_empty() {
    return Ok(Json(json!({ "docs": [] })));
  }
  let return_stored = wants_source(&body);
  let upstream = state
    .client()
    .mget(
      &index,
      &json!({ "ids": ids, "return_stored": return_stored }),
    )
    .await?;
  Ok(Json(translate_mget_response(&index, &upstream)))
}

/// `POST /_mget` — body has `docs: [{_index, _id}]`. We group by index and
/// dispatch one upstream call per group, then re-merge in request order.
pub async fn mget_global(
  State(state): State<Arc<AppState>>,
  Json(body): Json<Value>,
) -> ESResult<Json<Value>> {
  let docs = body
    .get("docs")
    .and_then(Value::as_array)
    .ok_or_else(|| ESError::bad_request("x_content_parse_exception", "missing `docs` array"))?
    .clone();

  // Honor top-level `_source` so global `_mget` matches the index-scoped
  // form's behaviour. Without this the upstream always returned stored
  // fields even when the caller asked for `_source: false`.
  let return_stored = wants_source(&body);

  // Track requested order so we re-emit in the original sequence.
  let mut grouped: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
  for (idx, doc) in docs.iter().enumerate() {
    let map = doc.as_object().ok_or_else(|| {
      ESError::bad_request(
        "x_content_parse_exception",
        "each `docs` entry must be an object",
      )
    })?;
    let index = map
      .get("_index")
      .and_then(Value::as_str)
      .ok_or_else(|| {
        ESError::bad_request(
          "x_content_parse_exception",
          "missing `_index` in docs entry",
        )
      })?
      .to_string();
    let id = map
      .get("_id")
      .and_then(Value::as_str)
      .ok_or_else(|| ESError::bad_request("x_content_parse_exception", "missing `_id`"))?
      .to_string();
    grouped.entry(index).or_default().push((idx, id));
  }

  let mut by_position: BTreeMap<usize, Value> = BTreeMap::new();
  for (index, entries) in grouped {
    let ids: Vec<String> = entries.iter().map(|(_, id)| id.clone()).collect();
    let upstream = state
      .client()
      .mget(
        &index,
        &json!({ "ids": &ids, "return_stored": return_stored }),
      )
      .await?;
    let translated = translate_mget_response(&index, &upstream);
    let docs_arr = translated
      .get("docs")
      .and_then(Value::as_array)
      .cloned()
      .unwrap_or_default();
    // Defense-in-depth: ES `_mget` guarantees one response entry per
    // requested id. If the upstream ever returns a different number we'd
    // silently drop trailing entries (or extras) when zipping; surface that
    // as a 502 so clients see the protocol violation instead of a malformed
    // response. Mirrors the equivalent check in `msearch` upstream-results.
    if docs_arr.len() != entries.len() {
      return Err(ESError::bad_gateway(
        "internal_server_error",
        format!(
          "upstream mget for index `{index}` returned {} docs for {} requested ids",
          docs_arr.len(),
          entries.len()
        ),
      ));
    }
    for ((position, _), translated_doc) in entries.into_iter().zip(docs_arr.into_iter()) {
      by_position.insert(position, translated_doc);
    }
  }

  let merged: Vec<Value> = by_position.into_values().collect();
  Ok(Json(json!({ "docs": merged })))
}

fn collect_ids(body: &Value, default_index: &str) -> ESResult<Vec<String>> {
  if let Some(ids) = body.get("ids").and_then(Value::as_array) {
    // Reject non-string entries explicitly. Silently filtering them out makes
    // malformed requests look like partial successes (fewer docs returned
    // than requested), which violates the one-response-per-id contract that
    // ES clients rely on.
    return ids
      .iter()
      .map(|v| {
        v.as_str().map(str::to_string).ok_or_else(|| {
          ESError::bad_request(
            "x_content_parse_exception",
            "every entry in `ids` must be a string",
          )
        })
      })
      .collect::<Result<Vec<_>, _>>();
  }
  if let Some(docs) = body.get("docs").and_then(Value::as_array) {
    let mut ids = Vec::with_capacity(docs.len());
    for doc in docs {
      let map = doc.as_object().ok_or_else(|| {
        ESError::bad_request(
          "x_content_parse_exception",
          "each `docs` entry must be an object",
        )
      })?;
      if let Some(idx) = map.get("_index").and_then(Value::as_str) {
        if idx != default_index {
          return Err(ESError::bad_request(
            "x_content_parse_exception",
            format!("doc _index `{idx}` does not match request index `{default_index}`"),
          ));
        }
      }
      let id = map
        .get("_id")
        .and_then(Value::as_str)
        .ok_or_else(|| ESError::bad_request("x_content_parse_exception", "missing `_id`"))?
        .to_string();
      ids.push(id);
    }
    return Ok(ids);
  }
  Err(ESError::bad_request(
    "x_content_parse_exception",
    "request must contain `ids` or `docs`",
  ))
}

fn wants_source(body: &Value) -> bool {
  match body.get("_source") {
    Some(Value::Bool(b)) => *b,
    Some(Value::Null) => false,
    Some(_) => true,
    None => true,
  }
}

fn translate_mget_response(index: &str, upstream: &Value) -> Value {
  let docs = upstream
    .get("docs")
    .and_then(Value::as_array)
    .cloned()
    .unwrap_or_default();
  let translated: Vec<Value> = docs.iter().map(|doc| translate_doc(index, doc)).collect();
  json!({ "docs": translated })
}

fn translate_doc(index: &str, doc: &Value) -> Value {
  let map = doc.as_object();
  let id = map
    .and_then(|m| m.get("doc_id"))
    .and_then(Value::as_str)
    .unwrap_or("")
    .to_string();
  let found = map
    .and_then(|m| m.get("found"))
    .and_then(Value::as_bool)
    .unwrap_or(false);
  let source = map.and_then(|m| m.get("_source"));

  let mut out = Map::new();
  out.insert("_index".into(), Value::String(index.to_string()));
  out.insert("_id".into(), Value::String(id));
  out.insert("found".into(), Value::Bool(found));
  if let Some(src) = source {
    out.insert("_source".into(), src.clone());
  }
  Value::Object(out)
}
