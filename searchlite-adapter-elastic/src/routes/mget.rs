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

  // Track requested order so we re-emit in the original sequence.
  let mut grouped: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
  for (idx, doc) in docs.iter().enumerate() {
    let map = doc.as_object().ok_or_else(|| {
      ESError::bad_request("x_content_parse_exception", "each `docs` entry must be an object")
    })?;
    let index = map
      .get("_index")
      .and_then(Value::as_str)
      .ok_or_else(|| {
        ESError::bad_request("x_content_parse_exception", "missing `_index` in docs entry")
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
        &json!({ "ids": &ids, "return_stored": true }),
      )
      .await?;
    let translated = translate_mget_response(&index, &upstream);
    let docs_arr = translated
      .get("docs")
      .and_then(Value::as_array)
      .cloned()
      .unwrap_or_default();
    for ((position, _), translated_doc) in entries.into_iter().zip(docs_arr.into_iter()) {
      by_position.insert(position, translated_doc);
    }
  }

  let merged: Vec<Value> = by_position.into_values().collect();
  Ok(Json(json!({ "docs": merged })))
}

fn collect_ids(body: &Value, default_index: &str) -> ESResult<Vec<String>> {
  if let Some(ids) = body.get("ids").and_then(Value::as_array) {
    return Ok(
      ids
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect(),
    );
  }
  if let Some(docs) = body.get("docs").and_then(Value::as_array) {
    let mut ids = Vec::with_capacity(docs.len());
    for doc in docs {
      let map = doc.as_object().ok_or_else(|| {
        ESError::bad_request("x_content_parse_exception", "each `docs` entry must be an object")
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
  let translated: Vec<Value> = docs
    .iter()
    .map(|doc| translate_doc(index, doc))
    .collect();
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
