use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use serde_json::{json, Map, Value};

use crate::error::{ESError, ESResult};
use crate::routes::indices::resolve_index_or_alias;
use crate::state::AppState;

/// `POST /{index}/_mget` — accepts `{ ids: [...] }` or `{ docs: [{_id, _source?}] }`.
pub async fn mget(
  State(state): State<Arc<AppState>>,
  Path(index): Path<String>,
  Json(body): Json<Value>,
) -> ESResult<Json<Value>> {
  // Resolve aliases so each returned doc carries `_index = <target>`,
  // matching ES; previously responses echoed the alias path token.
  let resolved = resolve_index_or_alias(&state, &index)
    .await?
    .ok_or_else(|| {
      ESError::not_found(
        "index_not_found_exception",
        format!("no such index [{index}]"),
      )
    })?;
  let ids = collect_ids(&body, &index)?;
  if ids.is_empty() {
    return Ok(Json(json!({ "docs": [] })));
  }
  let return_stored = wants_source(&body);
  let upstream = state
    .client()
    .mget(
      &resolved,
      &json!({ "ids": ids, "return_stored": return_stored }),
    )
    .await?;
  Ok(Json(translate_mget_response(&resolved, &upstream)))
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

  // Track requested order so we re-emit in the original sequence. Each
  // doc's `_index` is resolved through the alias table so groups (and the
  // `_index` stamping in the response) use the concrete target index, not
  // whatever path token the caller supplied.
  let mut grouped: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
  for (idx, doc) in docs.iter().enumerate() {
    let map = doc.as_object().ok_or_else(|| {
      ESError::bad_request(
        "x_content_parse_exception",
        "each `docs` entry must be an object",
      )
    })?;
    let raw_index = map.get("_index").and_then(Value::as_str).ok_or_else(|| {
      ESError::bad_request(
        "x_content_parse_exception",
        "missing `_index` in docs entry",
      )
    })?;
    let index = resolve_index_or_alias(&state, raw_index)
      .await?
      .ok_or_else(|| {
        ESError::not_found(
          "index_not_found_exception",
          format!("no such index [{raw_index}]"),
        )
      })?;
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
    validate_mget_order(&index, &entries, &docs_arr)?;
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

/// Validate that an upstream mget response lines up with the requested
/// positional batch.
///
/// Two checks, each surfaced as a 502 so callers see a clear protocol
/// violation rather than receiving a wrong-doc-at-wrong-slot response:
///
/// 1. Length parity — ES `_mget` (and our upstream) guarantees one response
///    entry per requested id; a mismatch would silently zip-truncate.
/// 2. Per-position `_id` parity — even with matching length, an upstream
///    that reordered docs would route the wrong source to each slot.
///    `searchlite-core`'s `Reader::mget` preserves order today (and there's
///    a regression test for it), but we don't want to bake that contract
///    into the adapter without a verifier.
fn validate_mget_order(
  index: &str,
  entries: &[(usize, String)],
  docs_arr: &[Value],
) -> ESResult<()> {
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
  for (i, ((_, requested_id), doc)) in entries.iter().zip(docs_arr.iter()).enumerate() {
    let actual = doc.get("_id").and_then(Value::as_str).unwrap_or("");
    if actual != requested_id {
      return Err(ESError::bad_gateway(
        "internal_server_error",
        format!(
          "upstream mget for index `{index}` returned out-of-order docs at position {i}: requested `{requested_id}`, got `{actual}`"
        ),
      ));
    }
  }
  Ok(())
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

#[cfg(test)]
mod tests {
  use super::*;
  use axum::http::StatusCode;

  fn entries(ids: &[&str]) -> Vec<(usize, String)> {
    ids
      .iter()
      .enumerate()
      .map(|(i, id)| (i, (*id).to_string()))
      .collect()
  }

  fn doc(id: &str) -> Value {
    json!({ "_index": "demo", "_id": id, "found": true })
  }

  #[test]
  fn validate_mget_order_accepts_matching_length_and_order() {
    let req = entries(&["a", "b", "c"]);
    let resp = vec![doc("a"), doc("b"), doc("c")];
    assert!(validate_mget_order("demo", &req, &resp).is_ok());
  }

  #[test]
  fn validate_mget_order_rejects_length_mismatch_with_502() {
    // Pre-existing length check; pinned here so a refactor of the helper
    // can't silently regress to a positional zip without parity.
    let req = entries(&["a", "b", "c"]);
    let resp = vec![doc("a"), doc("b")];
    let err = validate_mget_order("demo", &req, &resp).expect_err("should fail");
    assert_eq!(err.status, StatusCode::BAD_GATEWAY);
    assert!(
      err.reason.contains("returned 2 docs for 3 requested"),
      "reason: {}",
      err.reason
    );
  }

  #[test]
  fn validate_mget_order_rejects_per_position_id_mismatch_with_502() {
    // Defense-in-depth: even when lengths match, a reordered upstream
    // response would route the wrong document into each requested slot.
    // Surface as a clear 502 so the caller doesn't act on misrouted data.
    let req = entries(&["a", "b", "c"]);
    let resp = vec![doc("a"), doc("c"), doc("b")];
    let err = validate_mget_order("demo", &req, &resp).expect_err("should fail");
    assert_eq!(err.status, StatusCode::BAD_GATEWAY);
    assert!(
      err.reason.contains("position 1"),
      "reason should pinpoint the slot, got: {}",
      err.reason
    );
    assert!(
      err.reason.contains("requested `b`"),
      "reason should name the requested id, got: {}",
      err.reason
    );
    assert!(
      err.reason.contains("got `c`"),
      "reason should name the returned id, got: {}",
      err.reason
    );
  }

  #[test]
  fn validate_mget_order_rejects_doc_with_missing_id_field() {
    // A doc missing `_id` (e.g. an upstream that drops the field on a
    // not-found doc) should fail validation — the `_id` is what we use to
    // verify positional routing, so an absent value can't be trusted.
    let req = entries(&["a"]);
    let resp = vec![json!({ "_index": "demo", "found": false })];
    let err = validate_mget_order("demo", &req, &resp).expect_err("should fail");
    assert_eq!(err.status, StatusCode::BAD_GATEWAY);
  }
}
