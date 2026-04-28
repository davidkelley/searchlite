use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};

use crate::error::{ESError, ESResult};
use crate::state::AppState;
use crate::translate::schema_to_es;

pub async fn head_index(
  State(state): State<Arc<AppState>>,
  Path(index): Path<String>,
) -> StatusCode {
  match index_exists(&state, &index).await {
    Ok(true) => StatusCode::OK,
    Ok(false) => StatusCode::NOT_FOUND,
    Err(_) => StatusCode::SERVICE_UNAVAILABLE,
  }
}

pub async fn get_index(
  State(state): State<Arc<AppState>>,
  Path(index): Path<String>,
) -> ESResult<Json<Value>> {
  let mapping = get_mapping_for(&state, &index).await?;
  let settings = settings_payload(&index);
  let mappings = mapping
    .get(&index)
    .and_then(|v| v.get("mappings"))
    .cloned()
    .unwrap_or_else(|| json!({ "properties": {} }));
  let settings_value = settings
    .get(&index)
    .and_then(|v| v.get("settings"))
    .cloned()
    .unwrap_or_else(|| json!({}));

  let mut payload = serde_json::Map::new();
  payload.insert(
    index,
    json!({
      "aliases": {},
      "mappings": mappings,
      "settings": settings_value,
    }),
  );
  Ok(Json(Value::Object(payload)))
}

pub async fn get_mapping(
  State(state): State<Arc<AppState>>,
  Path(index): Path<String>,
) -> ESResult<Json<Value>> {
  Ok(Json(get_mapping_for(&state, &index).await?))
}

pub async fn mapping_all(State(state): State<Arc<AppState>>) -> ESResult<Json<Value>> {
  let listing = state.client().list_indexes().await?;
  let names = listing
    .get("indexes")
    .and_then(Value::as_array)
    .map(|items| {
      items
        .iter()
        .filter_map(|item| item.get("name").and_then(Value::as_str).map(str::to_string))
        .collect::<Vec<_>>()
    })
    .unwrap_or_default();

  let mut out = serde_json::Map::new();
  for name in names {
    let single = get_mapping_for(&state, &name).await?;
    if let Some(entry) = single.get(&name) {
      out.insert(name, entry.clone());
    }
  }
  Ok(Json(Value::Object(out)))
}

pub async fn get_settings(
  State(_state): State<Arc<AppState>>,
  Path(index): Path<String>,
) -> Json<Value> {
  Json(settings_payload(&index))
}

pub async fn aliases(State(state): State<Arc<AppState>>) -> ESResult<Json<Value>> {
  // Best-effort: resolve aliases from upstream listing if available.
  let listing = state.client().list_indexes().await?;
  let aliases = listing
    .get("aliases")
    .and_then(Value::as_array)
    .cloned()
    .unwrap_or_default();
  let mut by_index: serde_json::Map<String, Value> = serde_json::Map::new();
  for alias in aliases {
    let alias_name = alias
      .get("alias")
      .and_then(Value::as_str)
      .unwrap_or("")
      .to_string();
    let target = alias
      .get("target")
      .and_then(Value::as_str)
      .unwrap_or("")
      .to_string();
    if alias_name.is_empty() || target.is_empty() {
      continue;
    }
    let entry = by_index
      .entry(target)
      .or_insert_with(|| json!({ "aliases": {} }));
    if let Some(map) = entry.get_mut("aliases").and_then(Value::as_object_mut) {
      map.insert(alias_name, json!({}));
    }
  }
  Ok(Json(Value::Object(by_index)))
}

pub fn settings_payload(index: &str) -> Value {
  json!({
    index: {
      "settings": {
        "index": {
          "number_of_shards": "1",
          "number_of_replicas": "0",
          "provided_name": index,
          "creation_date": "0",
          "uuid": "_na_",
          "version": { "created": "8110000" },
        }
      }
    }
  })
}

async fn get_mapping_for(state: &Arc<AppState>, index: &str) -> ESResult<Value> {
  let inspect = state.client().inspect(index).await?;
  let schema = inspect
    .get("manifest")
    .and_then(|m| m.get("schema"))
    .ok_or_else(|| {
      ESError::internal(
        "internal_server_error",
        "upstream inspect response missing manifest.schema",
      )
    })?;
  Ok(schema_to_es(index, schema)?)
}

async fn index_exists(state: &Arc<AppState>, index: &str) -> ESResult<bool> {
  let listing = state.client().list_indexes().await?;
  let known = listing
    .get("indexes")
    .and_then(Value::as_array)
    .map(|items| {
      items
        .iter()
        .any(|item| item.get("name").and_then(Value::as_str) == Some(index))
    })
    .unwrap_or(false);
  if known {
    return Ok(true);
  }
  let alias_match = listing
    .get("aliases")
    .and_then(Value::as_array)
    .map(|items| {
      items
        .iter()
        .any(|item| item.get("alias").and_then(Value::as_str) == Some(index))
    })
    .unwrap_or(false);
  Ok(alias_match)
}
