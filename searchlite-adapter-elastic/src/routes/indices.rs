use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use futures_util::stream::{self, StreamExt};
use serde_json::{json, Value};

use crate::error::{ESError, ESResult};
use crate::state::AppState;
use crate::translate::schema_to_es;

/// Maximum number of concurrent upstream `inspect` calls fired by
/// `mapping_all`. Bounded so a `_mapping` request against a cluster with
/// many indexes doesn't open an unbounded number of connections to the
/// upstream all at once.
const MAPPING_ALL_CONCURRENCY: usize = 8;

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
  // Resolve aliases up front so the response is keyed by the concrete
  // target index, matching ES semantics for alias-based requests.
  let resolved = resolve_index_or_alias(&state, &index)
    .await?
    .ok_or_else(|| {
      ESError::not_found(
        "index_not_found_exception",
        format!("no such index [{index}]"),
      )
    })?;

  let mapping = get_mapping_for(&state, &resolved).await?;
  let mappings = mapping
    .get(&resolved)
    .and_then(|v| v.get("mappings"))
    .cloned()
    .unwrap_or_else(|| json!({ "properties": {} }));
  let settings = settings_payload(&resolved);
  let settings_value = settings
    .get(&resolved)
    .and_then(|v| v.get("settings"))
    .cloned()
    .unwrap_or_else(|| json!({}));

  // Surface the actual aliases pointing at the resolved target instead of
  // a hard-coded empty object, so management / discovery flows see the
  // real topology.
  let aliases_map = aliases_by_index(&state).await?;
  let aliases_for_target = aliases_map
    .get(&resolved)
    .and_then(|entry| entry.get("aliases"))
    .cloned()
    .unwrap_or_else(|| json!({}));

  let mut payload = serde_json::Map::new();
  payload.insert(
    resolved,
    json!({
      "aliases": aliases_for_target,
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
  // Resolve aliases here for parity with HEAD /{index} and
  // GET /{index}/_settings — calling with an alias name should return the
  // target's mapping keyed by the target.
  let resolved = resolve_index_or_alias(&state, &index)
    .await?
    .ok_or_else(|| {
      ESError::not_found(
        "index_not_found_exception",
        format!("no such index [{index}]"),
      )
    })?;
  Ok(Json(get_mapping_for(&state, &resolved).await?))
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

  // Run upstream `inspect` calls concurrently with bounded parallelism.
  // The previous serialized loop made `_mapping` latency proportional to
  // the index count and was a real problem on clusters with many indexes
  // (Kibana, ES SDKs hit it on every connect). `buffer_unordered` keeps
  // at most MAPPING_ALL_CONCURRENCY in flight at a time.
  let results: Vec<(String, ESResult<Value>)> = stream::iter(names.into_iter())
    .map(|name| {
      let state = state.clone();
      async move {
        let result = get_mapping_for(&state, &name).await;
        (name, result)
      }
    })
    .buffer_unordered(MAPPING_ALL_CONCURRENCY)
    .collect()
    .await;

  let mut out = serde_json::Map::new();
  for (name, result) in results {
    let single = result?;
    if let Some(entry) = single.get(&name) {
      out.insert(name, entry.clone());
    }
  }
  Ok(Json(Value::Object(out)))
}

pub async fn get_settings(
  State(state): State<Arc<AppState>>,
  Path(index): Path<String>,
) -> ESResult<Json<Value>> {
  // Resolve the path token through the alias listing so the response is
  // keyed by the concrete target index, not by the alias name. ES does
  // the same — alias requests return `{<target>: {settings: ...}}`, never
  // `{<alias>: ...}`.
  let resolved = resolve_index_or_alias(&state, &index)
    .await?
    .ok_or_else(|| {
      ESError::not_found(
        "index_not_found_exception",
        format!("no such index [{index}]"),
      )
    })?;
  Ok(Json(settings_payload(&resolved)))
}

pub async fn aliases(State(state): State<Arc<AppState>>) -> ESResult<Json<Value>> {
  Ok(Json(Value::Object(aliases_by_index(&state).await?)))
}

/// Index-scoped alias endpoint (`GET /{index}/_aliases`,
/// `GET /{index}/_alias`). Three cases, matching Elasticsearch:
///
/// 1. `{index}` is an index with aliases pointing at it → return that entry
/// 2. `{index}` is an index with no aliases → return `{<index>: {aliases: {}}}`
/// 3. `{index}` is an alias name → return `{<target>: {aliases: {<index>: {}}}}`
///
/// 404 only when `{index}` resolves neither as an index nor as an alias.
pub async fn aliases_for_index(
  State(state): State<Arc<AppState>>,
  Path(index): Path<String>,
) -> ESResult<Json<Value>> {
  if !index_exists(&state, &index).await? {
    return Err(ESError::not_found(
      "index_not_found_exception",
      format!("no such index [{index}]"),
    ));
  }
  let all = aliases_by_index(&state).await?;
  let mut filtered = serde_json::Map::new();

  if let Some(entry) = all.get(&index) {
    filtered.insert(index.clone(), entry.clone());
    return Ok(Json(Value::Object(filtered)));
  }

  // `{index}` may be an alias name itself — find the index it targets.
  let mut found_as_alias = false;
  for (target, entry) in &all {
    if let Some(aliases) = entry.get("aliases").and_then(Value::as_object) {
      if aliases.contains_key(&index) {
        let mut narrowed = serde_json::Map::new();
        narrowed.insert(index.clone(), json!({}));
        filtered.insert(target.clone(), json!({ "aliases": narrowed }));
        found_as_alias = true;
      }
    }
  }

  if !found_as_alias {
    // Index exists but has no aliases — ES returns an empty alias entry.
    filtered.insert(index.clone(), json!({ "aliases": {} }));
  }
  Ok(Json(Value::Object(filtered)))
}

/// Build the `{ <index>: { aliases: { <alias>: {} } } }` map from upstream's
/// alias listing. Shared between the global and index-scoped handlers.
async fn aliases_by_index(state: &Arc<AppState>) -> ESResult<serde_json::Map<String, Value>> {
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
  Ok(by_index)
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

/// Resolve `name` to the concrete target index it refers to.
///
/// - If `name` is a real index, returns `Some(name)`.
/// - If `name` is an alias, returns `Some(<target index name>)`.
/// - Otherwise returns `None`.
///
/// Returns a single target deliberately. The upstream `searchlite-http`
/// stores aliases as a `HashMap<String, String>` and rejects duplicate
/// alias names at startup (`searchlite-http/src/lib.rs::IndexRegistry::from_args`),
/// so an alias can refer to at most one target index. The `find_map` over
/// `aliases[*]` below is correct under that contract — at most one entry
/// can match a given alias name. If the upstream ever loosens to N:1
/// fan-out (matching real Elasticsearch's alias semantics), `_mapping`,
/// `_settings`, and `get_index` would need to fan out across all targets
/// rather than picking one.
pub(crate) async fn resolve_index_or_alias(
  state: &Arc<AppState>,
  name: &str,
) -> ESResult<Option<String>> {
  let listing = state.client().list_indexes().await?;
  let is_real_index = listing
    .get("indexes")
    .and_then(Value::as_array)
    .map(|items| {
      items
        .iter()
        .any(|item| item.get("name").and_then(Value::as_str) == Some(name))
    })
    .unwrap_or(false);
  if is_real_index {
    return Ok(Some(name.to_string()));
  }
  let target = listing
    .get("aliases")
    .and_then(Value::as_array)
    .and_then(|items| {
      items.iter().find_map(|item| {
        if item.get("alias").and_then(Value::as_str) == Some(name) {
          item
            .get("target")
            .and_then(Value::as_str)
            .map(str::to_string)
        } else {
          None
        }
      })
    });
  Ok(target)
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
