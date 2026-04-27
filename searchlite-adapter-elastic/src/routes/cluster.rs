use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde_json::Value;

use crate::compat::stub;
use crate::state::AppState;

pub async fn root(State(state): State<Arc<AppState>>) -> Json<Value> {
  Json(stub::version_banner(&state.args().version_banner))
}

pub async fn cluster_health(State(state): State<Arc<AppState>>) -> Json<Value> {
  let upstream_healthy = state.client().healthz().await.is_ok();
  Json(stub::cluster_health(upstream_healthy))
}

pub async fn cluster_state(State(state): State<Arc<AppState>>) -> Json<Value> {
  Json(stub::nodes_stub(&state.args().version_banner))
}

pub async fn nodes(State(state): State<Arc<AppState>>) -> Json<Value> {
  Json(stub::nodes_stub(&state.args().version_banner))
}
