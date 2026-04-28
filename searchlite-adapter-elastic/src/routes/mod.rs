use std::sync::Arc;

use axum::routing::{any, get, head, post, put};
use axum::Router;

use crate::state::AppState;

pub mod cluster;
pub mod indices;
pub mod mget;
pub mod msearch;
pub mod reject;
pub mod search;

pub fn router(state: Arc<AppState>) -> Router {
  Router::new()
    .route("/", get(cluster::root))
    .route("/_cluster/health", get(cluster::cluster_health))
    .route("/_cluster/state", get(cluster::cluster_state))
    .route("/_nodes", get(cluster::nodes))
    .route("/_nodes/stats", get(cluster::nodes))
    .route("/_mapping", get(indices::mapping_all))
    .route(
      "/_search",
      post(reject::cross_index_search).get(reject::cross_index_search),
    )
    .route("/_msearch", post(msearch::msearch))
    .route("/_mget", post(mget::mget_global))
    .route("/_bulk", post(reject::write_not_supported))
    .route(
      "/_aliases",
      post(reject::write_not_supported).get(indices::aliases),
    )
    .route(
      "/{index}",
      head(indices::head_index)
        .get(indices::get_index)
        .post(reject::write_not_supported)
        .put(reject::write_not_supported)
        .delete(reject::write_not_supported),
    )
    .route(
      "/{index}/_mapping",
      get(indices::get_mapping).put(reject::write_not_supported),
    )
    .route("/{index}/_settings", get(indices::get_settings))
    .route("/{index}/_aliases", get(indices::aliases))
    .route("/{index}/_alias", get(indices::aliases))
    .route("/{index}/_search", post(search::search).get(search::search))
    .route("/{index}/_count", post(search::count).get(search::count))
    .route("/{index}/_mget", post(mget::mget))
    .route("/{index}/_msearch", post(msearch::msearch))
    .route("/{index}/_refresh", any(reject::write_not_supported))
    .route("/{index}/_bulk", post(reject::write_not_supported))
    .route(
      "/{index}/_doc",
      post(reject::write_not_supported).put(reject::write_not_supported),
    )
    .route(
      "/{index}/_doc/{id}",
      post(reject::write_not_supported)
        .put(reject::write_not_supported)
        .delete(reject::write_not_supported)
        .get(reject::doc_get_not_supported),
    )
    .route("/{index}/_update/{id}", post(reject::write_not_supported))
    .route(
      "/{index}/_create/{id}",
      put(reject::write_not_supported).post(reject::write_not_supported),
    )
    .route(
      "/{index}/_delete_by_query",
      post(reject::write_not_supported),
    )
    .route(
      "/{index}/_update_by_query",
      post(reject::write_not_supported),
    )
    .with_state(state)
}
