use serde_json::{json, Value};

/// Minimal `GET /` banner mimicking Elasticsearch's root response so that
/// official ES clients accept the connection.
pub fn version_banner(version: &str) -> Value {
  json!({
    "name": "searchlite-adapter-elastic",
    "cluster_name": "searchlite",
    "cluster_uuid": "_na_",
    "version": {
      "number": version,
      "build_flavor": "default",
      "build_type": "binary",
      "build_hash": "searchlite",
      "build_date": "1970-01-01T00:00:00Z",
      "build_snapshot": false,
      "lucene_version": "9.0.0",
      "minimum_wire_compatibility_version": "7.17.0",
      "minimum_index_compatibility_version": "7.0.0",
    },
    "tagline": "You Know, for Search",
  })
}

/// `GET /_cluster/health` — green when upstream is reachable; advertises one
/// active primary so the response is internally consistent (Kibana / SDK
/// probes warn on green-with-zero-shards). When the upstream is unhealthy we
/// return red and keep the shard counts at zero so callers can distinguish
/// "fine but empty" from "broken".
pub fn cluster_health(upstream_healthy: bool) -> Value {
  let status = if upstream_healthy { "green" } else { "red" };
  let active = if upstream_healthy { 1 } else { 0 };
  json!({
    "cluster_name": "searchlite",
    "status": status,
    "timed_out": false,
    "number_of_nodes": 1,
    "number_of_data_nodes": 1,
    "active_primary_shards": active,
    "active_shards": active,
    "relocating_shards": 0,
    "initializing_shards": 0,
    "unassigned_shards": 0,
    "delayed_unassigned_shards": 0,
    "number_of_pending_tasks": 0,
    "number_of_in_flight_fetch": 0,
    "task_max_waiting_in_queue_millis": 0,
    "active_shards_percent_as_number": 100.0,
  })
}

/// `GET /_nodes` — minimal stub so ES clients that probe topology don't crash.
pub fn nodes_stub(version: &str) -> Value {
  json!({
    "_nodes": { "total": 1, "successful": 1, "failed": 0 },
    "cluster_name": "searchlite",
    "nodes": {
      "searchlite": {
        "name": "searchlite",
        "version": version,
        "roles": ["data", "master", "ingest"],
        "attributes": {},
      }
    }
  })
}
