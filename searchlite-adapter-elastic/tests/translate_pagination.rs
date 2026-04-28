use searchlite_adapter_elastic::translate::apply_pagination;
use serde_json::{json, Map, Value};

fn run(es_body: Value) -> Map<String, Value> {
  let mut out = Map::new();
  let map = es_body.as_object().unwrap().clone();
  apply_pagination(&map, &mut out).unwrap();
  out
}

#[test]
fn from_size_search_after_pass_through() {
  let out = run(json!({ "from": 5, "size": 20, "search_after": ["sort-key"] }));
  assert_eq!(out.get("from").unwrap(), &json!(5));
  assert_eq!(out.get("limit").unwrap(), &json!(20));
  assert_eq!(out.get("search_after").unwrap(), &json!(["sort-key"]));
}

#[test]
fn track_total_hits_true_is_forwarded() {
  let out = run(json!({ "track_total_hits": true }));
  assert_eq!(out.get("track_total_hits").unwrap(), &json!(true));
}

#[test]
fn track_total_hits_false_is_forwarded() {
  let out = run(json!({ "track_total_hits": false }));
  assert_eq!(out.get("track_total_hits").unwrap(), &json!(false));
}

#[test]
fn track_total_hits_positive_int_maps_to_true() {
  let out = run(json!({ "track_total_hits": 10000 }));
  assert_eq!(out.get("track_total_hits").unwrap(), &json!(true));
}

#[test]
fn track_total_hits_zero_int_maps_to_false() {
  let out = run(json!({ "track_total_hits": 0 }));
  assert_eq!(out.get("track_total_hits").unwrap(), &json!(false));
}

#[test]
fn track_total_hits_invalid_type_rejected() {
  let mut out = Map::new();
  let body = json!({ "track_total_hits": "true" });
  let map = body.as_object().unwrap().clone();
  let err = apply_pagination(&map, &mut out).unwrap_err();
  assert_eq!(err.feature, "track_total_hits");
}

#[test]
fn scroll_is_rejected() {
  let mut out = Map::new();
  let body = json!({ "scroll": "1m" });
  let map = body.as_object().unwrap().clone();
  let err = apply_pagination(&map, &mut out).unwrap_err();
  assert_eq!(err.feature, "scroll");
}
