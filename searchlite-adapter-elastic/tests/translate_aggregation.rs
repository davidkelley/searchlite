use searchlite_adapter_elastic::translate::translate_aggs;
use serde_json::{json, Map};

fn aggs(body: serde_json::Value) -> Map<String, serde_json::Value> {
  body.as_object().unwrap().clone()
}

#[test]
fn terms_agg_round_trips_field_and_size() {
  let es = aggs(json!({
    "by_category": { "terms": { "field": "category", "size": 5 } }
  }));
  let sl = translate_aggs(&es).unwrap();
  assert_eq!(
    sl.get("by_category").unwrap(),
    &json!({ "type": "terms", "field": "category", "size": 5 })
  );
}

#[test]
fn date_histogram_translates_calendar_interval() {
  let es = aggs(json!({
    "by_day": { "date_histogram": { "field": "ts", "calendar_interval": "day" } }
  }));
  let sl = translate_aggs(&es).unwrap();
  let translated = sl.get("by_day").unwrap();
  assert_eq!(translated.get("type").unwrap(), &json!("date_histogram"));
  assert_eq!(translated.get("field").unwrap(), &json!("ts"));
  assert_eq!(translated.get("calendar_interval").unwrap(), &json!("day"));
}

#[test]
fn range_agg_carries_keyed_default_false() {
  let es = aggs(json!({
    "by_price": {
      "range": {
        "field": "price",
        "ranges": [{"to": 10}, {"from": 10, "to": 100}, {"from": 100}]
      }
    }
  }));
  let sl = translate_aggs(&es).unwrap();
  let translated = sl.get("by_price").unwrap();
  assert_eq!(translated.get("type").unwrap(), &json!("range"));
  assert_eq!(translated.get("keyed").unwrap(), &json!(false));
  let ranges = translated.get("ranges").unwrap().as_array().unwrap();
  assert_eq!(ranges.len(), 3);
}

#[test]
fn stats_metric_agg() {
  let es = aggs(json!({ "p": { "stats": { "field": "price" } } }));
  let sl = translate_aggs(&es).unwrap();
  assert_eq!(
    sl.get("p").unwrap(),
    &json!({ "type": "stats", "field": "price", "missing": null })
  );
}

#[test]
fn nested_sub_aggs_recurse() {
  let es = aggs(json!({
    "by_category": {
      "terms": { "field": "category" },
      "aggs": {
        "avg_price": { "stats": { "field": "price" } }
      }
    }
  }));
  let sl = translate_aggs(&es).unwrap();
  let outer = sl.get("by_category").unwrap();
  let inner = outer.get("aggs").unwrap();
  assert_eq!(
    inner.get("avg_price").unwrap(),
    &json!({ "type": "stats", "field": "price", "missing": null })
  );
}

#[test]
fn unsupported_avg_returns_helpful_error() {
  let es = aggs(json!({ "avg_price": { "avg": { "field": "price" } } }));
  let err = translate_aggs(&es).unwrap_err();
  assert!(err.detail.contains("stats"), "got {err:?}");
}

#[test]
fn cardinality_translates() {
  let es = aggs(json!({ "uniq": { "cardinality": { "field": "user_id" } } }));
  let sl = translate_aggs(&es).unwrap();
  let translated = sl.get("uniq").unwrap();
  assert_eq!(translated.get("type").unwrap(), &json!("cardinality"));
  assert_eq!(translated.get("field").unwrap(), &json!("user_id"));
}
