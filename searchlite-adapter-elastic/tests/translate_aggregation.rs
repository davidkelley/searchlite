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

#[test]
fn top_hits_translates_with_size_from_sort() {
  let es = aggs(json!({
    "hits": {
      "top_hits": {
        "size": 5,
        "from": 1,
        "sort": [{ "price": "desc" }]
      }
    }
  }));
  let sl = translate_aggs(&es).unwrap();
  let translated = sl.get("hits").unwrap();
  assert_eq!(translated.get("type").unwrap(), &json!("top_hits"));
  assert_eq!(translated.get("size").unwrap(), &json!(5));
  assert_eq!(translated.get("from").unwrap(), &json!(1));
  let sort = translated.get("sort").unwrap().as_array().unwrap();
  assert_eq!(sort.len(), 1);
  assert_eq!(sort[0].get("field").unwrap(), &json!("price"));
  assert_eq!(sort[0].get("order").unwrap(), &json!("desc"));
}

#[test]
fn top_hits_uses_default_size_when_omitted() {
  let es = aggs(json!({ "hits": { "top_hits": {} } }));
  let sl = translate_aggs(&es).unwrap();
  let translated = sl.get("hits").unwrap();
  assert_eq!(translated.get("size").unwrap(), &json!(3));
  assert_eq!(translated.get("from").unwrap(), &json!(0));
}

#[test]
fn composite_agg_translates_terms_source() {
  let es = aggs(json!({
    "by_cat_then_brand": {
      "composite": {
        "size": 100,
        "sources": [
          { "cat": { "terms": { "field": "category" } } }
        ]
      }
    }
  }));
  let sl = translate_aggs(&es).unwrap();
  let translated = sl.get("by_cat_then_brand").unwrap();
  assert_eq!(translated.get("type").unwrap(), &json!("composite"));
  assert_eq!(translated.get("size").unwrap(), &json!(100));
  let sources = translated.get("sources").unwrap().as_array().unwrap();
  assert_eq!(sources.len(), 1);
  assert_eq!(sources[0].get("type").unwrap(), &json!("terms"));
  assert_eq!(sources[0].get("name").unwrap(), &json!("cat"));
  assert_eq!(sources[0].get("field").unwrap(), &json!("category"));
}

#[test]
fn composite_agg_with_histogram_source_carries_interval() {
  let es = aggs(json!({
    "by_price_bucket": {
      "composite": {
        "size": 50,
        "sources": [
          { "p": { "histogram": { "field": "price", "interval": 10.0 } } }
        ]
      }
    }
  }));
  let sl = translate_aggs(&es).unwrap();
  let sources = sl
    .get("by_price_bucket")
    .unwrap()
    .get("sources")
    .unwrap()
    .as_array()
    .unwrap();
  assert_eq!(sources[0].get("type").unwrap(), &json!("histogram"));
  assert_eq!(sources[0].get("interval").unwrap(), &json!(10.0));
}

#[test]
fn composite_agg_propagates_after_key() {
  let es = aggs(json!({
    "page2": {
      "composite": {
        "size": 10,
        "after": { "cat": "books" },
        "sources": [
          { "cat": { "terms": { "field": "category" } } }
        ]
      }
    }
  }));
  let sl = translate_aggs(&es).unwrap();
  assert_eq!(
    sl.get("page2").unwrap().get("after").unwrap(),
    &json!({ "cat": "books" })
  );
}

#[test]
fn pipeline_agg_avg_bucket_carries_buckets_path() {
  let es = aggs(json!({
    "avg_per_day": {
      "avg_bucket": { "buckets_path": "by_day>price.avg" }
    }
  }));
  let sl = translate_aggs(&es).unwrap();
  let translated = sl.get("avg_per_day").unwrap();
  assert_eq!(translated.get("type").unwrap(), &json!("avg_bucket"));
  assert_eq!(
    translated.get("buckets_path").unwrap(),
    &json!("by_day>price.avg")
  );
}

#[test]
fn pipeline_agg_sum_bucket_translates() {
  let es = aggs(json!({
    "sum_per_day": {
      "sum_bucket": { "buckets_path": "by_day>price.sum" }
    }
  }));
  let sl = translate_aggs(&es).unwrap();
  assert_eq!(
    sl.get("sum_per_day").unwrap().get("type").unwrap(),
    &json!("sum_bucket")
  );
}

#[test]
fn pipeline_agg_derivative_translates() {
  let es = aggs(json!({
    "deriv": { "derivative": { "buckets_path": "by_day>doc_count" } }
  }));
  let sl = translate_aggs(&es).unwrap();
  assert_eq!(
    sl.get("deriv").unwrap().get("type").unwrap(),
    &json!("derivative")
  );
  assert_eq!(
    sl.get("deriv").unwrap().get("buckets_path").unwrap(),
    &json!("by_day>doc_count")
  );
}

#[test]
fn pipeline_agg_moving_avg_carries_window_and_predict() {
  let es = aggs(json!({
    "smoothed": {
      "moving_avg": {
        "buckets_path": "by_day>doc_count",
        "window": 5,
        "model": "simple",
        "predict": 2
      }
    }
  }));
  let sl = translate_aggs(&es).unwrap();
  let translated = sl.get("smoothed").unwrap();
  assert_eq!(translated.get("type").unwrap(), &json!("moving_avg"));
  assert_eq!(translated.get("window").unwrap(), &json!(5));
  assert_eq!(translated.get("model").unwrap(), &json!("simple"));
  assert_eq!(translated.get("predict").unwrap(), &json!(2));
}

#[test]
fn pipeline_agg_bucket_script_passes_through_paths_and_script() {
  let es = aggs(json!({
    "ratio": {
      "bucket_script": {
        "buckets_path": { "a": "x>v", "b": "y>v" },
        "script": "params.a / params.b"
      }
    }
  }));
  let sl = translate_aggs(&es).unwrap();
  let translated = sl.get("ratio").unwrap();
  assert_eq!(translated.get("type").unwrap(), &json!("bucket_script"));
  assert_eq!(
    translated.get("buckets_path").unwrap(),
    &json!({ "a": "x>v", "b": "y>v" })
  );
  assert_eq!(
    translated.get("script").unwrap(),
    &json!("params.a / params.b")
  );
}

#[test]
fn bucket_sort_pipeline_agg_carries_from_size_and_sort() {
  let es = aggs(json!({
    "sorted": {
      "bucket_sort": {
        "from": 0,
        "size": 5,
        "sort": [{ "doc_count": "desc" }]
      }
    }
  }));
  let sl = translate_aggs(&es).unwrap();
  let translated = sl.get("sorted").unwrap();
  assert_eq!(translated.get("type").unwrap(), &json!("bucket_sort"));
  assert_eq!(translated.get("from").unwrap(), &json!(0));
  assert_eq!(translated.get("size").unwrap(), &json!(5));
  let sort = translated.get("sort").unwrap().as_array().unwrap();
  assert_eq!(sort[0].get("field").unwrap(), &json!("doc_count"));
  assert_eq!(sort[0].get("order").unwrap(), &json!("desc"));
}

#[test]
fn unsupported_avg_bucket_missing_buckets_path_rejected() {
  let es = aggs(json!({ "x": { "avg_bucket": {} } }));
  let err = translate_aggs(&es).unwrap_err();
  assert!(err.feature.contains("avg_bucket"), "got {err:?}");
}
