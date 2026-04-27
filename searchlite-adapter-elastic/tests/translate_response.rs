use searchlite_adapter_elastic::translate::translate_search_response;
use serde_json::json;

#[test]
fn empty_result_has_zero_hits_and_shards_envelope() {
  let sl = json!({
    "total_hits_estimate": 0,
    "hits": [],
    "aggregations": {},
  });
  let es = translate_search_response("books", &sl, 12);
  assert_eq!(es.get("took").unwrap(), &json!(12));
  assert_eq!(es.get("timed_out").unwrap(), &json!(false));
  assert_eq!(
    es.get("_shards").unwrap(),
    &json!({"total": 1, "successful": 1, "skipped": 0, "failed": 0})
  );
  let hits = es.get("hits").unwrap();
  assert_eq!(
    hits.get("total").unwrap(),
    &json!({"value": 0, "relation": "gte"})
  );
  assert_eq!(hits.get("hits").unwrap(), &json!([]));
}

#[test]
fn hits_have_index_id_score_source_fields() {
  let sl = json!({
    "total_hits_estimate": 2,
    "hits": [
      { "doc_id": "a", "score": 1.5, "fields": {"title": "rust"} },
      { "doc_id": "b", "score": 0.9, "fields": {"title": "search"} },
    ],
  });
  let es = translate_search_response("books", &sl, 0);
  let hits_arr = es
    .get("hits")
    .unwrap()
    .get("hits")
    .unwrap()
    .as_array()
    .unwrap();
  assert_eq!(hits_arr.len(), 2);
  let first = &hits_arr[0];
  assert_eq!(first.get("_index").unwrap(), &json!("books"));
  assert_eq!(first.get("_id").unwrap(), &json!("a"));
  assert_eq!(first.get("_score").unwrap(), &json!(1.5));
  assert_eq!(first.get("_source").unwrap(), &json!({"title": "rust"}));
  assert_eq!(
    es.get("hits").unwrap().get("max_score").unwrap(),
    &json!(1.5)
  );
}

#[test]
fn aggregations_terms_buckets_translated() {
  let sl = json!({
    "total_hits_estimate": 10,
    "hits": [],
    "aggregations": {
      "by_cat": {
        "type": "terms",
        "buckets": [
          {"key": "books", "doc_count": 7},
          {"key": "music", "doc_count": 3},
        ]
      }
    }
  });
  let es = translate_search_response("idx", &sl, 0);
  let aggs = es.get("aggregations").unwrap();
  let by_cat = aggs.get("by_cat").unwrap();
  let buckets = by_cat.get("buckets").unwrap().as_array().unwrap();
  assert_eq!(buckets.len(), 2);
  assert_eq!(buckets[0].get("key").unwrap(), &json!("books"));
  assert_eq!(buckets[0].get("doc_count").unwrap(), &json!(7));
  assert_eq!(buckets[0].get("key_as_string").unwrap(), &json!("books"));
}

#[test]
fn stats_aggregation_translates_to_es_envelope() {
  let sl = json!({
    "total_hits_estimate": 5,
    "hits": [],
    "aggregations": {
      "p": {
        "type": "stats",
        "count": 5,
        "min": 1.0,
        "max": 10.0,
        "sum": 25.0,
        "avg": 5.0
      }
    }
  });
  let es = translate_search_response("idx", &sl, 0);
  let aggs = es.get("aggregations").unwrap();
  assert_eq!(
    aggs.get("p").unwrap(),
    &json!({"count": 5, "min": 1.0, "max": 10.0, "sum": 25.0, "avg": 5.0})
  );
}
