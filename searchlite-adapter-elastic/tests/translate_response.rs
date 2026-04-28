use searchlite_adapter_elastic::translate::translate_search_response;
use serde_json::json;

fn translate(sl: &serde_json::Value) -> serde_json::Value {
  // Default-tracking helper for legacy tests that don't care about
  // track_total_hits semantics (treated as unset).
  translate_search_response("idx", sl, 0, None)
}

#[test]
fn empty_result_has_zero_hits_and_shards_envelope() {
  let sl = json!({
    "total_hits_estimate": 0,
    "hits": [],
    "aggregations": {},
  });
  let es = translate_search_response("books", &sl, 12, None);
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

// --- track_total_hits semantics ---------------------------------------------

#[test]
fn track_total_hits_true_emits_relation_eq() {
  // Regression: response always emitted relation:"gte" regardless of the
  // request's track_total_hits. When the caller asked for exact totals,
  // ES emits relation:"eq".
  let sl = json!({
    "total_hits_estimate": 42,
    "hits": [],
  });
  let es = translate_search_response("idx", &sl, 0, Some(true));
  assert_eq!(
    es.pointer("/hits/total").unwrap(),
    &json!({"value": 42, "relation": "eq"})
  );
}

#[test]
fn track_total_hits_false_omits_total() {
  // ES omits hits.total entirely when track_total_hits=false, signaling
  // that totals were not computed. Clients that rely on the presence of
  // `total` to distinguish exact-vs-approximate must not see a fabricated
  // value here.
  let sl = json!({
    "total_hits_estimate": 17,
    "hits": [],
  });
  let es = translate_search_response("idx", &sl, 0, Some(false));
  assert!(
    es.pointer("/hits/total").is_none(),
    "hits.total should be omitted when track_total_hits=false; got: {es}"
  );
  // The hits array should still be present.
  assert!(es.pointer("/hits/hits").is_some());
}

#[test]
fn track_total_hits_unset_keeps_relation_gte() {
  // No explicit signal from the caller → preserve historical default
  // (lower-bound semantics).
  let sl = json!({
    "total_hits_estimate": 9,
    "hits": [],
  });
  let es = translate_search_response("idx", &sl, 0, None);
  assert_eq!(
    es.pointer("/hits/total").unwrap(),
    &json!({"value": 9, "relation": "gte"})
  );
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
  let es = translate_search_response("books", &sl, 0, None);
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
fn keyed_range_aggregation_emits_object_buckets() {
  // Regression: ES returns range/date_range buckets as an object map keyed
  // by each bucket's `key` when `keyed: true` is requested. Previously the
  // adapter always emitted an array, breaking clients that destructure the
  // keyed shape.
  let sl = json!({
    "total_hits_estimate": 0,
    "hits": [],
    "aggregations": {
      "by_price": {
        "type": "range",
        "keyed": true,
        "buckets": [
          {"key": "cheap", "doc_count": 5},
          {"key": "premium", "doc_count": 3},
        ]
      }
    }
  });
  let es = translate(&sl);
  let buckets = es.pointer("/aggregations/by_price/buckets").unwrap();
  assert!(
    buckets.is_object(),
    "keyed range should emit object buckets, got: {buckets}"
  );
  assert_eq!(
    buckets.pointer("/cheap/doc_count").unwrap(),
    &json!(5),
    "got: {buckets}"
  );
  assert_eq!(buckets.pointer("/premium/doc_count").unwrap(), &json!(3));
  // The key should NOT appear inside the value (it's the outer map key).
  assert!(
    buckets.pointer("/cheap/key").is_none(),
    "key should not be duplicated inside the keyed bucket value"
  );
}

#[test]
fn unkeyed_range_aggregation_emits_array_buckets() {
  let sl = json!({
    "total_hits_estimate": 0,
    "hits": [],
    "aggregations": {
      "by_price": {
        "type": "range",
        "keyed": false,
        "buckets": [{"key": "cheap", "doc_count": 5}]
      }
    }
  });
  let es = translate(&sl);
  let buckets = es.pointer("/aggregations/by_price/buckets").unwrap();
  assert!(
    buckets.is_array(),
    "non-keyed range should emit array buckets"
  );
}

#[test]
fn keyed_date_range_aggregation_emits_object_buckets() {
  let sl = json!({
    "total_hits_estimate": 0,
    "hits": [],
    "aggregations": {
      "by_month": {
        "type": "date_range",
        "keyed": true,
        "buckets": [
          {"key": "2024-Q1", "doc_count": 100},
        ]
      }
    }
  });
  let es = translate(&sl);
  let buckets = es.pointer("/aggregations/by_month/buckets").unwrap();
  assert!(buckets.is_object(), "got: {buckets}");
  assert_eq!(buckets.pointer("/2024-Q1/doc_count").unwrap(), &json!(100));
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
  let es = translate(&sl);
  let aggs = es.get("aggregations").unwrap();
  let by_cat = aggs.get("by_cat").unwrap();
  let buckets = by_cat.get("buckets").unwrap().as_array().unwrap();
  assert_eq!(buckets.len(), 2);
  assert_eq!(buckets[0].get("key").unwrap(), &json!("books"));
  assert_eq!(buckets[0].get("doc_count").unwrap(), &json!(7));
  assert_eq!(buckets[0].get("key_as_string").unwrap(), &json!("books"));
}

#[test]
fn hit_with_only_snippet_emits_snippet_under_sentinel_key() {
  let sl = json!({
    "total_hits_estimate": 1,
    "hits": [{ "doc_id": "a", "score": 1.0, "snippet": "rust …safety" }],
  });
  let es = translate(&sl);
  let hits = es.pointer("/hits/hits").unwrap().as_array().unwrap();
  assert_eq!(
    hits[0].get("highlight").unwrap(),
    &json!({ "_snippet": ["rust …safety"] })
  );
}

#[test]
fn hit_with_only_highlights_emits_highlights_verbatim() {
  let sl = json!({
    "total_hits_estimate": 1,
    "hits": [{
      "doc_id": "a",
      "score": 1.0,
      "highlights": { "title": ["<em>rust</em> safety"] }
    }],
  });
  let es = translate(&sl);
  let hits = es.pointer("/hits/hits").unwrap().as_array().unwrap();
  assert_eq!(
    hits[0].get("highlight").unwrap(),
    &json!({ "title": ["<em>rust</em> safety"] })
  );
}

#[test]
fn hit_with_both_snippet_and_highlights_prefers_highlights_without_dropping_snippet() {
  // Regression for review feedback: previously the second `out.insert("highlight", …)`
  // silently overwrote the snippet entry. Either the structured `highlights` or the
  // legacy `snippet` is informative on its own; if both are present we keep the
  // structured one (richer) and surface the snippet alongside it under `_snippet`
  // so neither field is silently lost.
  let sl = json!({
    "total_hits_estimate": 1,
    "hits": [{
      "doc_id": "a",
      "score": 1.0,
      "snippet": "rust …safety",
      "highlights": { "title": ["<em>rust</em> safety"] }
    }],
  });
  let es = translate(&sl);
  let highlight = es.pointer("/hits/hits").unwrap().as_array().unwrap()[0]
    .get("highlight")
    .unwrap();
  assert_eq!(
    highlight.get("title").unwrap(),
    &json!(["<em>rust</em> safety"]),
    "structured highlights should be preserved"
  );
  assert_eq!(
    highlight.get("_snippet").unwrap(),
    &json!(["rust …safety"]),
    "legacy snippet should also be preserved alongside structured highlights"
  );
}

#[test]
fn hit_without_highlight_omits_highlight_field() {
  let sl = json!({
    "total_hits_estimate": 1,
    "hits": [{ "doc_id": "a", "score": 1.0 }],
  });
  let es = translate(&sl);
  let hit = &es.pointer("/hits/hits").unwrap().as_array().unwrap()[0];
  assert!(hit.get("highlight").is_none());
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
  let es = translate(&sl);
  let aggs = es.get("aggregations").unwrap();
  assert_eq!(
    aggs.get("p").unwrap(),
    &json!({"count": 5, "min": 1.0, "max": 10.0, "sum": 25.0, "avg": 5.0})
  );
}
