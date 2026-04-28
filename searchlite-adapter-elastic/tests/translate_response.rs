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
fn hit_with_only_snippet_emits_snippet_under_sentinel_key() {
  let sl = json!({
    "total_hits_estimate": 1,
    "hits": [{ "doc_id": "a", "score": 1.0, "snippet": "rust …safety" }],
  });
  let es = translate_search_response("idx", &sl, 0);
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
  let es = translate_search_response("idx", &sl, 0);
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
  let es = translate_search_response("idx", &sl, 0);
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
  let es = translate_search_response("idx", &sl, 0);
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
  let es = translate_search_response("idx", &sl, 0);
  let aggs = es.get("aggregations").unwrap();
  assert_eq!(
    aggs.get("p").unwrap(),
    &json!({"count": 5, "min": 1.0, "max": 10.0, "sum": 25.0, "avg": 5.0})
  );
}
