use searchlite_adapter_elastic::translate::query::{translate_query, translate_to_filter};
use serde_json::json;

#[test]
fn match_all_translates_to_match_all() {
  let es = json!({"match_all": {}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(sl, json!({"type": "match_all"}));
}

#[test]
fn match_all_carries_boost() {
  let es = json!({"match_all": {"boost": 2.5}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(sl, json!({"type": "match_all", "boost": 2.5}));
}

#[test]
fn match_with_string_value_uses_query_string() {
  let es = json!({"match": {"title": "rust"}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({"type": "query_string", "query": "rust", "fields": ["title"]})
  );
}

#[test]
fn match_with_object_value_picks_query_field() {
  let es = json!({"match": {"title": {"query": "rust", "boost": 1.5}}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "query_string",
      "query": "rust",
      "fields": ["title"],
      "boost": 1.5,
    })
  );
}

#[test]
fn term_with_string_value() {
  let es = json!({"term": {"status": "active"}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({"type": "term", "field": "status", "value": "active"})
  );
}

#[test]
fn term_with_object_value_and_boost() {
  let es = json!({"term": {"status": {"value": "active", "boost": 2.0}}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({"type": "term", "field": "status", "value": "active", "boost": 2.0})
  );
}

#[test]
fn terms_translates_to_should_bool() {
  let es = json!({"terms": {"tag": ["rust", "search"]}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "bool",
      "should": [
        {"type": "term", "field": "tag", "value": "rust"},
        {"type": "term", "field": "tag", "value": "search"},
      ],
      "minimum_should_match": 1,
    })
  );
}

#[test]
fn prefix_translates_directly() {
  let es = json!({"prefix": {"name": "ja"}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({"type": "prefix", "field": "name", "value": "ja"})
  );
}

#[test]
fn wildcard_translates_directly() {
  let es = json!({"wildcard": {"name": "j*n"}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({"type": "wildcard", "field": "name", "value": "j*n"})
  );
}

#[test]
fn regexp_translates_to_regex() {
  let es = json!({"regexp": {"name": "j.*n"}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({"type": "regex", "field": "name", "value": "j.*n"})
  );
}

#[test]
fn match_phrase_string_form() {
  let es = json!({"match_phrase": {"title": "to be or not to be"}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "phrase",
      "field": "title",
      "terms": ["to", "be", "or", "not", "to", "be"],
    })
  );
}

#[test]
fn match_phrase_object_form_with_slop() {
  let es = json!({"match_phrase": {"title": {"query": "rust safety", "slop": 2}}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "phrase",
      "field": "title",
      "terms": ["rust", "safety"],
      "slop": 2,
    })
  );
}

#[test]
fn range_wraps_in_constant_score() {
  let es = json!({"range": {"price": {"gte": 10, "lt": 100}}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "constant_score",
      "filter": { "I64Range": { "field": "price", "min": 10, "max": 99 } },
    })
  );
}

#[test]
fn range_picks_f64_when_floats() {
  let es = json!({"range": {"score": {"gte": 0.5, "lte": 0.9}}});
  let sl = translate_query(&es).unwrap();
  let inner = sl.get("filter").unwrap();
  assert_eq!(
    inner,
    &json!({ "F64Range": { "field": "score", "min": 0.5, "max": 0.9 } })
  );
}

#[test]
fn bool_translates_recursively() {
  let es = json!({
    "bool": {
      "must": [{"term": {"a": "1"}}],
      "should": [{"term": {"b": "2"}}],
      "must_not": [{"term": {"c": "3"}}],
      "filter": [{"term": {"d": "4"}}],
      "minimum_should_match": 1,
    }
  });
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "bool",
      "must": [{"type": "term", "field": "a", "value": "1"}],
      "should": [{"type": "term", "field": "b", "value": "2"}],
      "must_not": [{"type": "term", "field": "c", "value": "3"}],
      "filter": [{ "KeywordEq": { "field": "d", "value": "4" } }],
      "minimum_should_match": 1,
    })
  );
}

#[test]
fn multi_match_translates_with_field_boost_parsing() {
  let es = json!({
    "multi_match": {
      "query": "rust",
      "fields": ["title^2", "body"],
      "type": "best_fields",
    }
  });
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "multi_match",
      "query": "rust",
      "fields": [
        {"field": "title", "boost": 2.0},
        {"field": "body"},
      ],
      "match_type": "best_fields",
    })
  );
}

#[test]
fn query_string_passes_through() {
  let es = json!({"query_string": {"query": "title:rust OR body:safety"}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "query_string",
      "query": "title:rust OR body:safety",
    })
  );
}

#[test]
fn dis_max_translates() {
  let es = json!({
    "dis_max": {
      "queries": [
        {"term": {"a": "x"}},
        {"term": {"b": "y"}},
      ],
      "tie_breaker": 0.3,
    }
  });
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "dis_max",
      "queries": [
        {"type": "term", "field": "a", "value": "x"},
        {"type": "term", "field": "b", "value": "y"},
      ],
      "tie_breaker": 0.3,
    })
  );
}

#[test]
fn unsupported_clause_returns_error() {
  let es = json!({"geo_distance": {"point": {"lat": 0.0, "lon": 0.0}}});
  let err = translate_query(&es).unwrap_err();
  assert!(err.feature.contains("geo_distance"), "got {err:?}");
}

#[test]
fn term_in_filter_context_emits_keyword_eq() {
  let es = json!({"term": {"status": "active"}});
  let sl = translate_to_filter(&es).unwrap();
  assert_eq!(
    sl,
    json!({ "KeywordEq": { "field": "status", "value": "active" } })
  );
}

#[test]
fn terms_in_filter_context_emits_keyword_in() {
  let es = json!({"terms": {"tag": ["a", "b"]}});
  let sl = translate_to_filter(&es).unwrap();
  assert_eq!(
    sl,
    json!({ "KeywordIn": { "field": "tag", "values": ["a", "b"] } })
  );
}

#[test]
fn bool_filter_context_with_must_not_emits_not() {
  let es = json!({
    "bool": {
      "must": [{"term": {"a": "1"}}],
      "must_not": [{"term": {"b": "2"}}],
    }
  });
  let sl = translate_to_filter(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "And": [
        { "KeywordEq": { "field": "a", "value": "1" } },
        { "Not": { "KeywordEq": { "field": "b", "value": "2" } } },
      ]
    })
  );
}

// ── Numeric `term` handling ─────────────────────────────────────────────

#[test]
fn term_with_integer_emits_constant_score_with_i64range() {
  let es = json!({"term": {"price": 25}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "constant_score",
      "filter": { "I64Range": { "field": "price", "min": 25, "max": 25 } }
    })
  );
}

#[test]
fn term_with_float_emits_constant_score_with_f64range() {
  let es = json!({"term": {"rating": 4.5}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "constant_score",
      "filter": { "F64Range": { "field": "rating", "min": 4.5, "max": 4.5 } }
    })
  );
}

#[test]
fn term_with_integer_in_filter_context_emits_i64range() {
  let es = json!({"term": {"price": 25}});
  let sl = translate_to_filter(&es).unwrap();
  assert_eq!(
    sl,
    json!({ "I64Range": { "field": "price", "min": 25, "max": 25 } })
  );
}

#[test]
fn term_with_float_in_filter_context_emits_f64range() {
  let es = json!({"term": {"rating": 4.5}});
  let sl = translate_to_filter(&es).unwrap();
  assert_eq!(
    sl,
    json!({ "F64Range": { "field": "rating", "min": 4.5, "max": 4.5 } })
  );
}

#[test]
fn term_with_object_integer_value_carries_boost() {
  let es = json!({"term": {"price": {"value": 25, "boost": 2.0}}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "constant_score",
      "filter": { "I64Range": { "field": "price", "min": 25, "max": 25 } },
      "boost": 2.0
    })
  );
}

// ── Validation: terms filter rejects multi-field input ─────────────────

#[test]
fn terms_in_filter_context_rejects_multiple_field_entries() {
  let es = json!({"terms": {"a": ["x"], "b": ["y"]}});
  let err = translate_to_filter(&es).unwrap_err();
  assert!(err.feature == "terms", "got {err:?}");
  assert!(err.detail.contains("exactly one"), "got {err:?}");
}
