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
fn match_phrase_string_form_uses_query_string_for_analyzer_aware_tokenization() {
  // Slop=0 (default) is delegated to `query_string` with a quoted phrase so
  // SearchLite's per-field analyzer drives tokenization. Previously we split
  // on whitespace and emitted a `phrase` query — that diverged from ES on
  // any input with punctuation, contractions, or non-ASCII text.
  let es = json!({"match_phrase": {"title": "to be or not to be"}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "query_string",
      "query": "\"to be or not to be\"",
      "fields": ["title"],
    })
  );
}

#[test]
fn match_phrase_with_punctuation_is_quoted_intact_for_analyzer() {
  let es = json!({"match_phrase": {"title": "U.S.A. today"}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "query_string",
      "query": "\"U.S.A. today\"",
      "fields": ["title"],
    })
  );
}

#[test]
fn match_phrase_escapes_quote_and_backslash() {
  let es = json!({"match_phrase": {"title": "say \"hi\" \\here"}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(sl.get("query").unwrap(), &json!(r#""say \"hi\" \\here""#));
}

#[test]
fn match_phrase_with_slop_keeps_phrase_form_for_token_proximity() {
  // Slop > 0 controls token proximity, which `query_string` quoted phrases
  // don't model. Fall back to the literal-token `phrase` translation; the
  // tokenization-divergence limitation is documented for this path.
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
fn multi_match_phrase_type_rejection_does_not_recommend_a_substitute_with_different_semantics() {
  // Regression: previously the error said "use best_fields/most_fields/cross_fields",
  // but those are bag-of-words match modes — copying the suggestion would silently
  // change phrase semantics and return wrong results without warning.
  let es = json!({
    "multi_match": {
      "query": "exact phrase",
      "fields": ["title", "description"],
      "type": "phrase"
    }
  });
  let err = translate_query(&es).unwrap_err();
  let detail = err.detail.to_lowercase();
  assert!(
    !detail.contains("best_fields"),
    "rejection should not suggest a substitute that changes match semantics: {err:?}"
  );
  assert!(
    !detail.contains("most_fields") && !detail.contains("cross_fields"),
    "rejection should not suggest a substitute that changes match semantics: {err:?}"
  );
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

// ── Numeric `terms` handling (mainline + filter) ───────────────────────

#[test]
fn terms_with_integer_values_emits_constant_score_should_clauses() {
  let es = json!({"terms": {"price": [10, 20]}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "bool",
      "should": [
        { "type": "constant_score", "filter": { "I64Range": { "field": "price", "min": 10, "max": 10 } } },
        { "type": "constant_score", "filter": { "I64Range": { "field": "price", "min": 20, "max": 20 } } },
      ],
      "minimum_should_match": 1,
    })
  );
}

#[test]
fn terms_with_float_values_emits_f64range_should_clauses() {
  let es = json!({"terms": {"rating": [4.5, 4.8]}});
  let sl = translate_query(&es).unwrap();
  let shoulds = sl.get("should").unwrap().as_array().unwrap();
  assert_eq!(shoulds.len(), 2);
  assert_eq!(
    shoulds[0],
    json!({"type": "constant_score", "filter": {"F64Range": {"field": "rating", "min": 4.5, "max": 4.5}}})
  );
}

#[test]
fn terms_with_mixed_string_and_number_dispatches_per_value() {
  let es = json!({"terms": {"f": ["a", 1]}});
  let sl = translate_query(&es).unwrap();
  let shoulds = sl.get("should").unwrap().as_array().unwrap();
  assert_eq!(shoulds.len(), 2);
  assert_eq!(
    shoulds[0],
    json!({"type": "term", "field": "f", "value": "a"})
  );
  assert_eq!(
    shoulds[1],
    json!({"type": "constant_score", "filter": {"I64Range": {"field": "f", "min": 1, "max": 1}}})
  );
}

#[test]
fn terms_filter_all_strings_uses_keyword_in() {
  let es = json!({"terms": {"category": ["books", "music"]}});
  let sl = translate_to_filter(&es).unwrap();
  assert_eq!(
    sl,
    json!({ "KeywordIn": { "field": "category", "values": ["books", "music"] } })
  );
}

#[test]
fn terms_filter_all_integers_uses_or_of_i64ranges() {
  let es = json!({"terms": {"price": [10, 20]}});
  let sl = translate_to_filter(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "Or": [
        { "I64Range": { "field": "price", "min": 10, "max": 10 } },
        { "I64Range": { "field": "price", "min": 20, "max": 20 } },
      ]
    })
  );
}

#[test]
fn terms_filter_single_integer_emits_bare_filter() {
  let es = json!({"terms": {"price": [42]}});
  let sl = translate_to_filter(&es).unwrap();
  assert_eq!(
    sl,
    json!({ "I64Range": { "field": "price", "min": 42, "max": 42 } })
  );
}

#[test]
fn terms_with_field_literally_named_boost_is_queryable() {
  // Regression for review: `terms` translation filtered out any key named
  // `boost` so a schema field literally named `boost` was unqueryable. The
  // filter should only treat `boost` as the meta-key when its value is a
  // numeric boost — when it's an array it's a field selector.
  let es = json!({"terms": {"boost": ["a", "b"]}});
  let sl = translate_query(&es).unwrap();
  assert_eq!(
    sl,
    json!({
      "type": "bool",
      "should": [
        {"type": "term", "field": "boost", "value": "a"},
        {"type": "term", "field": "boost", "value": "b"},
      ],
      "minimum_should_match": 1,
    })
  );
}

#[test]
fn terms_filter_with_field_literally_named_boost_is_queryable() {
  let es = json!({"terms": {"boost": ["a"]}});
  let sl = translate_to_filter(&es).unwrap();
  assert_eq!(
    sl,
    json!({ "KeywordIn": { "field": "boost", "values": ["a"] } })
  );
}

#[test]
fn bool_minimum_should_match_percentage_resolves_to_floor() {
  // Regression: previously ANY non-integer minimum_should_match was rejected,
  // including ES's common `"75%"` form. Now we resolve it adapter-side
  // (count should clauses, floor multiply) since core's Bool variant only
  // accepts a usize.
  let es = json!({
    "bool": {
      "should": [
        {"term": {"a": "1"}},
        {"term": {"b": "2"}},
        {"term": {"c": "3"}},
        {"term": {"d": "4"}},
      ],
      "minimum_should_match": "75%"
    }
  });
  let sl = translate_query(&es).unwrap();
  // 4 should-clauses * 0.75 = 3.0 → floor = 3
  assert_eq!(sl.get("minimum_should_match").unwrap(), &json!(3));
}

#[test]
fn bool_minimum_should_match_percentage_floors_fractional() {
  // 3 clauses * 75% = 2.25 → floor = 2
  let es = json!({
    "bool": {
      "should": [
        {"term": {"a": "1"}},
        {"term": {"b": "2"}},
        {"term": {"c": "3"}},
      ],
      "minimum_should_match": "75%"
    }
  });
  let sl = translate_query(&es).unwrap();
  assert_eq!(sl.get("minimum_should_match").unwrap(), &json!(2));
}

#[test]
fn bool_minimum_should_match_integer_string_accepted() {
  let es = json!({
    "bool": {
      "should": [{"term": {"a": "1"}}, {"term": {"b": "2"}}],
      "minimum_should_match": "1"
    }
  });
  let sl = translate_query(&es).unwrap();
  assert_eq!(sl.get("minimum_should_match").unwrap(), &json!(1));
}

#[test]
fn bool_minimum_should_match_complex_syntax_rejected() {
  // ES supports "3<90%" combinator syntax. We don't (yet) — reject loudly
  // with a descriptive error rather than silently mis-translating.
  let es = json!({
    "bool": {
      "should": [{"term": {"a": "1"}}],
      "minimum_should_match": "3<90%"
    }
  });
  let err = translate_query(&es).unwrap_err();
  assert!(
    err.feature.starts_with("bool.minimum_should_match"),
    "got {err:?}"
  );
}

#[test]
fn terms_with_real_boost_meta_key_still_works() {
  // Sanity check that the boost-as-meta-key path still works when boost is
  // a number alongside a real field array.
  let es = json!({"terms": {"category": ["books"], "boost": 2.0}});
  let sl = translate_query(&es).unwrap();
  // Bool.should should target `category`, not `boost`.
  let shoulds = sl.get("should").unwrap().as_array().unwrap();
  assert_eq!(shoulds.len(), 1);
  assert_eq!(shoulds[0].get("field").unwrap(), &json!("category"));
}
