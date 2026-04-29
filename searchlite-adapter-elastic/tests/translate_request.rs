use searchlite_adapter_elastic::translate::translate_search_body;
use serde_json::{json, Value};

#[test]
fn empty_object_body_translates_to_match_all() {
  let sl = translate_search_body(&json!({})).unwrap();
  assert_eq!(sl.get("query").unwrap(), &json!({ "type": "match_all" }));
}

#[test]
fn null_body_translates_to_match_all() {
  // Some ES clients send a literal `null` body when no search criteria are
  // supplied. ES treats that the same as an empty body / match-all, and so
  // do we.
  let sl = translate_search_body(&Value::Null).unwrap();
  assert_eq!(sl.get("query").unwrap(), &json!({ "type": "match_all" }));
}

#[test]
fn array_body_is_rejected() {
  // Regression: previously a non-object body was silently coerced into an
  // empty map and run as match_all. For `_msearch` that meant a malformed
  // body line silently scanned the whole index.
  let err = translate_search_body(&json!([])).unwrap_err();
  assert!(err.feature.contains("body"), "got {err:?}");
}

#[test]
fn string_body_is_rejected() {
  let err = translate_search_body(&json!("not a body")).unwrap_err();
  assert!(err.feature.contains("body"), "got {err:?}");
}

#[test]
fn number_body_is_rejected() {
  let err = translate_search_body(&json!(42)).unwrap_err();
  assert!(err.feature.contains("body"), "got {err:?}");
}

#[test]
fn boolean_body_is_rejected() {
  let err = translate_search_body(&json!(true)).unwrap_err();
  assert!(err.feature.contains("body"), "got {err:?}");
}

// --- _source.includes shape handling ---------------------------------------

#[test]
fn source_includes_string_form_is_accepted() {
  // Regression: previously only the array form was handled, so the
  // single-string shape silently widened the response (no field restriction
  // applied) instead of returning just `["title"]`.
  let sl = translate_search_body(&json!({
    "_source": { "includes": "title" }
  }))
  .unwrap();
  assert_eq!(sl.get("fields").unwrap(), &json!(["title"]));
}

#[test]
fn source_includes_array_with_non_string_is_rejected() {
  // Silently dropping the non-string element (as filter_map did) widens the
  // payload with whatever fields the dropped entry would have restricted.
  let err = translate_search_body(&json!({
    "_source": { "includes": ["title", 42] }
  }))
  .unwrap_err();
  assert_eq!(err.feature, "_source.includes");
}

#[test]
fn source_includes_non_string_non_array_value_rejected() {
  let err = translate_search_body(&json!({
    "_source": { "includes": 42 }
  }))
  .unwrap_err();
  assert_eq!(err.feature, "_source.includes");
}

#[test]
fn source_includes_array_of_strings_still_works() {
  let sl = translate_search_body(&json!({
    "_source": { "includes": ["title", "category"] }
  }))
  .unwrap();
  assert_eq!(sl.get("fields").unwrap(), &json!(["title", "category"]));
}
