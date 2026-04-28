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
