use searchlite_adapter_elastic::translate::translate_sort;
use serde_json::json;

#[test]
fn bare_field_string_defaults_to_asc() {
  let es = json!("price");
  let sl = translate_sort(&es).unwrap();
  assert_eq!(sl, vec![json!({"field": "price", "order": "asc"})]);
}

#[test]
fn bare_score_string_defaults_to_desc() {
  let es = json!("_score");
  let sl = translate_sort(&es).unwrap();
  assert_eq!(sl, vec![json!({"field": "_score", "order": "desc"})]);
}

#[test]
fn score_object_without_explicit_order_defaults_to_desc() {
  let es = json!([{"_score": {}}]);
  let sl = translate_sort(&es).unwrap();
  assert_eq!(sl, vec![json!({"field": "_score", "order": "desc"})]);
}

#[test]
fn field_object_without_explicit_order_defaults_to_asc() {
  let es = json!([{"price": {}}]);
  let sl = translate_sort(&es).unwrap();
  assert_eq!(sl, vec![json!({"field": "price", "order": "asc"})]);
}

#[test]
fn explicit_order_overrides_field_default() {
  let es = json!([{"_score": {"order": "asc"}}, {"price": "desc"}]);
  let sl = translate_sort(&es).unwrap();
  assert_eq!(
    sl,
    vec![
      json!({"field": "_score", "order": "asc"}),
      json!({"field": "price", "order": "desc"}),
    ]
  );
}

#[test]
fn array_form_with_mixed_entries() {
  let es = json!(["price", "_score"]);
  let sl = translate_sort(&es).unwrap();
  assert_eq!(
    sl,
    vec![
      json!({"field": "price", "order": "asc"}),
      json!({"field": "_score", "order": "desc"}),
    ]
  );
}
