use searchlite_adapter_elastic::translate::translate_highlight;
use serde_json::json;

#[test]
fn basic_per_field_highlight() {
  let es = json!({
    "fields": {
      "title": {}
    }
  });
  let sl = translate_highlight(&es).unwrap();
  let title = sl
    .pointer("/fields/title")
    .expect("title field")
    .as_object()
    .unwrap();
  // No global tags, no per-field options → empty per-field config.
  assert!(
    title.is_empty() || (!title.contains_key("pre_tag") && !title.contains_key("post_tag")),
    "got {title:?}"
  );
}

#[test]
fn global_pre_post_tags_apply_to_all_fields() {
  let es = json!({
    "pre_tags": ["<em>"],
    "post_tags": ["</em>"],
    "fields": {
      "title": {},
      "description": {}
    }
  });
  let sl = translate_highlight(&es).unwrap();
  for field in ["title", "description"] {
    let pre = sl.pointer(&format!("/fields/{field}/pre_tag")).unwrap();
    let post = sl.pointer(&format!("/fields/{field}/post_tag")).unwrap();
    assert_eq!(pre, &json!("<em>"));
    assert_eq!(post, &json!("</em>"));
  }
}

#[test]
fn picks_first_tag_when_array_supplied() {
  // ES allows arrays; SearchLite uses singular pre_tag/post_tag — first wins.
  let es = json!({
    "pre_tags": ["<a>", "<b>", "<c>"],
    "post_tags": ["</a>", "</b>"],
    "fields": { "title": {} }
  });
  let sl = translate_highlight(&es).unwrap();
  assert_eq!(sl.pointer("/fields/title/pre_tag").unwrap(), &json!("<a>"));
  assert_eq!(
    sl.pointer("/fields/title/post_tag").unwrap(),
    &json!("</a>")
  );
}

#[test]
fn per_field_tags_override_global() {
  let es = json!({
    "pre_tags": ["<em>"],
    "post_tags": ["</em>"],
    "fields": {
      "title": { "pre_tags": ["<b>"], "post_tags": ["</b>"] },
      "description": {}
    }
  });
  let sl = translate_highlight(&es).unwrap();
  assert_eq!(sl.pointer("/fields/title/pre_tag").unwrap(), &json!("<b>"));
  assert_eq!(
    sl.pointer("/fields/description/pre_tag").unwrap(),
    &json!("<em>")
  );
}

#[test]
fn fragment_size_and_number_of_fragments_propagate() {
  let es = json!({
    "fields": {
      "title": { "fragment_size": 200, "number_of_fragments": 3 }
    }
  });
  let sl = translate_highlight(&es).unwrap();
  assert_eq!(
    sl.pointer("/fields/title/fragment_size").unwrap(),
    &json!(200)
  );
  assert_eq!(
    sl.pointer("/fields/title/number_of_fragments").unwrap(),
    &json!(3)
  );
}

#[test]
fn missing_fields_object_rejected() {
  let es = json!({ "pre_tags": ["<em>"] });
  let err = translate_highlight(&es).unwrap_err();
  assert!(err.feature.starts_with("highlight"), "got {err:?}");
}

#[test]
fn non_object_input_rejected() {
  let es = json!("not an object");
  let err = translate_highlight(&es).unwrap_err();
  assert_eq!(err.feature, "highlight");
}
