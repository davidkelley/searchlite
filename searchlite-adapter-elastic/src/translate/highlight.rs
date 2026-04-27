use serde_json::{json, Map, Value};

use super::unsupported::Unsupported;

/// Translate ES highlight spec into SearchLite's HighlightRequest JSON.
///
/// SearchLite uses singular `pre_tag`/`post_tag`; ES allows arrays. We pick the
/// first element if a list is supplied.
pub fn translate_highlight(es_highlight: &Value) -> Result<Value, Unsupported> {
  let map = es_highlight.as_object().ok_or_else(|| {
    Unsupported::with_detail("highlight", "expected an object with `fields`")
  })?;
  let global_pre = pick_first_tag(map.get("pre_tags"));
  let global_post = pick_first_tag(map.get("post_tags"));
  let fields_in = map
    .get("fields")
    .and_then(Value::as_object)
    .ok_or_else(|| Unsupported::with_detail("highlight.fields", "expected an object"))?;

  let mut fields_out = Map::new();
  for (field, spec) in fields_in {
    let field_obj = spec.as_object();
    let pre_tag = field_obj
      .and_then(|m| pick_first_tag(m.get("pre_tags")))
      .or_else(|| global_pre.clone());
    let post_tag = field_obj
      .and_then(|m| pick_first_tag(m.get("post_tags")))
      .or_else(|| global_post.clone());
    let fragment_size = field_obj
      .and_then(|m| m.get("fragment_size"))
      .and_then(Value::as_u64);
    let number_of_fragments = field_obj
      .and_then(|m| m.get("number_of_fragments"))
      .and_then(Value::as_u64);

    let mut entry = Map::new();
    if let Some(tag) = pre_tag {
      entry.insert("pre_tag".to_string(), Value::String(tag));
    }
    if let Some(tag) = post_tag {
      entry.insert("post_tag".to_string(), Value::String(tag));
    }
    if let Some(size) = fragment_size {
      entry.insert("fragment_size".to_string(), Value::from(size));
    }
    if let Some(n) = number_of_fragments {
      entry.insert("number_of_fragments".to_string(), Value::from(n));
    }
    fields_out.insert(field.clone(), Value::Object(entry));
  }

  Ok(json!({ "fields": Value::Object(fields_out) }))
}

fn pick_first_tag(value: Option<&Value>) -> Option<String> {
  match value? {
    Value::String(s) => Some(s.clone()),
    Value::Array(items) => items.first().and_then(Value::as_str).map(str::to_string),
    _ => None,
  }
}
