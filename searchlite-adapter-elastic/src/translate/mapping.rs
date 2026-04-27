use serde_json::{json, Map, Value};

use super::unsupported::Unsupported;

/// Convert a SearchLite JSON Schema (with `searchlite:*` extensions) into an
/// Elasticsearch mapping JSON of the form:
///
/// ```json
/// { "<index>": { "mappings": { "properties": { ... } } } }
/// ```
pub fn schema_to_es(index: &str, schema: &Value) -> Result<Value, Unsupported> {
  let properties = schema
    .as_object()
    .and_then(|m| m.get("properties"))
    .and_then(Value::as_object);
  let mapping_props = match properties {
    Some(map) => translate_properties(map)?,
    None => Map::new(),
  };
  Ok(json!({
    index: {
      "mappings": {
        "properties": Value::Object(mapping_props),
      }
    }
  }))
}

fn translate_properties(props: &Map<String, Value>) -> Result<Map<String, Value>, Unsupported> {
  let mut out = Map::new();
  for (name, value) in props {
    let prop = value
      .as_object()
      .ok_or_else(|| Unsupported::with_detail("mapping", format!("property `{name}` must be an object")))?;
    out.insert(name.clone(), translate_field(prop)?);
  }
  Ok(out)
}

fn translate_field(prop: &Map<String, Value>) -> Result<Value, Unsupported> {
  let raw_type = prop
    .get("type")
    .ok_or_else(|| Unsupported::with_detail("mapping", "property missing `type`"))?;
  let base_type = primary_type(raw_type)?;
  let kind = prop.get("searchlite:kind").and_then(Value::as_str);

  let mut out = Map::new();
  match base_type {
    "string" => match kind {
      Some("keyword") => {
        out.insert("type".into(), Value::String("keyword".into()));
      }
      _ => {
        out.insert("type".into(), Value::String("text".into()));
        if let Some(analyzer) = prop.get("searchlite:analyzer").and_then(Value::as_str) {
          out.insert("analyzer".into(), Value::String(analyzer.to_string()));
        }
        if let Some(search_analyzer) =
          prop.get("searchlite:searchAnalyzer").and_then(Value::as_str)
        {
          out.insert(
            "search_analyzer".into(),
            Value::String(search_analyzer.to_string()),
          );
        }
      }
    },
    "integer" => {
      out.insert("type".into(), Value::String("long".into()));
    }
    "number" => {
      out.insert("type".into(), Value::String("double".into()));
    }
    "boolean" => {
      out.insert("type".into(), Value::String("boolean".into()));
    }
    "array" => {
      // Nested fields are emitted as JSON Schema arrays of objects.
      let items = prop
        .get("items")
        .and_then(Value::as_object)
        .ok_or_else(|| Unsupported::with_detail("mapping", "array property missing `items`"))?;
      let item_type = items
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("object");
      if item_type != "object" {
        return Err(Unsupported::with_detail(
          "mapping",
          "array fields must have items.type=object (nested)",
        ));
      }
      out.insert("type".into(), Value::String("nested".into()));
      if let Some(child_props) = items.get("properties").and_then(Value::as_object) {
        out.insert(
          "properties".into(),
          Value::Object(translate_properties(child_props)?),
        );
      }
    }
    "object" => {
      // Vector fields and other object-typed extensions are surfaced as
      // dense_vector when the marker is present.
      if let Some(vec_cfg) = prop.get("searchlite:vector").and_then(Value::as_object) {
        out.insert("type".into(), Value::String("dense_vector".into()));
        if let Some(dim) = vec_cfg.get("dim") {
          out.insert("dims".into(), dim.clone());
        }
        if let Some(metric) = vec_cfg.get("metric").and_then(Value::as_str) {
          let similarity = match metric {
            "Cosine" | "cosine" => "cosine",
            "L2" | "l2" => "l2_norm",
            other => {
              return Err(Unsupported::with_detail(
                "mapping.vector.metric",
                format!("unknown metric `{other}`"),
              ))
            }
          };
          out.insert("similarity".into(), Value::String(similarity.into()));
        }
      } else {
        out.insert("type".into(), Value::String("object".into()));
      }
    }
    other => {
      return Err(Unsupported::with_detail(
        "mapping",
        format!("type `{other}` not supported"),
      ));
    }
  }
  Ok(Value::Object(out))
}

fn primary_type(value: &Value) -> Result<&str, Unsupported> {
  match value {
    Value::String(s) => Ok(primary_str(s)),
    Value::Array(items) => {
      for v in items {
        if let Some(s) = v.as_str() {
          if s != "null" {
            return Ok(primary_str(s));
          }
        }
      }
      Err(Unsupported::with_detail(
        "mapping",
        "type array contained no concrete type",
      ))
    }
    _ => Err(Unsupported::with_detail(
      "mapping",
      "`type` must be a string or array of strings",
    )),
  }
}

fn primary_str(s: &str) -> &str {
  match s {
    "string" => "string",
    "integer" => "integer",
    "number" => "number",
    "boolean" => "boolean",
    "array" => "array",
    "object" => "object",
    _ => s,
  }
}
