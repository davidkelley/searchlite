//! Conversion between JSON Schema (with `searchlite:` vocabulary) and the
//! internal [`Schema`] representation.
//!
//! The user-facing schema format is standard JSON Schema 2020-12 annotated with
//! `searchlite:` prefixed keywords that configure search-engine behaviour.  The
//! internal representation keeps flat, type-specific arrays for engine
//! efficiency.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Map, Value};

use super::manifest::{
  default_doc_id_field, KeywordField, NestedField, NestedProperty, NumericField, Schema,
  SearchAsYouType, TextField,
};
use crate::analysis::analyzer::AnalyzerDef;

#[cfg(feature = "vectors")]
use super::manifest::{VectorField, VectorMetric};

// ── Constants ────────────────────────────────────────────────────────────────

const META_SCHEMA: &str = "https://searchlite.dev/draft/2025/schema";
const PREFIX: &str = "searchlite:";

// ── Parsing (JSON Schema → internal Schema) ──────────────────────────────────

/// Parse a JSON Schema document with `searchlite:` vocabulary into an internal
/// [`Schema`].
pub fn parse_json_schema(root: &Value) -> Result<Schema> {
  let obj = root
    .as_object()
    .ok_or_else(|| anyhow!("schema must be a JSON object"))?;

  // Reject old-format schemas with a helpful message.
  if obj.contains_key("text_fields")
    || obj.contains_key("keyword_fields")
    || obj.contains_key("numeric_fields")
  {
    bail!(
      "this appears to be a legacy field-array schema (text_fields/keyword_fields/numeric_fields). \
       Searchlite now uses JSON Schema with `searchlite:` vocabulary keywords. \
       See https://searchlite.dev/docs/schema for the new format."
    );
  }

  // Enforce root `type: "object"` when present (the meta-schema requires it,
  // so the parser should not silently accept other values).
  match obj.get("type") {
    Some(Value::String(s)) if s == "object" => {}
    Some(_) => bail!("root `type` must be \"object\""),
    None => {}
  }

  // Reject unknown root-level `searchlite:` keywords so malformed schemas
  // surface a clear error instead of being silently ignored.
  validate_root_searchlite_keys(obj)?;

  let doc_id_field = match obj.get("searchlite:docIdField") {
    Some(Value::String(s)) if !s.is_empty() => s.clone(),
    Some(Value::String(_)) => bail!("searchlite:docIdField must be a non-empty string"),
    Some(_) => bail!("searchlite:docIdField must be a string"),
    None => default_doc_id_field(),
  };

  let analyzers: Vec<AnalyzerDef> = match obj.get("searchlite:analyzers") {
    Some(v) => serde_json::from_value(v.clone()).context("parsing searchlite:analyzers")?,
    None => Vec::new(),
  };

  let props = match obj.get("properties") {
    Some(Value::Object(m)) => m,
    Some(_) => bail!("`properties` must be a JSON object"),
    None => {
      return Ok(Schema {
        doc_id_field,
        analyzers,
        text_fields: Vec::new(),
        keyword_fields: Vec::new(),
        numeric_fields: Vec::new(),
        nested_fields: Vec::new(),
        #[cfg(feature = "vectors")]
        vector_fields: Vec::new(),
      });
    }
  };

  let mut text_fields = Vec::new();
  let mut keyword_fields = Vec::new();
  let mut numeric_fields = Vec::new();
  let mut nested_fields = Vec::new();
  #[cfg(feature = "vectors")]
  let mut vector_fields = Vec::new();

  for (name, prop_val) in props {
    let prop = prop_val
      .as_object()
      .ok_or_else(|| anyhow!("property `{name}` must be a JSON object"))?;

    validate_searchlite_keys(name, prop)?;

    let (base_type, nullable) = resolve_type(name, prop)?;

    match base_type.as_str() {
      "string" => {
        let kind = sl_str(prop, "kind")?;
        if kind.as_deref() == Some("keyword") {
          keyword_fields.push(parse_keyword_field(name, prop, nullable)?);
        } else if kind.is_some() {
          bail!(
            "property `{name}`: unknown searchlite:kind value `{}`",
            kind.unwrap()
          );
        } else {
          text_fields.push(parse_text_field(name, prop, nullable)?);
        }
      }
      "integer" => {
        numeric_fields.push(parse_numeric_field(name, prop, nullable, true)?);
      }
      "number" => {
        numeric_fields.push(parse_numeric_field(name, prop, nullable, false)?);
      }
      "object" => {
        nested_fields.push(parse_nested_field(name, prop, nullable)?);
      }
      "array" => {
        let items = prop
          .get("items")
          .and_then(|v| v.as_object())
          .ok_or_else(|| anyhow!("property `{name}`: array type requires an `items` object"))?;

        #[cfg(feature = "vectors")]
        if prop.contains_key("searchlite:vector") {
          let items_type = items.get("type").and_then(|v| v.as_str()).unwrap_or("");
          if items_type != "number" {
            bail!(
              "property `{name}`: vector fields require items.type to be \"number\", got \"{items_type}\""
            );
          }
          vector_fields.push(parse_vector_field(name, prop, items)?);
          continue;
        }
        #[cfg(not(feature = "vectors"))]
        if prop.contains_key("searchlite:vector") {
          bail!("property `{name}`: vector fields require the `vectors` feature flag");
        }

        let items_type = items.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if items_type == "object" {
          nested_fields.push(parse_nested_from_array(name, items, nullable)?);
        } else {
          bail!(
            "property `{name}`: unsupported array items type `{items_type}`. \
             Arrays must contain objects (nested fields) or numbers with searchlite:vector."
          );
        }
      }
      other => {
        bail!("property `{name}`: unsupported JSON Schema type `{other}`");
      }
    }
  }

  Ok(Schema {
    doc_id_field,
    analyzers,
    text_fields,
    keyword_fields,
    numeric_fields,
    nested_fields,
    #[cfg(feature = "vectors")]
    vector_fields,
  })
}

// ── Serialization (internal Schema → JSON Schema) ────────────────────────────

/// Serialize an internal [`Schema`] to JSON Schema with `searchlite:` vocabulary.
pub fn schema_to_json_schema(schema: &Schema) -> Value {
  let mut root = Map::new();
  root.insert("$schema".into(), json!(META_SCHEMA));
  root.insert("type".into(), json!("object"));

  if schema.doc_id_field != "_id" {
    root.insert("searchlite:docIdField".into(), json!(schema.doc_id_field));
  }

  if !schema.analyzers.is_empty() {
    root.insert(
      "searchlite:analyzers".into(),
      serde_json::to_value(&schema.analyzers).unwrap_or(json!([])),
    );
  }

  let mut properties = Map::new();

  for f in &schema.text_fields {
    properties.insert(f.name.clone(), text_field_to_json(f));
  }
  for f in &schema.keyword_fields {
    properties.insert(f.name.clone(), keyword_field_to_json(f));
  }
  for f in &schema.numeric_fields {
    properties.insert(f.name.clone(), numeric_field_to_json(f));
  }
  for f in &schema.nested_fields {
    properties.insert(f.name.clone(), nested_field_to_json(f));
  }
  #[cfg(feature = "vectors")]
  for f in &schema.vector_fields {
    properties.insert(f.name.clone(), vector_field_to_json(f));
  }

  if !properties.is_empty() {
    root.insert("properties".into(), Value::Object(properties));
  }

  Value::Object(root)
}

// ── Field parsers ────────────────────────────────────────────────────────────

fn parse_text_field(name: &str, prop: &Map<String, Value>, nullable: bool) -> Result<TextField> {
  let nullable = sl_bool(prop, "nullable")?.unwrap_or(nullable);
  let analyzer = sl_str(prop, "analyzer")?.unwrap_or_else(|| "default".to_string());
  if analyzer.is_empty() {
    bail!("property `{name}`: searchlite:analyzer must not be empty");
  }
  let search_analyzer = sl_str(prop, "searchAnalyzer")?;
  if let Some(ref sa) = search_analyzer {
    if sa.is_empty() {
      bail!("property `{name}`: searchlite:searchAnalyzer must not be empty");
    }
  }
  Ok(TextField {
    name: name.to_string(),
    analyzer,
    search_analyzer,
    stored: sl_bool(prop, "stored")?.unwrap_or(true),
    indexed: sl_bool(prop, "indexed")?.unwrap_or(true),
    nullable,
    search_as_you_type: parse_search_as_you_type(prop)?,
  })
}

fn parse_keyword_field(
  name: &str,
  prop: &Map<String, Value>,
  nullable: bool,
) -> Result<KeywordField> {
  let nullable = sl_bool(prop, "nullable")?.unwrap_or(nullable);
  Ok(KeywordField {
    name: name.to_string(),
    stored: sl_bool(prop, "stored")?.unwrap_or(true),
    indexed: sl_bool(prop, "indexed")?.unwrap_or(true),
    fast: sl_bool(prop, "fast")?.unwrap_or(true),
    nullable,
  })
}

fn parse_numeric_field(
  name: &str,
  prop: &Map<String, Value>,
  nullable: bool,
  is_i64: bool,
) -> Result<NumericField> {
  let nullable = sl_bool(prop, "nullable")?.unwrap_or(nullable);
  Ok(NumericField {
    name: name.to_string(),
    i64: is_i64,
    fast: sl_bool(prop, "fast")?.unwrap_or(true),
    stored: sl_bool(prop, "stored")?.unwrap_or(false),
    nullable,
  })
}

fn parse_nested_field(
  name: &str,
  prop: &Map<String, Value>,
  nullable: bool,
) -> Result<NestedField> {
  let nullable = sl_bool(prop, "nullable")?.unwrap_or(nullable);
  let fields = match prop.get("properties") {
    Some(Value::Object(m)) => parse_nested_properties(m)?,
    _ => Vec::new(),
  };
  Ok(NestedField {
    name: name.to_string(),
    fields,
    nullable,
  })
}

fn parse_nested_from_array(
  name: &str,
  items: &Map<String, Value>,
  nullable: bool,
) -> Result<NestedField> {
  let fields = match items.get("properties") {
    Some(Value::Object(m)) => parse_nested_properties(m)?,
    _ => Vec::new(),
  };
  Ok(NestedField {
    name: name.to_string(),
    fields,
    nullable,
  })
}

fn parse_nested_properties(props: &Map<String, Value>) -> Result<Vec<NestedProperty>> {
  let mut out = Vec::new();
  for (name, val) in props {
    let prop = val
      .as_object()
      .ok_or_else(|| anyhow!("nested property `{name}` must be a JSON object"))?;

    validate_searchlite_keys(name, prop)?;
    let (base_type, nullable) = resolve_type(name, prop)?;

    match base_type.as_str() {
      "string" => {
        let kind = sl_str(prop, "kind")?;
        if kind.as_deref() == Some("keyword") {
          out.push(NestedProperty::Keyword(parse_keyword_field(
            name, prop, nullable,
          )?));
        } else if kind.is_some() {
          bail!(
            "nested property `{name}`: unknown searchlite:kind value `{}`",
            kind.unwrap()
          );
        } else {
          out.push(NestedProperty::Text(parse_text_field(
            name, prop, nullable,
          )?));
        }
      }
      "integer" => {
        out.push(NestedProperty::Numeric(parse_numeric_field(
          name, prop, nullable, true,
        )?));
      }
      "number" => {
        out.push(NestedProperty::Numeric(parse_numeric_field(
          name, prop, nullable, false,
        )?));
      }
      "object" => {
        out.push(NestedProperty::Object(parse_nested_field(
          name, prop, nullable,
        )?));
      }
      "array" => {
        let items = prop
          .get("items")
          .and_then(|v| v.as_object())
          .ok_or_else(|| {
            anyhow!("nested property `{name}`: array type requires an `items` object")
          })?;
        let items_type = items.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if items_type == "object" {
          out.push(NestedProperty::Object(parse_nested_from_array(
            name, items, nullable,
          )?));
        } else {
          bail!("nested property `{name}`: unsupported array items type `{items_type}`");
        }
      }
      other => {
        bail!("nested property `{name}`: unsupported type `{other}`");
      }
    }
  }
  Ok(out)
}

fn parse_search_as_you_type(prop: &Map<String, Value>) -> Result<Option<SearchAsYouType>> {
  let val = match prop.get("searchlite:searchAsYouType") {
    Some(v) => v,
    None => return Ok(None),
  };
  let obj = val
    .as_object()
    .ok_or_else(|| anyhow!("searchlite:searchAsYouType must be an object"))?;
  let min_gram = obj
    .get("minGram")
    .and_then(|v| v.as_u64())
    .map(|v| v as usize)
    .unwrap_or(1);
  let max_gram = obj
    .get("maxGram")
    .and_then(|v| v.as_u64())
    .map(|v| v as usize)
    .unwrap_or(15);
  if min_gram == 0 || max_gram == 0 {
    bail!("searchlite:searchAsYouType: minGram and maxGram must be > 0");
  }
  if min_gram > max_gram {
    bail!("searchlite:searchAsYouType: minGram must be <= maxGram");
  }
  Ok(Some(SearchAsYouType { min_gram, max_gram }))
}

#[cfg(feature = "vectors")]
fn parse_vector_field(
  name: &str,
  prop: &Map<String, Value>,
  _items: &Map<String, Value>,
) -> Result<VectorField> {
  let vec_val = prop
    .get("searchlite:vector")
    .ok_or_else(|| anyhow!("property `{name}`: missing searchlite:vector"))?;
  let vec_obj = vec_val
    .as_object()
    .ok_or_else(|| anyhow!("property `{name}`: searchlite:vector must be an object"))?;

  let dim = vec_obj
    .get("dim")
    .and_then(|v| v.as_u64())
    .map(|v| v as usize)
    .ok_or_else(|| {
      anyhow!("property `{name}`: searchlite:vector.dim is required and must be a positive integer")
    })?;

  let metric_str = vec_obj
    .get("metric")
    .and_then(|v| v.as_str())
    .ok_or_else(|| anyhow!("property `{name}`: searchlite:vector.metric is required"))?;
  let metric = match metric_str {
    "Cosine" => VectorMetric::Cosine,
    "L2" => VectorMetric::L2,
    other => bail!("property `{name}`: unknown vector metric `{other}` (expected Cosine or L2)"),
  };

  let hnsw = match vec_obj.get("hnsw") {
    Some(v) => Some(
      serde_json::from_value(v.clone())
        .context(format!("property `{name}`: parsing searchlite:vector.hnsw"))?,
    ),
    None => None,
  };

  Ok(VectorField {
    name: name.to_string(),
    dim,
    metric,
    hnsw,
  })
}

// ── Field serializers ────────────────────────────────────────────────────────

fn text_field_to_json(f: &TextField) -> Value {
  let mut prop = Map::new();
  let type_val = if f.nullable {
    json!(["string", "null"])
  } else {
    json!("string")
  };
  prop.insert("type".into(), type_val);

  if f.analyzer != "default" {
    prop.insert("searchlite:analyzer".into(), json!(f.analyzer));
  }
  if let Some(sa) = &f.search_analyzer {
    prop.insert("searchlite:searchAnalyzer".into(), json!(sa));
  }
  // Omit stored/indexed when they match text defaults (true/true).
  if !f.stored {
    prop.insert("searchlite:stored".into(), json!(false));
  }
  if !f.indexed {
    prop.insert("searchlite:indexed".into(), json!(false));
  }
  if let Some(saty) = &f.search_as_you_type {
    let mut saty_obj = Map::new();
    if saty.min_gram != 1 {
      saty_obj.insert("minGram".into(), json!(saty.min_gram));
    }
    if saty.max_gram != 15 {
      saty_obj.insert("maxGram".into(), json!(saty.max_gram));
    }
    prop.insert("searchlite:searchAsYouType".into(), Value::Object(saty_obj));
  }
  Value::Object(prop)
}

fn keyword_field_to_json(f: &KeywordField) -> Value {
  let mut prop = Map::new();
  let type_val = if f.nullable {
    json!(["string", "null"])
  } else {
    json!("string")
  };
  prop.insert("type".into(), type_val);
  prop.insert("searchlite:kind".into(), json!("keyword"));

  // Omit when matching keyword defaults (true/true/true).
  if !f.stored {
    prop.insert("searchlite:stored".into(), json!(false));
  }
  if !f.indexed {
    prop.insert("searchlite:indexed".into(), json!(false));
  }
  if !f.fast {
    prop.insert("searchlite:fast".into(), json!(false));
  }
  Value::Object(prop)
}

fn numeric_field_to_json(f: &NumericField) -> Value {
  let mut prop = Map::new();
  let base_type = if f.i64 { "integer" } else { "number" };
  let type_val = if f.nullable {
    json!([base_type, "null"])
  } else {
    json!(base_type)
  };
  prop.insert("type".into(), type_val);

  // Omit when matching numeric defaults (fast: true, stored: false).
  if !f.fast {
    prop.insert("searchlite:fast".into(), json!(false));
  }
  if f.stored {
    prop.insert("searchlite:stored".into(), json!(true));
  }
  Value::Object(prop)
}

fn nested_field_to_json(f: &NestedField) -> Value {
  let mut items = Map::new();
  items.insert("type".into(), json!("object"));

  if !f.fields.is_empty() {
    let mut properties = Map::new();
    for child in &f.fields {
      match child {
        NestedProperty::Text(tf) => {
          properties.insert(tf.name.clone(), text_field_to_json(tf));
        }
        NestedProperty::Keyword(kf) => {
          properties.insert(kf.name.clone(), keyword_field_to_json(kf));
        }
        NestedProperty::Numeric(nf) => {
          properties.insert(nf.name.clone(), numeric_field_to_json(nf));
        }
        NestedProperty::Object(nested) => {
          properties.insert(nested.name.clone(), nested_field_to_json(nested));
        }
      }
    }
    items.insert("properties".into(), Value::Object(properties));
  }

  let mut prop = Map::new();
  let type_val = if f.nullable {
    json!(["array", "null"])
  } else {
    json!("array")
  };
  prop.insert("type".into(), type_val);
  prop.insert("items".into(), Value::Object(items));
  Value::Object(prop)
}

#[cfg(feature = "vectors")]
fn vector_field_to_json(f: &VectorField) -> Value {
  let mut vec_config = Map::new();
  vec_config.insert("dim".into(), json!(f.dim));
  vec_config.insert(
    "metric".into(),
    json!(match f.metric {
      VectorMetric::Cosine => "Cosine",
      VectorMetric::L2 => "L2",
    }),
  );
  if let Some(hnsw) = &f.hnsw {
    vec_config.insert(
      "hnsw".into(),
      serde_json::to_value(hnsw).unwrap_or(json!({})),
    );
  }

  let mut prop = Map::new();
  prop.insert("type".into(), json!("array"));
  prop.insert("items".into(), json!({"type": "number"}));
  prop.insert("searchlite:vector".into(), Value::Object(vec_config));
  Value::Object(prop)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Extract the base type and nullable flag from a property's `type` field.
///
/// Supports `"type": "string"` and `"type": ["string", "null"]`.
fn resolve_type(name: &str, prop: &Map<String, Value>) -> Result<(String, bool)> {
  let type_val = prop
    .get("type")
    .ok_or_else(|| anyhow!("property `{name}` is missing a `type` field"))?;

  match type_val {
    Value::String(s) => Ok((s.clone(), false)),
    Value::Array(arr) => {
      let types: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
      let nullable = types.contains(&"null");
      let base: Vec<&&str> = types.iter().filter(|t| **t != "null").collect();
      if base.len() != 1 {
        bail!(
          "property `{name}`: `type` array must contain exactly one base type plus optional \"null\""
        );
      }
      Ok((base[0].to_string(), nullable))
    }
    _ => bail!("property `{name}`: `type` must be a string or array"),
  }
}

/// Read a `searchlite:` boolean keyword, rejecting wrong types.
fn sl_bool(prop: &Map<String, Value>, key: &str) -> Result<Option<bool>> {
  let full_key = format!("{PREFIX}{key}");
  match prop.get(&full_key) {
    Some(Value::Bool(b)) => Ok(Some(*b)),
    Some(_) => bail!("`{full_key}` must be a boolean"),
    None => Ok(None),
  }
}

/// Read a `searchlite:` string keyword, rejecting wrong types.
fn sl_str(prop: &Map<String, Value>, key: &str) -> Result<Option<String>> {
  let full_key = format!("{PREFIX}{key}");
  match prop.get(&full_key) {
    Some(Value::String(s)) => Ok(Some(s.clone())),
    Some(_) => bail!("`{full_key}` must be a string"),
    None => Ok(None),
  }
}

/// Known `searchlite:` keywords that may appear on individual properties.
const KNOWN_PROPERTY_KEYS: &[&str] = &[
  "searchlite:kind",
  "searchlite:stored",
  "searchlite:indexed",
  "searchlite:fast",
  "searchlite:analyzer",
  "searchlite:searchAnalyzer",
  "searchlite:searchAsYouType",
  "searchlite:nullable",
  "searchlite:vector",
];

/// Known `searchlite:` keywords that may appear at the root of a schema.
const KNOWN_ROOT_KEYS: &[&str] = &["searchlite:docIdField", "searchlite:analyzers"];

/// Validate that all `searchlite:` keys on a property are known.
fn validate_searchlite_keys(name: &str, prop: &Map<String, Value>) -> Result<()> {
  for key in prop.keys() {
    if key.starts_with(PREFIX) && !KNOWN_PROPERTY_KEYS.contains(&key.as_str()) {
      bail!("property `{name}`: unknown keyword `{key}`");
    }
  }
  Ok(())
}

/// Validate that all `searchlite:` keys at the root of the schema are known.
fn validate_root_searchlite_keys(obj: &Map<String, Value>) -> Result<()> {
  for key in obj.keys() {
    if key.starts_with(PREFIX) && !KNOWN_ROOT_KEYS.contains(&key.as_str()) {
      bail!("unknown root-level keyword `{key}`");
    }
  }
  Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;
  use serde_json::json;

  fn round_trip(schema: &Schema) -> Schema {
    let json_val = schema_to_json_schema(schema);
    parse_json_schema(&json_val).expect("round-trip parse failed")
  }

  #[test]
  fn text_field_defaults() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "title": { "type": "string" }
      }
    }))
    .unwrap();

    assert_eq!(schema.text_fields.len(), 1);
    let f = &schema.text_fields[0];
    assert_eq!(f.name, "title");
    assert_eq!(f.analyzer, "default");
    assert!(f.stored);
    assert!(f.indexed);
    assert!(!f.nullable);
    assert!(f.search_analyzer.is_none());
    assert!(f.search_as_you_type.is_none());
  }

  #[test]
  fn keyword_inference() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "tag": { "type": "string", "searchlite:kind": "keyword" }
      }
    }))
    .unwrap();

    assert_eq!(schema.keyword_fields.len(), 1);
    let f = &schema.keyword_fields[0];
    assert_eq!(f.name, "tag");
    assert!(f.stored);
    assert!(f.indexed);
    assert!(f.fast);
    assert!(!f.nullable);
  }

  #[test]
  fn integer_type() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "count": { "type": "integer" }
      }
    }))
    .unwrap();

    assert_eq!(schema.numeric_fields.len(), 1);
    let f = &schema.numeric_fields[0];
    assert_eq!(f.name, "count");
    assert!(f.i64);
    assert!(f.fast);
    assert!(!f.stored);
    assert!(!f.nullable);
  }

  #[test]
  fn number_type() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "price": { "type": "number" }
      }
    }))
    .unwrap();

    let f = &schema.numeric_fields[0];
    assert!(!f.i64);
  }

  #[test]
  fn nullable_via_type_array() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "bio": { "type": ["string", "null"] }
      }
    }))
    .unwrap();

    assert!(schema.text_fields[0].nullable);
  }

  #[test]
  fn nullable_via_keyword() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "bio": { "type": "string", "searchlite:nullable": true }
      }
    }))
    .unwrap();

    assert!(schema.text_fields[0].nullable);
  }

  #[test]
  fn nested_from_object_properties() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "author": {
          "type": "object",
          "properties": {
            "name": { "type": "string", "searchlite:kind": "keyword" },
            "age": { "type": "integer" }
          }
        }
      }
    }))
    .unwrap();

    assert_eq!(schema.nested_fields.len(), 1);
    let n = &schema.nested_fields[0];
    assert_eq!(n.name, "author");
    assert_eq!(n.fields.len(), 2);
    // BTreeMap ordering: "age" < "name"
    assert!(n
      .fields
      .iter()
      .any(|f| matches!(f, NestedProperty::Keyword(kf) if kf.name == "name")));
    assert!(n
      .fields
      .iter()
      .any(|f| matches!(f, NestedProperty::Numeric(nf) if nf.name == "age")));
  }

  #[test]
  fn nested_from_array_of_objects() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "tags": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "label": { "type": "string", "searchlite:kind": "keyword" }
            }
          }
        }
      }
    }))
    .unwrap();

    assert_eq!(schema.nested_fields.len(), 1);
    assert_eq!(schema.nested_fields[0].name, "tags");
    assert_eq!(schema.nested_fields[0].fields.len(), 1);
  }

  #[test]
  fn custom_text_field_overrides() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "body": {
          "type": "string",
          "searchlite:analyzer": "english",
          "searchlite:searchAnalyzer": "standard",
          "searchlite:stored": false,
          "searchlite:indexed": true
        }
      }
    }))
    .unwrap();

    let f = &schema.text_fields[0];
    assert_eq!(f.analyzer, "english");
    assert_eq!(f.search_analyzer.as_deref(), Some("standard"));
    assert!(!f.stored);
    assert!(f.indexed);
  }

  #[test]
  fn custom_keyword_overrides() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "status": {
          "type": "string",
          "searchlite:kind": "keyword",
          "searchlite:fast": false,
          "searchlite:stored": false
        }
      }
    }))
    .unwrap();

    let f = &schema.keyword_fields[0];
    assert!(!f.fast);
    assert!(!f.stored);
  }

  #[test]
  fn search_as_you_type() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "title": {
          "type": "string",
          "searchlite:searchAsYouType": { "minGram": 2, "maxGram": 10 }
        }
      }
    }))
    .unwrap();

    let saty = schema.text_fields[0].search_as_you_type.as_ref().unwrap();
    assert_eq!(saty.min_gram, 2);
    assert_eq!(saty.max_gram, 10);
  }

  #[test]
  fn doc_id_field_default() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {}
    }))
    .unwrap();
    assert_eq!(schema.doc_id_field, "_id");
  }

  #[test]
  fn doc_id_field_custom() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "searchlite:docIdField": "pk",
      "properties": {}
    }))
    .unwrap();
    assert_eq!(schema.doc_id_field, "pk");
  }

  #[test]
  fn analyzers_pass_through() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "searchlite:analyzers": [
        { "name": "english", "tokenizer": "default", "filters": [{"stemmer": "english"}] }
      ],
      "properties": {
        "body": { "type": "string", "searchlite:analyzer": "english" }
      }
    }))
    .unwrap();

    assert_eq!(schema.analyzers.len(), 1);
    assert_eq!(schema.analyzers[0].name, "english");
  }

  #[test]
  fn error_on_non_string_doc_id_field() {
    let err = parse_json_schema(&json!({
      "type": "object",
      "searchlite:docIdField": 123,
      "properties": {}
    }))
    .unwrap_err();
    assert!(
      err.to_string().contains("must be a string"),
      "expected doc_id_field type error, got: {err}"
    );
  }

  #[cfg(feature = "vectors")]
  #[test]
  fn error_on_vector_with_wrong_items_type() {
    let err = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "embedding": {
          "type": "array",
          "items": { "type": "string" },
          "searchlite:vector": { "dim": 384, "metric": "Cosine" }
        }
      }
    }))
    .unwrap_err();
    assert!(
      err.to_string().contains("items.type") && err.to_string().contains("number"),
      "expected vector items.type error, got: {err}"
    );
  }

  #[test]
  fn error_on_wrong_type_for_stored() {
    let err = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "x": { "type": "string", "searchlite:stored": "yes" }
      }
    }))
    .unwrap_err();
    assert!(
      err.to_string().contains("must be a boolean"),
      "expected type error for stored, got: {err}"
    );
  }

  #[test]
  fn error_on_wrong_type_for_kind() {
    let err = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "x": { "type": "string", "searchlite:kind": true }
      }
    }))
    .unwrap_err();
    assert!(
      err.to_string().contains("must be a string"),
      "expected type error for kind, got: {err}"
    );
  }

  #[test]
  fn error_on_empty_analyzer() {
    let err = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "x": { "type": "string", "searchlite:analyzer": "" }
      }
    }))
    .unwrap_err();
    assert!(
      err.to_string().contains("must not be empty"),
      "expected empty analyzer error, got: {err}"
    );
  }

  #[test]
  fn error_on_old_format() {
    let err = parse_json_schema(&json!({
      "text_fields": [],
      "keyword_fields": [],
      "numeric_fields": []
    }))
    .unwrap_err();
    assert!(err.to_string().contains("legacy field-array schema"));
  }

  #[test]
  fn error_on_unknown_searchlite_key() {
    let err = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "x": { "type": "string", "searchlite:bogus": true }
      }
    }))
    .unwrap_err();
    assert!(err.to_string().contains("unknown keyword"));
  }

  #[test]
  fn error_on_unknown_root_searchlite_key() {
    let err = parse_json_schema(&json!({
      "type": "object",
      "searchlite:bogus": true,
      "properties": {}
    }))
    .unwrap_err();
    assert!(
      err.to_string().contains("unknown root-level keyword"),
      "expected unknown root-level keyword error, got: {err}"
    );
  }

  #[test]
  fn error_on_wrong_root_type() {
    let err = parse_json_schema(&json!({
      "type": "array",
      "properties": {}
    }))
    .unwrap_err();
    assert!(
      err.to_string().contains("root `type` must be \"object\""),
      "expected root type error, got: {err}"
    );
  }

  #[test]
  fn error_on_empty_doc_id_field() {
    let err = parse_json_schema(&json!({
      "type": "object",
      "searchlite:docIdField": "",
      "properties": {}
    }))
    .unwrap_err();
    assert!(
      err.to_string().contains("non-empty string"),
      "expected empty doc_id_field error, got: {err}"
    );
  }

  #[test]
  fn round_trip_text_body() {
    let original = Schema::default_text_body();
    let rt = round_trip(&original);
    assert_eq!(rt.doc_id_field, original.doc_id_field);
    assert_eq!(rt.text_fields.len(), original.text_fields.len());
    assert_eq!(rt.text_fields[0].name, "body");
    assert_eq!(rt.text_fields[0].analyzer, "default");
    assert_eq!(rt.text_fields[0].stored, original.text_fields[0].stored);
    assert_eq!(rt.text_fields[0].indexed, original.text_fields[0].indexed);
  }

  #[test]
  fn round_trip_mixed_fields() {
    let original = Schema {
      doc_id_field: "pk".to_string(),
      analyzers: Vec::new(),
      text_fields: vec![TextField {
        name: "body".into(),
        analyzer: "english".into(),
        search_analyzer: Some("standard".into()),
        stored: true,
        indexed: true,
        nullable: false,
        search_as_you_type: None,
      }],
      keyword_fields: vec![KeywordField {
        name: "tag".into(),
        stored: true,
        indexed: true,
        fast: false,
        nullable: true,
      }],
      numeric_fields: vec![NumericField {
        name: "score".into(),
        i64: false,
        fast: true,
        stored: true,
        nullable: false,
      }],
      nested_fields: vec![NestedField {
        name: "comments".into(),
        fields: vec![
          NestedProperty::Keyword(KeywordField {
            name: "author".into(),
            stored: true,
            indexed: true,
            fast: true,
            nullable: false,
          }),
          NestedProperty::Numeric(NumericField {
            name: "rating".into(),
            i64: true,
            fast: true,
            stored: true,
            nullable: false,
          }),
        ],
        nullable: false,
      }],
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };

    let rt = round_trip(&original);
    assert_eq!(rt.doc_id_field, "pk");
    assert_eq!(rt.text_fields[0].analyzer, "english");
    assert_eq!(
      rt.text_fields[0].search_analyzer.as_deref(),
      Some("standard")
    );
    assert!(rt.keyword_fields[0].nullable);
    assert!(!rt.keyword_fields[0].fast);
    assert!(!rt.numeric_fields[0].i64);
    assert!(rt.numeric_fields[0].stored);
    assert_eq!(rt.nested_fields[0].fields.len(), 2);
  }

  #[test]
  fn serialization_omits_defaults() {
    let schema = Schema {
      doc_id_field: "_id".to_string(),
      analyzers: Vec::new(),
      text_fields: vec![TextField {
        name: "body".into(),
        analyzer: "default".into(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: false,
        search_as_you_type: None,
      }],
      keyword_fields: Vec::new(),
      numeric_fields: Vec::new(),
      nested_fields: Vec::new(),
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };

    let val = schema_to_json_schema(&schema);
    let root = val.as_object().unwrap();
    // doc_id_field is default, so omitted
    assert!(!root.contains_key("searchlite:docIdField"));
    // analyzers is empty, so omitted
    assert!(!root.contains_key("searchlite:analyzers"));

    let body = root["properties"]["body"].as_object().unwrap();
    // analyzer is default, so omitted
    assert!(!body.contains_key("searchlite:analyzer"));
    // stored=true is default for text, so omitted
    assert!(!body.contains_key("searchlite:stored"));
    // indexed=true is default for text, so omitted
    assert!(!body.contains_key("searchlite:indexed"));
  }

  #[cfg(feature = "vectors")]
  #[test]
  fn vector_field_round_trip() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "embedding": {
          "type": "array",
          "items": { "type": "number" },
          "searchlite:vector": {
            "dim": 384,
            "metric": "Cosine"
          }
        }
      }
    }))
    .unwrap();

    assert_eq!(schema.vector_fields.len(), 1);
    let vf = &schema.vector_fields[0];
    assert_eq!(vf.name, "embedding");
    assert_eq!(vf.dim, 384);
    assert!(matches!(vf.metric, VectorMetric::Cosine));

    let rt = round_trip(&schema);
    assert_eq!(rt.vector_fields.len(), 1);
    assert_eq!(rt.vector_fields[0].dim, 384);
  }

  #[cfg(feature = "vectors")]
  #[test]
  fn vector_field_missing_dim() {
    let err = parse_json_schema(&json!({
      "type": "object",
      "properties": {
        "embedding": {
          "type": "array",
          "items": { "type": "number" },
          "searchlite:vector": { "metric": "Cosine" }
        }
      }
    }))
    .unwrap_err();
    assert!(err.to_string().contains("dim"));
  }

  #[test]
  fn empty_properties_is_valid() {
    let schema = parse_json_schema(&json!({
      "type": "object",
      "properties": {}
    }))
    .unwrap();
    assert!(schema.text_fields.is_empty());
    assert!(schema.keyword_fields.is_empty());
  }

  #[test]
  fn no_properties_key_is_valid() {
    let schema = parse_json_schema(&json!({
      "type": "object"
    }))
    .unwrap();
    assert!(schema.text_fields.is_empty());
  }
}
