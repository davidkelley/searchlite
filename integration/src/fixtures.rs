use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use searchlite_core::api::types::{NestedField, NestedProperty, Schema, SearchRequest};
use serde_json::{json, Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DatasetName {
  Recipes,
  VideoGames,
}

impl DatasetName {
  fn relative_dir(self) -> &'static str {
    match self {
      Self::Recipes => "examples/recipes",
      Self::VideoGames => "examples/video-games",
    }
  }

  fn all() -> [Self; 2] {
    [Self::Recipes, Self::VideoGames]
  }
}

#[derive(Debug, Clone)]
pub struct QueryFixture {
  pub name: String,
  pub raw: Value,
  pub request: SearchRequest,
}

#[derive(Debug, Clone)]
pub struct UpdateFixture {
  pub id: String,
  pub set: Map<String, Value>,
  pub unset: Vec<String>,
  /// The field name that was modified (derived from schema).
  pub updated_field: String,
}

#[derive(Debug, Clone)]
pub struct MutationFixtures {
  pub insert_docs: Vec<Value>,
  pub update_docs: Vec<UpdateFixture>,
  pub delete_ids: Vec<String>,
  pub mget_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DatasetFixture {
  pub schema: Schema,
  pub seed_docs: Vec<Value>,
  pub queries: Vec<QueryFixture>,
  pub mutations: MutationFixtures,
}

#[derive(Debug, Clone)]
pub struct ExampleFixtures {
  pub datasets: BTreeMap<DatasetName, DatasetFixture>,
}

pub fn load_example_fixtures() -> Result<ExampleFixtures> {
  let mut datasets = BTreeMap::new();
  let workspace_root = workspace_root()?;

  for dataset in DatasetName::all() {
    let fixture = load_dataset_fixture(workspace_root.as_path(), dataset)?;
    datasets.insert(dataset, fixture);
  }

  Ok(ExampleFixtures { datasets })
}

fn load_dataset_fixture(root: &Path, dataset: DatasetName) -> Result<DatasetFixture> {
  let dataset_dir = root.join(dataset.relative_dir());
  let schema = load_schema(dataset_dir.as_path())?;
  let mut seed_docs = load_jsonl_docs(dataset_dir.as_path())?;
  sanitize_seed_docs(&mut seed_docs, &schema);
  let queries = load_query_fixtures(dataset_dir.as_path())?;
  let mutations = derive_mutations(&seed_docs, &schema.doc_id_field, &schema);

  Ok(DatasetFixture {
    schema,
    seed_docs,
    queries,
    mutations,
  })
}

fn load_schema(dataset_dir: &Path) -> Result<Schema> {
  let schema_path = dataset_dir.join("schema.json");
  let schema_str = fs::read_to_string(&schema_path)
    .with_context(|| format!("reading schema from {}", schema_path.display()))?;
  let mut schema: Schema = serde_json::from_str(&schema_str)
    .with_context(|| format!("parsing schema from {}", schema_path.display()))?;
  normalize_schema(&mut schema);
  Ok(schema)
}

fn load_jsonl_docs(dataset_dir: &Path) -> Result<Vec<Value>> {
  let data_path = dataset_dir.join("data.jsonl");
  let body = fs::read_to_string(&data_path)
    .with_context(|| format!("reading JSONL docs from {}", data_path.display()))?;
  let mut docs = Vec::new();
  for (line_no, line) in body.lines().enumerate() {
    if line.trim().is_empty() {
      continue;
    }
    let value: Value = serde_json::from_str(line).with_context(|| {
      format!(
        "parsing document JSON on line {} from {}",
        line_no + 1,
        data_path.display()
      )
    })?;
    docs.push(value);
  }
  Ok(docs)
}

fn load_query_fixtures(dataset_dir: &Path) -> Result<Vec<QueryFixture>> {
  let queries_dir = dataset_dir.join("queries");
  let mut entries = fs::read_dir(&queries_dir)
    .with_context(|| format!("reading queries dir {}", queries_dir.display()))?
    .collect::<std::io::Result<Vec<_>>>()
    .with_context(|| format!("iterating queries dir {}", queries_dir.display()))?;
  entries.sort_by_key(|entry| entry.path());

  let mut queries = Vec::new();
  for entry in entries {
    let path = entry.path();
    if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
      continue;
    }
    let body = fs::read_to_string(&path)
      .with_context(|| format!("reading query fixture {}", path.display()))?;
    let mut raw: Value = serde_json::from_str(&body)
      .with_context(|| format!("parsing query fixture {}", path.display()))?;
    normalize_request_value(&mut raw);
    let request: SearchRequest = serde_json::from_value(raw.clone())
      .with_context(|| format!("parsing SearchRequest from {}", path.display()))?;
    let name = path
      .file_stem()
      .and_then(|s| s.to_str())
      .unwrap_or_default()
      .to_string();

    queries.push(QueryFixture { name, raw, request });
  }

  Ok(queries)
}

fn derive_mutations(seed_docs: &[Value], doc_id_field: &str, schema: &Schema) -> MutationFixtures {
  let mut insert_docs = Vec::new();
  let mut delete_ids = Vec::new();

  for (idx, source) in seed_docs.iter().take(2).enumerate() {
    let mut cloned = source.clone();
    let id = format!("integration-generated-{doc_id_field}-{idx}");
    set_doc_id(&mut cloned, doc_id_field, &id);
    insert_docs.push(cloned);
    delete_ids.push(id);
  }

  let update_field = schema
    .text_fields
    .first()
    .map(|f| f.name.clone())
    .unwrap_or_else(|| "text".to_string());

  let mut update_docs = Vec::new();
  if let Some(first) = seed_docs.first() {
    let id = extract_doc_id(first, doc_id_field)
      .or_else(|| extract_doc_id(first, "_id"))
      .unwrap_or_else(|| "missing-id".to_string());
    let mut set = Map::new();
    set.insert(
      update_field.clone(),
      json!(format!("integration update marker for {id}")),
    );
    update_docs.push(UpdateFixture {
      id: id.clone(),
      set,
      unset: Vec::new(),
      updated_field: update_field,
    });
  }

  let mut mget_ids = Vec::new();
  if let Some(first) = seed_docs.first() {
    if let Some(id) = extract_doc_id(first, doc_id_field).or_else(|| extract_doc_id(first, "_id")) {
      mget_ids.push(id.clone());
      mget_ids.push(id);
    }
  }
  if let Some(second) = seed_docs.get(1) {
    if let Some(id) = extract_doc_id(second, doc_id_field).or_else(|| extract_doc_id(second, "_id"))
    {
      mget_ids.push(id);
    }
  }
  mget_ids.push("missing-doc-id".to_string());

  MutationFixtures {
    insert_docs,
    update_docs,
    delete_ids,
    mget_ids,
  }
}

fn set_doc_id(value: &mut Value, doc_id_field: &str, new_id: &str) {
  if let Some(obj) = value.as_object_mut() {
    if obj.contains_key(doc_id_field) {
      obj.insert(doc_id_field.to_string(), Value::String(new_id.to_string()));
      return;
    }
    if obj.contains_key("_id") {
      obj.insert("_id".to_string(), Value::String(new_id.to_string()));
      return;
    }
    obj.insert(doc_id_field.to_string(), Value::String(new_id.to_string()));
  }
}

fn extract_doc_id(value: &Value, field: &str) -> Option<String> {
  value
    .get(field)
    .and_then(|v| v.as_str())
    .map(ToString::to_string)
}

fn workspace_root() -> Result<PathBuf> {
  let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  crate_root.parent().map(Path::to_path_buf).with_context(|| {
    format!(
      "computing workspace root from crate root {}",
      crate_root.display()
    )
  })
}

fn normalize_schema(schema: &mut Schema) {
  let doc_id = schema.doc_id_field.clone();
  schema.text_fields.retain(|field| field.name != doc_id);
  schema.keyword_fields.retain(|field| field.name != doc_id);
  schema.numeric_fields.retain(|field| field.name != doc_id);
  schema.nested_fields.retain(|field| field.name != doc_id);
}

fn sanitize_seed_docs(seed_docs: &mut [Value], schema: &Schema) {
  for doc in seed_docs.iter_mut() {
    sanitize_doc_nested_fields(doc, schema);
  }
}

fn sanitize_doc_nested_fields(doc: &mut Value, schema: &Schema) {
  let Some(obj) = doc.as_object_mut() else {
    return;
  };
  for nested in schema.nested_fields.iter() {
    if let Some(value) = obj.get_mut(&nested.name) {
      sanitize_nested_value(value, nested);
    }
  }
}

fn sanitize_nested_value(value: &mut Value, nested: &NestedField) {
  match value {
    Value::Null => {
      if !nested.nullable {
        *value = Value::Array(Vec::new());
      }
    }
    Value::Array(items) => {
      for item in items.iter_mut() {
        sanitize_nested_object(item, nested);
      }
    }
    Value::Object(_) => sanitize_nested_object(value, nested),
    _ => {
      *value = Value::Array(Vec::new());
    }
  }
}

fn sanitize_nested_object(value: &mut Value, nested: &NestedField) {
  if !value.is_object() {
    *value = Value::Object(Map::new());
  }
  let map = value.as_object_mut().expect("object ensured");

  for prop in nested.fields.iter() {
    match map.get_mut(prop.name()) {
      Some(existing) => sanitize_property_value(existing, prop),
      None if !prop.is_nullable() => {
        map.insert(prop.name().to_string(), default_property_value(prop));
      }
      None => {}
    }
  }
}

fn sanitize_property_value(value: &mut Value, prop: &NestedProperty) {
  match prop {
    NestedProperty::Text(field) => {
      if value.is_null() && !field.nullable {
        *value = Value::String(String::new());
      }
    }
    NestedProperty::Keyword(field) => {
      if value.is_null() && !field.nullable {
        *value = Value::String(String::new());
      }
    }
    NestedProperty::Numeric(field) => {
      if value.is_null() && !field.nullable {
        *value = Value::from(0);
      }
    }
    NestedProperty::Object(nested) => {
      if value.is_null() && !nested.nullable {
        *value = Value::Object(Map::new());
      }
      sanitize_nested_value(value, nested);
    }
  }
}

fn default_property_value(prop: &NestedProperty) -> Value {
  match prop {
    NestedProperty::Text(_) | NestedProperty::Keyword(_) => Value::String(String::new()),
    NestedProperty::Numeric(field) => {
      if field.i64 {
        Value::from(0_i64)
      } else {
        Value::from(0.0_f64)
      }
    }
    NestedProperty::Object(_) => Value::Object(Map::new()),
  }
}

fn normalize_request_value(value: &mut Value) {
  if let Some(obj) = value.as_object_mut() {
    if let Some(aggs) = obj.get_mut("aggs") {
      normalize_aggs(aggs);
    }
  }
}

fn normalize_aggs(value: &mut Value) {
  let Some(aggs) = value.as_object_mut() else {
    return;
  };
  for agg_value in aggs.values_mut() {
    let Some(obj) = agg_value.as_object_mut() else {
      continue;
    };
    let agg_type = obj
      .get("type")
      .and_then(|v| v.as_str())
      .unwrap_or_default()
      .to_string();
    if (agg_type == "range" || agg_type == "date_range") && !obj.contains_key("keyed") {
      obj.insert("keyed".to_string(), Value::Bool(false));
    }
    if let Some(child_aggs) = obj.get_mut("aggs") {
      normalize_aggs(child_aggs);
    }
  }
}
