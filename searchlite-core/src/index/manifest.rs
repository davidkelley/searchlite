use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::analyzer::{Analyzer, AnalyzerDef, AnalyzerRegistry};
use crate::storage::Storage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
  pub version: u32,
  pub uuid: Uuid,
  pub segments: Vec<SegmentMeta>,
  pub committed_at: String,
  pub schema: Schema,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMeta {
  pub id: String,
  pub generation: u32,
  pub paths: SegmentPaths,
  pub doc_count: u32,
  pub max_doc_id: u32,
  pub blockmax: bool,
  #[serde(default)]
  pub deleted_docs: Vec<u32>,
  pub avg_field_lengths: HashMap<String, f32>,
  pub checksums: HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentPaths {
  pub terms: String,
  pub postings: String,
  pub docstore: String,
  pub fast: String,
  pub meta: String,
}

impl Manifest {
  pub fn new(schema: Schema) -> Self {
    Self {
      version: 1,
      uuid: Uuid::new_v4(),
      segments: Vec::new(),
      committed_at: Utc::now().to_rfc3339(),
      schema,
    }
  }

  pub fn load(storage: &dyn Storage, path: &Path) -> Result<Self> {
    let data = storage
      .read_to_end(path)
      .with_context(|| format!("reading manifest at {:?}", path))?;
    let manifest: Manifest =
      serde_json::from_slice(&data).with_context(|| format!("parsing manifest at {:?}", path))?;
    Ok(manifest)
  }

  pub fn store(&self, storage: &dyn Storage, path: &Path) -> Result<()> {
    let data = serde_json::to_vec_pretty(self)?;
    storage
      .atomic_write(path, &data)
      .with_context(|| format!("writing manifest at {:?}", path))
  }

  pub fn manifest_path(root: &Path) -> PathBuf {
    root.join("MANIFEST.json")
  }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Schema {
  #[serde(default = "default_doc_id_field")]
  pub doc_id_field: String,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub analyzers: Vec<AnalyzerDef>,
  pub text_fields: Vec<TextField>,
  pub keyword_fields: Vec<KeywordField>,
  pub numeric_fields: Vec<NumericField>,
  #[serde(default)]
  pub nested_fields: Vec<NestedField>,
  #[cfg(feature = "vectors")]
  pub vector_fields: Vec<VectorField>,
}

pub fn default_doc_id_field() -> String {
  "_id".to_string()
}

impl Schema {
  pub fn default_text_body() -> Self {
    Self {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: vec![TextField {
        name: "body".to_string(),
        analyzer: "default".to_string(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: false,
      }],
      keyword_fields: Vec::new(),
      numeric_fields: Vec::new(),
      nested_fields: Vec::new(),
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    }
  }

  pub fn is_indexed_field(&self, field: &str) -> bool {
    self
      .resolved_fields()
      .iter()
      .any(|f| f.path == field && f.indexed)
  }

  pub fn is_stored_field(&self, field: &str) -> bool {
    self
      .resolved_fields()
      .iter()
      .any(|f| f.path == field && f.stored)
  }

  pub fn validate_config(&self) -> anyhow::Result<()> {
    if self.doc_id_field.contains('.') {
      anyhow::bail!("doc_id_field `{}` cannot be nested", self.doc_id_field);
    }
    self.build_analyzers()?;
    if self
      .resolved_fields()
      .iter()
      .any(|f| f.path == self.doc_id_field)
    {
      anyhow::bail!(
        "doc_id_field `{}` must not overlap with other schema fields",
        self.doc_id_field
      );
    }
    Ok(())
  }

  pub fn build_analyzers(&self) -> anyhow::Result<SchemaAnalyzers> {
    let registry = AnalyzerRegistry::from_defs(&self.analyzers)?;
    let mut field_map = HashMap::new();
    for (path, field) in self.text_field_map().into_iter() {
      if registry.get(&field.analyzer).is_none() {
        anyhow::bail!(
          "field `{path}` references unknown analyzer `{}`",
          field.analyzer
        );
      }
      let search_name = field
        .search_analyzer
        .clone()
        .unwrap_or_else(|| field.analyzer.clone());
      if registry.get(&search_name).is_none() {
        anyhow::bail!("field `{path}` references unknown search analyzer `{search_name}`");
      }
      if field_map
        .insert(
          path.clone(),
          FieldAnalyzerRefs {
            analyzer: field.analyzer.clone(),
            search_analyzer: search_name,
          },
        )
        .is_some()
      {
        anyhow::bail!("duplicate field `{path}` in analyzer map");
      }
    }
    Ok(SchemaAnalyzers {
      registry,
      field_map,
    })
  }

  fn text_field_map(&self) -> Vec<(String, &TextField)> {
    let mut out = Vec::new();
    for field in self.text_fields.iter() {
      out.push((field.name.clone(), field));
    }
    for nested in self.nested_fields.iter() {
      collect_nested_text_fields(nested, None, &mut out);
    }
    out
  }

  pub fn fast_fields(&self) -> Vec<String> {
    self
      .resolved_fields()
      .into_iter()
      .filter(|f| f.fast)
      .map(|f| f.path)
      .collect()
  }

  pub fn field_kind(&self, field: &str) -> FieldKind {
    self
      .resolved_fields()
      .into_iter()
      .find(|f| f.path == field)
      .map(|f| f.kind)
      .unwrap_or(FieldKind::Unknown)
  }

  pub fn field_meta(&self, field: &str) -> Option<ResolvedField> {
    self.resolved_fields().into_iter().find(|f| f.path == field)
  }

  pub fn resolved_fields(&self) -> Vec<ResolvedField> {
    let mut fields = Vec::new();
    for f in self.text_fields.iter() {
      fields.push(ResolvedField {
        path: f.name.clone(),
        kind: FieldKind::Text,
        indexed: f.indexed,
        stored: f.stored,
        fast: false,
        numeric_i64: None,
      });
    }
    for f in self.keyword_fields.iter() {
      fields.push(ResolvedField {
        path: f.name.clone(),
        kind: FieldKind::Keyword,
        indexed: f.indexed,
        stored: f.stored,
        fast: f.fast,
        numeric_i64: None,
      });
    }
    for f in self.numeric_fields.iter() {
      fields.push(ResolvedField {
        path: f.name.clone(),
        kind: FieldKind::Numeric,
        indexed: true,
        stored: f.stored,
        fast: f.fast,
        numeric_i64: Some(f.i64),
      });
    }
    for nested in self.nested_fields.iter() {
      nested.collect_fields(None, &mut fields);
    }
    fields
  }

  pub fn doc_id_field(&self) -> &str {
    &self.doc_id_field
  }

  pub fn validate_document(&self, doc: &crate::api::types::Document) -> anyhow::Result<()> {
    if doc
      .fields
      .get(self.doc_id_field())
      .and_then(|v| v.as_str())
      .map(|s| s.trim())
      .filter(|s| !s.is_empty())
      .is_none()
    {
      anyhow::bail!(
        "missing or empty required document id field `{}`",
        self.doc_id_field()
      );
    }
    for (name, value) in doc.fields.iter() {
      if let Some(nested) = self.nested_fields.iter().find(|n| n.name == *name) {
        nested
          .validate(value)
          .with_context(|| format!("validating nested field {name}"))?;
      }
    }
    Ok(())
  }

  #[cfg(feature = "vectors")]
  pub fn vector_field(&self, field: &str) -> Option<VectorField> {
    self.vector_fields.iter().find(|f| f.name == field).cloned()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[test]
  fn doc_id_field_defaults_and_validates_presence() {
    let schema = Schema::default_text_body();
    assert_eq!(schema.doc_id_field(), "_id");
    let doc = crate::api::types::Document::default();
    let err = schema.validate_document(&doc).unwrap_err();
    assert!(err
      .to_string()
      .contains("missing or empty required document id field"));
  }

  #[test]
  fn doc_id_field_rejects_empty() {
    let schema = Schema::default_text_body();
    for value in ["", "   "] {
      let doc = crate::api::types::Document {
        fields: [("_id".into(), serde_json::json!(value))]
          .into_iter()
          .collect(),
      };
      let err = schema.validate_document(&doc).unwrap_err();
      assert!(err
        .to_string()
        .contains("missing or empty required document id field"));
    }
  }

  #[test]
  fn persists_manifest_and_schema_helpers() {
    let dir = tempdir().unwrap();
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let schema = Schema {
      doc_id_field: "pk".into(),
      analyzers: Vec::new(),
      text_fields: vec![TextField {
        name: "body".into(),
        analyzer: "default".into(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: false,
      }],
      keyword_fields: vec![KeywordField {
        name: "tag".into(),
        stored: true,
        indexed: true,
        fast: true,
        nullable: false,
      }],
      numeric_fields: vec![NumericField {
        name: "year".into(),
        i64: true,
        fast: true,
        stored: true,
        nullable: false,
      }],
      nested_fields: vec![NestedField {
        name: "comment".into(),
        fields: vec![NestedProperty::Keyword(KeywordField {
          name: "author".into(),
          stored: true,
          indexed: true,
          fast: true,
          nullable: false,
        })],
        nullable: false,
      }],
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };
    let manifest = Manifest::new(schema.clone());
    let path = Manifest::manifest_path(dir.path());
    manifest.store(&storage, &path).unwrap();
    let loaded = Manifest::load(&storage, &path).unwrap();
    assert!(loaded.schema.is_indexed_field("body"));
    assert!(loaded.schema.is_stored_field("year"));
    let mut fast_fields = loaded.schema.fast_fields();
    fast_fields.sort();
    assert_eq!(
      fast_fields,
      vec![
        "comment.author".to_string(),
        "tag".to_string(),
        "year".to_string()
      ]
    );
    assert!(matches!(
      loaded.schema.field_kind("year"),
      FieldKind::Numeric
    ));
    assert!(matches!(
      loaded.schema.field_kind("comment.author"),
      FieldKind::Keyword
    ));
  }

  #[test]
  fn nested_nullable_fields_are_explicit() {
    let base_schema = Schema {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: Vec::new(),
      keyword_fields: Vec::new(),
      numeric_fields: Vec::new(),
      nested_fields: vec![NestedField {
        name: "game".into(),
        fields: vec![
          NestedProperty::Keyword(KeywordField {
            name: "name".into(),
            stored: true,
            indexed: true,
            fast: true,
            nullable: false,
          }),
          NestedProperty::Keyword(KeywordField {
            name: "franchise".into(),
            stored: true,
            indexed: false,
            fast: true,
            nullable: true,
          }),
        ],
        nullable: false,
      }],
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };

    let ok = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("game-1")),
        (
          "game".into(),
          serde_json::json!({ "name": "Skyline of Void", "franchise": null }),
        ),
      ]
      .into_iter()
      .collect(),
    };
    base_schema.validate_document(&ok).expect("nullable ok");

    let bad_null = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("game-2")),
        (
          "game".into(),
          serde_json::json!({ "name": null, "franchise": "Series" }),
        ),
      ]
      .into_iter()
      .collect(),
    };
    assert!(base_schema.validate_document(&bad_null).is_err());

    let nullable_game_schema = Schema {
      nested_fields: vec![NestedField {
        name: "game".into(),
        fields: vec![],
        nullable: true,
      }],
      ..base_schema.clone()
    };
    let null_game = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("game-3")),
        ("game".into(), serde_json::Value::Null),
      ]
      .into_iter()
      .collect(),
    };
    nullable_game_schema
      .validate_document(&null_game)
      .expect("nullable container ok");
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldKind {
  Text,
  Keyword,
  Numeric,
  Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedField {
  pub path: String,
  pub kind: FieldKind,
  pub indexed: bool,
  pub stored: bool,
  pub fast: bool,
  pub numeric_i64: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct FieldAnalyzerRefs {
  pub analyzer: String,
  pub search_analyzer: String,
}

#[derive(Debug, Clone)]
pub struct SchemaAnalyzers {
  pub(crate) registry: AnalyzerRegistry,
  pub(crate) field_map: HashMap<String, FieldAnalyzerRefs>,
}

impl SchemaAnalyzers {
  pub fn index_analyzer(&self, field: &str) -> Option<&Analyzer> {
    self
      .field_map
      .get(field)
      .and_then(|f| self.registry.get(&f.analyzer))
  }

  pub fn search_analyzer(&self, field: &str) -> Option<&Analyzer> {
    self
      .field_map
      .get(field)
      .and_then(|f| self.registry.get(&f.search_analyzer))
  }
}

#[derive(Debug, Clone)]
pub struct TextField {
  pub name: String,
  pub analyzer: String,
  pub search_analyzer: Option<String>,
  pub stored: bool,
  pub indexed: bool,
  pub nullable: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct TextFieldSerde {
  pub name: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub tokenizer: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub analyzer: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub search_analyzer: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub search_tokenizer: Option<String>,
  pub stored: bool,
  pub indexed: bool,
  #[serde(default)]
  pub nullable: bool,
}

impl From<TextField> for TextFieldSerde {
  fn from(value: TextField) -> Self {
    Self {
      name: value.name,
      tokenizer: Some(value.analyzer),
      analyzer: None,
      search_analyzer: value.search_analyzer,
      search_tokenizer: None,
      stored: value.stored,
      indexed: value.indexed,
      nullable: value.nullable,
    }
  }
}

impl TryFrom<TextFieldSerde> for TextField {
  type Error = serde::de::value::Error;

  fn try_from(value: TextFieldSerde) -> Result<Self, Self::Error> {
    let primary = match (value.analyzer, value.tokenizer) {
      (Some(a), None) => a,
      (None, Some(t)) => t,
      (Some(_), Some(_)) => {
        return Err(serde::de::Error::custom(
          "text field cannot set both `tokenizer` and `analyzer`",
        ));
      }
      (None, None) => {
        return Err(serde::de::Error::custom(
          "text field must set `analyzer` (or `tokenizer` as an alias)",
        ));
      }
    };
    let search_analyzer = match (value.search_analyzer, value.search_tokenizer) {
      (Some(a), None) => Some(a),
      (None, Some(t)) => Some(t),
      (Some(_), Some(_)) => {
        return Err(serde::de::Error::custom(
          "text field cannot set both `search_analyzer` and `search_tokenizer`",
        ));
      }
      (None, None) => None,
    };
    Ok(TextField {
      name: value.name,
      analyzer: primary,
      search_analyzer,
      stored: value.stored,
      indexed: value.indexed,
      nullable: value.nullable,
    })
  }
}

impl Serialize for TextField {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    let helper = TextFieldSerde {
      name: self.name.clone(),
      tokenizer: Some(self.analyzer.clone()),
      analyzer: None,
      search_analyzer: self.search_analyzer.clone(),
      search_tokenizer: None,
      stored: self.stored,
      indexed: self.indexed,
      nullable: self.nullable,
    };
    helper.serialize(serializer)
  }
}

impl<'de> Deserialize<'de> for TextField {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let helper = TextFieldSerde::deserialize(deserializer)?;
    TextField::try_from(helper).map_err(serde::de::Error::custom)
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordField {
  pub name: String,
  pub stored: bool,
  pub indexed: bool,
  pub fast: bool,
  #[serde(default)]
  pub nullable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericField {
  pub name: String,
  pub i64: bool,
  pub fast: bool,
  #[serde(default)]
  pub stored: bool,
  #[serde(default)]
  pub nullable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedField {
  pub name: String,
  #[serde(default)]
  pub fields: Vec<NestedProperty>,
  #[serde(default)]
  pub nullable: bool,
}

impl NestedField {
  fn validate(&self, value: &serde_json::Value) -> anyhow::Result<()> {
    match value {
      serde_json::Value::Null => {
        if self.nullable {
          return Ok(());
        }
        Err(anyhow!("nested field {} cannot be null", self.name))
      }
      serde_json::Value::Array(arr) => {
        for v in arr.iter() {
          self.validate(v)?;
        }
        Ok(())
      }
      serde_json::Value::Object(map) => {
        for (k, v) in map.iter() {
          let Some(prop) = self.fields.iter().find(|p| p.name() == k) else {
            return Err(anyhow!("unknown nested field {k}"));
          };
          prop.validate_value(k, v)?;
        }
        Ok(())
      }
      _ => Err(anyhow!(
        "nested field {} must be object or array",
        self.name
      )),
    }
  }

  fn collect_fields(&self, prefix: Option<&str>, out: &mut Vec<ResolvedField>) {
    let mut full_prefix = String::new();
    if let Some(p) = prefix {
      full_prefix.push_str(p);
      full_prefix.push('.');
    }
    full_prefix.push_str(&self.name);
    for f in self.fields.iter() {
      f.collect_fields(&full_prefix, out);
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NestedProperty {
  Text(TextField),
  Keyword(KeywordField),
  Numeric(NumericField),
  Object(NestedField),
}

impl NestedProperty {
  pub fn name(&self) -> &str {
    match self {
      NestedProperty::Text(f) => &f.name,
      NestedProperty::Keyword(f) => &f.name,
      NestedProperty::Numeric(f) => &f.name,
      NestedProperty::Object(f) => &f.name,
    }
  }

  fn validate_value(&self, key: &str, v: &serde_json::Value) -> anyhow::Result<()> {
    match self {
      NestedProperty::Text(f) => {
        if v.is_null() {
          if f.nullable {
            return Ok(());
          }
          return Err(anyhow!("nested field {key} cannot be null"));
        }
        if !(v.is_string() || v.is_array()) {
          return Err(anyhow!("nested field {key} must be string or array"));
        }
        Ok(())
      }
      NestedProperty::Keyword(f) => {
        if v.is_null() {
          if f.nullable {
            return Ok(());
          }
          return Err(anyhow!("nested field {key} cannot be null"));
        }
        if !(v.is_string() || v.is_array()) {
          return Err(anyhow!("nested field {key} must be string or array"));
        }
        Ok(())
      }
      NestedProperty::Numeric(f) => {
        if v.is_null() {
          if f.nullable {
            return Ok(());
          }
          return Err(anyhow!("nested field {key} cannot be null"));
        }
        if !(v.is_number() || v.is_array()) {
          return Err(anyhow!("nested field {key} must be number or array"));
        }
        Ok(())
      }
      NestedProperty::Object(obj) => {
        if v.is_null() {
          if obj.nullable {
            return Ok(());
          }
          return Err(anyhow!("nested field {key} cannot be null"));
        }
        obj.validate(v)
      }
    }
  }

  fn collect_fields(&self, prefix: &str, out: &mut Vec<ResolvedField>) {
    match self {
      NestedProperty::Text(f) => out.push(ResolvedField {
        path: format!("{prefix}.{}", f.name),
        kind: FieldKind::Text,
        indexed: f.indexed,
        stored: f.stored,
        fast: false,
        numeric_i64: None,
      }),
      NestedProperty::Keyword(f) => out.push(ResolvedField {
        path: format!("{prefix}.{}", f.name),
        kind: FieldKind::Keyword,
        indexed: f.indexed,
        stored: f.stored,
        fast: f.fast,
        numeric_i64: None,
      }),
      NestedProperty::Numeric(f) => out.push(ResolvedField {
        path: format!("{prefix}.{}", f.name),
        kind: FieldKind::Numeric,
        indexed: true,
        stored: f.stored,
        fast: f.fast,
        numeric_i64: Some(f.i64),
      }),
      NestedProperty::Object(obj) => obj.collect_fields(Some(prefix), out),
    }
  }
}

fn collect_nested_text_fields<'a>(
  nested: &'a NestedField,
  prefix: Option<&str>,
  out: &mut Vec<(String, &'a TextField)>,
) {
  let mut full_prefix = String::new();
  if let Some(p) = prefix {
    full_prefix.push_str(p);
    full_prefix.push('.');
  }
  full_prefix.push_str(&nested.name);
  for f in nested.fields.iter() {
    match f {
      NestedProperty::Text(field) => {
        out.push((format!("{full_prefix}.{}", field.name), field));
      }
      NestedProperty::Object(obj) => collect_nested_text_fields(obj, Some(&full_prefix), out),
      _ => {}
    }
  }
}

#[cfg(feature = "vectors")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorField {
  pub name: String,
  pub dim: usize,
  pub metric: VectorMetric,
}

#[cfg(feature = "vectors")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorMetric {
  Cosine,
  L2,
}

#[cfg(feature = "vectors")]
impl From<crate::api::types::VectorMetric> for VectorMetric {
  fn from(v: crate::api::types::VectorMetric) -> Self {
    match v {
      crate::api::types::VectorMetric::Cosine => VectorMetric::Cosine,
      crate::api::types::VectorMetric::L2 => VectorMetric::L2,
    }
  }
}

#[cfg(feature = "vectors")]
impl From<VectorMetric> for crate::api::types::VectorMetric {
  fn from(v: VectorMetric) -> Self {
    match v {
      VectorMetric::Cosine => crate::api::types::VectorMetric::Cosine,
      VectorMetric::L2 => crate::api::types::VectorMetric::L2,
    }
  }
}
