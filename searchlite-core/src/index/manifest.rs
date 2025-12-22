use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
  pub text_fields: Vec<TextField>,
  pub keyword_fields: Vec<KeywordField>,
  pub numeric_fields: Vec<NumericField>,
  #[serde(default)]
  pub nested_fields: Vec<NestedField>,
  #[cfg(feature = "vectors")]
  pub vector_fields: Vec<VectorField>,
}

impl Schema {
  pub fn default_text_body() -> Self {
    Self {
      text_fields: vec![TextField {
        name: "body".to_string(),
        tokenizer: "default".to_string(),
        stored: true,
        indexed: true,
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

  pub fn validate_document(&self, doc: &crate::api::types::Document) -> anyhow::Result<()> {
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
  fn persists_manifest_and_schema_helpers() {
    let dir = tempdir().unwrap();
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let schema = Schema {
      text_fields: vec![TextField {
        name: "body".into(),
        tokenizer: "default".into(),
        stored: true,
        indexed: true,
      }],
      keyword_fields: vec![KeywordField {
        name: "tag".into(),
        stored: true,
        indexed: true,
        fast: true,
      }],
      numeric_fields: vec![NumericField {
        name: "year".into(),
        i64: true,
        fast: true,
        stored: true,
      }],
      nested_fields: vec![NestedField {
        name: "comment".into(),
        fields: vec![NestedProperty::Keyword(KeywordField {
          name: "author".into(),
          stored: true,
          indexed: true,
          fast: true,
        })],
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextField {
  pub name: String,
  pub tokenizer: String,
  pub stored: bool,
  pub indexed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordField {
  pub name: String,
  pub stored: bool,
  pub indexed: bool,
  pub fast: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericField {
  pub name: String,
  pub i64: bool,
  pub fast: bool,
  #[serde(default)]
  pub stored: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedField {
  pub name: String,
  #[serde(default)]
  pub fields: Vec<NestedProperty>,
}

impl NestedField {
  fn validate(&self, value: &serde_json::Value) -> anyhow::Result<()> {
    match value {
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
          match prop {
            NestedProperty::Text(_) | NestedProperty::Keyword(_) => {
              if !(v.is_string() || v.is_array()) {
                return Err(anyhow!("nested field {k} must be string or array"));
              }
            }
            NestedProperty::Numeric(_) => {
              if !(v.is_number() || v.is_array()) {
                return Err(anyhow!("nested field {k} must be number or array"));
              }
            }
            NestedProperty::Object(obj) => {
              obj.validate(v)?;
            }
          }
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
