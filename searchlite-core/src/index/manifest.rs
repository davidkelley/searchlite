use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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

  pub fn load(path: &Path) -> Result<Self> {
    let data =
      fs::read_to_string(path).with_context(|| format!("reading manifest at {:?}", path))?;
    let manifest: Manifest =
      serde_json::from_str(&data).with_context(|| format!("parsing manifest at {:?}", path))?;
    Ok(manifest)
  }

  pub fn store(&self, path: &Path) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    let data = serde_json::to_vec_pretty(self)?;
    {
      let mut file =
        fs::File::create(&tmp).with_context(|| format!("writing manifest tmp at {:?}", tmp))?;
      file.write_all(&data)?;
      file.sync_all()?;
    }
    fs::rename(&tmp, path).with_context(|| format!("atomic rename manifest to {:?}", path))?;
    Ok(())
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
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    }
  }

  pub fn is_indexed_field(&self, field: &str) -> bool {
    self
      .text_fields
      .iter()
      .any(|f| f.name == field && f.indexed)
      || self
        .keyword_fields
        .iter()
        .any(|f| f.name == field && f.indexed)
      || self.numeric_fields.iter().any(|f| f.name == field)
  }

  pub fn is_stored_field(&self, field: &str) -> bool {
    self.text_fields.iter().any(|f| f.name == field && f.stored)
      || self
        .keyword_fields
        .iter()
        .any(|f| f.name == field && f.stored)
      || self
        .numeric_fields
        .iter()
        .any(|f| f.name == field && f.stored)
  }

  pub fn fast_fields(&self) -> Vec<String> {
    self
      .numeric_fields
      .iter()
      .filter(|f| f.fast)
      .map(|f| f.name.clone())
      .chain(
        self
          .keyword_fields
          .iter()
          .filter(|f| f.fast)
          .map(|f| f.name.clone()),
      )
      .collect()
  }

  pub fn field_kind(&self, field: &str) -> FieldKind {
    if self.text_fields.iter().any(|f| f.name == field) {
      FieldKind::Text
    } else if self.keyword_fields.iter().any(|f| f.name == field) {
      FieldKind::Keyword
    } else if self.numeric_fields.iter().any(|f| f.name == field) {
      FieldKind::Numeric
    } else {
      FieldKind::Unknown
    }
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
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };
    let manifest = Manifest::new(schema.clone());
    let path = Manifest::manifest_path(dir.path());
    manifest.store(&path).unwrap();
    let loaded = Manifest::load(&path).unwrap();
    assert!(loaded.schema.is_indexed_field("body"));
    assert!(loaded.schema.is_stored_field("year"));
    assert_eq!(
      loaded.schema.fast_fields(),
      vec!["year".to_string(), "tag".to_string()]
    );
    assert!(matches!(
      loaded.schema.field_kind("year"),
      FieldKind::Numeric
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
