use std::fs;
use std::path::Path;

use anyhow::Context;

use crate::api::types::IndexOptions;
use crate::index::manifest::Schema;
use crate::index::Index;

pub struct IndexBuilder;

impl IndexBuilder {
  pub fn create(path: &Path, schema: Schema, opts: IndexOptions) -> anyhow::Result<Index> {
    Index::create(path, schema, opts)
  }

  pub fn create_from_schema_file(
    path: &Path,
    schema_path: &Path,
    opts: IndexOptions,
  ) -> anyhow::Result<Index> {
    let data = fs::read_to_string(schema_path)
      .with_context(|| format!("reading schema file {:?}", schema_path))?;
    let schema: Schema = serde_json::from_str(&data)
      .with_context(|| format!("parsing schema file {:?}", schema_path))?;
    Self::create(path, schema, opts)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::api::types::Schema;
  use tempfile::tempdir;

  #[test]
  fn creates_index_from_schema_file() {
    let dir = tempdir().unwrap();
    let schema_path = dir.path().join("schema.json");
    let schema = Schema::default_text_body();
    std::fs::write(&schema_path, serde_json::to_string(&schema).unwrap()).unwrap();
    let index_path = dir.path().join("idx");
    let opts = IndexOptions {
      path: index_path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 1.1,
      bm25_b: 0.4,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    };
    let idx = IndexBuilder::create_from_schema_file(&index_path, &schema_path, opts).unwrap();
    assert!(idx.manifest().segments.is_empty());
  }
}
