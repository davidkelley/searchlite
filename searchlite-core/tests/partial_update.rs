use std::collections::BTreeMap;

use searchlite_core::api::types::{Document, IndexOptions, Schema, StorageType};
use searchlite_core::Index;
use tempfile::tempdir;

fn opts(path: &std::path::Path) -> IndexOptions {
  IndexOptions {
    path: path.to_path_buf(),
    create_if_missing: true,
    enable_positions: true,
    bm25_k1: 1.2,
    bm25_b: 0.75,
    storage: StorageType::Filesystem,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  }
}

#[test]
fn update_set_unset_top_level_fields() {
  let dir = tempdir().unwrap();
  let schema: Schema = serde_json::from_value(serde_json::json!({
    "doc_id_field": "_id",
    "text_fields": [
      { "name": "body", "analyzer": "default", "stored": true, "indexed": true, "nullable": false }
    ],
    "keyword_fields": [],
    "numeric_fields": [
      { "name": "count", "i64": true, "fast": false, "stored": true, "nullable": false }
    ],
    "nested_fields": []
  }))
  .unwrap();
  let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

  let mut writer = idx.writer().unwrap();
  writer
    .add_document(&Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("hello")),
        ("count".into(), serde_json::json!(5)),
      ]
      .into_iter()
      .collect(),
    })
    .unwrap();
  writer.commit().unwrap();

  let mut set = BTreeMap::new();
  set.insert("count".to_string(), serde_json::json!(10));
  let unset = vec!["body".to_string()];

  let mut writer = idx.writer().unwrap();
  writer.apply_patch("doc-1", &set, &unset).unwrap();
  writer.commit().unwrap();

  let reader = idx.reader().unwrap();
  let res = reader.mget(&["doc-1".to_string()], true).unwrap();
  let doc = res[0]._source.clone().unwrap();
  assert_eq!(doc["count"], 10);
  assert!(doc.get("body").is_none());
}

#[test]
fn update_supports_nested_paths() {
  let dir = tempdir().unwrap();
  let schema: Schema = serde_json::from_value(serde_json::json!({
    "doc_id_field": "_id",
    "text_fields": [{ "name": "body", "analyzer": "default", "stored": true, "indexed": true, "nullable": false }],
    "keyword_fields": [],
    "numeric_fields": [],
    "nested_fields": [
      { "name": "metadata", "nullable": true, "fields": [
          { "type": "keyword", "name": "alt", "stored": true, "indexed": true, "fast": false, "nullable": true }
        ]
      }
    ]
  }))
  .unwrap();
  let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

  let mut writer = idx.writer().unwrap();
  writer
    .add_document(&Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-2")),
        ("body".into(), serde_json::json!("hello")),
        ("metadata".into(), serde_json::json!({ "alt": "v1" })),
      ]
      .into_iter()
      .collect(),
    })
    .unwrap();
  writer.commit().unwrap();

  let mut set = BTreeMap::new();
  set.insert("metadata.alt".to_string(), serde_json::json!("v2"));

  let mut writer = idx.writer().unwrap();
  writer.apply_patch("doc-2", &set, &[]).unwrap();
  writer.commit().unwrap();

  let reader = idx.reader().unwrap();
  let res = reader.mget(&["doc-2".to_string()], true).unwrap();
  let doc = res[0]._source.clone().unwrap();
  assert_eq!(doc["metadata"]["alt"], "v2");
}
