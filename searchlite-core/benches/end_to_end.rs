use std::collections::BTreeMap;

use criterion::{criterion_group, criterion_main, Criterion};
use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Document, ExecutionStrategy, IndexOptions, Schema, SearchRequest, StorageType,
};
use searchlite_core::api::Index;

fn bench_indexing(c: &mut Criterion) {
  c.bench_function("index_small", |b| {
    b.iter(|| {
      let path = tempfile::tempdir().unwrap().into_path();
      let schema = Schema::default_text_body();
      let opts = IndexOptions {
        path: path.clone(),
        create_if_missing: true,
        enable_positions: true,
        bm25_k1: 0.9,
        bm25_b: 0.4,
        storage: StorageType::Filesystem,
        #[cfg(feature = "vectors")]
        vector_defaults: None,
      };
      let idx = IndexBuilder::create(&path, schema, opts).unwrap();
      let mut writer = idx.writer().unwrap();
      for i in 0..50u32 {
        let doc = Document {
          fields: [
            (
              "body".to_string(),
              serde_json::json!(format!("rust language {}", i)),
            ),
            ("year".to_string(), serde_json::json!(2020 + (i % 3))),
          ]
          .into_iter()
          .collect(),
        };
        writer.add_document(&doc).unwrap();
      }
      writer.commit().unwrap();
    });
  });
}

fn bench_search(c: &mut Criterion) {
  c.bench_function("search_small", |b| {
    let path = tempfile::tempdir().unwrap().into_path();
    let schema = Schema::default_text_body();
    let opts = IndexOptions {
      path: path.clone(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    };
    let idx = IndexBuilder::create(&path, schema, opts).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      for i in 0..20u32 {
        let doc = Document {
          fields: [
            ("body".to_string(), serde_json::json!(format!("rust {}", i))),
            ("year".to_string(), serde_json::json!(2020 + (i % 3))),
          ]
          .into_iter()
          .collect(),
        };
        writer.add_document(&doc).unwrap();
      }
      writer.commit().unwrap();
    }
    b.iter(|| {
      let reader = idx.reader().unwrap();
      let req = SearchRequest {
        query: "rust".to_string(),
        fields: None,
        filters: vec![],
        limit: 5,
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        #[cfg(feature = "vectors")]
        vector_query: None,
        return_stored: true,
        highlight_field: None,
        aggs: BTreeMap::new(),
      };
      let _ = reader.search(&req).unwrap();
    });
  });
}

criterion_group!(benches, bench_indexing, bench_search);
criterion_main!(benches);
