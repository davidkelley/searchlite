use std::collections::BTreeMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Document, ExecutionStrategy, IndexOptions, KeywordField, NestedField, NestedProperty,
  NumericField, Schema, SearchRequest, StorageType,
};
use searchlite_core::api::Filter;
use serde_json::json;

fn bench_indexing(c: &mut Criterion) {
  c.bench_function("index_small", |b| {
    b.iter(|| {
      let tempdir = tempfile::tempdir().unwrap();
      let path = tempdir.path().to_path_buf();
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
              "_id".to_string(),
              serde_json::json!(format!("bench-add-{i}")),
            ),
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
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().to_path_buf();
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
            (
              "_id".to_string(),
              serde_json::json!(format!("bench-search-{i}")),
            ),
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
        query: "rust".into(),
        fields: None,
        filter: None,
        filters: vec![],
        limit: 5,
        sort: Vec::new(),
        cursor: None,
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        fuzzy: None,
        #[cfg(feature = "vectors")]
        vector_query: None,
        return_stored: true,
        highlight_field: None,
        aggs: BTreeMap::new(),
        suggest: BTreeMap::new(),
        rescore: None,
        explain: false,
        profile: false,
      };
      let _ = reader.search(&req).unwrap();
    });
  });
}

fn bench_nested_filters(c: &mut Criterion) {
  c.bench_function("search_nested_filters", |b| {
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().to_path_buf();
    let mut schema = Schema::default_text_body();
    schema.nested_fields.push(NestedField {
      name: "review".into(),
      fields: vec![
        NestedProperty::Keyword(KeywordField {
          name: "user".into(),
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
        NestedProperty::Numeric(NumericField {
          name: "score".into(),
          i64: false,
          fast: true,
          stored: false,
          nullable: false,
        }),
      ],
      nullable: false,
    });
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
      for i in 0..40u32 {
        let reviews = json!([
          { "user": "user_a", "rating": (i % 6) + 1, "score": 0.5 + (i as f64 % 5.0) * 0.05 },
          { "user": "user_b", "rating": 9, "score": 0.2 }
        ]);
        writer
          .add_document(&Document {
            fields: [
              ("_id".into(), json!(format!("nested-{i}"))),
              ("body".into(), json!(format!("rust nested {}", i))),
              ("review".into(), reviews),
            ]
            .into_iter()
            .collect(),
          })
          .unwrap();
      }
      writer.commit().unwrap();
    }
    b.iter(|| {
      let reader = idx.reader().unwrap();
      let req = SearchRequest {
        query: "rust".into(),
        fields: None,
        filter: None,
        filters: vec![
          Filter::Nested {
            path: "review".into(),
            filter: Box::new(Filter::KeywordEq {
              field: "user".into(),
              value: "user_a".into(),
            }),
          },
          Filter::Nested {
            path: "review".into(),
            filter: Box::new(Filter::I64Range {
              field: "rating".into(),
              min: 3,
              max: 6,
            }),
          },
        ],
        limit: 5,
        sort: Vec::new(),
        cursor: None,
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        fuzzy: None,
        #[cfg(feature = "vectors")]
        vector_query: None,
        return_stored: true,
        highlight_field: None,
        aggs: BTreeMap::new(),
        suggest: BTreeMap::new(),
        rescore: None,
        explain: false,
        profile: false,
      };
      let _ = reader.search(&req).unwrap();
    });
  });
}

fn bench_cursor_pagination(c: &mut Criterion) {
  c.bench_function("search_cursor_pagination", |b| {
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().to_path_buf();
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
    for batch in 0..5 {
      let mut writer = idx.writer().unwrap();
      for i in 0..500u32 {
        let doc = Document {
          fields: [
            (
              "body".to_string(),
              serde_json::json!(format!("rust {}", i + batch * 500)),
            ),
            ("year".to_string(), serde_json::json!(2020 + (i % 5))),
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
      let mut cursor = None;
      let mut total = 0usize;
      loop {
        let req = SearchRequest {
          query: "rust".into(),
          fields: None,
          filter: None,
          filters: vec![],
          limit: 20,
          sort: Vec::new(),
          cursor,
          execution: ExecutionStrategy::Wand,
          bmw_block_size: None,
          fuzzy: None,
          #[cfg(feature = "vectors")]
          vector_query: None,
          return_stored: false,
          highlight_field: None,
          aggs: BTreeMap::new(),
          suggest: BTreeMap::new(),
          rescore: None,
          explain: false,
          profile: false,
        };
        let res = reader.search(&req).unwrap();
        total += res.hits.len();
        if let Some(next) = res.next_cursor {
          cursor = Some(next);
        } else {
          break;
        }
      }
      black_box(total);
    });
  });
}

criterion_group!(
  benches,
  bench_indexing,
  bench_search,
  bench_nested_filters,
  bench_cursor_pagination
);
criterion_main!(benches);
