use crate::api::types::Filter;
use crate::index::fastfields::FastFieldsReader;
use crate::DocId;

pub fn passes_filters(reader: &FastFieldsReader, doc_id: DocId, filters: &[Filter]) -> bool {
  passes_filters_at(reader, doc_id, filters, "", None)
}

pub fn passes_filter(reader: &FastFieldsReader, doc_id: DocId, filter: &Filter) -> bool {
  filter_matches(reader, doc_id, filter, "", None)
}

fn passes_filters_at<'a>(
  reader: &FastFieldsReader,
  doc_id: DocId,
  filters: &'a [Filter],
  base_path: &str,
  object_idx: Option<usize>,
) -> bool {
  let mut nested: std::collections::HashMap<&str, Vec<&'a Filter>> =
    std::collections::HashMap::new();
  for filter in filters.iter() {
    match filter {
      Filter::Nested { path, filter } => {
        nested
          .entry(path.as_str())
          .or_default()
          .push(filter.as_ref());
      }
      _ => {
        if !filter_matches(reader, doc_id, filter, base_path, object_idx) {
          return false;
        }
      }
    }
  }
  for (path, group) in nested.into_iter() {
    if !nested_group_passes(
      reader,
      doc_id,
      base_path,
      path,
      object_idx,
      group.as_slice(),
    ) {
      return false;
    }
  }
  true
}

fn nested_group_passes(
  reader: &FastFieldsReader,
  doc_id: DocId,
  base_path: &str,
  path: &str,
  parent_idx: Option<usize>,
  filters: &[&Filter],
) -> bool {
  let full_path = if base_path.is_empty() {
    path.to_string()
  } else {
    format!("{base_path}.{path}")
  };
  let object_count = reader.nested_object_count(&full_path, doc_id);
  if object_count == 0 {
    return false;
  }
  let owned: Vec<Filter> = filters.iter().map(|f| (*f).clone()).collect();
  let parents = reader.nested_parents(&full_path, doc_id);
  for idx in 0..object_count {
    if let Some(p) = parent_idx {
      if parents.get(idx).and_then(|v| *v) != Some(p) {
        continue;
      }
    }
    if passes_filters_at(reader, doc_id, &owned, &full_path, Some(idx)) {
      return true;
    }
  }
  false
}

fn filter_matches(
  reader: &FastFieldsReader,
  doc_id: DocId,
  filter: &Filter,
  base_path: &str,
  object_idx: Option<usize>,
) -> bool {
  match filter {
    Filter::KeywordEq { field, value } => {
      let full = qualified_field(base_path, field);
      match object_idx {
        Some(idx) => reader
          .nested_str_values(&full, doc_id)
          .get(idx)
          .map(|vals| vals.iter().any(|v| v.eq_ignore_ascii_case(value)))
          .unwrap_or(false),
        None => reader.matches_keyword(&full, doc_id, value),
      }
    }
    Filter::KeywordIn { field, values } => {
      let full = qualified_field(base_path, field);
      match object_idx {
        Some(idx) => reader
          .nested_str_values(&full, doc_id)
          .get(idx)
          .map(|vals| {
            vals
              .iter()
              .any(|v| values.iter().any(|t| t.eq_ignore_ascii_case(v)))
          })
          .unwrap_or(false),
        None => reader.matches_keyword_in(&full, doc_id, values),
      }
    }
    Filter::I64Range { field, min, max } => {
      let full = qualified_field(base_path, field);
      match object_idx {
        Some(idx) => reader
          .nested_i64_values(&full, doc_id)
          .get(idx)
          .map(|vals| vals.iter().any(|v| *v >= *min && *v <= *max))
          .unwrap_or(false),
        None => reader.matches_i64_range(&full, doc_id, *min, *max),
      }
    }
    Filter::F64Range { field, min, max } => {
      let full = qualified_field(base_path, field);
      match object_idx {
        Some(idx) => reader
          .nested_f64_values(&full, doc_id)
          .get(idx)
          .map(|vals| vals.iter().any(|v| *v >= *min && *v <= *max))
          .unwrap_or(false),
        None => reader.matches_f64_range(&full, doc_id, *min, *max),
      }
    }
    Filter::Nested { path, filter } => {
      nested_filter_passes(reader, doc_id, base_path, path, object_idx, filter.as_ref())
    }
    Filter::And(filters) => passes_filters_at(reader, doc_id, filters, base_path, object_idx),
    Filter::Or(filters) => filters
      .iter()
      .any(|child| filter_matches(reader, doc_id, child, base_path, object_idx)),
    Filter::Not(filter) => !filter_matches(reader, doc_id, filter.as_ref(), base_path, object_idx),
  }
}

fn nested_filter_passes(
  reader: &FastFieldsReader,
  doc_id: DocId,
  base_path: &str,
  path: &str,
  parent_idx: Option<usize>,
  filter: &Filter,
) -> bool {
  let full_path = if base_path.is_empty() {
    path.to_string()
  } else {
    format!("{base_path}.{path}")
  };
  let object_count = reader.nested_object_count(&full_path, doc_id);
  if object_count == 0 {
    return false;
  }
  let parents = reader.nested_parents(&full_path, doc_id);
  for idx in 0..object_count {
    if let Some(p) = parent_idx {
      if parents.get(idx).and_then(|v| *v) != Some(p) {
        continue;
      }
    }
    if filter_matches(reader, doc_id, filter, &full_path, Some(idx)) {
      return true;
    }
  }
  false
}

fn qualified_field(base: &str, field: &str) -> String {
  if base.is_empty() {
    field.to_string()
  } else {
    format!("{base}.{field}")
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::index::fastfields::{
    nested_count_key, nested_parent_key, FastFieldsWriter, FastValue,
  };
  use tempfile::tempdir;

  #[test]
  fn keyword_filters_are_case_insensitive() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.json");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut writer = FastFieldsWriter::new();
    writer.set("cat", 0, FastValue::Str("News".into()));
    writer.set(
      &nested_count_key("comment"),
      0,
      FastValue::NestedCount { objects: 1 },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 0,
        values: vec!["Alice".into()],
      },
    );
    writer.write_to(&storage, &path).unwrap();
    let reader = FastFieldsReader::open(&storage, &path).unwrap();

    let filters = vec![
      Filter::KeywordEq {
        field: "cat".into(),
        value: "news".into(),
      },
      Filter::KeywordIn {
        field: "cat".into(),
        values: vec!["sports".into(), "NEWS".into()],
      },
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "author".into(),
          value: "alice".into(),
        }),
      },
    ];
    assert!(passes_filters(&reader, 0, &filters));

    let rejecting = vec![Filter::KeywordEq {
      field: "cat".into(),
      value: "other".into(),
    }];
    assert!(!passes_filters(&reader, 0, &rejecting));
  }

  #[test]
  fn evaluates_all_filter_types() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.json");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut writer = FastFieldsWriter::new();
    writer.set("cat", 0, FastValue::Str("news".into()));
    writer.set(
      &nested_count_key("comment"),
      0,
      FastValue::NestedCount { objects: 1 },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 0,
        values: vec!["alice".into()],
      },
    );
    writer.set("year", 0, FastValue::I64(2024));
    writer.set("score", 0, FastValue::F64(0.75));
    writer.write_to(&storage, &path).unwrap();
    let reader = FastFieldsReader::open(&storage, &path).unwrap();

    let filters = vec![
      Filter::KeywordEq {
        field: "cat".into(),
        value: "news".into(),
      },
      Filter::KeywordIn {
        field: "cat".into(),
        values: vec!["sports".into(), "news".into()],
      },
      Filter::I64Range {
        field: "year".into(),
        min: 2020,
        max: 2025,
      },
      Filter::F64Range {
        field: "score".into(),
        min: 0.5,
        max: 1.0,
      },
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "author".into(),
          value: "alice".into(),
        }),
      },
    ];
    assert!(passes_filters(&reader, 0, &filters));

    let rejecting = vec![Filter::I64Range {
      field: "year".into(),
      min: 2025,
      max: 2030,
    }];
    assert!(!passes_filters(&reader, 0, &rejecting));
  }

  #[test]
  fn nested_filters_require_shared_object() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.json");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut writer = FastFieldsWriter::new();
    writer.set(
      &nested_count_key("comment"),
      0,
      FastValue::NestedCount { objects: 2 },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 0,
        values: vec!["alice".into()],
      },
    );
    writer.set(
      "comment.tag",
      0,
      FastValue::StrNested {
        object: 1,
        values: vec!["rust".into()],
      },
    );
    writer.write_to(&storage, &path).unwrap();
    let reader = FastFieldsReader::open(&storage, &path).unwrap();

    let filters = vec![
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "author".into(),
          value: "alice".into(),
        }),
      },
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "tag".into(),
          value: "rust".into(),
        }),
      },
    ];

    assert!(!passes_filters(&reader, 0, &filters));

    let aligned = vec![Filter::Nested {
      path: "comment".into(),
      filter: Box::new(Filter::KeywordIn {
        field: "author".into(),
        values: vec!["alice".into()],
      }),
    }];
    assert!(passes_filters(&reader, 0, &aligned));
  }

  #[test]
  fn nested_filters_bind_to_parent_objects() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.json");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut writer = FastFieldsWriter::new();
    writer.set(
      &nested_count_key("comment"),
      0,
      FastValue::NestedCount { objects: 2 },
    );
    writer.set(
      &nested_parent_key("comment"),
      0,
      FastValue::NestedParent {
        object: 0,
        parent: u32::MAX as usize,
      },
    );
    writer.set(
      &nested_parent_key("comment"),
      0,
      FastValue::NestedParent {
        object: 1,
        parent: u32::MAX as usize,
      },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 0,
        values: vec!["alice".into()],
      },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 1,
        values: vec!["bob".into()],
      },
    );
    writer.set(
      &nested_count_key("comment.reply"),
      0,
      FastValue::NestedCount { objects: 2 },
    );
    writer.set(
      &nested_parent_key("comment.reply"),
      0,
      FastValue::NestedParent {
        object: 0,
        parent: 0,
      },
    );
    writer.set(
      &nested_parent_key("comment.reply"),
      0,
      FastValue::NestedParent {
        object: 1,
        parent: 1,
      },
    );
    writer.set(
      "comment.reply.tag",
      0,
      FastValue::StrNested {
        object: 0,
        values: vec!["x".into()],
      },
    );
    writer.set(
      "comment.reply.tag",
      0,
      FastValue::StrNested {
        object: 1,
        values: vec!["y".into()],
      },
    );
    writer.write_to(&storage, &path).unwrap();
    let reader = FastFieldsReader::open(&storage, &path).unwrap();

    let filters = vec![
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::Nested {
          path: "reply".into(),
          filter: Box::new(Filter::KeywordEq {
            field: "tag".into(),
            value: "y".into(),
          }),
        }),
      },
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "author".into(),
          value: "alice".into(),
        }),
      },
    ];

    // Should not pass because the reply with tag "y" belongs to the bob comment.
    assert!(!passes_filters(&reader, 0, &filters));

    let aligned = vec![
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "author".into(),
          value: "bob".into(),
        }),
      },
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::Nested {
          path: "reply".into(),
          filter: Box::new(Filter::KeywordEq {
            field: "tag".into(),
            value: "y".into(),
          }),
        }),
      },
    ];
    assert!(passes_filters(&reader, 0, &aligned));
  }

  #[test]
  fn nested_numeric_filters_scope_to_same_object() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.json");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut writer = FastFieldsWriter::new();
    writer.set(
      &nested_count_key("comment"),
      0,
      FastValue::NestedCount { objects: 2 },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 0,
        values: vec!["alice".into()],
      },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 1,
        values: vec!["bob".into()],
      },
    );
    writer.set(
      "comment.stars",
      0,
      FastValue::I64Nested {
        object: 0,
        values: vec![5],
      },
    );
    writer.set(
      "comment.stars",
      0,
      FastValue::I64Nested {
        object: 1,
        values: vec![9],
      },
    );
    writer.set(
      "comment.score",
      0,
      FastValue::F64Nested {
        object: 0,
        values: vec![0.72],
      },
    );
    writer.set(
      "comment.score",
      0,
      FastValue::F64Nested {
        object: 1,
        values: vec![0.21],
      },
    );
    writer.write_to(&storage, &path).unwrap();
    let reader = FastFieldsReader::open(&storage, &path).unwrap();

    let passing = vec![
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "author".into(),
          value: "alice".into(),
        }),
      },
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::I64Range {
          field: "stars".into(),
          min: 5,
          max: 7,
        }),
      },
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::F64Range {
          field: "score".into(),
          min: 0.7,
          max: 0.8,
        }),
      },
    ];
    assert!(passes_filters(&reader, 0, &passing));

    let failing = vec![
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "author".into(),
          value: "alice".into(),
        }),
      },
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::I64Range {
          field: "stars".into(),
          min: 8,
          max: 10,
        }),
      },
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::F64Range {
          field: "score".into(),
          min: 0.2,
          max: 0.3,
        }),
      },
    ];
    assert!(!passes_filters(&reader, 0, &failing));
  }

  #[test]
  fn nested_and_filters_require_shared_object_in_and() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.json");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut writer = FastFieldsWriter::new();
    writer.set(
      &nested_count_key("comment"),
      0,
      FastValue::NestedCount { objects: 2 },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 0,
        values: vec!["alice".into()],
      },
    );
    writer.set(
      "comment.tag",
      0,
      FastValue::StrNested {
        object: 1,
        values: vec!["rust".into()],
      },
    );
    writer.write_to(&storage, &path).unwrap();
    let reader = FastFieldsReader::open(&storage, &path).unwrap();

    let filter = Filter::And(vec![
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "author".into(),
          value: "alice".into(),
        }),
      },
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "tag".into(),
          value: "rust".into(),
        }),
      },
    ]);

    assert!(!passes_filter(&reader, 0, &filter));
  }

  #[test]
  fn nested_and_filters_match_shared_object_in_and() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.json");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut writer = FastFieldsWriter::new();
    writer.set(
      &nested_count_key("comment"),
      0,
      FastValue::NestedCount { objects: 1 },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 0,
        values: vec!["alice".into()],
      },
    );
    writer.set(
      "comment.tag",
      0,
      FastValue::StrNested {
        object: 0,
        values: vec!["rust".into()],
      },
    );
    writer.write_to(&storage, &path).unwrap();
    let reader = FastFieldsReader::open(&storage, &path).unwrap();

    let filter = Filter::And(vec![
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "author".into(),
          value: "alice".into(),
        }),
      },
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "tag".into(),
          value: "rust".into(),
        }),
      },
    ]);

    assert!(passes_filter(&reader, 0, &filter));
  }

  #[test]
  fn nested_or_filters_match_any_object() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.json");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut writer = FastFieldsWriter::new();
    writer.set(
      &nested_count_key("comment"),
      0,
      FastValue::NestedCount { objects: 2 },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 0,
        values: vec!["alice".into()],
      },
    );
    writer.set(
      "comment.tag",
      0,
      FastValue::StrNested {
        object: 1,
        values: vec!["rust".into()],
      },
    );
    writer.write_to(&storage, &path).unwrap();
    let reader = FastFieldsReader::open(&storage, &path).unwrap();

    let filter = Filter::Or(vec![
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "author".into(),
          value: "alice".into(),
        }),
      },
      Filter::Nested {
        path: "comment".into(),
        filter: Box::new(Filter::KeywordEq {
          field: "tag".into(),
          value: "rust".into(),
        }),
      },
    ]);

    assert!(passes_filter(&reader, 0, &filter));
  }

  #[test]
  fn nested_not_filters_negate_nested() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.json");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut writer = FastFieldsWriter::new();
    writer.set(
      &nested_count_key("comment"),
      0,
      FastValue::NestedCount { objects: 1 },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 0,
        values: vec!["alice".into()],
      },
    );
    writer.write_to(&storage, &path).unwrap();
    let reader = FastFieldsReader::open(&storage, &path).unwrap();

    let rejecting = Filter::Not(Box::new(Filter::Nested {
      path: "comment".into(),
      filter: Box::new(Filter::KeywordEq {
        field: "author".into(),
        value: "alice".into(),
      }),
    }));
    assert!(!passes_filter(&reader, 0, &rejecting));

    let accepting = Filter::Not(Box::new(Filter::Nested {
      path: "comment".into(),
      filter: Box::new(Filter::KeywordEq {
        field: "author".into(),
        value: "bob".into(),
      }),
    }));
    assert!(passes_filter(&reader, 0, &accepting));
  }
}
