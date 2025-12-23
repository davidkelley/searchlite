use crate::api::types::Filter;
use crate::index::fastfields::FastFieldsReader;
use crate::DocId;

pub fn passes_filters(reader: &FastFieldsReader, doc_id: DocId, filters: &[Filter]) -> bool {
  passes_filters_at(reader, doc_id, filters, "", None)
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
        if !filter_passes_flat(reader, doc_id, filter, base_path, object_idx) {
          return false;
        }
      }
    }
  }
  for (path, group) in nested.into_iter() {
    if !nested_group_passes(reader, doc_id, base_path, path, group.as_slice()) {
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
  for idx in 0..object_count {
    if passes_filters_at(reader, doc_id, &owned, &full_path, Some(idx)) {
      return true;
    }
  }
  false
}

fn filter_passes_flat(
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
          .map(|vals| vals.iter().any(|v| v == value))
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
          .map(|vals| vals.iter().any(|v| values.iter().any(|t| t == v)))
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
    Filter::Nested { .. } => false,
  }
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
  use crate::index::fastfields::{nested_count_key, FastFieldsWriter, FastValue};
  use tempfile::tempdir;

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
}
