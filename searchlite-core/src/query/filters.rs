use crate::api::types::Filter;
use crate::index::fastfields::FastFieldsReader;
use crate::DocId;

pub fn passes_filters(reader: &FastFieldsReader, doc_id: DocId, filters: &[Filter]) -> bool {
  filters.iter().all(|f| filter_passes(reader, doc_id, f))
}

fn filter_passes(reader: &FastFieldsReader, doc_id: DocId, filter: &Filter) -> bool {
  match filter {
    Filter::KeywordEq { field, value } => reader.matches_keyword(field, doc_id, value),
    Filter::KeywordIn { field, values } => reader.matches_keyword_in(field, doc_id, values),
    Filter::I64Range { field, min, max } => reader.matches_i64_range(field, doc_id, *min, *max),
    Filter::F64Range { field, min, max } => reader.matches_f64_range(field, doc_id, *min, *max),
    Filter::Nested { path, filter } => {
      let prefixed = prefix_filter(path, filter);
      filter_passes(reader, doc_id, &prefixed)
    }
  }
}

fn prefix_filter(prefix: &str, filter: &Filter) -> Filter {
  match filter {
    Filter::KeywordEq { field, value } => Filter::KeywordEq {
      field: format!("{prefix}.{field}"),
      value: value.clone(),
    },
    Filter::KeywordIn { field, values } => Filter::KeywordIn {
      field: format!("{prefix}.{field}"),
      values: values.clone(),
    },
    Filter::I64Range { field, min, max } => Filter::I64Range {
      field: format!("{prefix}.{field}"),
      min: *min,
      max: *max,
    },
    Filter::F64Range { field, min, max } => Filter::F64Range {
      field: format!("{prefix}.{field}"),
      min: *min,
      max: *max,
    },
    Filter::Nested { path, filter } => {
      let nested_prefix = format!("{prefix}.{path}");
      prefix_filter(&nested_prefix, filter)
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::index::fastfields::{FastFieldsWriter, FastValue};
  use tempfile::tempdir;

  #[test]
  fn evaluates_all_filter_types() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.json");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut writer = FastFieldsWriter::new();
    writer.set("cat", 0, FastValue::Str("news".into()));
    writer.set("comment.author", 0, FastValue::Str("alice".into()));
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
}
