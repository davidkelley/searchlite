use crate::api::types::Filter;
use crate::index::fastfields::FastFieldsReader;
use crate::DocId;

pub fn passes_filters(reader: &FastFieldsReader, doc_id: DocId, filters: &[Filter]) -> bool {
  for f in filters {
    match f {
      Filter::KeywordEq { field, value } => {
        if !reader.matches_keyword(field, doc_id, value) {
          return false;
        }
      }
      Filter::KeywordIn { field, values } => {
        if !reader.matches_keyword_in(field, doc_id, values) {
          return false;
        }
      }
      Filter::I64Range { field, min, max } => {
        if !reader.matches_i64_range(field, doc_id, *min, *max) {
          return false;
        }
      }
      Filter::F64Range { field, min, max } => {
        if !reader.matches_f64_range(field, doc_id, *min, *max) {
          return false;
        }
      }
    }
  }
  true
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
