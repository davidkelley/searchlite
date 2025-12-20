use searchlite_core::api::types::Document;
use searchlite_core::storage::FsStorage;
use searchlite_core::util::varint;
use searchlite_core::wal::Wal;
use std::sync::Arc;

#[test]
fn wal_roundtrip() {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("wal.log");
  let storage = Arc::new(FsStorage::new(dir.path().to_path_buf()));
  let mut wal = Wal::open(storage.clone(), &path).unwrap();
  let doc = Document {
    fields: [("body".to_string(), serde_json::json!("hello wal"))]
      .into_iter()
      .collect(),
  };
  wal.append_add_doc(&doc).unwrap();
  wal.append_commit().unwrap();
  let entries = Wal::replay(storage.as_ref(), &path).unwrap();
  assert!(!entries.is_empty());
}

#[test]
fn varint_roundtrip() {
  let mut buf = Vec::new();
  varint::write_u64(300, &mut buf);
  let (v, len) = varint::read_u64(&buf).unwrap();
  assert_eq!(v, 300);
  assert!(len <= buf.len());
}
