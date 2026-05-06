use std::path::{Path, PathBuf};

use anyhow::Result;

use super::manifest::SegmentPaths;
use crate::storage::Storage;

pub fn ensure_root(storage: &dyn Storage, path: &Path) -> Result<()> {
  storage.ensure_dir(path)
}

pub fn wal_path(root: &Path) -> PathBuf {
  root.join("wal.log")
}

/// Stage 9a: emit **relative-to-root keys** rather than absolute
/// paths. Each field is a per-file (or per-directory, for vectors)
/// key under the index root. Resolution against an actual filesystem
/// root happens at every read/write call site via
/// [`SegmentPaths::resolve`]. The `_root` parameter is preserved for
/// API stability but is no longer dereferenced here — keys never
/// embed the root, which is what makes manifests portable.
pub fn segment_paths(_root: &Path, id: &str) -> SegmentPaths {
  SegmentPaths {
    terms: format!("seg_{id}.terms"),
    postings: format!("seg_{id}.post"),
    docstore: format!("seg_{id}.docs"),
    fast: format!("seg_{id}.fast"),
    meta: format!("seg_{id}.meta"),
    #[cfg(feature = "vectors")]
    vector_dir: Some(format!("seg_{id}_vectors")),
  }
}

#[allow(dead_code)]
pub fn segment_meta_path(root: &Path, id: &str) -> PathBuf {
  root.join(format!("seg_{id}.meta"))
}

#[allow(dead_code)]
pub fn manifest_path(root: &Path) -> PathBuf {
  root.join("MANIFEST.json")
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[test]
  fn builds_relative_keys_independent_of_root() {
    // Stage 9a: keys must NOT embed the root. Two different roots
    // produce identical key strings — this is the property that makes
    // manifests relocatable.
    let dir_a = tempdir().unwrap();
    let dir_b = tempdir().unwrap();
    let storage_a = crate::storage::FsStorage::new(dir_a.path().to_path_buf());
    ensure_root(&storage_a, dir_a.path()).unwrap();
    let paths_a = segment_paths(dir_a.path(), "abc");
    let paths_b = segment_paths(dir_b.path(), "abc");
    assert_eq!(paths_a.terms, "seg_abc.terms");
    assert_eq!(paths_a.postings, "seg_abc.post");
    assert_eq!(paths_a.terms, paths_b.terms);
    assert_eq!(paths_a.postings, paths_b.postings);
    assert_eq!(wal_path(dir_a.path()), dir_a.path().join("wal.log"));
    paths_a.validate_v2_relative().unwrap();
    #[cfg(feature = "vectors")]
    {
      let vector_dir = paths_a.vector_dir.as_deref().expect("vector dir set");
      assert_eq!(vector_dir, "seg_abc_vectors");
    }
  }
}
