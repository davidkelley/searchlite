use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;

use super::manifest::SegmentPaths;

pub fn ensure_root(path: &Path) -> Result<()> {
  if !path.exists() {
    fs::create_dir_all(path)?;
  }
  Ok(())
}

pub fn wal_path(root: &Path) -> PathBuf {
  root.join("wal.log")
}

pub fn segment_paths(root: &Path, id: &str) -> SegmentPaths {
  SegmentPaths {
    terms: root
      .join(format!("seg_{}.terms", id))
      .to_string_lossy()
      .into(),
    postings: root
      .join(format!("seg_{}.post", id))
      .to_string_lossy()
      .into(),
    docstore: root
      .join(format!("seg_{}.docs", id))
      .to_string_lossy()
      .into(),
    fast: root
      .join(format!("seg_{}.fast", id))
      .to_string_lossy()
      .into(),
    meta: root
      .join(format!("seg_{}.meta", id))
      .to_string_lossy()
      .into(),
  }
}

#[allow(dead_code)]
pub fn segment_meta_path(root: &Path, id: &str) -> PathBuf {
  root.join(format!("seg_{}.meta", id))
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
  fn builds_paths_under_root() {
    let dir = tempdir().unwrap();
    ensure_root(dir.path()).unwrap();
    let paths = segment_paths(dir.path(), "abc");
    assert!(paths.terms.ends_with("seg_abc.terms"));
    assert!(paths.postings.ends_with("seg_abc.post"));
    assert_eq!(wal_path(dir.path()), dir.path().join("wal.log"));
  }
}
