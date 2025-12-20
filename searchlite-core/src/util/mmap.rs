use std::fs::File;
use std::path::Path;

use anyhow::Result;
use memmap2::Mmap;

pub fn mmap_read(path: &Path) -> Result<Mmap> {
  let file = File::open(path)?;
  unsafe { Mmap::map(&file).map_err(Into::into) }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::io::Write;
  use tempfile::NamedTempFile;

  #[test]
  fn maps_file_contents() {
    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "hello mmap").unwrap();
    let mmap = mmap_read(tmp.path()).unwrap();
    assert_eq!(&mmap[..], b"hello mmap");
  }
}
