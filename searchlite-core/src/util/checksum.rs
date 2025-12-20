use crc32fast::Hasher;

pub fn checksum(data: &[u8]) -> u32 {
  let mut hasher = Hasher::new();
  hasher.update(data);
  hasher.finalize()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn computes_consistent_checksum() {
    let data = b"searchlite";
    let first = checksum(data);
    let second = checksum(data);
    assert_eq!(first, second);
    assert_ne!(first, checksum(b"searchlite!"));
  }
}
