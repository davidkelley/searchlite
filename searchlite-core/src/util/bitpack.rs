pub fn pack_u32(values: &[u32]) -> Vec<u8> {
  let mut out = Vec::with_capacity(values.len() * 4);
  for v in values {
    out.extend_from_slice(&v.to_le_bytes());
  }
  out
}

pub fn unpack_u32(bytes: &[u8]) -> Vec<u32> {
  bytes
    .chunks_exact(4)
    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn roundtrip_pack() {
    let vals = vec![0u32, 1, 2, 255, 65_535, u32::MAX];
    let packed = pack_u32(&vals);
    let unpacked = unpack_u32(&packed);
    assert_eq!(vals, unpacked);
  }
}
