use std::io::Read;

use anyhow::{anyhow, Result};

pub fn write_u64(mut v: u64, out: &mut Vec<u8>) {
  while v >= 0x80 {
    out.push(((v as u8) & 0x7F) | 0x80);
    v >>= 7;
  }
  out.push(v as u8);
}

pub fn write_u32_var(v: u32, out: &mut Vec<u8>) {
  write_u64(v as u64, out);
}

pub fn read_u64(buf: &[u8]) -> Result<(u64, usize)> {
  let mut shift = 0u32;
  let mut value = 0u64;
  for (i, b) in buf.iter().enumerate() {
    let part = (b & 0x7F) as u64;
    value |= part << shift;
    if b & 0x80 == 0 {
      return Ok((value, i + 1));
    }
    shift += 7;
  }
  Err(anyhow!("unterminated varint"))
}

pub fn read_u32_var<R: Read>(r: &mut R) -> Result<u32> {
  let mut shift = 0u32;
  let mut value = 0u32;
  loop {
    let mut byte = [0u8; 1];
    if let Err(e) = r.read_exact(&mut byte) {
      return Err(anyhow!(e));
    }
    let b = byte[0];
    value |= ((b & 0x7F) as u32) << shift;
    if b & 0x80 == 0 {
      return Ok(value);
    }
    shift += 7;
    if shift > 28 {
      return Err(anyhow!("varint too long"));
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn roundtrip() {
    for val in [0u32, 1, 127, 128, 16384, u32::MAX] {
      let mut buf = Vec::new();
      write_u32_var(val, &mut buf);
      let (decoded, _len) = read_u64(&buf).unwrap();
      assert_eq!(decoded as u32, val);
    }
  }
}
