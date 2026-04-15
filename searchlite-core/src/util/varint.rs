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
    // On the final possible byte (shift == 63) only the low bit of `part`
    // can fit in a u64; higher bits would be silently truncated by the
    // shift. Reject such inputs as overflowing u64 rather than decoding
    // to a lossy value.
    if shift == 63 && part > 1 {
      return Err(anyhow!("varint overflows u64"));
    }
    value |= part << shift;
    if b & 0x80 == 0 {
      return Ok((value, i + 1));
    }
    shift += 7;
    if shift > 63 {
      return Err(anyhow!("varint too long"));
    }
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
    let part = (b & 0x7F) as u32;
    // On the final possible byte (shift == 28) only the low 4 bits of
    // `part` fit in a u32; any higher bit would be silently truncated by
    // the shift. Reject such inputs as overflowing u32 rather than
    // decoding to a lossy value (mirrors the shift == 63 guard in
    // read_u64; see BUG-002 / #140).
    if shift == 28 && part > 0x0F {
      return Err(anyhow!("varint overflows u32"));
    }
    value |= part << shift;
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

  #[test]
  fn roundtrip_u64() {
    for val in [0u64, 1, 127, 128, 16384, u32::MAX as u64, u64::MAX] {
      let mut buf = Vec::new();
      write_u64(val, &mut buf);
      let (decoded, len) = read_u64(&buf).unwrap();
      assert_eq!(decoded, val);
      assert_eq!(len, buf.len());
    }
  }

  #[test]
  fn read_u64_rejects_overlong_varint() {
    // A stream of continuation-only bytes (no value bits set) would shift
    // past 63 bits of a u64 without ever terminating. Must return an
    // error instead of panicking (debug) or silently corrupting the
    // value (release).
    let buf = vec![0x80u8; 20];
    let err = read_u64(&buf).expect_err("overlong varint must be rejected");
    assert_eq!(err.to_string(), "varint too long");
  }

  #[test]
  fn read_u64_rejects_unterminated_varint() {
    // A short buffer whose bytes all have the continuation bit set but is
    // not long enough to trigger the shift guard must still report the
    // underlying framing problem.
    let buf = vec![0x80u8; 3];
    let err = read_u64(&buf).expect_err("unterminated varint must be rejected");
    assert_eq!(err.to_string(), "unterminated varint");
  }

  #[test]
  fn read_u64_empty_buffer_errors() {
    let err = read_u64(&[]).expect_err("empty buffer must be rejected");
    assert_eq!(err.to_string(), "unterminated varint");
  }

  #[test]
  fn read_u64_accepts_max_length_valid_varint() {
    // u64::MAX encodes to exactly 10 LEB128 bytes — the boundary case.
    let mut buf = Vec::new();
    write_u64(u64::MAX, &mut buf);
    assert_eq!(buf.len(), 10);
    let (decoded, len) = read_u64(&buf).unwrap();
    assert_eq!(decoded, u64::MAX);
    assert_eq!(len, 10);
  }

  #[test]
  fn read_u32_var_roundtrip_boundary_values() {
    // u32::MAX encodes to exactly 5 LEB128 bytes — the boundary case that
    // must round-trip through the stricter overflow guard without error.
    use std::io::Cursor;
    for val in [0u32, 1, 127, 128, 16383, 16384, u32::MAX - 1, u32::MAX] {
      let mut buf = Vec::new();
      write_u32_var(val, &mut buf);
      let mut cur = Cursor::new(&buf);
      let decoded = read_u32_var(&mut cur).unwrap();
      assert_eq!(decoded, val, "round-trip failed for {val}");
    }
  }

  #[test]
  fn read_u32_var_rejects_final_byte_overflow() {
    // 5-byte varint: four continuation bytes (value bits all zero) + final
    // byte 0x10. The final byte's bit 4 sits at position 32 once shifted,
    // i.e. one bit above u32::MAX. The decoded value is exactly 2^32 and
    // does NOT fit in a u32. read_u32_var must error rather than silently
    // dropping the high bit through the u32 left shift.
    use std::io::Cursor;
    let buf = vec![0x80u8, 0x80, 0x80, 0x80, 0x10];
    let mut cur = Cursor::new(&buf);
    let err = read_u32_var(&mut cur).expect_err("u32-overflowing varint must be rejected");
    assert_eq!(err.to_string(), "varint overflows u32");

    // Same shape, even more value bits set in the final byte (bits 4..6).
    let buf = vec![0x80u8, 0x80, 0x80, 0x80, 0x70];
    let mut cur = Cursor::new(&buf);
    let err = read_u32_var(&mut cur).expect_err("u32-overflowing varint must be rejected");
    assert_eq!(err.to_string(), "varint overflows u32");
  }

  #[test]
  fn read_u32_var_rejects_overlong_varint() {
    // A stream of continuation-only bytes (no value bits set) would shift
    // past 28 bits of a u32 without ever terminating. Must return an
    // error instead of panicking or silently corrupting the value.
    use std::io::Cursor;
    let buf = vec![0x80u8; 10];
    let mut cur = Cursor::new(&buf);
    let err = read_u32_var(&mut cur).expect_err("overlong varint must be rejected");
    assert_eq!(err.to_string(), "varint too long");
  }

  #[test]
  fn read_u32_var_accepts_max_length_valid_varint() {
    // u32::MAX encodes to exactly 5 LEB128 bytes whose final byte is 0x0F —
    // the boundary the new guard must accept.
    use std::io::Cursor;
    let mut buf = Vec::new();
    write_u32_var(u32::MAX, &mut buf);
    assert_eq!(buf.len(), 5);
    assert_eq!(buf[4], 0x0F);
    let mut cur = Cursor::new(&buf);
    let decoded = read_u32_var(&mut cur).unwrap();
    assert_eq!(decoded, u32::MAX);
  }

  #[test]
  fn read_u64_rejects_final_byte_overflow() {
    // A 10-byte varint whose first 9 bytes are continuation (shift = 63)
    // and whose final byte has any value bit above bit 0 set represents
    // a value > u64::MAX. It must be rejected rather than silently
    // truncating the high bits.
    let mut buf = vec![0x80u8; 9];
    // Final byte: no continuation bit, but value bits beyond bit 0 set —
    // 0x02 would shift to bit 64, which doesn't fit in a u64.
    buf.push(0x02);
    let err = read_u64(&buf).expect_err("u64-overflowing varint must be rejected");
    assert_eq!(err.to_string(), "varint overflows u64");

    // Also reject when the overflow bits coexist with a continuation bit.
    let mut buf = vec![0x80u8; 9];
    buf.push(0x82);
    let err = read_u64(&buf).expect_err("u64-overflowing varint must be rejected");
    assert_eq!(err.to_string(), "varint overflows u64");
  }
}
