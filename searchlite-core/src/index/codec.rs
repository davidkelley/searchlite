use std::io::{Read, Write};

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

pub fn write_u32<W: Write + ?Sized>(w: &mut W, v: u32) -> Result<()> {
  w.write_u32::<LittleEndian>(v)?;
  Ok(())
}

pub fn write_f32<W: Write + ?Sized>(w: &mut W, v: f32) -> Result<()> {
  w.write_f32::<LittleEndian>(v)?;
  Ok(())
}

pub fn read_u32<R: Read + ?Sized>(r: &mut R) -> Result<u32> {
  r.read_u32::<LittleEndian>().context("reading u32")
}

pub fn read_f32<R: Read + ?Sized>(r: &mut R) -> Result<f32> {
  r.read_f32::<LittleEndian>().context("reading f32")
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::io::Cursor;

  #[test]
  fn roundtrips_numbers() {
    let mut buf = Vec::new();
    write_u32(&mut buf, 42).unwrap();
    write_f32(&mut buf, 1.25).unwrap();
    let mut cursor = Cursor::new(buf);
    assert_eq!(read_u32(&mut cursor).unwrap(), 42);
    assert!((read_f32(&mut cursor).unwrap() - 1.25).abs() < 1e-6);
  }
}
