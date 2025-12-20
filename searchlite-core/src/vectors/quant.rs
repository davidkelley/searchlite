pub fn quantize(vec: &[f32]) -> Vec<u8> {
  vec.iter().map(|v| (*v * 127.0) as i8 as u8).collect()
}
