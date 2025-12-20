pub fn normalize_token(token: &str) -> String {
  token.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn lowercases_ascii() {
    assert_eq!(normalize_token("HelloWORLD"), "helloworld");
    assert_eq!(normalize_token("123-ABC"), "123-abc");
  }
}
