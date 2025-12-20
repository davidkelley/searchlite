pub fn tokenize(text: &str) -> Vec<String> {
  let mut tokens = Vec::new();
  let mut current = String::new();
  for ch in text.chars() {
    if ch.is_alphanumeric() {
      current.push(ch.to_ascii_lowercase());
    } else if !current.is_empty() {
      tokens.push(current.clone());
      current.clear();
    }
  }
  if !current.is_empty() {
    tokens.push(current);
  }
  tokens
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn tokenizes_words() {
    let out = tokenize("Rust: systems programming language");
    assert_eq!(out, vec!["rust", "systems", "programming", "language"]);
  }
}
