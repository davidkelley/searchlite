use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

use crate::analysis::analyzer::Token;

/// Legacy tokenizer that splits on ASCII and lowercases ASCII characters.
pub fn default_tokenize(text: &str) -> Vec<Token> {
  let mut tokens = Vec::new();
  let mut current = String::new();
  let mut position = 0u32;
  for ch in text.chars() {
    if ch.is_alphanumeric() {
      current.push(ch.to_ascii_lowercase());
    } else if !current.is_empty() {
      tokens.push(Token {
        text: std::mem::take(&mut current),
        position,
      });
      position += 1;
    }
  }
  if !current.is_empty() {
    tokens.push(Token {
      text: current,
      position,
    });
  }
  tokens
}

/// Unicode-aware tokenizer that normalizes to NFKC and case folds tokens.
pub fn unicode_tokenize(text: &str) -> Vec<Token> {
  let normalized: String = text.nfkc().collect();
  normalized
    .unicode_words()
    .enumerate()
    .map(|(idx, word)| Token {
      text: word.to_lowercase(),
      position: idx as u32,
    })
    .collect()
}

/// Tokenizer that splits on Unicode whitespace without extra normalization.
pub fn whitespace_tokenize(text: &str) -> Vec<Token> {
  text
    .split_whitespace()
    .enumerate()
    .map(|(idx, word)| Token {
      text: word.to_string(),
      position: idx as u32,
    })
    .collect()
}

/// Backwards-compatible helper returning plain strings from the default analyzer.
pub fn tokenize(text: &str) -> Vec<String> {
  default_tokenize(text).into_iter().map(|t| t.text).collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn tokenizes_words() {
    let out = default_tokenize("Rust: systems programming language");
    assert_eq!(
      out,
      vec![
        Token {
          text: "rust".into(),
          position: 0
        },
        Token {
          text: "systems".into(),
          position: 1
        },
        Token {
          text: "programming".into(),
          position: 2
        },
        Token {
          text: "language".into(),
          position: 3
        }
      ]
    );
  }
}
