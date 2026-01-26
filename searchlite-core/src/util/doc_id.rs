use anyhow::{bail, Result};

/// Validate a document identifier used across ingest and delete paths.
///
/// Rules:
/// - must not be empty or all whitespace
/// - must not contain control characters (including newlines and tabs)
pub fn validate_doc_id(id: &str) -> Result<()> {
  if id.trim().is_empty() {
    bail!("document id cannot be empty or whitespace");
  }
  if id.chars().any(|c| c.is_control()) {
    bail!("document id cannot contain control characters");
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::validate_doc_id;

  #[test]
  fn accepts_normal_and_padded_ids() {
    for id in ["abc", " abc", "abc ", " a b c "] {
      validate_doc_id(id).expect("id should be accepted");
    }
  }

  #[test]
  fn rejects_empty_or_whitespace_only() {
    for id in ["", "   ", "\t", "\n"] {
      assert!(
        validate_doc_id(id).is_err(),
        "expected `{id}` to be rejected"
      );
    }
  }

  #[test]
  fn rejects_control_characters() {
    for id in ["abc\n", "carriage\rreturn", "tab\tid", "nul\u{0000}id"] {
      assert!(
        validate_doc_id(id).is_err(),
        "expected `{id}` to be rejected"
      );
    }
  }
}
