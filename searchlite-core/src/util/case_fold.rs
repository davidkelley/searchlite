use std::borrow::Cow;

/// Case-fold a keyword/term value for indexing and searching.
///
/// This is the single source of truth for how keyword values are folded. Both
/// the postings path (indexing and `match` queries over keyword fields) and
/// the fast-field equality path (`case_insensitive_equals`) must agree on the
/// folded form so their results line up for ASCII and non-ASCII alike.
///
/// Previously the postings path used `str::to_ascii_lowercase`, which only
/// rewrites the 26 uppercase ASCII letters and leaves every other byte
/// untouched. That diverged from `case_insensitive_equals`, which does real
/// Unicode folding via `char::to_lowercase`, so a keyword value whose
/// uppercase form contained non-ASCII code points (e.g. `RÉSUMÉ`) was
/// matched by `Filter::KeywordEq` but silently missed by a `match` query on
/// the same field.
///
/// The helper keeps the ASCII fast path for the common case: inputs that are
/// pure ASCII and already lowercase are returned borrowed, without
/// allocating.
pub fn fold_keyword(value: &str) -> Cow<'_, str> {
  if value.is_ascii() {
    if value.bytes().any(|b| b.is_ascii_uppercase()) {
      Cow::Owned(value.to_ascii_lowercase())
    } else {
      Cow::Borrowed(value)
    }
  } else {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
      for lc in ch.to_lowercase() {
        out.push(lc);
      }
    }
    Cow::Owned(out)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn ascii_lowercase_borrows() {
    let folded = fold_keyword("tag");
    assert_eq!(folded, "tag");
    assert!(matches!(folded, Cow::Borrowed(_)));
  }

  #[test]
  fn ascii_uppercase_lowercases() {
    let folded = fold_keyword("TAG");
    assert_eq!(folded, "tag");
    assert!(matches!(folded, Cow::Owned(_)));
  }

  #[test]
  fn ascii_mixed_case_lowercases() {
    assert_eq!(fold_keyword("Tag"), "tag");
  }

  #[test]
  fn non_ascii_uppercase_lowercases() {
    assert_eq!(fold_keyword("RÉSUMÉ"), "résumé");
    assert_eq!(fold_keyword("Résumé"), "résumé");
    assert_eq!(fold_keyword("résumé"), "résumé");
  }

  #[test]
  fn non_ascii_cyrillic_and_greek() {
    // Cyrillic capital Ж -> lowercase ж
    assert_eq!(fold_keyword("ЖУК"), "жук");
    // Greek capital Σ folds to lowercase σ in non-final positions.
    assert_eq!(fold_keyword("ΣΟΦΙΑ"), "σοφια");
  }

  #[test]
  fn non_ascii_one_to_many_folding() {
    // U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE lowercases to two code points.
    assert_eq!(fold_keyword("İ"), "i\u{0307}");
  }

  #[test]
  fn empty_input_is_borrowed() {
    let folded = fold_keyword("");
    assert_eq!(folded, "");
    assert!(matches!(folded, Cow::Borrowed(_)));
  }
}
