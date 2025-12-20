pub fn make_snippet(text: &str, terms: &[String]) -> Option<String> {
  if text.is_empty() || terms.is_empty() {
    return None;
  }
  let lower = text.to_ascii_lowercase();
  let mut best_pos = None;
  for term in terms {
    if let Some(pos) = lower.find(&term.to_ascii_lowercase()) {
      best_pos = Some(pos);
      break;
    }
  }
  let pos = best_pos.unwrap_or(0);
  let start = pos.saturating_sub(40);
  let end = usize::min(text.len(), pos + 80);
  let mut snippet = String::from(&text[start..end]);
  for term in terms {
    let pat = term;
    snippet = snippet.replace(pat, &format!("**{}**", pat));
  }
  Some(snippet)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn highlights_first_term() {
    let text = "Rust is a systems programming language";
    let snippet = make_snippet(text, &[String::from("systems")]).unwrap();
    assert!(snippet.contains("**systems**"));
    let none = make_snippet("", &[String::from("systems")]);
    assert!(none.is_none());
  }
}
