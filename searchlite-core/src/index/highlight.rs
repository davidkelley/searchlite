use regex::RegexBuilder;

pub struct HighlightOptions<'a> {
  pub pre_tag: &'a str,
  pub post_tag: &'a str,
  pub fragment_size: usize,
  pub number_of_fragments: usize,
}

pub fn highlight_fragments(
  text: &str,
  terms: &[String],
  opts: HighlightOptions<'_>,
) -> Vec<String> {
  if text.is_empty() || terms.is_empty() {
    return Vec::new();
  }
  let pattern = terms
    .iter()
    .map(|t| regex::escape(t))
    .collect::<Vec<_>>()
    .join("|");
  let Ok(re) = RegexBuilder::new(&pattern).case_insensitive(true).build() else {
    return Vec::new();
  };
  let mut out = Vec::new();
  let mut offset = 0usize;
  for _ in 0..opts.number_of_fragments {
    if let Some(m) = re.find_at(text, offset) {
      let start = m.start().saturating_sub(opts.fragment_size / 2);
      let end = usize::min(text.len(), start.saturating_add(opts.fragment_size));
      let fragment = text.get(start..end).unwrap_or("").to_string();
      let highlighted = re
        .replace_all(&fragment, |caps: &regex::Captures<'_>| {
          format!("{}{}{}", opts.pre_tag, &caps[0], opts.post_tag)
        })
        .into_owned();
      out.push(highlighted);
      offset = m.end();
    } else {
      break;
    }
  }
  out
}

pub fn make_snippet(text: &str, terms: &[String]) -> Option<String> {
  let mut frags = highlight_fragments(
    text,
    terms,
    HighlightOptions {
      pre_tag: "**",
      post_tag: "**",
      fragment_size: 120,
      number_of_fragments: 1,
    },
  );
  frags.pop()
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
