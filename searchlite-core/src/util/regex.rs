use anyhow::Result;
use regex::Regex;

/// Builds an anchored regex used for term expansion, wrapping the pattern with ^(?:...)$.
pub fn anchored_regex(pattern: &str) -> Result<Regex> {
  let anchored = format!("^(?:{pattern})$");
  Regex::new(&anchored).map_err(|e| anyhow::anyhow!("invalid regex `{pattern}`: {e}"))
}
