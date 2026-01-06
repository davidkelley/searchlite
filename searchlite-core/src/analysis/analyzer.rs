use std::borrow::Cow;

use crate::analysis::tokenizer::{default_tokenize, unicode_tokenize, whitespace_tokenize};
use anyhow::{anyhow, bail, Result};
use hashbrown::{HashMap, HashSet};
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
  pub text: String,
  pub position: u32,
}

#[derive(Debug, Clone)]
pub struct Analyzer {
  tokenizer: TokenizerKind,
  filters: Vec<TokenFilter>,
}

impl Analyzer {
  pub fn analyze(&self, text: &str) -> Vec<Token> {
    let mut tokens = self.tokenizer.tokenize(text);
    for filter in self.filters.iter() {
      tokens = filter.apply(tokens);
    }
    resequence_positions(&mut tokens);
    tokens
  }

  /// Applies inexpensive, structure-preserving normalization suitable for patterns
  /// (e.g., wildcard/regex) without re-tokenizing or stripping delimiters.
  pub fn normalize_pattern(&self, pattern: &str) -> String {
    let lowercases = matches!(
      self.tokenizer,
      TokenizerKind::Default | TokenizerKind::Unicode
    ) || self
      .filters
      .iter()
      .any(|f| matches!(f, TokenFilter::Lowercase));
    if lowercases {
      pattern.to_lowercase()
    } else {
      pattern.to_string()
    }
  }
}

#[derive(Debug, Clone)]
enum TokenizerKind {
  Default,
  Unicode,
  Whitespace,
}

impl TokenizerKind {
  fn from_name(name: &str) -> Result<Self> {
    match name {
      "default" => Ok(Self::Default),
      "unicode" => Ok(Self::Unicode),
      "whitespace" => Ok(Self::Whitespace),
      other => bail!("unknown tokenizer `{other}`"),
    }
  }

  fn tokenize(&self, text: &str) -> Vec<Token> {
    match self {
      TokenizerKind::Default => default_tokenize(text),
      TokenizerKind::Unicode => unicode_tokenize(text),
      TokenizerKind::Whitespace => whitespace_tokenize(text),
    }
  }
}

#[derive(Debug, Clone)]
enum TokenFilter {
  Lowercase,
  Stopwords(HashSet<String>),
  Stemmer(Algorithm),
  Synonyms(Vec<SynonymRule>),
  EdgeNgram(EdgeNgramConfig),
}

impl TokenFilter {
  fn apply(&self, tokens: Vec<Token>) -> Vec<Token> {
    match self {
      TokenFilter::Lowercase => lowercase(tokens),
      TokenFilter::Stopwords(stopwords) => filter_stopwords(tokens, stopwords),
      TokenFilter::Stemmer(stemmer) => stem_tokens(tokens, stemmer),
      TokenFilter::Synonyms(rules) => expand_synonyms(tokens, rules),
      TokenFilter::EdgeNgram(cfg) => edge_ngrams(tokens, cfg),
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerDef {
  pub name: String,
  pub tokenizer: String,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub filters: Vec<TokenFilterDef>,
}

#[derive(Debug, Clone)]
pub enum TokenFilterDef {
  Lowercase,
  Stopwords(StopwordsConfig),
  Stemmer(StemmerConfig),
  Synonyms(Vec<SynonymRule>),
  EdgeNgram(EdgeNgramConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynonymRule {
  pub from: Vec<String>,
  pub to: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeNgramConfig {
  pub min: usize,
  pub max: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopwordsConfig {
  Named(String),
  List(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StemmerConfig {
  Language(String),
}

impl TokenFilterDef {
  fn parse_type(value: &serde_json::Value) -> Option<String> {
    value
      .get("type")
      .and_then(|v| v.as_str())
      .map(|s| s.to_string())
  }

  fn from_kind(kind: &str, value: &serde_json::Value) -> Result<Self> {
    match kind {
      "lowercase" => Ok(TokenFilterDef::Lowercase),
      "stopwords" => Ok(TokenFilterDef::Stopwords(
        serde_json::from_value(
          value
            .get("stopwords")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        )
        .map_err(|e| anyhow!(e))?,
      )),
      "stemmer" => Ok(TokenFilterDef::Stemmer(
        serde_json::from_value(value.get("stemmer").cloned().unwrap_or_else(|| {
          value
            .get("language")
            .cloned()
            .unwrap_or(serde_json::Value::Null)
        }))
        .map_err(|e| anyhow!(e))?,
      )),
      "synonyms" => Ok(TokenFilterDef::Synonyms(
        serde_json::from_value(
          value
            .get("synonyms")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        )
        .map_err(|e| anyhow!(e))?,
      )),
      "edge_ngram" => Ok(TokenFilterDef::EdgeNgram(
        serde_json::from_value(
          value
            .get("edge_ngram")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        )
        .map_err(|e| anyhow!(e))?,
      )),
      other => bail!("unknown token filter `{other}`"),
    }
  }
}

impl Serialize for TokenFilterDef {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    use serde::ser::SerializeMap;
    let mut map = serializer.serialize_map(Some(1))?;
    match self {
      TokenFilterDef::Lowercase => {
        map.serialize_entry("lowercase", &true)?;
      }
      TokenFilterDef::Stopwords(cfg) => {
        map.serialize_entry("stopwords", cfg)?;
      }
      TokenFilterDef::Stemmer(cfg) => {
        map.serialize_entry("stemmer", cfg)?;
      }
      TokenFilterDef::Synonyms(rules) => {
        map.serialize_entry("synonyms", rules)?;
      }
      TokenFilterDef::EdgeNgram(cfg) => {
        map.serialize_entry("edge_ngram", cfg)?;
      }
    }
    map.end()
  }
}

impl<'de> Deserialize<'de> for TokenFilterDef {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let value = serde_json::Value::deserialize(deserializer)?;
    match &value {
      serde_json::Value::String(s) => {
        TokenFilterDef::from_kind(s, &serde_json::Value::Object(serde_json::Map::new()))
          .map_err(serde::de::Error::custom)
      }
      serde_json::Value::Object(obj) => {
        if obj.len() == 1 && obj.keys().all(|k| k == "lowercase") {
          if let Some(val) = obj.get("lowercase") {
            if val.as_bool() == Some(false) {
              return Err(serde::de::Error::custom("lowercase filter expects `true`"));
            }
          }
          return Ok(TokenFilterDef::Lowercase);
        }
        if obj.contains_key("stopwords") && !obj.contains_key("type") {
          return TokenFilterDef::from_kind("stopwords", &value).map_err(serde::de::Error::custom);
        }
        if obj.contains_key("synonyms") && !obj.contains_key("type") {
          return TokenFilterDef::from_kind("synonyms", &value).map_err(serde::de::Error::custom);
        }
        if obj.contains_key("edge_ngram") && !obj.contains_key("type") {
          return TokenFilterDef::from_kind("edge_ngram", &value).map_err(serde::de::Error::custom);
        }
        if obj.contains_key("stemmer") && !obj.contains_key("type") {
          return TokenFilterDef::from_kind("stemmer", &value).map_err(serde::de::Error::custom);
        }
        if let Some(kind) = TokenFilterDef::parse_type(&value) {
          return TokenFilterDef::from_kind(&kind, &value).map_err(serde::de::Error::custom);
        }
        Err(serde::de::Error::custom(
          "token filter must declare `type` or one of `lowercase`, `stopwords`, `stemmer`, `synonyms`, `edge_ngram` keys",
        ))
      }
      _ => Err(serde::de::Error::custom(
        "token filter must be string or object",
      )),
    }
  }
}

#[derive(Debug, Clone)]
pub struct AnalyzerRegistry {
  analyzers: HashMap<String, Analyzer>,
}

impl AnalyzerRegistry {
  pub fn with_default() -> Self {
    let mut registry = Self {
      analyzers: HashMap::new(),
    };
    registry
      .analyzers
      .insert("default".to_string(), Analyzer::default_analyzer());
    registry
  }

  pub fn insert(&mut self, name: &str, analyzer: Analyzer) -> Result<()> {
    if self.analyzers.contains_key(name) {
      bail!("duplicate analyzer `{name}`");
    }
    self.analyzers.insert(name.to_string(), analyzer);
    Ok(())
  }

  pub fn from_defs(defs: &[AnalyzerDef]) -> Result<Self> {
    let mut registry = AnalyzerRegistry::with_default();
    for def in defs {
      if def.name == "default" {
        bail!("analyzer name `default` is reserved");
      }
      let tokenizer = TokenizerKind::from_name(&def.tokenizer)?;
      let mut filters = Vec::with_capacity(def.filters.len());
      for filter in def.filters.iter() {
        filters.push(build_filter(filter)?);
      }
      registry.insert(&def.name, Analyzer { tokenizer, filters })?;
    }
    Ok(registry)
  }

  pub fn get(&self, name: &str) -> Option<&Analyzer> {
    self.analyzers.get(name)
  }
}

impl Analyzer {
  fn default_analyzer() -> Self {
    Self {
      tokenizer: TokenizerKind::Default,
      filters: Vec::new(),
    }
  }
}

fn build_filter(def: &TokenFilterDef) -> Result<TokenFilter> {
  match def {
    TokenFilterDef::Lowercase => Ok(TokenFilter::Lowercase),
    TokenFilterDef::Stopwords(cfg) => Ok(TokenFilter::Stopwords(stopword_set(cfg)?)),
    TokenFilterDef::Stemmer(cfg) => Ok(TokenFilter::Stemmer(build_stemmer(cfg)?)),
    TokenFilterDef::Synonyms(rules) => Ok(TokenFilter::Synonyms(rules.clone())),
    TokenFilterDef::EdgeNgram(cfg) => {
      if cfg.min == 0 || cfg.max == 0 {
        bail!("edge_ngram min and max must be positive");
      }
      if cfg.min > cfg.max {
        bail!("edge_ngram min must be <= max");
      }
      Ok(TokenFilter::EdgeNgram(cfg.clone()))
    }
  }
}

fn build_stemmer(cfg: &StemmerConfig) -> Result<Algorithm> {
  match cfg {
    StemmerConfig::Language(lang) => match lang.to_ascii_lowercase().as_str() {
      "en" | "eng" | "english" => Ok(Algorithm::English),
      other => bail!("unsupported stemmer language `{other}`"),
    },
  }
}

fn stopword_set(cfg: &StopwordsConfig) -> Result<HashSet<String>> {
  let words: Vec<String> = match cfg {
    StopwordsConfig::Named(name) => match name.to_ascii_lowercase().as_str() {
      "en" | "english" => ENGLISH_STOPWORDS.iter().map(|s| s.to_string()).collect(),
      other => bail!("unsupported stopword list `{other}`"),
    },
    StopwordsConfig::List(list) => list.clone(),
  };
  Ok(words.into_iter().collect())
}

fn lowercase(mut tokens: Vec<Token>) -> Vec<Token> {
  for token in tokens.iter_mut() {
    token.text = token.text.to_lowercase();
  }
  tokens
}

fn filter_stopwords(tokens: Vec<Token>, stopwords: &HashSet<String>) -> Vec<Token> {
  tokens
    .into_iter()
    .filter(|t| !stopwords.contains(&t.text))
    .collect()
}

fn stem_tokens(mut tokens: Vec<Token>, algo: &Algorithm) -> Vec<Token> {
  let stemmer = Stemmer::create(*algo);
  for token in tokens.iter_mut() {
    token.text = stemmer.stem(&token.text).into_owned();
  }
  tokens
}

fn expand_synonyms(tokens: Vec<Token>, rules: &[SynonymRule]) -> Vec<Token> {
  if rules.is_empty() {
    return tokens;
  }
  let mut out = Vec::with_capacity(tokens.len());
  let mut i = 0usize;
  while i < tokens.len() {
    let mut matched = false;
    for rule in rules.iter() {
      if rule.from.is_empty() || i + rule.from.len() > tokens.len() {
        continue;
      }
      if rule
        .from
        .iter()
        .zip(tokens[i..].iter())
        .all(|(from, tok)| from == &tok.text)
      {
        for offset in 0..rule.from.len() {
          out.push(tokens[i + offset].clone());
        }
        if !rule.to.is_empty() {
          let pos = tokens[i].position;
          for to in rule.to.iter() {
            out.push(Token {
              text: to.clone(),
              position: pos,
            });
          }
        }
        i += rule.from.len();
        matched = true;
        break;
      }
    }
    if !matched {
      out.push(tokens[i].clone());
      i += 1;
    }
  }
  out
}

fn edge_ngrams(tokens: Vec<Token>, cfg: &EdgeNgramConfig) -> Vec<Token> {
  let mut out = Vec::new();
  for token in tokens.into_iter() {
    let len = token.text.chars().count();
    let max = usize::min(cfg.max, len);
    let min = usize::min(cfg.min, max);
    if min == 0 || max == 0 {
      continue;
    }
    for size in min..=max {
      let prefix = char_prefix(&token.text, size);
      out.push(Token {
        text: prefix.to_string(),
        position: token.position,
      });
    }
  }
  out
}

fn resequence_positions(tokens: &mut [Token]) {
  let mut last_source: Option<u32> = None;
  let mut next: u32 = 0;
  for token in tokens.iter_mut() {
    let original = token.position;
    if last_source != Some(original) {
      token.position = next;
      last_source = Some(original);
      next = next.saturating_add(1);
    } else {
      token.position = next.saturating_sub(1);
    }
  }
}

fn char_prefix(input: &str, len: usize) -> Cow<'_, str> {
  if len == 0 {
    return Cow::Borrowed("");
  }
  match input.char_indices().nth(len) {
    Some((idx, _)) => Cow::Borrowed(&input[..idx]),
    None => Cow::Borrowed(input),
  }
}

const ENGLISH_STOPWORDS: &[&str] = &[
  "a", "about", "after", "all", "also", "an", "and", "another", "any", "are", "as", "at", "be",
  "because", "been", "before", "being", "between", "both", "but", "by", "came", "can", "come",
  "could", "did", "do", "each", "for", "from", "get", "got", "had", "has", "have", "he", "her",
  "here", "him", "himself", "his", "how", "if", "in", "into", "is", "it", "like", "make", "many",
  "me", "might", "more", "most", "much", "must", "my", "never", "now", "of", "on", "only", "or",
  "other", "our", "out", "over", "said", "same", "see", "should", "since", "some", "still", "such",
  "take", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
  "those", "through", "to", "too", "under", "up", "use", "very", "want", "was", "way", "we",
  "well", "were", "what", "when", "where", "which", "while", "who", "will", "with", "would", "you",
  "your",
];

#[cfg(test)]
mod tests {
  use super::*;

  fn analyze_with(filters: Vec<TokenFilterDef>, text: &str) -> Vec<Token> {
    let analyzer = Analyzer {
      tokenizer: TokenizerKind::Default,
      filters: filters.iter().map(|f| build_filter(f).unwrap()).collect(),
    };
    analyzer.analyze(text)
  }

  #[test]
  fn default_analyzer_matches_legacy_tokenize() {
    let analyzer = Analyzer::default_analyzer();
    let tokens = analyzer.analyze("Rust: systems programming language");
    let out: Vec<String> = tokens.into_iter().map(|t| t.text).collect();
    assert_eq!(out, vec!["rust", "systems", "programming", "language"]);
  }

  #[test]
  fn unicode_tokenizer_normalizes_and_folds() {
    let analyzer = Analyzer {
      tokenizer: TokenizerKind::Unicode,
      filters: vec![],
    };
    let tokens = analyzer.analyze("CAFÉ ﬂavor");
    let out: Vec<String> = tokens.into_iter().map(|t| t.text).collect();
    assert_eq!(out, vec!["café", "flavor"]);
  }

  #[test]
  fn stopwords_filter_removes_tokens() {
    let tokens = analyze_with(
      vec![TokenFilterDef::Stopwords(StopwordsConfig::Named(
        "en".into(),
      ))],
      "the quick brown fox",
    );
    let out: Vec<String> = tokens.into_iter().map(|t| t.text).collect();
    assert_eq!(out, vec!["quick", "brown", "fox"]);
  }

  #[test]
  fn stemmer_reduces_tokens() {
    let tokens = analyze_with(
      vec![TokenFilterDef::Stemmer(StemmerConfig::Language(
        "english".into(),
      ))],
      "running runners",
    );
    let out: Vec<String> = tokens.into_iter().map(|t| t.text).collect();
    assert_eq!(out, vec!["run", "runner"]);
  }

  #[test]
  fn synonyms_expand_without_position_gap() {
    let tokens = analyze_with(
      vec![TokenFilterDef::Synonyms(vec![SynonymRule {
        from: vec!["nyc".into()],
        to: vec!["new".into(), "york".into()],
      }])],
      "nyc subway",
    );
    let texts: Vec<(String, u32)> = tokens.into_iter().map(|t| (t.text, t.position)).collect();
    assert_eq!(
      texts,
      vec![
        ("nyc".into(), 0),
        ("new".into(), 0),
        ("york".into(), 0),
        ("subway".into(), 1)
      ]
    );
  }

  #[test]
  fn edge_ngram_expands_prefixes() {
    let tokens = analyze_with(
      vec![TokenFilterDef::EdgeNgram(EdgeNgramConfig {
        min: 1,
        max: 3,
      })],
      "rust",
    );
    let texts: Vec<String> = tokens.into_iter().map(|t| t.text).collect();
    assert_eq!(
      texts,
      vec!["r".to_string(), "ru".to_string(), "rus".to_string()]
    );
  }
}
