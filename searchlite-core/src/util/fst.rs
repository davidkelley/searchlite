use hashbrown::HashMap;

/// Compact term dictionary with O(1) exact lookups and efficient prefix iteration.
///
/// Uses a `HashMap` for exact term lookups (the hot path during search) and a
/// sorted `Vec` for ordered iteration and prefix queries. The sorted vec is
/// built once at construction time and enables binary-search-based prefix
/// scans without per-call allocations.
#[derive(Debug, Clone, Default)]
pub struct TinyFst {
  map: HashMap<String, u64>,
  sorted: Vec<(String, u64)>,
}

impl TinyFst {
  pub fn from_terms(terms: &[(String, u64)]) -> Self {
    let map: HashMap<String, u64> = terms.iter().cloned().collect();
    let mut sorted: Vec<(String, u64)> = terms.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    Self { map, sorted }
  }

  pub fn get(&self, term: &str) -> Option<u64> {
    self.map.get(term).copied()
  }

  pub fn iter(&self) -> impl Iterator<Item = (&String, &u64)> {
    self.sorted.iter().map(|(k, v)| (k, v))
  }

  pub fn iter_prefix<'a>(
    &'a self,
    prefix: &'a str,
  ) -> impl Iterator<Item = (&'a String, &'a u64)> + 'a {
    // Binary search for the first entry >= prefix, then take while matching.
    let start = self.sorted.partition_point(|(k, _)| k.as_str() < prefix);
    self.sorted[start..]
      .iter()
      .take_while(move |(k, _)| k.starts_with(prefix))
      .map(|(k, v)| (k, v))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn builds_and_queries_terms() {
    let fst = TinyFst::from_terms(&[
      ("alpha".to_string(), 1),
      ("beta".to_string(), 2),
      ("gamma".to_string(), 3),
    ]);
    assert_eq!(fst.get("beta"), Some(2));
    assert_eq!(fst.get("missing"), None);
    let collected: Vec<_> = fst.iter().map(|(k, v)| (k.clone(), *v)).collect();
    assert_eq!(
      collected,
      vec![
        ("alpha".to_string(), 1),
        ("beta".to_string(), 2),
        ("gamma".to_string(), 3)
      ]
    );
  }

  #[test]
  fn prefix_iteration_finds_matching_terms() {
    let fst = TinyFst::from_terms(&[
      ("body:apple".to_string(), 10),
      ("body:application".to_string(), 20),
      ("body:banana".to_string(), 30),
      ("title:apple".to_string(), 40),
    ]);
    let prefixed: Vec<_> = fst
      .iter_prefix("body:app")
      .map(|(k, v)| (k.as_str(), *v))
      .collect();
    assert_eq!(
      prefixed,
      vec![("body:apple", 10), ("body:application", 20)]
    );
    assert_eq!(fst.iter_prefix("missing").count(), 0);
  }
}
