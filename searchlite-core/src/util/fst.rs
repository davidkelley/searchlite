use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct TinyFst {
  map: BTreeMap<String, u64>,
}

impl TinyFst {
  pub fn from_terms(terms: &[(String, u64)]) -> Self {
    let mut map = BTreeMap::new();
    for (t, off) in terms {
      map.insert(t.clone(), *off);
    }
    Self { map }
  }

  pub fn get(&self, term: &str) -> Option<u64> {
    self.map.get(term).copied()
  }

  pub fn iter(&self) -> impl Iterator<Item = (&String, &u64)> {
    self.map.iter()
  }

  pub fn iter_prefix<'a>(
    &'a self,
    prefix: &str,
  ) -> impl Iterator<Item = (&'a String, &'a u64)> + 'a {
    let prefix_owned = prefix.to_string();
    self
      .map
      .range(prefix_owned.clone()..)
      .take_while(move |(k, _)| k.starts_with(&prefix_owned))
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
}
