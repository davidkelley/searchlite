use hashbrown::HashSet;

pub fn intersect(all: &[HashSet<u32>]) -> HashSet<u32> {
  if all.is_empty() {
    return HashSet::new();
  }
  let mut iter = all.iter();
  let mut acc = iter.next().cloned().unwrap_or_default();
  for set in iter {
    acc = acc.intersection(set).copied().collect();
  }
  acc
}

pub fn union(all: &[HashSet<u32>]) -> HashSet<u32> {
  let mut acc = HashSet::new();
  for set in all {
    acc.extend(set.iter().copied());
  }
  acc
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn computes_intersection_and_union() {
    let a = HashSet::from([1, 2, 3]);
    let b = HashSet::from([2, 3, 4]);
    let c = HashSet::from([3, 4, 5]);
    let all = vec![a.clone(), b.clone(), c.clone()];
    assert_eq!(intersect(&all), HashSet::from([3]));
    assert_eq!(union(&all), HashSet::from([1, 2, 3, 4, 5]));
    assert!(intersect(&[]).is_empty());
  }
}
