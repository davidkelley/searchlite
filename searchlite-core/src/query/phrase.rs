use crate::index::postings::PostingEntry;
use crate::DocId;

pub fn matches_phrase(postings: &[Vec<PostingEntry>], doc_id: DocId) -> bool {
  if postings.is_empty() {
    return true;
  }
  let mut positions_per_term = Vec::new();
  for term_posts in postings {
    if let Some(entry) = term_posts.iter().find(|p| p.doc_id == doc_id) {
      positions_per_term.push(entry.positions.clone());
    } else {
      return false;
    }
  }
  if positions_per_term.iter().any(|p| p.is_empty()) {
    return false;
  }
  for start in positions_per_term[0].iter() {
    let mut ok = true;
    for (i, term_positions) in positions_per_term.iter().enumerate().skip(1) {
      let target = start + i as u32;
      if !term_positions.contains(&target) {
        ok = false;
        break;
      }
    }
    if ok {
      return true;
    }
  }
  false
}

#[cfg(test)]
mod tests {
  use super::*;
  use smallvec::smallvec;

  #[test]
  fn matches_consecutive_positions() {
    let postings = vec![
      vec![PostingEntry {
        doc_id: 1,
        term_freq: 2,
        positions: smallvec![1, 4],
      }],
      vec![PostingEntry {
        doc_id: 1,
        term_freq: 1,
        positions: smallvec![2],
      }],
      vec![PostingEntry {
        doc_id: 1,
        term_freq: 1,
        positions: smallvec![3],
      }],
    ];
    assert!(matches_phrase(&postings, 1));
    assert!(!matches_phrase(&postings, 2));
  }

  #[test]
  fn rejects_non_consecutive_positions() {
    let postings = vec![
      vec![PostingEntry {
        doc_id: 7,
        term_freq: 1,
        positions: smallvec![1],
      }],
      vec![PostingEntry {
        doc_id: 7,
        term_freq: 1,
        positions: smallvec![3],
      }],
    ];
    assert!(!matches_phrase(&postings, 7));
  }
}
