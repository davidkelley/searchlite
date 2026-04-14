use crate::index::postings::PostingEntry;
use crate::DocId;

pub fn matches_phrase(postings: &[Vec<PostingEntry>], doc_id: DocId, slop: u32) -> bool {
  if postings.is_empty() {
    return true;
  }
  let mut positions_per_term: Vec<Vec<u32>> = Vec::new();
  for term_posts in postings {
    if let Some(entry) = term_posts.iter().find(|p| p.doc_id == doc_id) {
      positions_per_term.push(entry.positions.iter().copied().collect());
    } else {
      return false;
    }
  }
  if positions_per_term.iter().any(|p| p.is_empty()) {
    return false;
  }
  if positions_per_term.len() == 1 {
    return true;
  }
  fn search(positions: &[Vec<u32>], idx: usize, prev: u32, remaining: i32) -> bool {
    if idx >= positions.len() {
      return true;
    }
    for &pos in positions[idx].iter() {
      if pos <= prev {
        continue;
      }
      let gap = pos.saturating_sub(prev.saturating_add(1)) as i32;
      if gap > remaining {
        // positions are sorted; no later entry will shrink the gap
        break;
      }
      if search(positions, idx + 1, pos, remaining - gap) {
        return true;
      }
    }
    false
  }
  // Saturate to i32::MAX: callers already saturate at the query planning
  // boundary (see planner::MAX_PHRASE_SLOP), but using `as i32` here would
  // wrap values >= 2^31 to a negative remaining budget and silently reject
  // every document — the opposite of the caller's intent — so we keep the
  // saturating cast as defence-in-depth for any future caller that reaches
  // the matcher without going through the planner.
  let remaining = i32::try_from(slop).unwrap_or(i32::MAX);
  for start in positions_per_term[0].iter().copied() {
    if search(&positions_per_term, 1, start, remaining) {
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
    assert!(matches_phrase(&postings, 1, 0));
    assert!(!matches_phrase(&postings, 2, 0));
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
    assert!(!matches_phrase(&postings, 7, 0));
  }

  #[test]
  fn allows_sloppy_phrase() {
    let postings = vec![
      vec![PostingEntry {
        doc_id: 3,
        term_freq: 1,
        positions: smallvec![1],
      }],
      vec![PostingEntry {
        doc_id: 3,
        term_freq: 1,
        positions: smallvec![4],
      }],
      vec![PostingEntry {
        doc_id: 3,
        term_freq: 1,
        positions: smallvec![6],
      }],
    ];
    assert!(!matches_phrase(&postings, 3, 0));
    assert!(matches_phrase(&postings, 3, 3));
  }

  #[test]
  fn saturates_large_slop_instead_of_wrapping_to_negative() {
    // Regression test for BUG-026. Previously `slop as i32` wrapped any value
    // >= 2^31 into a negative "remaining" budget, so the phrase matched zero
    // documents — the exact opposite of the caller's "very loose match"
    // intent. The saturating cast keeps the loose-match semantics.
    let postings = vec![
      vec![PostingEntry {
        doc_id: 1,
        term_freq: 1,
        positions: smallvec![1],
      }],
      vec![PostingEntry {
        doc_id: 1,
        term_freq: 1,
        positions: smallvec![2],
      }],
    ];
    assert!(matches_phrase(&postings, 1, 0));
    assert!(matches_phrase(&postings, 1, u32::MAX));
    assert!(matches_phrase(&postings, 1, (i32::MAX as u32) + 1));
  }
}
