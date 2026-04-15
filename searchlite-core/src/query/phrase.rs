use crate::index::postings::PostingEntry;
use crate::DocId;

pub fn matches_phrase(postings: &[Vec<PostingEntry>], doc_id: DocId, slop: u32) -> bool {
  if postings.is_empty() {
    return true;
  }
  let mut positions_per_term: Vec<Vec<u32>> = Vec::new();
  for term_posts in postings {
    // Postings are stored sorted by `doc_id` (see
    // `InvertedIndexBuilder::add_term` in postings.rs), so a binary search
    // is O(log N) vs. the previous O(N) linear scan. `matches_phrase` is
    // invoked once per candidate doc per phrase term, so the difference
    // dominates phrase-query latency when any term has a long posting list.
    match term_posts.binary_search_by_key(&doc_id, |p| p.doc_id) {
      Ok(idx) => positions_per_term.push(term_posts[idx].positions.iter().copied().collect()),
      Err(_) => return false,
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

  #[test]
  fn finds_doc_in_long_sorted_postings() {
    // Regression test for BUG-021. `matches_phrase` used to walk each term's
    // postings linearly to locate the current `doc_id`; for high-frequency
    // terms that produced per-doc O(N) behaviour when postings are known to
    // be sorted. This test asserts correctness across a long posting list
    // and exercises targets near the middle and at both ends so that the
    // binary-search path is covered end-to-end.
    const N: u32 = 10_000;
    let term_a: Vec<PostingEntry> = (0..N)
      .map(|doc| PostingEntry {
        doc_id: doc,
        term_freq: 1,
        positions: smallvec![1],
      })
      .collect();
    let term_b: Vec<PostingEntry> = (0..N)
      .map(|doc| PostingEntry {
        doc_id: doc,
        term_freq: 1,
        positions: smallvec![2],
      })
      .collect();
    let postings = vec![term_a, term_b];
    assert!(matches_phrase(&postings, 0, 0));
    assert!(matches_phrase(&postings, N / 2, 0));
    assert!(matches_phrase(&postings, N - 1, 0));
    assert!(!matches_phrase(&postings, N, 0));
  }

  #[test]
  fn missing_doc_returns_false_without_matching_neighbour() {
    // The new binary-search path returns `Err(insertion_idx)` when the
    // target doc is absent; guard against a future refactor that forgets
    // to treat `Err` as "no match" and accidentally indexes into the
    // neighbouring entry's positions instead.
    let term_a = vec![
      PostingEntry {
        doc_id: 10,
        term_freq: 1,
        positions: smallvec![1],
      },
      PostingEntry {
        doc_id: 30,
        term_freq: 1,
        positions: smallvec![1],
      },
    ];
    let term_b = vec![
      PostingEntry {
        doc_id: 10,
        term_freq: 1,
        positions: smallvec![2],
      },
      PostingEntry {
        doc_id: 30,
        term_freq: 1,
        positions: smallvec![2],
      },
    ];
    let postings = vec![term_a, term_b];
    assert!(matches_phrase(&postings, 10, 0));
    assert!(matches_phrase(&postings, 30, 0));
    // doc_id 20 is absent — must not silently succeed on a neighbour.
    assert!(!matches_phrase(&postings, 20, 0));
  }
}
