use hashbrown::HashMap;
use std::ops::Range;
use std::sync::Arc;

/// Compact term dictionary with O(1) exact lookups and efficient prefix iteration.
///
/// Uses a `HashMap` for exact term lookups (the hot path during search) and a
/// sorted `Vec` for ordered iteration and prefix queries. The sorted vec is
/// built once at construction time and enables binary-search-based prefix
/// scans without per-call allocations.
///
/// Both structures share term strings via `Arc<str>` to avoid duplicating
/// every term in memory (this struct is created per-segment).
#[derive(Debug, Clone, Default)]
pub struct TinyFst {
  map: HashMap<Arc<str>, u64>,
  sorted: Vec<(Arc<str>, u64)>,
}

impl TinyFst {
  pub fn from_terms(terms: &[(String, u64)]) -> Self {
    let map: HashMap<Arc<str>, u64> = terms
      .iter()
      .map(|(k, v)| (Arc::from(k.as_str()), *v))
      .collect();
    // Build sorted from the deduplicated map so iter/iter_prefix stay
    // consistent with get — HashMap::collect already resolved duplicate keys.
    let mut sorted: Vec<(Arc<str>, u64)> = map.iter().map(|(k, v)| (Arc::clone(k), *v)).collect();
    sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Self { map, sorted }
  }

  pub fn get(&self, term: &str) -> Option<u64> {
    self.map.get(term).copied()
  }

  pub fn iter(&self) -> impl Iterator<Item = (&str, &u64)> {
    self.sorted.iter().map(|(k, v)| (k.as_ref(), v))
  }

  pub fn iter_prefix<'a>(
    &'a self,
    prefix: &'a str,
  ) -> impl Iterator<Item = (&'a str, &'a u64)> + 'a {
    // Binary search for the first entry >= prefix, then take while matching.
    let start = self.sorted.partition_point(|(k, _)| k.as_ref() < prefix);
    self.sorted[start..]
      .iter()
      .take_while(move |(k, _)| k.starts_with(prefix))
      .map(|(k, v)| (k.as_ref(), v))
  }

  /// Byte range `[start, end)` of the postings list for `term` inside the
  /// segment's postings file, derived from the next sorted term's offset (or
  /// `postings_len` for the last term).
  ///
  /// The postings writer emits each term's list sequentially in sorted-term
  /// order, so consecutive offsets define each list's exact byte span. This
  /// lets a future object-storage backend (Stage 8) issue a bounded
  /// `read_range` instead of opening the whole file or reading from `offset`
  /// to EOF for short lists late in the dictionary.
  ///
  /// Returns `None` when:
  /// - the term is not present, or
  /// - the derived `end` is at or below `start` (a zero-length range; a
  ///   valid postings list always carries at least its multi-byte header,
  ///   so this can only happen for malformed FST data), or
  /// - the derived `end` exceeds `postings_len` (the next-term offset or
  ///   `postings_len` itself is wrong relative to the actual postings file
  ///   size; a Stage 8 caller would otherwise issue an out-of-bounds range
  ///   read against the object store).
  pub fn range_for(&self, term: &str, postings_len: u64) -> Option<Range<u64>> {
    let pos = self
      .sorted
      .binary_search_by(|(k, _)| k.as_ref().cmp(term))
      .ok()?;
    let start = self.sorted[pos].1;
    let end = self
      .sorted
      .get(pos + 1)
      .map(|(_, off)| *off)
      .unwrap_or(postings_len);
    if end <= start || end > postings_len {
      return None;
    }
    Some(start..end)
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
    let collected: Vec<_> = fst.iter().map(|(k, v)| (k.to_owned(), *v)).collect();
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
    let prefixed: Vec<_> = fst.iter_prefix("body:app").map(|(k, v)| (k, *v)).collect();
    assert_eq!(prefixed, vec![("body:apple", 10), ("body:application", 20)]);
    assert_eq!(fst.iter_prefix("missing").count(), 0);
  }

  #[test]
  fn duplicate_terms_are_deduplicated() {
    let fst = TinyFst::from_terms(&[
      ("alpha".to_string(), 1),
      ("alpha".to_string(), 99),
      ("beta".to_string(), 2),
    ]);
    // HashMap::collect keeps the last value for duplicate keys
    assert_eq!(fst.get("alpha"), Some(99));
    // iter must not yield duplicates — sorted is built from the deduplicated map
    let collected: Vec<_> = fst.iter().map(|(k, _)| k).collect();
    assert_eq!(collected, vec!["alpha", "beta"]);
  }

  #[test]
  fn range_for_middle_term_uses_next_sorted_offset() {
    let fst = TinyFst::from_terms(&[
      ("alpha".to_string(), 0),
      ("beta".to_string(), 100),
      ("gamma".to_string(), 250),
    ]);
    // postings_len here is the file size — irrelevant for non-last terms.
    assert_eq!(fst.range_for("alpha", 1000), Some(0..100));
    assert_eq!(fst.range_for("beta", 1000), Some(100..250));
  }

  #[test]
  fn range_for_last_term_uses_postings_len() {
    let fst = TinyFst::from_terms(&[
      ("alpha".to_string(), 0),
      ("beta".to_string(), 100),
      ("gamma".to_string(), 250),
    ]);
    // The last term's range runs from its offset to the end of the file.
    assert_eq!(fst.range_for("gamma", 1000), Some(250..1000));
  }

  #[test]
  fn range_for_single_term_spans_whole_postings_file() {
    let fst = TinyFst::from_terms(&[("solo".to_string(), 0)]);
    assert_eq!(fst.range_for("solo", 42), Some(0..42));
  }

  #[test]
  fn range_for_returns_none_for_missing_term() {
    let fst = TinyFst::from_terms(&[("alpha".to_string(), 0), ("beta".to_string(), 100)]);
    assert_eq!(fst.range_for("missing", 1000), None);
  }

  #[test]
  fn range_for_returns_none_for_empty_fst() {
    let fst = TinyFst::default();
    assert_eq!(fst.range_for("anything", 1000), None);
  }

  #[test]
  fn range_for_returns_none_when_postings_len_smaller_than_last_offset() {
    // Defensive: a malformed call where `postings_len` < last term's offset
    // would otherwise produce `start > end`. Surface as None rather than
    // returning a nonsense range to the caller.
    let fst = TinyFst::from_terms(&[("alpha".to_string(), 0), ("beta".to_string(), 500)]);
    assert_eq!(fst.range_for("beta", 100), None);
  }

  #[test]
  fn range_for_returns_none_when_neighbor_offset_exceeds_postings_len() {
    // For a non-last term, the derived `end` comes from the next sorted
    // term's offset. If that next offset is past `postings_len` (a
    // tampered manifest, a stale `postings_len`, or a writer bug), the
    // returned range would extend beyond the postings file — a Stage 8
    // caller plumbing this into `BlobStore::get_range` would issue an
    // out-of-bounds range read. Reject it here so the failure surfaces as
    // a missing range, not an opaque storage-layer error.
    let fst = TinyFst::from_terms(&[("alpha".to_string(), 0), ("beta".to_string(), 500)]);
    assert_eq!(
      fst.range_for("alpha", 100),
      None,
      "alpha's neighbor offset (500) exceeds postings_len (100); \
       must reject rather than return an out-of-bounds range"
    );
  }

  #[test]
  fn range_for_returns_none_for_zero_length_range() {
    // Two distinct terms with the same offset should not happen for a
    // well-formed segment — every postings list carries at least a
    // multi-byte header — but if it ever does (corrupt FST data, a future
    // writer bug), the API must reject it. A zero-byte range GET would
    // decode as an empty postings list and silently mask the corruption.
    let fst = TinyFst::from_terms(&[("alpha".to_string(), 50), ("beta".to_string(), 50)]);
    assert_eq!(
      fst.range_for("alpha", 1000),
      None,
      "two terms sharing an offset is malformed; range_for must reject \
       the zero-length range rather than return Some(50..50)"
    );
    // The lexicographically-greater term's range still works because its
    // `end` falls back to `postings_len`, not a duplicate offset.
    assert_eq!(fst.range_for("beta", 1000), Some(50..1000));
  }

  #[test]
  fn range_for_handles_unicode_and_lexicographic_neighbors() {
    // Terms with multi-byte UTF-8 characters must sort and locate by the same
    // byte-level Ord that `binary_search_by` uses, so a term whose Unicode
    // codepoint sorts later than its lexicographic byte successor still gets
    // the correct neighbor offset.
    let fst = TinyFst::from_terms(&[
      ("apple".to_string(), 0),
      ("banana".to_string(), 30),
      ("café".to_string(), 70),
      ("zebra".to_string(), 200),
    ]);
    assert_eq!(fst.range_for("banana", 1000), Some(30..70));
    assert_eq!(fst.range_for("café", 1000), Some(70..200));
  }
}
