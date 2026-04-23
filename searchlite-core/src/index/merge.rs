use crate::index::manifest::SegmentMeta;

/// Tiered merge policy inspired by Lucene's TieredMergePolicy.
///
/// Segments are grouped into size tiers (each tier spans roughly 10x the
/// previous one). When any tier contains more segments than `segments_per_tier`,
/// the smallest segments in that tier are selected for merging.
pub struct TieredMergePolicy {
  /// Maximum number of segments to merge in one pass.
  pub max_merge_at_once: usize,
  /// Target number of segments per size tier.
  pub segments_per_tier: usize,
  /// Segments with fewer docs than this are always eligible for merging
  /// (treated as belonging to the smallest tier).
  pub floor_segment_docs: u32,
  /// Cap on the doc count of a merged segment. Segments at or above this
  /// size will not be selected for further merging.
  pub max_merged_segment_docs: u32,
}

impl Default for TieredMergePolicy {
  fn default() -> Self {
    Self {
      max_merge_at_once: 10,
      segments_per_tier: 10,
      floor_segment_docs: 1_000,
      max_merged_segment_docs: 5_000_000,
    }
  }
}

impl TieredMergePolicy {
  /// Evaluate the current set of segments and return groups of segment IDs
  /// that should be merged together. Typically returns zero or one group.
  pub fn find_merges(&self, segments: &[SegmentMeta]) -> Vec<Vec<String>> {
    if segments.len() <= 1 {
      return Vec::new();
    }

    // Effective doc count: live docs only (doc_count minus deleted).
    let effective_doc_count =
      |seg: &SegmentMeta| -> u32 { seg.doc_count.saturating_sub(seg.deleted_docs.len() as u32) };

    // Sort segments by effective doc count ascending.
    let mut sorted: Vec<&SegmentMeta> = segments.iter().collect();
    sorted.sort_by_key(|s| effective_doc_count(s));

    // Filter out segments that are already at or above the max merged size.
    let eligible: Vec<&SegmentMeta> = sorted
      .into_iter()
      .filter(|s| effective_doc_count(s) < self.max_merged_segment_docs)
      .collect();

    if eligible.len() <= 1 {
      return Vec::new();
    }

    // Group segments into tiers. The floor tier covers segments with fewer
    // than floor_segment_docs live docs (exclusive boundary). Each
    // subsequent tier spans 10x the previous tier boundary.
    let mut tiers: Vec<Vec<&SegmentMeta>> = Vec::new();
    let mut tier_max = self.floor_segment_docs.max(1) as u64;

    let mut remaining: &[&SegmentMeta] = &eligible;
    loop {
      let split_pos = remaining
        .iter()
        .position(|s| (effective_doc_count(s) as u64) >= tier_max);
      match split_pos {
        Some(0) => {
          // No segments in this tier range, advance to the next tier.
          tier_max = tier_max.saturating_mul(10);
          if tier_max > self.max_merged_segment_docs as u64 {
            // Remaining segments all exceed the previous tier boundary
            // but are still below max_merged_segment_docs (they passed
            // the eligibility filter). Include them as the final tier.
            if !remaining.is_empty() {
              tiers.push(remaining.to_vec());
            }
            break;
          }
        }
        Some(pos) => {
          tiers.push(remaining[..pos].to_vec());
          remaining = &remaining[pos..];
          tier_max = tier_max.saturating_mul(10);
          if tier_max > self.max_merged_segment_docs as u64 {
            // Remaining segments are within the final tier.
            if !remaining.is_empty() {
              tiers.push(remaining.to_vec());
            }
            break;
          }
        }
        None => {
          // All remaining segments fall in this tier.
          tiers.push(remaining.to_vec());
          break;
        }
      }
    }

    // Find the first tier that has too many segments.
    for tier in tiers.iter() {
      if tier.len() <= self.segments_per_tier {
        continue;
      }
      // Greedy pick of the smallest `take` segments may overshoot
      // `max_merged_segment_docs`. Because `tier` is sorted by effective doc
      // count ascending, the cumulative total grows monotonically with `take`,
      // so shrinking the batch until the total fits yields the largest valid
      // prefix. Only give up on the tier if even the two smallest segments
      // together exceed the cap.
      let mut take = tier.len().min(self.max_merge_at_once);
      while take >= 2 {
        let total_docs: u64 = tier[..take]
          .iter()
          .map(|s| effective_doc_count(s) as u64)
          .sum();
        if total_docs <= self.max_merged_segment_docs as u64 {
          let ids: Vec<String> = tier[..take].iter().map(|s| s.id.clone()).collect();
          return vec![ids];
        }
        take -= 1;
      }
    }

    Vec::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::index::manifest::{SegmentMeta, SegmentPaths};
  use std::collections::HashMap;

  fn make_segment(id: &str, doc_count: u32) -> SegmentMeta {
    SegmentMeta {
      id: id.to_string(),
      generation: 1,
      paths: SegmentPaths {
        terms: String::new(),
        postings: String::new(),
        docstore: String::new(),
        fast: String::new(),
        meta: String::new(),
        #[cfg(feature = "vectors")]
        vector_dir: None,
      },
      doc_count,
      max_doc_id: doc_count,
      blockmax: false,
      deleted_docs: Vec::new(),
      avg_field_lengths: HashMap::new(),
      checksums: HashMap::new(),
      write_binding_b64: None,
    }
  }

  #[test]
  fn no_merge_when_few_segments() {
    let policy = TieredMergePolicy::default();
    let segments = vec![make_segment("a", 100)];
    assert!(policy.find_merges(&segments).is_empty());
  }

  #[test]
  fn no_merge_when_under_threshold() {
    let policy = TieredMergePolicy {
      segments_per_tier: 10,
      ..Default::default()
    };
    // 5 segments in the floor tier -- under the threshold of 10.
    let segments: Vec<_> = (0..5)
      .map(|i| make_segment(&format!("s{i}"), 100))
      .collect();
    assert!(policy.find_merges(&segments).is_empty());
  }

  #[test]
  fn merge_when_floor_tier_overflows() {
    let policy = TieredMergePolicy {
      segments_per_tier: 3,
      max_merge_at_once: 4,
      floor_segment_docs: 1_000,
      ..Default::default()
    };
    // 5 small segments, threshold is 3 => should select up to 4.
    let segments: Vec<_> = (0..5)
      .map(|i| make_segment(&format!("s{i}"), 100))
      .collect();
    let merges = policy.find_merges(&segments);
    assert_eq!(merges.len(), 1);
    assert_eq!(merges[0].len(), 4);
  }

  #[test]
  fn skips_segments_at_max_size() {
    let policy = TieredMergePolicy {
      segments_per_tier: 2,
      max_merge_at_once: 10,
      max_merged_segment_docs: 1_000,
      floor_segment_docs: 100,
    };
    // Two big segments at the cap + three small ones.
    let mut segments = vec![make_segment("big1", 1_000), make_segment("big2", 1_000)];
    for i in 0..3 {
      segments.push(make_segment(&format!("s{i}"), 50));
    }
    let merges = policy.find_merges(&segments);
    assert_eq!(merges.len(), 1);
    // Only the small segments should be selected.
    for id in merges[0].iter() {
      assert!(id.starts_with('s'), "unexpected segment in merge: {id}");
    }
  }

  #[test]
  fn selects_smallest_in_tier() {
    let policy = TieredMergePolicy {
      segments_per_tier: 2,
      max_merge_at_once: 3,
      floor_segment_docs: 100,
      max_merged_segment_docs: 5_000_000,
    };
    let segments: Vec<_> = (0..5)
      .map(|i| make_segment(&format!("s{i}"), 10 * (i as u32 + 1)))
      .collect();
    let merges = policy.find_merges(&segments);
    assert_eq!(merges.len(), 1);
    // Should pick the 3 smallest.
    assert_eq!(merges[0], vec!["s0", "s1", "s2"]);
  }

  #[test]
  fn respects_deleted_docs() {
    let policy = TieredMergePolicy {
      segments_per_tier: 2,
      max_merge_at_once: 10,
      max_merged_segment_docs: 1_000,
      floor_segment_docs: 100,
    };
    // A segment with 900 total but 850 deleted => 50 effective.
    let mut seg = make_segment("mostly_deleted", 900);
    seg.deleted_docs = (0..850).collect();
    let segments = vec![seg, make_segment("small1", 50), make_segment("small2", 50)];
    let merges = policy.find_merges(&segments);
    assert_eq!(merges.len(), 1);
    assert!(merges[0].contains(&"mostly_deleted".to_string()));
  }

  #[test]
  fn shrinks_batch_when_greedy_pick_overshoots_max_merged_docs() {
    // Regression for BUG-008: previously the policy abandoned the tier when
    // the first `max_merge_at_once` segments summed above the cap. It must
    // instead shrink the batch until the cumulative total fits so overflowing
    // tiers do not stall indefinitely.
    let policy = TieredMergePolicy {
      max_merge_at_once: 10,
      segments_per_tier: 10,
      floor_segment_docs: 100_000,
      max_merged_segment_docs: 5_000_000,
    };
    // 11 × 800k: first 10 sum to 8M (> 5M cap). A batch of 6 sums to 4.8M,
    // which is the largest prefix that fits.
    let segments: Vec<_> = (0..11)
      .map(|i| make_segment(&format!("s{i}"), 800_000))
      .collect();
    let merges = policy.find_merges(&segments);
    assert_eq!(
      merges.len(),
      1,
      "oversized greedy pick must not stall the tier"
    );
    assert_eq!(merges[0].len(), 6);
    // The 6 smallest segments (which, at equal size, is a deterministic prefix
    // after the stable sort) should be selected and their total must fit.
    let selected: std::collections::HashSet<_> = merges[0].iter().cloned().collect();
    assert_eq!(selected.len(), 6, "no duplicate segment ids");
  }

  #[test]
  fn gives_up_tier_when_even_two_smallest_exceed_cap() {
    // If even the two smallest segments together overshoot the cap, no valid
    // merge can come from this tier. Policy must not emit a degenerate single-
    // segment "merge" and must not loop forever.
    let policy = TieredMergePolicy {
      max_merge_at_once: 10,
      segments_per_tier: 2,
      floor_segment_docs: 100_000,
      max_merged_segment_docs: 1_000_000,
    };
    // 3 × 700k: all still below the 1M cap (so they pass the eligibility
    // filter) but any pair sums to 1.4M (> 1M cap).
    let segments: Vec<_> = (0..3)
      .map(|i| make_segment(&format!("s{i}"), 700_000))
      .collect();
    let merges = policy.find_merges(&segments);
    assert!(
      merges.is_empty(),
      "no valid batch exists when pairs exceed the cap"
    );
  }

  #[test]
  fn shrinks_batch_to_include_deleted_docs_in_cap_check() {
    // The shrink loop must use *effective* doc counts (i.e. honour tombstones),
    // matching the rest of the policy. A tier where the live docs fit but the
    // raw totals wouldn't should still merge.
    let policy = TieredMergePolicy {
      max_merge_at_once: 10,
      segments_per_tier: 2,
      floor_segment_docs: 100,
      max_merged_segment_docs: 1_000,
    };
    // 3 segments, each 900 total with 500 deleted (live = 400). Raw totals
    // would sum to 2700 (> 1000 cap), but effective totals sum to 1200 (> 1000
    // too, so take=3 is out). A batch of 2 sums to 800 effective docs => fits.
    let segments: Vec<_> = (0..3)
      .map(|i| {
        let mut s = make_segment(&format!("s{i}"), 900);
        s.deleted_docs = (0..500).collect();
        s
      })
      .collect();
    let merges = policy.find_merges(&segments);
    assert_eq!(
      merges.len(),
      1,
      "effective doc counts must drive the shrink"
    );
    assert_eq!(merges[0].len(), 2);
  }

  #[test]
  fn merges_segments_in_upper_tier_when_lower_tiers_empty() {
    // All segments sit above the floor tier. Advancing through empty lower
    // tiers must still include them in the final tier so overflow is detected.
    let policy = TieredMergePolicy {
      segments_per_tier: 2,
      max_merge_at_once: 5,
      floor_segment_docs: 100,
      max_merged_segment_docs: 5_000_000,
    };
    // 4 segments at ~120K docs each — above floor (100) and above 1K, 10K
    // tiers. They should land in the 100K tier and overflow (4 > 2).
    let segments: Vec<_> = (0..4)
      .map(|i| make_segment(&format!("s{i}"), 120_000))
      .collect();
    let merges = policy.find_merges(&segments);
    assert_eq!(merges.len(), 1, "should detect overflow in upper tier");
    assert_eq!(merges[0].len(), 4);
  }
}
