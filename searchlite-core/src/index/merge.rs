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
        let effective_doc_count = |seg: &SegmentMeta| -> u32 {
            seg.doc_count.saturating_sub(seg.deleted_docs.len() as u32)
        };

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

        // Group segments into tiers. The floor tier covers everything up to
        // floor_segment_docs. Each subsequent tier spans 10x the previous
        // tier boundary.
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
            if tier.len() > self.segments_per_tier {
                let take = tier.len().min(self.max_merge_at_once);
                let ids: Vec<String> = tier[..take].iter().map(|s| s.id.clone()).collect();
                // Check that the merged result would not exceed the max size.
                let total_docs: u64 = tier[..take]
                    .iter()
                    .map(|s| effective_doc_count(s) as u64)
                    .sum();
                if total_docs <= self.max_merged_segment_docs as u64 {
                    return vec![ids];
                }
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
        let mut segments = vec![
            make_segment("big1", 1_000),
            make_segment("big2", 1_000),
        ];
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
        let segments = vec![
            seg,
            make_segment("small1", 50),
            make_segment("small2", 50),
        ];
        let merges = policy.find_merges(&segments);
        assert_eq!(merges.len(), 1);
        assert!(merges[0].contains(&"mostly_deleted".to_string()));
    }
}
