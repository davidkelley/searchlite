use std::collections::BTreeMap;

use anyhow::Result;

use crate::api::types::{Aggregation, AggregationResponse};
use crate::index::segment::SegmentReader;
use crate::query::aggs::{
  merge_aggregation_results, AggregationContext, AggregationIntermediate, AggregationNode,
  SegmentAggregationCollector,
};

/// AggregationPipeline wires query execution to per-segment aggregation
/// collectors. Future aggregation implementations will replace the placeholder
/// collectors used here.
pub struct AggregationPipeline {
  aggs: BTreeMap<String, Aggregation>,
  highlight_terms: Vec<String>,
}

impl AggregationPipeline {
  pub fn new(aggs: BTreeMap<String, Aggregation>, highlight_terms: Vec<String>) -> Self {
    Self {
      aggs,
      highlight_terms,
    }
  }

  pub fn from_request(
    aggs: &BTreeMap<String, Aggregation>,
    highlight_terms: &[String],
  ) -> Option<Self> {
    if aggs.is_empty() {
      None
    } else {
      Some(Self::new(aggs.clone(), highlight_terms.to_vec()))
    }
  }

  pub(crate) fn for_segment<'a>(
    &'a self,
    segment: &'a SegmentReader,
  ) -> Result<SegmentAggregationCollector<'a>> {
    let ctx = AggregationContext {
      fast_fields: segment.fast_fields(),
      segment,
      highlight_terms: &self.highlight_terms,
    };
    let aggs = self
      .aggs
      .iter()
      .map(|(name, agg)| {
        (
          name.clone(),
          AggregationNode::from_request(ctx.clone(), agg),
        )
      })
      .collect();
    Ok(SegmentAggregationCollector::new(aggs))
  }

  pub fn merge(
    &self,
    segments: Vec<BTreeMap<String, AggregationIntermediate>>,
  ) -> Result<BTreeMap<String, AggregationResponse>> {
    let _ = &self.aggs;
    Ok(merge_aggregation_results(segments))
  }
}
