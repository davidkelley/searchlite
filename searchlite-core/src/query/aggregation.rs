use std::collections::BTreeMap;

use anyhow::Result;

use crate::api::types::{Aggregation, AggregationResponse};
use crate::index::segment::SegmentReader;
use crate::query::aggs::{
  merge_aggregation_results, AggregationContext, AggregationIntermediate, AggregationNode,
  SegmentAggregationCollector,
};
use crate::query::collector::AggregationSegmentCollector;

/// AggregationPipeline wires query execution to per-segment aggregation
/// collectors. Future aggregation implementations will replace the placeholder
/// collectors used here.
pub struct AggregationPipeline {
  aggs: BTreeMap<String, Aggregation>,
}

impl AggregationPipeline {
  pub fn new(aggs: BTreeMap<String, Aggregation>) -> Self {
    Self { aggs }
  }

  pub fn from_request(aggs: &BTreeMap<String, Aggregation>) -> Option<Self> {
    if aggs.is_empty() {
      None
    } else {
      Some(Self::new(aggs.clone()))
    }
  }

  pub fn for_segment(
    &self,
    segment: &SegmentReader,
  ) -> Result<impl AggregationSegmentCollector<Output = BTreeMap<String, AggregationIntermediate>>>
  {
    let ctx = AggregationContext {
      fast_fields: segment.fast_fields(),
      segment,
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
