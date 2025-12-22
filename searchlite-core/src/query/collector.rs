use crate::DocId;

/// A lightweight callback-style collector for matched documents.
///
/// Aggregations stream every accepted document through this trait so that
/// per-segment collectors can update their state without materializing all
/// postings in memory.
pub trait DocCollector {
  fn collect(&mut self, doc_id: DocId, score: f32);
}

/// Represents a segment-scoped aggregation collector that can be finalized
/// after all documents have been streamed.
pub trait AggregationSegmentCollector: DocCollector {
  type Output;

  /// Finalizes the collector for a segment and returns its partial output.
  fn finish(self) -> Self::Output;
}

/// A simple segment collector that only counts matched documents. It exists to
/// validate the document streaming path until richer aggregation collectors are
/// wired in.
#[derive(Default)]
pub struct MatchCountingCollector {
  matches: u64,
}

impl DocCollector for MatchCountingCollector {
  fn collect(&mut self, _doc_id: DocId, _score: f32) {
    self.matches += 1;
  }
}

impl AggregationSegmentCollector for MatchCountingCollector {
  type Output = u64;

  fn finish(self) -> Self::Output {
    self.matches
  }
}

#[cfg(test)]
#[derive(Default)]
pub struct RecordingCollector {
  pub docs: Vec<(DocId, f32)>,
}

#[cfg(test)]
impl DocCollector for RecordingCollector {
  fn collect(&mut self, doc_id: DocId, score: f32) {
    self.docs.push((doc_id, score));
  }
}
