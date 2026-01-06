pub mod builder;
pub mod errors;
pub mod query;
pub mod reader;
pub mod types;
pub mod writer;

pub use crate::index::Index;
pub use builder::IndexBuilder;
pub use errors::AggregationError;
pub use reader::{
  ExecutionProfile, FunctionExplanation, Hit, HitExplanation, IndexReader, ProfileResult,
  RescoreExplanation, SearchResult,
};
pub use types::{
  Aggregation, AggregationResponse, Aggregations, Document, Filter, FunctionBoostMode,
  FunctionScoreMode, FunctionSpec, FuzzyOptions, IndexOptions, Query, QueryNode, RescoreMode,
  RescoreRequest, SearchRequest, SortOrder, SortSpec, StorageType, SuggestOption, SuggestRequest,
  SuggestResult,
};
pub use writer::IndexWriter;
