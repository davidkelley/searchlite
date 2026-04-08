pub mod builder;
pub mod errors;
pub mod query;
mod materialization;
mod pagination;
mod phrase;
mod query_eval;
pub mod reader;
mod scoring;
mod suggestion;
mod term_expansion;
pub mod types;
pub mod writer;

pub use crate::index::Index;
pub use builder::IndexBuilder;
pub use errors::{AggregationError, PatchError};
pub use reader::{
  ExecutionProfile, FunctionExplanation, Hit, HitExplanation, IndexReader, MultiSearchResponse,
  ProfileResult, RescoreExplanation, SearchResult,
};
pub use types::{
  Aggregation, AggregationResponse, Aggregations, CollapseRequest, DecayFunction, Document,
  FieldValueModifier, Filter, FunctionBoostMode, FunctionScoreMode, FunctionSpec, FuzzyOptions,
  HighlightField, HighlightRequest, IndexOptions, InnerHitsRequest, MgetDoc, MgetRequest,
  MgetResponse, MultiSearchRequest, Query, QueryNode, RankFeatureModifier, RescoreMode,
  RescoreRequest, SearchRequest, SortOrder, SortSpec, StorageType, SuggestOption, SuggestRequest,
  SuggestResult,
};
pub use writer::IndexWriter;
