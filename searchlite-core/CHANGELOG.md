# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/davidkelley/searchlite/releases/tag/searchlite-core-v0.1.0) - 2026-01-08

### Added

- add P6 aggregations collapse highlight
- initial pass at expanded aggregations and collapsing fields
- improve vector search api and perf
- implemented vector search
- initial implementation for function scoring
- initial implementation of search-as-you-type
- initial implementation of multi-field matching
- tokenizer pipeline
- implemented improved structured query/filter ast
- implemented fuzzy searching
- implemented update/delete actions
- sorting implementation
- improved performance of cursor-based pagination
- add cursor-based pagination
- completed implementation for nested filters
- improved performance for nested-field filters
- adding receipes example
- implemented release-plz
- initial release version
- support in-memory option
- initial commit

### Fixed

- ensure correct workflow artifacts are released
- address clippy regressions
- address review feedback on highlighting and collapse
- address clippy warnings in aggregations
- address review feedback on aggs and collapse
- improve highlighting and add request schema
- address review feedback
- dedupe segment writer and vector filter defaults
- Consolidated write_segment_from_iter so there’s only one definition
- make segment iter writer fallible and non-exact
- add iterator-friendly segment writer entrypoint
- tighten vector query handling
- restore bench build without vectors
- enforce alpha on bm25-only hybrid hits
- honor function_score min_score semantics
- validate function score params and clarify scoring edges
- finalize P4 scoring tweaks and docs
- apply function scoring hooks
- tidy search_as_you_type errors and regex prefix handling
- address additional PR feedback
- address PR review feedback
- address latest review feedback
- address review feedback
- address review feedback
- address PR review feedback
- address PR review feedback
- validate doc_id and cover delete WAL replay
- re-ordering commit flow
- hold the writer lock so WAL replay and live doc loading cant race
- alterations for copilot feedback
- implemented match based sorting algorithm
- improving cursor implementation
- addressing feedback on performance
- updated rust to 1.92.0

### Other

- Merge pull request #34 from davidkelley/fix/ci-builds-and-releases
- Merge branch 'main' into feat/vector-search
- cap vector search work to available vectors
- clarify multi-field section and planner TODO
- Merge branch 'main' into feat/wasm
- fixing formatting error
- Merge branch 'main' into feat/plan-implementation-for-nested-property-filters
- Merge pull request #2 from davidkelley/feat/plan-elasticsearch-like-aggregations-implementation
- using `contains()` instead of `iter().any()` is more efficient
