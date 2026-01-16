# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2](https://github.com/davidkelley/searchlite/compare/searchlite-ffi-v0.1.1...searchlite-ffi-v0.1.2) - 2026-01-15

### Other

- remove filters final ([#62](https://github.com/davidkelley/searchlite/pull/62))

## [0.1.1](https://github.com/davidkelley/searchlite/compare/searchlite-ffi-v0.1.0...searchlite-ffi-v0.1.1) - 2026-01-09

### Added

- add multi-vector ranking and scoring hooks

## [0.1.0](https://github.com/davidkelley/searchlite/releases/tag/searchlite-ffi-v0.1.0) - 2026-01-08

### Added

- add P6 aggregations collapse highlight
- initial implementation for function scoring
- initial implementation of search-as-you-type
- implemented improved structured query/filter ast
- implemented fuzzy searching
- implemented update/delete actions
- sorting implementation
- add cursor-based pagination
- implemented release-plz
- initial release version
- support in-memory option
- initial commit

### Fixed

- ensure correct workflow artifacts are released
- harden compaction and sync ffi header
- updated rust to 1.92.0

### Other

- Merge branch 'main' into feat/vector-search
- Merge pull request #2 from davidkelley/feat/plan-elasticsearch-like-aggregations-implementation
