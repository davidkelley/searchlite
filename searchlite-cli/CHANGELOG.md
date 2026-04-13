# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.10](https://github.com/davidkelley/searchlite/compare/searchlite-cli-v0.1.9...searchlite-cli-v0.1.10) - 2026-04-13

### Other

- update Cargo.lock dependencies

## [0.1.9](https://github.com/davidkelley/searchlite/compare/searchlite-cli-v0.1.8...searchlite-cli-v0.1.9) - 2026-04-13

### Other

- update Cargo.lock dependencies

## [0.1.8](https://github.com/davidkelley/searchlite/compare/searchlite-cli-v0.1.7...searchlite-cli-v0.1.8) - 2026-04-10

### Other

- update Cargo.lock dependencies

## [0.1.7](https://github.com/davidkelley/searchlite/compare/searchlite-cli-v0.1.6...searchlite-cli-v0.1.7) - 2026-04-10

### Added

- product readiness — modularize reader, scale benchmarks, merge policy ([#107](https://github.com/davidkelley/searchlite/pull/107))
- Fuzzy/cross-field search + exact total hits ([#103](https://github.com/davidkelley/searchlite/pull/103))
- mget & `search_after` ([#95](https://github.com/davidkelley/searchlite/pull/95))

### Fixed

- inline format args for clippy on Rust 1.88.0 ([#113](https://github.com/davidkelley/searchlite/pull/113))

## [0.1.6](https://github.com/davidkelley/searchlite/compare/searchlite-cli-v0.1.5...searchlite-cli-v0.1.6) - 2026-01-30

### Added

- multi index routing ([#93](https://github.com/davidkelley/searchlite/pull/93))
- optional write keys ([#85](https://github.com/davidkelley/searchlite/pull/85))

## [0.1.5](https://github.com/davidkelley/searchlite/compare/searchlite-cli-v0.1.4...searchlite-cli-v0.1.5) - 2026-01-28

### Added

- vector cap env

### Fixed

- address remaining limit=0 review feedback
- unify doc id validation   across ingest and delete paths

## [0.1.4](https://github.com/davidkelley/searchlite/compare/searchlite-cli-v0.1.3...searchlite-cli-v0.1.4) - 2026-01-16

### Other

- remove filters final ([#62](https://github.com/davidkelley/searchlite/pull/62))

## [0.1.3](https://github.com/davidkelley/searchlite/compare/searchlite-cli-v0.1.2...searchlite-cli-v0.1.3) - 2026-01-14

### Other

- update Cargo.lock dependencies

## [0.1.2](https://github.com/davidkelley/searchlite/compare/searchlite-cli-v0.1.1...searchlite-cli-v0.1.2) - 2026-01-14

### Other

- update Cargo.lock dependencies

## [0.1.1](https://github.com/davidkelley/searchlite/compare/searchlite-cli-v0.1.0...searchlite-cli-v0.1.1) - 2026-01-09

### Added

- supporting http server via cli

### Other

- Merge branch 'main' into feat/p8-ranking-controls

## [0.1.0](https://github.com/davidkelley/searchlite/releases/tag/searchlite-cli-v0.1.0) - 2026-01-08

### Added

- add P6 aggregations collapse highlight
- initial pass at expanded aggregations and collapsing fields
- improve vector search api and perf
- initial implementation for function scoring
- initial implementation of search-as-you-type
- implemented improved structured query/filter ast
- implemented fuzzy searching
- implemented update/delete actions
- sorting implementation
- add cursor-based pagination
- improved performance for nested-field filters
- adding receipes example
- implemented release-plz
- initial release version
- support in-memory option
- initial commit

### Fixed

- ensure correct workflow artifacts are released
- dedupe segment writer and vector filter defaults
- harden compaction and sync ffi header
- updated rust to 1.92.0

### Other

- Merge branch 'main' into feat/vector-search
- Merge pull request #2 from davidkelley/feat/plan-elasticsearch-like-aggregations-implementation
