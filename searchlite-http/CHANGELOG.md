# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.7](https://github.com/davidkelley/searchlite/compare/searchlite-http-v0.2.6...searchlite-http-v0.2.7) - 2026-05-07

### Other

- publish 5 searchlite crates to crates.io ([#491](https://github.com/davidkelley/searchlite/pull/491))
- *(deps)* bump the rust-minor-patch group across 1 directory with 6 updates ([#487](https://github.com/davidkelley/searchlite/pull/487))

## [0.2.5](https://github.com/davidkelley/searchlite/compare/searchlite-http-v0.2.4...searchlite-http-v0.2.5) - 2026-04-27

### Other

- *(deps)* bump axum from 0.7.9 to 0.8.9 ([#432](https://github.com/davidkelley/searchlite/pull/432))
- *(deps)* bump the rust-minor-patch group across 1 directory with 16 updates ([#448](https://github.com/davidkelley/searchlite/pull/448))

## [0.2.4](https://github.com/davidkelley/searchlite/compare/searchlite-http-v0.2.3...searchlite-http-v0.2.4) - 2026-04-25

### Fixed

- *(http)* classify write-key errors via typed downcast (BUG-406) ([#407](https://github.com/davidkelley/searchlite/pull/407))
- *(http)* drop path from /indexes response (BUG-219) ([#220](https://github.com/davidkelley/searchlite/pull/220))
- *(http)* redact internal error detail from client responses (BUG-016) ([#197](https://github.com/davidkelley/searchlite/pull/197))
- *(http)* drop index_path from /stats response (BUG-015) ([#178](https://github.com/davidkelley/searchlite/pull/178))

### Other

- *(http)* share IndexReader pool across parallel multi_search ([#282](https://github.com/davidkelley/searchlite/pull/282))

## [0.2.3](https://github.com/davidkelley/searchlite/compare/searchlite-http-v0.2.2...searchlite-http-v0.2.3) - 2026-04-13

### Other

- update Cargo.lock dependencies

## [0.2.2](https://github.com/davidkelley/searchlite/compare/searchlite-http-v0.2.1...searchlite-http-v0.2.2) - 2026-04-13

### Added

- replace custom index schema with JSON Schema + searchlite: vocabulary ([#118](https://github.com/davidkelley/searchlite/pull/118))

## [0.2.0](https://github.com/davidkelley/searchlite/compare/searchlite-http-v0.1.5...searchlite-http-v0.2.0) - 2026-04-10

### Added

- product readiness — modularize reader, scale benchmarks, merge policy ([#107](https://github.com/davidkelley/searchlite/pull/107))
- auto commit refresh indexes ([#104](https://github.com/davidkelley/searchlite/pull/104))
- Fuzzy/cross-field search + exact total hits ([#103](https://github.com/davidkelley/searchlite/pull/103))
- feat/nested aggregations ([#102](https://github.com/davidkelley/searchlite/pull/102))
- partial update APIs ([#101](https://github.com/davidkelley/searchlite/pull/101))
- feat/mget documentation ([#100](https://github.com/davidkelley/searchlite/pull/100))
- mget & `search_after` ([#95](https://github.com/davidkelley/searchlite/pull/95))

### Fixed

- inline format args for clippy on Rust 1.88.0 ([#113](https://github.com/davidkelley/searchlite/pull/113))

## [0.1.5](https://github.com/davidkelley/searchlite/compare/searchlite-http-v0.1.4...searchlite-http-v0.1.5) - 2026-01-30

### Added

- multi index routing ([#93](https://github.com/davidkelley/searchlite/pull/93))
- optional write keys ([#85](https://github.com/davidkelley/searchlite/pull/85))

## [0.1.4](https://github.com/davidkelley/searchlite/compare/searchlite-http-v0.1.3...searchlite-http-v0.1.4) - 2026-01-28

### Added

- vector cap env

### Fixed

- address remaining limit=0 review feedback
- unify doc id validation   across ingest and delete paths
- stream ndjson batches with single writer ([#65](https://github.com/davidkelley/searchlite/pull/65))

## [0.1.3](https://github.com/davidkelley/searchlite/compare/searchlite-http-v0.1.2...searchlite-http-v0.1.3) - 2026-01-16

### Other

- perf/optimize wand and http ([#63](https://github.com/davidkelley/searchlite/pull/63))
- remove filters final ([#62](https://github.com/davidkelley/searchlite/pull/62))

## [0.1.2](https://github.com/davidkelley/searchlite/compare/searchlite-http-v0.1.1...searchlite-http-v0.1.2) - 2026-01-14

### Other

- update Cargo.lock dependencies

## [0.1.1](https://github.com/davidkelley/searchlite/compare/searchlite-http-v0.1.0...searchlite-http-v0.1.1) - 2026-01-14

### Added

- add return_hits toggle and limit validation
- supporting http server via cli

### Fixed

- address review feedback on return_hits

### Other

- Merge branch 'main' into feat/p8-ranking-controls

## [0.1.0](https://github.com/davidkelley/searchlite/releases/tag/searchlite-http-v0.1.0) - 2026-01-08

### Added

- implemented http service

### Fixed

- address http review feedback
- various fixes for the branch
