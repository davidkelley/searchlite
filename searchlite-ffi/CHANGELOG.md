# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.8](https://github.com/davidkelley/searchlite/compare/searchlite-ffi-v0.1.7...searchlite-ffi-v0.1.8) - 2026-04-27

### Other

- update Cargo.toml dependencies

## [0.1.7](https://github.com/davidkelley/searchlite/compare/searchlite-ffi-v0.1.6...searchlite-ffi-v0.1.7) - 2026-04-23

### Fixed

- *(ffi)* classify write-key auth errors via typed downcast (BUG-020) ([#237](https://github.com/davidkelley/searchlite/pull/237))
- *(ffi)* signal buffer-too-small instead of truncating JSON (BUG-029) ([#229](https://github.com/davidkelley/searchlite/pull/229))

## [0.1.6](https://github.com/davidkelley/searchlite/compare/searchlite-ffi-v0.1.5...searchlite-ffi-v0.1.6) - 2026-04-10

### Added

- product readiness — modularize reader, scale benchmarks, merge policy ([#107](https://github.com/davidkelley/searchlite/pull/107))
- Fuzzy/cross-field search + exact total hits ([#103](https://github.com/davidkelley/searchlite/pull/103))
- feat/mget documentation ([#100](https://github.com/davidkelley/searchlite/pull/100))
- mget & `search_after` ([#95](https://github.com/davidkelley/searchlite/pull/95))

## [0.1.5](https://github.com/davidkelley/searchlite/compare/searchlite-ffi-v0.1.4...searchlite-ffi-v0.1.5) - 2026-01-30

### Added

- optional write keys ([#85](https://github.com/davidkelley/searchlite/pull/85))

## [0.1.4](https://github.com/davidkelley/searchlite/compare/searchlite-ffi-v0.1.3...searchlite-ffi-v0.1.4) - 2026-01-28

### Fixed

- make ffi/wasm search requests atomic and align   return_stored defaults ([#73](https://github.com/davidkelley/searchlite/pull/73))

## [0.1.3](https://github.com/davidkelley/searchlite/compare/searchlite-ffi-v0.1.2...searchlite-ffi-v0.1.3) - 2026-01-28

### Added

- vector cap env

### Fixed

- *(ffi)* contain panics across   C boundary and stabilize ffi tests

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
