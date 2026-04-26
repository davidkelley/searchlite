# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.9](https://github.com/davidkelley/searchlite/compare/searchlite-wasm-v0.1.8...searchlite-wasm-v0.1.9) - 2026-04-26

### Other

- *(deps)* bump the rust-minor-patch group across 1 directory with 16 updates ([#448](https://github.com/davidkelley/searchlite/pull/448))

## [0.1.8](https://github.com/davidkelley/searchlite/compare/searchlite-wasm-v0.1.7...searchlite-wasm-v0.1.8) - 2026-04-25

### Added

- *(wasm)* harden browser runtime and production workflows ([#179](https://github.com/davidkelley/searchlite/pull/179))

### Fixed

- *(wasm)* release cached IndexedDB connections at end of each lib test ([#444](https://github.com/davidkelley/searchlite/pull/444))

## [0.1.7](https://github.com/davidkelley/searchlite/compare/searchlite-wasm-v0.1.6...searchlite-wasm-v0.1.7) - 2026-04-23

### Added

- *(node)* Zod-native schema authoring ([#244](https://github.com/davidkelley/searchlite/pull/244))

### Fixed

- *(wasm)* bound search limit/from/candidate_size against unbounded allocation (BUG-163) ([#227](https://github.com/davidkelley/searchlite/pull/227))

## [0.1.6](https://github.com/davidkelley/searchlite/compare/searchlite-wasm-v0.1.5...searchlite-wasm-v0.1.6) - 2026-04-10

### Added

- mget & `search_after` ([#95](https://github.com/davidkelley/searchlite/pull/95))

## [0.1.5](https://github.com/davidkelley/searchlite/compare/searchlite-wasm-v0.1.4...searchlite-wasm-v0.1.5) - 2026-01-28

### Fixed

- make ffi/wasm search requests atomic and align   return_stored defaults ([#73](https://github.com/davidkelley/searchlite/pull/73))

## [0.1.4](https://github.com/davidkelley/searchlite/compare/searchlite-wasm-v0.1.3...searchlite-wasm-v0.1.4) - 2026-01-28

### Added

- vector cap env

## [0.1.3](https://github.com/davidkelley/searchlite/compare/searchlite-wasm-v0.1.2...searchlite-wasm-v0.1.3) - 2026-01-16

### Other

- remove filters final ([#62](https://github.com/davidkelley/searchlite/pull/62))

## [0.1.2](https://github.com/davidkelley/searchlite/compare/searchlite-wasm-v0.1.1...searchlite-wasm-v0.1.2) - 2026-01-14

### Fixed

- updated wasm build

## [0.1.1](https://github.com/davidkelley/searchlite/compare/searchlite-wasm-v0.1.0...searchlite-wasm-v0.1.1) - 2026-01-09

### Added

- add multi-vector ranking and scoring hooks

## [0.1.0](https://github.com/davidkelley/searchlite/releases/tag/searchlite-wasm-v0.1.0) - 2026-01-08

### Added

- initial pass at expanded aggregations and collapsing fields
- implemented vector search
- initial implementation for function scoring
- initial implementation of search-as-you-type
- implemented improved structured query/filter ast
- improve wasm init and worker support
- completed, working wasm implementation
- initial experimental wasm implementation

### Fixed

- ensure correct workflow artifacts are released
- updated wasm module index.html example
- address PR review feedback
- updated wasm build path so it compiles for wasm32
- address wasm review feedback
- address wasm review feedback
