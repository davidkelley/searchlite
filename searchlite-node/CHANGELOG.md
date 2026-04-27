# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.5](https://github.com/davidkelley/searchlite/compare/searchlite-node-v0.2.4...searchlite-node-v0.2.5) - 2026-04-27

### Fixed

- *(node)* align SearchRequestSchema with Rust wire format ([#440](https://github.com/davidkelley/searchlite/pull/440))

### Other

- *(deps-dev)* bump @biomejs/biome from 1.9.4 to 2.4.13 in /searchlite-node ([#424](https://github.com/davidkelley/searchlite/pull/424))
- *(deps-dev)* bump typescript from 5.9.3 to 6.0.3 ([#456](https://github.com/davidkelley/searchlite/pull/456))
- *(deps-dev)* bump vitest from 3.2.4 to 4.1.5 in /searchlite-node ([#428](https://github.com/davidkelley/searchlite/pull/428))

## [0.2.4](https://github.com/davidkelley/searchlite/compare/searchlite-node-v0.2.3...searchlite-node-v0.2.4) - 2026-04-25

### Added

- *(node)* Zod-native schema authoring ([#244](https://github.com/davidkelley/searchlite/pull/244))

### Fixed

- *(schema)* reject documents that omit non-nullable top-level fields (BUG-224) ([#226](https://github.com/davidkelley/searchlite/pull/226))

### Other

- *(deps-dev)* bump the npm-minor-patch group across 1 directory with 4 updates ([#421](https://github.com/davidkelley/searchlite/pull/421))
- *(deps-dev)* bump postcss from 8.5.9 to 8.5.10 in /searchlite-node ([#442](https://github.com/davidkelley/searchlite/pull/442))

## [0.2.2](https://github.com/davidkelley/searchlite/compare/searchlite-node-v0.2.1...searchlite-node-v0.2.2) - 2026-04-13

### Added

- replace custom index schema with JSON Schema + searchlite: vocabulary ([#118](https://github.com/davidkelley/searchlite/pull/118))

## [0.2.1](https://github.com/davidkelley/searchlite/compare/searchlite-node-v0.2.0...searchlite-node-v0.2.1) - 2026-04-10

### Other

- add Node.js/TypeScript to quickstart and enhance searchlite-js README ([#115](https://github.com/davidkelley/searchlite/pull/115))

## [0.2.0](https://github.com/davidkelley/searchlite/compare/searchlite-node-v0.1.5...searchlite-node-v0.2.0) - 2026-04-09

### Added

- add searchlite-js Node.js native addon ([#109](https://github.com/davidkelley/searchlite/pull/109))
