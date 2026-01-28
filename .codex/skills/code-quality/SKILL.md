---
name: code-quality
description: Guardrails for keeping the Searchlite Rust workspace compliant with formatting, lint, build, test, and perf expectations; use when preparing or reviewing changes to ensure they meet project quality bars.
---

# Code Quality Guardrails

## What this skill is for

Use this skill any time you:

- touch core indexing/query code, storage/durability, schema parsing, or the JSON request model
- add/modify CLI flags or HTTP endpoints
- change behavior that could affect correctness, durability, or performance
- add feature-gated functionality (`vectors`, `gpu`, `zstd`, `ffi`, wasm/IndexedDB storage)

This is not a “nice-to-have.” The point is to make changes **reviewable, reproducible, and safe**.

---

## Non-negotiables (definition of done)

A change is “done” only when:

1. **Formatting** is applied to the whole workspace
2. **Clippy** passes with warnings denied
3. **Build** passes with `--all-features` (and ideally default features too)
4. **Tests** pass with `--all-features`
5. **Perf-sensitive changes** have at least one concrete validation (bench, profiling, or a before/after metric)
6. **API/schema changes** update the artifacts that define the contract (`openapi.yaml`, `search-request.schema.json`, and docs/README examples)

---

## Required commands (workspace root)

### Canonical Cargo commands

Run these in the repo root:

- Format:
  - `cargo fmt --all`

- Lint (deny warnings):
  - `cargo clippy --all --all-features --all-targets -- -D warnings`

- Build:
  - `cargo build --all --all-features`

- Test:
  - `cargo test --all --all-features`

- Bench (when touching ranking/index/query hot paths):
  - `cargo bench -p searchlite-core`

### `just` shortcuts

If you use the workspace `Justfile`, keep the “one-liner” workflows working and equivalent:

- `just fmt`
- `just lint`
- `just build`
- `just test`
- `just bench`

> If you update one side (Cargo vs `just`), update the other.

---

## Toolchain policy (MSRV/pinned toolchain)

- The repo pins a specific Rust toolchain via `rust-toolchain.toml` and documents that setup explicitly (currently `1.92.0`).
- Treat that as the “works everywhere” baseline unless you intentionally raise it.

**If you need a newer language/library feature:**

- Prefer an alternative implementation compatible with the pinned toolchain.
- If you must bump the toolchain (currently pinned in `rust-toolchain.toml` to `1.92.0`):
  - change the toolchain file
  - update README “Development setup”
  - ensure CI (if any) uses the same toolchain
  - call out the bump explicitly in the changelog/release notes

---

## Feature matrix expectations

Searchlite is explicitly feature-gated in key areas:

- `vectors` (vector fields + ANN/HNSW behavior)
- `gpu` (GPU stubs / rerank integration points)
- `zstd` (docstore compression behavior)
- `ffi` (C ABI)
- wasm bindings (`searchlite-wasm`, IndexedDB-backed storage)

**Quality bar:**

- Don’t break non-default combinations.
- If you add code under a feature flag, add at least one:
  - unit test gated with `#[cfg(feature = "...")]`, and/or
  - integration test that runs only when the feature is enabled.

---

## “Hot path” performance rules

Searchlite advertises small footprint + fast top-k via WAND/BMW pruning and block maxes. That implies real constraints:

### When touching query evaluation / ranking

- Avoid allocations in inner loops.
- Prefer reusing buffers (e.g., keep scratch vectors on the struct, clear not reallocate).
- Avoid iterator-heavy abstractions in the deepest loops if they obscure costs.
- Measure with Searchlite’s built-in profiling modes when possible (see debugging-playbook).

### Validate pruning modes didn’t regress

Search supports multiple execution strategies:

- `bm25` (full evaluation)
- `wand` (exact WAND pruning)
- `bmw` (block-max WAND)

When changing ranking/pruning logic, verify correctness/perf by:

- running the pruning example:
  - `cargo run -p searchlite-core --example pruning`
- and/or comparing the same query under `execution=bm25` vs `wand` vs `bmw`.

---

## Durability & storage invariants (do not weaken)

Searchlite’s durability story is a core product claim. Preserve it:

### Core guarantees

- Segment files/manifests are flushed (`fsync`) on write.
- WAL is truncated **only after** the manifest is persisted and synced.
- Manifest updates use atomic rename + directory fsync.
- Crash window is explicitly documented: a crash after manifest write but before WAL truncation causes WAL replay to reapply the last batch (no data loss; compaction cleans it up).

### What this means for PRs that touch storage/commit logic

If you change anything in:

- segment writing
- manifest writing
- WAL appending/truncation
- compaction + cleanup
- storage backends (FS vs in-memory vs IndexedDB)

…you must:

- state the durability invariant(s) you preserved in the PR description
- add/adjust tests that cover the crash window / replay behavior
- keep atomic rename + directory fsync semantics for on-disk storage
- keep in-memory storage explicitly “no durability” (don’t accidentally add fsync-ish behavior there)

---

## API contract sync checklist

When you change any of:

- HTTP endpoints, request/response shapes, error format
- CLI flags or subcommand behavior
- JSON request schema (query/filter/aggs/suggest/collapse/etc)

You must update:

- `openapi.yaml` (HTTP contract)
- `search-request.schema.json` (JSON request contract)
- at least one runnable example (README or docs) that exercises the new behavior

Also preserve:

- HTTP errors return the structured form:
  - `{"error":{"type":"...","reason":"..."}}`

---

## Review checklist (copy into PRs)

- [ ] `cargo fmt --all` clean
- [ ] `cargo clippy --all --all-features --all-targets -- -D warnings` clean
- [ ] `cargo build --all --all-features` clean
- [ ] `cargo test --all --all-features` clean
- [ ] If perf-sensitive: bench/profiling evidence included
- [ ] If API/schema touched: `openapi.yaml` + schema JSON updated
- [ ] If durability touched: atomic rename/fsync/WAL ordering preserved + test coverage updated
- [ ] Feature flags respected (vectors/gpu/zstd/ffi/wasm)
- [ ] Docs/examples updated and copy/paste runnable

---

## Commit & changelog hygiene

- Prefer Conventional Commit prefixes: `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `chore:`.
- If change impacts compatibility or durability semantics, call it out explicitly.
