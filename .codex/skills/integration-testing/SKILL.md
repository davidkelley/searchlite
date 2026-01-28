---
name: integration-testing
description: Guidance for end-to-end and integration tests across CLI/HTTP surfaces; use when exercising Searchlite flows through external interfaces.
---

# Integration Testing Guide

## What counts as an integration test here?

An integration test should:

- create a real index (filesystem or in-memory, depending on surface)
- drive Searchlite through a public interface (CLI args or HTTP requests)
- validate observable behavior (hits, counts, errors, lifecycle)

Avoid “testing internals through integration.” If you need to validate internals, write a unit test.

---

## Core end-to-end flows to cover

### HTTP flow (minimum happy path)

1. Start server on ephemeral port bound to localhost.
2. `POST /init` with schema JSON.
3. `POST /add` with NDJSON (or `POST /bulk` with JSON).
4. `POST /commit`.
5. `POST /refresh` (unless refresh-on-commit is enabled).
6. `POST /search` with `return_stored: true`.
7. `GET /stats` and `GET /inspect` sanity assertions.
8. `POST /compact` then re-check stats/inspect invariants.

Assertions that matter:

- status codes
- error payload shape: `{"error":{"type":"...","reason":"..."}}`
- `stats` documents/segments move in expected direction
- search returns `_id` and stored fields when requested
- highlight presence when requested

### CLI flow (minimum happy path)

1. `init <index> <schema>`
2. `add <index> <docs.ndjson>`
3. `commit <index>`
4. `search <index> ...`
5. `inspect <index>`
6. `compact <index>`

Recommended harness:

- `assert_cmd` + temp directories
- parse JSON output and assert on structure not just string contains

---

## Integration tests for “easy to break” features

### Filters + fast fields

- Ensure filtering on keyword + numeric fields works only when `fast: true`.
- Include at least one negative case (filter on non-fast field returns an error or empty, depending on contract).

### Aggregations

- Terms agg on a fast keyword field
- Percentiles/stats on a fast numeric field
- Composite pagination (assert `after_key` behavior if exposed)

### Nested fields

- Nested filter that requires binding to the same nested object
- Deeper nested (nested inside nested) if supported

### Debug flags (HTTP/JSON API)

- `explain: true` returns a score breakdown blob
- `profile: true` returns timing/stats fields
- assert flags are off by default (no extra fields unless requested)

### Execution modes

- Same query under `bm25`, `wand`, `bmw` returns same top-k doc ids for deterministic fixtures
- (allow differences only if the contract explicitly permits)

### Vector search (feature-gated)

- Guard with `#[cfg(feature = "vectors")]`
- Assert dimension mismatch errors
- Assert `vector_score` presence when vector path runs

---

## Resource-limit and shutdown coverage (HTTP)

These are common production footguns; cover at least one:

- max body bytes (expect rejection when exceeded)
- request timeout (simulate with slow client / large payload)
- max concurrency (hammer with parallel requests; expect graceful rejection/backpressure)
- shutdown grace (if the test harness can trigger shutdown)

---

## Determinism rules (don’t write flaky tests)

- Use fixed doc IDs and small, explicit corpora.
- Avoid time-based assertions.
- When testing sampling, fix seeds and assert “shape” (e.g., `sampled: true`) rather than exact counts unless contract guarantees exactness.
- Sort results before comparing unless stable ordering is guaranteed by request `sort`.

---

## Running and maintaining the suite

- Integration tests must pass with `cargo test --all --all-features`.
- Keep runtime reasonable:
  - prefer small fixtures
  - avoid huge corpora
  - use targeted assertions
