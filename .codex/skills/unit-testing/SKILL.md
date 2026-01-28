---
name: unit-testing
description: Patterns for adding and maintaining unit tests in the Searchlite workspace; use when writing or updating Rust tests for core behaviors.
---

# Unit Testing Patterns

## Principles

Unit tests should be:

- **Hermetic**: no network, no global state, no shared directories.
- **Deterministic**: fixed ids, fixed inputs, stable assertions.
- **Small**: quick to run; integration tests cover full flows.
- **Behavior-focused**: assert externally visible behavior of the module/API you’re testing.

---

## Recommended fixture setup

- Use `tempfile::tempdir()` for filesystem-backed tests.
- Prefer a minimal schema:
  - `_id` as `doc_id_field`
  - one text field (indexed + stored if needed)
  - one keyword fast field (filters/aggs)
  - one numeric fast field (ranges/aggs)
  - add nested/vector only when needed

- Prefer NDJSON-style docs for realism, but for unit tests:
  - construct docs via serde_json where possible
  - keep docs minimal and explicit

---

## Common test categories to maintain

### A) Schema validation

- Missing `doc_id_field` in docs should be rejected.
- Analyzer references must exist.
- `fast` fields required for filters/aggs paths that depend on them.
- Nested fields flattening/dotted paths behave as documented.

### B) Write lifecycle correctness

- Add docs → commit → search sees docs
- Delete docs → commit → search hides docs
- Upsert same `_id` → commit → newest version is returned

### C) Durability/WAL replay semantics

- Write + commit; reopen index; ensure data present.
- Simulate crash-window behavior if the API allows:
  - manifest persisted but WAL not truncated → replay creates extra generation
  - compaction cleans duplicates / reduces segments

These tests are crucial if you touch commit ordering, fsync behavior, or WAL encoding.

### D) Query correctness (core matching)

Include at least one test each for:

- query_string on multiple fields
- phrase queries (slop if supported)
- fuzzy matching (max_edits/prefix_length behavior)
- dis_max / multi_match “best fields” semantics (if implemented)

### E) Ranking correctness and stability

- Use a tiny corpus where expected top-k is obvious.
- If testing WAND/BMW pruning:
  - assert results match the full-evaluation baseline (bm25) for deterministic corpora.

### F) Filters and aggregations

Filters:

- Keyword equality
- Numeric range
- And/Or composition

Aggregations:

- terms on keyword fast field
- stats/percentiles on numeric fast field
- test multi-valued semantics if supported

### G) Nested filters binding semantics

- Construct a doc with multiple nested objects.
- Assert that nested constraints bind to the same object instance (not “cross-object” matches).
- Add deeper nesting tests if supported (nested inside nested).

### H) Debug outputs

If you expose `explain` or `profile` at the core layer:

- assert fields appear only when requested
- assert they don’t panic on edge cases (no hits, empty segments, etc.)

### I) Vector search (feature gated)

- Wrap tests in `#[cfg(feature = "vectors")]`
- Dimension mismatch rejects
- ANN knobs don’t panic and return sensible results on small corpora

---

## Assertion style guidelines

- Prefer structured assertions:
  - parse JSON responses into structs or serde_json::Value
  - compare sorted lists for unordered fields
- When checking floats (scores):
  - assert relative ordering, not exact values, unless values are contractually stable
- Avoid “string contains” unless you’re testing human-readable CLI output intentionally.

---

## Where to place tests

- Small module-specific tests: `#[cfg(test)] mod tests { … }` alongside code.
- Cross-module but still “core”: `searchlite-core/tests/*.rs`
- End-to-end surface tests: prefer integration-testing skill instead.

---

## Red flags (rewrite the test if you see these)

- Sleeps/timeouts used for “eventual consistency”
- Assertions that depend on hash iteration order
- Tests that pass only when run alone
- Huge fixtures that slow the suite
- Tests that depend on external binaries or network
