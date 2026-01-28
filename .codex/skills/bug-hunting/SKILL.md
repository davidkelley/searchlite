---
name: bug-hunting
description: Structured workflow for discovering, reproducing, and fixing bugs in Searchlite by inspecting real code paths (CLI/HTTP/core), validating durability invariants, and producing regression tests; use when investigating incorrect results, crashes, data loss risks, or “it sometimes fails” reports.
---

# Bug Hunting Workflow

## Intent

This skill is for **finding real bugs in the implementation**, not guessing. It forces a discovery phase (build a mental model, reproduce, narrow scope) before making changes.

You must prioritize:

1. correctness and durability
2. clear reproduction
3. minimal fix + regression test

---

## Non-negotiable rules

- **No fix without a reproduction** (a failing test, a minimal repro program, or a deterministic HTTP/CLI script).
- **No durability regressions**: do not weaken WAL/manifest/atomic rename/fsync ordering.
- **No “drive-by refactors”** while bug hunting; keep patches tight and reviewable.
- After changes, you must run the full quality suite from `code-quality`.

---

## Discovery phase (required)

### 0) Establish the boundaries

Before opening files, answer (in notes) these questions:

- Which surface is affected? **CLI**, **HTTP**, **embedded API**, **wasm**, or **ffi**?
- Is it a **correctness** issue (wrong results), **durability** issue (lost/duplicated data), or **stability** issue (panic/hang)?
- Does it require a specific feature flag (`vectors`, `zstd`, `gpu`, wasm/IndexedDB, ffi)?
- Is the failure deterministic or intermittent?

### 1) Baseline the repo (don’t trust your local state)

Run in workspace root:

- `cargo fmt --all`
- `cargo clippy --all --all-features --all-targets -- -D warnings`
- `cargo test --all --all-features`

If anything fails here, you may be chasing a known failure; fix/resolve that first.

### 2) Map the architecture you’re about to debug

Create a quick map in notes (1–2 minutes):

- `searchlite-core`: schema, indexing, WAL/manifest, segments, query execution, filters/aggs, vectors
- `searchlite-cli`: user-facing commands and JSON output formatting
- `searchlite-http`: endpoints (/init, /add, /bulk, /delete, /commit, /refresh, /search, /inspect, /stats, /compact), limits/timeouts/concurrency
- `searchlite-wasm`: bindings + IndexedDB storage semantics (no fsync)
- `searchlite-ffi`: C ABI / safety edges

This map guides where you look first.

### 3) Reproduce the bug with the smallest possible corpus

Produce a “minimal reproducible case” artifact set:

- a schema JSON (small)
- a small NDJSON/doc list (5–50 docs)
- one request (HTTP JSON or CLI command) that fails

For HTTP:

- include the server flags/env (bind addr, index path, max-body-bytes, max-concurrency, request-timeout, refresh-on-commit)

For CLI:

- include exact commands and any flags

**Goal:** a script a teammate can run in <60 seconds.

### 4) Turn on observability (don’t add prints yet)

Use existing debugging features first:

- For search correctness: add `explain: true` to the request.
- For performance/odd slowdowns while reproducing: add `profile: true`.
- For server behavior: run with `RUST_LOG=debug` (or more targeted module filters) and capture the relevant lines.

If your issue is “writes not visible,” verify lifecycle:

- you must `commit`
- you may need `/refresh` in HTTP depending on config

### 5) Narrow the fault line by bisecting the pipeline

Pick one axis at a time:

**Correctness axis**

- Does the doc get ingested successfully?
- Does it appear in `/stats` documents?
- Does `/inspect` show a new segment after commit?
- Does the query match in bm25 mode vs wand/bmw mode?

**Durability axis**

- Is WAL appended correctly?
- Is manifest updated atomically?
- Is WAL truncated only after manifest persistence?
- After restart, does replay do the expected thing?

**HTTP axis**

- Is the request being rejected due to max body / timeouts / concurrency?
- Is NDJSON streaming partial and rolling back?
- Are errors correctly surfaced as `{"error":{"type","reason"}}`?

### 6) Perform a targeted static sweep of the suspected modules

Once you know the rough area, do a deliberate scan for bug smells:

Search patterns to grep/ripgrep for in the relevant crate:

- panics: `unwrap()`, `expect(`, `panic!`, `unreachable!`, `todo!`
- suspicious error handling: `map_err(|_|`, `ok()?` in non-optional contexts, silent `let _ =`
- integer edge cases: casts (`as usize`), underflow (`- 1`), unchecked indexing
- concurrency hazards: `Mutex`/`RwLock` lock ordering, `.await` while holding locks, channels without bounds
- IO hazards: missing `fsync`, non-atomic file replace, partial writes not handled, directory fsync omissions
- schema mismatch hazards: stringly-typed field lookups, analyzer lookup defaults, missing nested path validation
- vector hazards: dimension mismatch, normalization assumptions, unchecked candidate sizing, `NaN` propagation

You are not “fixing” in this step—just identifying suspects.

### 7) Dynamic bug probes by subsystem (choose what fits)

#### A) WAL / manifest / commit / rollback

If you suspect commit/durability issues:

- Reproduce a crash-window scenario:
  1. ingest docs
  2. start commit
  3. force termination mid-way (or simulate via test hooks if present)
  4. restart and observe replay
- Validate the invariants:
  - manifest atomic update (rename) and directory sync
  - WAL truncation happens after manifest sync
  - rollback cleans new segment artifacts

If you can’t safely crash in tests, create a unit test around writer/commit phases and validate state.

#### B) Segment lifecycle / compaction / cleanup

If you see duplicates, ghosts, or bloat:

- Inspect segments via `inspect` and compare to `stats`
- Run compaction and verify:
  - live docs preserved
  - deleted/tombstoned dropped
  - old segment files removed (or expected to remain until cleanup)

#### C) Query execution modes

If you suspect scoring/pruning bugs:

- Run the same query with `execution=bm25`, `execution=wand`, `execution=bmw`
- Results should match for deterministic corpora (unless the contract explicitly says otherwise)
- If they diverge, you likely have a pruning-bound bug or block-max metadata mismatch

#### D) Filters/aggs/nested correctness

If filters/aggs are wrong:

- Verify required fields are `fast: true`
- For nested, verify the filter uses a `Nested` block and that clauses bind to the same object instance
- Add a repro doc with two nested objects that would catch “cross-object” leakage

#### E) Vectors (feature `vectors`)

If vector behavior is wrong:

- Check dimension mismatch handling
- Check cosine normalization expectations
- Confirm ANN parameters (`candidate_size`, `ef_search`) don’t overflow or accept nonsensical values
- Verify hybrid search doesn’t double-count or drop vector scores

---

## Fix phase (required)

### 1) Write the failing regression test first

- Prefer a unit test in the most local crate if possible.
- If the bug is surface-level behavior (HTTP contract, CLI output), add an integration test.
- Make it deterministic and small.
- Assert on structure, not strings (parse JSON output).

### 2) Implement the minimal fix

- Tight patch, minimal diff.
- Keep durability semantics intact.
- Avoid introducing allocations/locks in hot paths unless justified.

### 3) Prove it

Run:

- `cargo test --all --all-features`
- Any relevant examples (e.g. pruning example if you touched pruning logic)
- If the bug touched performance-sensitive code, run the relevant bench before/after.

### 4) Add a “root cause” note

In the PR description or internal notes, include:

- symptom
- root cause
- why the fix is correct
- what test prevents regressions

---

## Output format (what Codex should produce)

When using this skill, produce:

1. **Repro artifact** (schema + docs + request/command)
2. **Fault localization** (files + functions involved)
3. **Root cause analysis**
4. **Fix plan** (minimal changes)
5. **Regression test** (where it lives and what it asserts)
6. **Validation commands run** (from code-quality)
