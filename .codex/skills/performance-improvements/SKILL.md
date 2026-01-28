---
name: performance-improvements
description: Methodical workflow for discovering and fixing performance issues in Searchlite by measuring real workloads, using built-in profiling/explain tools, and optimizing hot paths without compromising durability or correctness; use when tuning indexing speed, query latency, memory, or throughput.
---

# Performance Improvements Workflow

## Intent

This skill is for improving performance by:

- running a discovery phase to identify true hotspots
- measuring before/after
- making targeted optimizations that preserve correctness and durability
- adding regression coverage (benchmarks or tests) when practical

---

## Non-negotiable rules

- **Measure first**. No optimization without a baseline.
- **One change at a time** (or a small coherent set) with before/after numbers.
- **Do not weaken durability** (WAL/manifest/fsync/atomic write ordering stays intact).
- After changes, run the full suite from `code-quality`.

---

## Discovery phase (required)

### 0) Define what “better” means (pick a target)

Pick one primary metric:

- Search p50/p95 latency (ms)
- Indexing throughput (docs/sec) or commit time
- Memory usage (RSS / peak allocations)
- HTTP ingestion throughput (MB/s) or request timeouts/rejections
- Compaction runtime and resulting segment count/size

Write it down as:  
**Target:** “Improve X by ~Y without regressing correctness.”

### 1) Choose a representative workload

You need a workload that resembles reality:

- For search: a set of queries including filters/aggs/highlights if used
- For ingest: a doc set and commit cadence that matches production
- For vectors: a vector corpus and ANN parameters

Prefer:

- small but realistic corpus for quick iteration (thousands of docs)
- optional larger corpus for final validation (hundreds of thousands+)

### 2) Build and run in release mode

Performance work must be done in `--release`:

- `cargo build --release --all-features`
- run your workload with the release binary

If you’re benchmarking, ensure the environment is stable (avoid noisy background work).

### 3) Use Searchlite’s built-in query profiling tools

For search workloads:

- Add `profile: true` to capture execution stats and timing breakdowns.
- Use `explain: true` only when investigating scoring/candidate issues (it adds overhead).

Compare execution strategies:

- `execution=bm25` (baseline)
- `execution=wand` (default exact pruning)
- `execution=bmw` (block-max pruning)

If bm25 is similar to wand/bmw, pruning isn’t working as intended or your query pattern defeats it.

### 4) Establish baselines and record them

Capture:

- command lines / request JSON used
- corpus size (docs, fields, terms, vectors)
- timings (at least 5 runs if feasible) and report min/median

For `searchlite-core` hot path changes:

- run `cargo bench -p searchlite-core`
- store the before results in your notes/PR

### 5) Identify where time is really going (don’t guess)

Classify the hotspot into one bucket:

#### A) Query evaluation CPU

Symptoms: profile shows time in postings traversal, scoring, pruning, intersections.

#### B) Candidate expansion / memory churn

Symptoms: lots of allocations, large intermediate vectors, repeated sorting/heap ops.

#### C) Docstore / highlight / JSON serialization

Symptoms: time after scoring, fetching stored fields dominates, highlight is expensive.

#### D) Filters/aggs

Symptoms: time in aggregations; large cardinality terms aggs; missing fast fields causes slow fallback.

#### E) IO / fsync overhead

Symptoms: commit latency spikes; indexing throughput limited by disk flush behavior.

#### F) HTTP ingestion

Symptoms: parsing costs, backpressure stalls, concurrency limits hit, 413/timeout, high per-request overhead.

#### G) Vectors (ANN)

Symptoms: recall vs latency tradeoffs; ef_search/candidate_size too high; vector filtering not used.

Pick the dominant bucket first; do not optimize secondary paths until the main bucket is improved.

---

## Implementation analysis checklist (required)

Once you know the bucket, do a focused scan of the code involved.

### General scan patterns (performance smells)

Look for:

- allocations inside tight loops:
  - `Vec::new()` / `String::new()` inside per-doc/per-posting loops
  - `.to_string()` / `.clone()` on hot data
  - `collect::<Vec<_>>()` just to iterate again
- repeated sorting where incremental selection would work (top-k heaps vs sort-all)
- repeated hashing of the same keys (`HashMap` in inner loops)
- converting between owned/borrowed forms repeatedly (String <-> &str)
- lock contention:
  - locks held across `.await`
  - coarse locks around scoring or per-hit processing
- IO inefficiencies:
  - reopening files repeatedly
  - small reads without buffering
  - syncing too frequently (but do not weaken durability rules)

### Query path specifics

- Verify WAND/BMW pruning metadata is used effectively.
- Ensure block-max metadata is not recomputed repeatedly.
- Ensure per-hit work is minimized:
  - fetch stored fields only if requested
  - highlight only when requested
  - avoid decoding docstore for filtered-out candidates

### Aggregations specifics

- Ensure aggs operate on fast fields and avoid per-doc dynamic dispatch.
- Watch high-cardinality terms aggs:
  - reduce allocations, reuse buffers
  - avoid repeated string materialization if possible
- If composite pagination exists, avoid re-sorting full bucket sets each page.

### Indexing/commit specifics

- Segment writing should be streaming and avoid holding the entire docset in memory.
- Validate compression choices (`zstd`) only impact the intended paths.
- Ensure fsync/atomic rename/directory sync semantics remain intact (don’t “optimize” them away).

### HTTP specifics

- NDJSON streaming should use bounded channels/backpressure.
- Avoid buffering full bodies unnecessarily (respect `max-body-bytes`).
- Ensure request timeouts and concurrency limits are enforced without excessive overhead.
- Reduce serde overhead in hot request paths where possible (but keep correctness).

### Vector specifics (`vectors`)

- ANN parameters control CPU directly:
  - lower `ef_search` reduces latency at cost of recall
  - lower `candidate_size` reduces rescoring costs
- Prefer vector filtering to shrink the candidate set before ANN where supported.
- Ensure normalization happens once and doesn’t allocate repeatedly.

---

## Optimization playbook (apply in order)

### 1) Remove unnecessary work

- Short-circuit early (filters before expensive scoring if valid).
- Don’t compute or serialize fields not requested.
- Avoid repeated decoding of the same stored content.

### 2) Reduce allocations

- Reuse buffers on structs (clear instead of reallocate).
- Pre-allocate vectors with `with_capacity` when sizes are known/estimable.
- Prefer slices/borrowed views over owned clones in inner loops.

### 3) Improve algorithmic complexity

- For top-k: avoid “sort everything” when a heap/partial selection is enough.
- For intersections: ensure the smallest postings list drives the loop.
- For high-cardinality aggs: reduce key materialization and avoid repeated hashing.

### 4) Optimize data locality

- Keep hot structs small and contiguous.
- Reduce indirection in tight loops (avoid iterator chains that hide branching/allocs).
- Use integer IDs instead of strings where safe (e.g., field IDs vs field names).

### 5) IO improvements (with durability preserved)

- Batch reads/writes where safe.
- Avoid redundant fsync calls, but never remove required ones:
  - segment files must be persisted
  - manifest update must be atomic and synced
  - WAL truncation ordering must remain correct

---

## Validation phase (required)

### 1) Prove correctness didn’t change

- Run: `cargo test --all --all-features`
- For query-path changes:
  - verify same results across `bm25` vs `wand` vs `bmw` on a deterministic corpus
  - use `explain` on a couple of hits to ensure scoring logic remains consistent

### 2) Prove performance improved

You must provide at least one of:

- `cargo bench -p searchlite-core` before/after results (preferred)
- recorded request latencies with `profile: true` before/after
- ingest/commit timings before/after on a fixed corpus

Include:

- baseline numbers
- new numbers
- percent change
- environment notes (release build, feature flags)

### 3) Add a regression guard when feasible

- Add/adjust a benchmark for the hot path you improved, OR
- Add a test that prevents the pathological behavior (e.g., “does not allocate unboundedly” via size caps; “does not decode stored fields when return_stored=false”; etc.)

---

## Output format (what Codex should produce)

When using this skill, produce:

1. **Workload definition** (schema + corpus + queries/requests)
2. **Baseline measurements**
3. **Hotspot classification** (A–G bucket)
4. **Code-level findings** (files/functions + why they’re hot)
5. **Optimization plan** (ordered steps)
6. **Patch summary** (what changed and why)
7. **After measurements**
8. **Risk assessment** (correctness/durability/feature-flag impact)
9. **Commands run** (from code-quality)
