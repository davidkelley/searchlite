---
name: debugging-playbook
description: Playbook for diagnosing Searchlite issues across CLI/HTTP, storage, and indexing; use when triaging failures or odd search results.
---

# Debugging Playbook

## High-level mental model

Searchlite is a single-index, single-writer system with:

- buffered writes (CLI or HTTP)
- durability via WAL + atomic manifest updates
- immutable segments + periodic compaction
- “refresh” semantics for making new commits visible to readers (especially in HTTP mode)

Most “weirdness” is one of:

1. **you didn’t commit** (writes are buffered)
2. **you didn’t refresh** (reader is stale)
3. **your schema isn’t configured for what you’re asking** (e.g., missing `fast` fields for filters/aggs)
4. **you’re seeing crash-window replay artifacts** (extra segments until compaction)
5. **you’re hitting HTTP limits** (body size, concurrency, timeouts)

---

## 10-minute triage checklist

1. **Identify surface area**
   - CLI? embedded Rust API? HTTP server?

2. **Capture exact versions + features**
   - git SHA/tag (if known)
   - build flags / enabled features (`vectors` changes behavior materially)

3. **Get a minimal reproduction**
   - schema JSON
   - the smallest NDJSON/JSON doc set that reproduces
   - the exact search request JSON or CLI flags

4. **Check index state**
   - CLI: `searchlite inspect <index>` and `searchlite compact <index>`
   - HTTP: `GET /inspect`, `GET /stats`, optionally `POST /compact`

5. **Confirm lifecycle steps**
   - Writes are queued → must call `commit`
   - Depending on config, may need `refresh` for immediate visibility

---

## CLI debugging moves

### A) “I added docs but search returns nothing”

Common causes:

- Documents missing the required ID field (`doc_id_field`, default `_id`)
- You didn’t run `commit`
- Query is hitting an analyzer mismatch or field mismatch

Checklist:

- Verify you ran:
  - `searchlite add …`
  - `searchlite commit …`
- Run:
  - `searchlite inspect <index>`
  - `searchlite search <index> --q "..." --return-stored` (or request JSON with `return_stored: true`)
- If segments ballooned after a crash, run:
  - `searchlite compact <index>`

### B) “Filters/aggs don’t work / always empty”

Filters and aggregations depend on `fast` fields.

- Ensure the target keyword/numeric fields are `"fast": true` in schema.
- For nested filters, ensure nested subfields are `fast` too and use `Nested` filters properly.

---

## HTTP debugging moves

### A) Safety + environment sanity

HTTP server has **no auth / no rate limiting**; don’t expose it to untrusted networks.
When debugging production-ish behavior, always capture:

- bind addr (`--bind` / `SEARCHLITE_BIND_ADDR`)
- index path (`--index` / `SEARCHLITE_INDEX_PATH`)
- resource limits:
  - `--max-body-bytes`
  - `--max-concurrency`
  - `--request-timeout-secs`
  - shutdown grace
- refresh behavior:
  - `--refresh-on-commit`

### B) Confirm lifecycle and visibility

Writes issued via:

- `POST /add` (NDJSON streaming)
- `POST /bulk` (JSON bulk)
- `POST /delete`

…are queued and only become durable/searchable after:

- `POST /commit`

Then visibility depends on staleness needs:

- if `--refresh-on-commit` is enabled, new data becomes visible immediately
- otherwise call:
  - `POST /refresh`

### C) Use maintenance endpoints for ground truth

- `POST /refresh` — reload readers
- `POST /compact` — merge segments + clean up fragmentation
- `GET  /inspect` — manifest + segments info
- `GET  /stats` — doc/segment counts

### D) Interpret HTTP errors

All errors return:

- `{"error":{"type":"...","reason":"..."}}`

If you see a 413/timeout-ish behavior:

- check `--max-body-bytes` and `--request-timeout-secs`
- for NDJSON streaming, reduce batch sizes / request size

---

## Ranking correctness + “why did this score happen?”

Search supports explicit debugging aids:

### `explain: true`

- Returns per-hit score breakdowns.
- Use this for correctness (“why did doc X win?”), not for performance runs.

### `profile: true`

- Adds execution stats like:
  - candidates examined
  - scored docs
  - postings advances
  - timing buckets (search vs rescore)
- Use this for performance triage (“why is this query slow?”).

Both are off by default for overhead reasons.

---

## “This query is slow” playbook

1. Enable `profile: true` on the request and capture the profile blob.
2. Compare execution modes:
   - `execution=bm25` (baseline full scoring)
   - `execution=wand` (default exact pruning)
   - `execution=bmw` (block-max pruning)
3. Confirm you’re not forcing expensive features:
   - rescore windows
   - heavy highlighting
   - large aggregations on high-cardinality fields
4. If using aggregations:
   - confirm required fields are `fast`
   - consider sampling knobs where supported
5. If using vector search (feature `vectors`):
   - remember ANN is approximate; raising `candidate_size`/`ef_search` increases recall but costs CPU

---

## Vector search “it errors / results look wrong”

Common causes:

- wrong vector dimension (rejected)
- using cosine without normalization assumptions (Searchlite normalizes cosine vectors automatically)
- expecting exact results from ANN (HNSW is approximate)

Knobs to tune:

- `candidate_size` (oversampling)
- `ef_search` (beam width)
- `vector_filter` (reduce candidate set)

---

## Nested filter “it matches too much / too little”

Key rule: nested filters are evaluated per object, not across the entire document.

Correct pattern:

- wrap clauses under `Nested` with `path`
- for deeper nesting, use a nested `Nested` inside the parent nested filter

Common gotcha:

- forgetting to mark nested filter fields as `fast: true`

---

## Storage/durability anomalies

Symptoms:

- “extra segments” after crash
- “duplicate generation” feeling after restart

Remember the documented crash window:

- if the process dies after manifest persisted but before WAL truncation, WAL replay reapplies last batch
- data isn’t lost, but you may see extra segment generations until compaction

Fix:

- run `compact` (CLI or HTTP) to merge and clean up

---

## What to attach to a bug report

- schema JSON
- small docs ndjson/json
- exact request JSON (include `execution`, `profile`, `explain` if relevant)
- output of `inspect` + `stats`
- server flags/env (HTTP)
- if durability related: note whether crash/restart occurred
