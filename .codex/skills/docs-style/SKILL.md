---
name: docs-style
description: Style guide for writing Searchlite documentation and examples; use when creating or editing docs/ guides.
---

# Documentation Style

## Goal

Docs should be **copy/paste runnable** and should teach the lifecycle:
init → ingest → commit → (refresh) → search → maintain (inspect/compact)

Assume a junior/mid developer integrating Searchlite for the first time.

---

## Required conventions (use everywhere)

### 1) Always introduce an index path variable

Prefer:

- CLI docs: `INDEX=/tmp/searchlite_idx`
- HTTP docs: `SEARCHLITE_INDEX_PATH=/tmp/searchlite_idx`

### 2) Always mention the required ID field

- Every document must include the schema’s `doc_id_field` (default `_id`)
- If a doc is missing it, it will be rejected

### 3) Always include an explicit **commit** step after writes

Searchlite buffers writes. Your examples must show:

- `commit` after `add`/`bulk`/`delete`

### 4) Always call out refresh semantics for HTTP

- If `--refresh-on-commit` is disabled, readers may be stale until `POST /refresh`.

### 5) Always call out fast vs stored

- `stored: true` controls what can be returned
- `fast: true` controls what can be filtered/aggregated efficiently

---

## Safety warnings (must appear in HTTP docs)

The HTTP server:

- provides **no authentication**
- provides **no authorization**
- provides **no rate limiting**

Docs must include a bold warning:

- “Do not expose directly to untrusted networks; place behind a proxy/API gateway that enforces auth + rate limiting.”

Also include the “bind safely” recommendation:

- default bind is localhost; show how to bind `0.0.0.0` only when behind a firewall/proxy.

---

## Examples must be complete

Every guide/example must include all three artifacts:

1. **schema JSON**
2. **documents** (NDJSON or JSON)
3. **request JSON** (search / filter / aggs / etc)

Do not show “…”-style pseudo snippets for the main flow. If you must omit details, provide:

- a minimal working version
- then an “extended” version

---

## Code block conventions

- Use fenced blocks with language tags:
  - `bash`, `json`, `yaml`
- Prefer 1 command per line
- When showing multi-step flows, number them

---

## Docs structure templates

### Quickstart / tutorial

1. Install / build
2. Init index
3. Add docs
4. Commit
5. Search (show return_stored + highlight)
6. Inspect / stats
7. Compact (explain why)

### HTTP serving guide

1. Security warning (no auth)
2. Run server (CLI flag form + env var form)
3. `/init` example
4. `/add` (NDJSON streaming) + `/commit` (+ optional `/refresh`)
5. `/search` example
6. Maintenance endpoints (`/inspect`, `/stats`, `/compact`, `/refresh`)
7. Troubleshooting section (body size, timeouts, commit/refresh)

### Reference docs (query/filter/aggs)

- Start with “Field requirements”
  - aggs require fast fields (terms = keyword fast; percentiles = numeric fast; etc.)
- Include at least one runnable example per feature:
  - fuzzy
  - phrase/slop
  - collapse + inner_hits
  - rescore
  - profile/explain
  - vectors (feature-gated)

---

## Consistency + contract sync

Whenever docs mention HTTP behavior:

- ensure it matches `openapi.yaml`
- errors should be shown as `{"error":{"type":"...","reason":"..."}}`

Whenever docs mention request JSON shape:

- ensure it matches `search-request.schema.json`

---

## “Docs review” checklist

- [ ] Uses `INDEX` or `SEARCHLITE_INDEX_PATH`
- [ ] Includes schema + docs + request JSON
- [ ] Includes commit step
- [ ] HTTP docs include security warning + refresh semantics
- [ ] Notes fast vs stored requirements
- [ ] Commands work as written (no placeholders)
- [ ] Cross-links or references to related guides/endpoints
