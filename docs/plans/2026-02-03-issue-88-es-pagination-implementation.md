# Issue 88 ES-Style Pagination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify the existing mget/from/search_after/multi_search implementation and align the public API contract (OpenAPI, JSON schema, docs) with the shipped behavior.

**Architecture:** The reader and HTTP layers already implement the functionality; this plan focuses on validation and specification. We will add schema and OpenAPI entries for new pagination fields and endpoints, document the limits and pagination precedence, and add/adjust tests only if gaps are found.

**Tech Stack:** Rust (searchlite-core/searchlite-http), OpenAPI YAML, JSON Schema.

### Task 1: Verify current behavior and capture gaps

**Files:**
- Read: `searchlite-core/src/api/reader.rs`
- Read: `searchlite-core/src/api/types.rs`
- Read: `searchlite-http/src/lib.rs`
- Read: `openapi.yaml`
- Read: `search-request.schema.json`

**Step 1: Write a short verification checklist (notes only)**

Checklist (no code change):
- Confirm `SearchRequest` has `from` and `search_after` and `size` alias.
- Confirm `Hit.sort_key` and `SearchResult.next_search_after` exist.
- Confirm `/indexes/{name}/mget` and `/indexes/{name}/multi_search` handlers exist.
- Identify OpenAPI and schema fields that are missing.

**Step 2: Run a focused build/test to ensure baseline passes**

Run: `cargo build`
Expected: PASS (no compile errors); warnings acceptable but note any clippy issues.

**Step 3: If test gaps are suspected, run HTTP tests only**

Run: `cargo test -p searchlite-http -- tests::http_supports_mget_and_missing_order tests::http_supports_from_and_search_after tests::http_supports_multi_search`
Expected: PASS.

**Step 4: Commit notes only if a new doc or file was added**

If no files changed, skip commit.

### Task 2: Update search request JSON schema

**Files:**
- Modify: `search-request.schema.json`

**Step 1: Write a failing schema assertion (optional)**

If you prefer a test, add a small JSON schema validation test (likely none exist). Otherwise skip.

**Step 2: Add missing properties**

Add to top-level `properties`:
- `from`: integer, minimum 0, default 0, description of offset paging.
- `search_after`: array (nullable), description indicating it must contain sort values + doc_id + segment ordinal.
- Add note to `limit` description that `size` is an alias (already implemented by serde `alias = "size"`).

Example snippet (inline JSON):
```json
"from": { "type": "integer", "minimum": 0, "default": 0, "description": "Offset into hits (use with size/limit)." },
"search_after": { "type": ["array", "null"], "description": "Pagination token from the previous page (sort values + doc_id + segment ord)." }
```

**Step 3: Run a JSON formatting check (if applicable)**

No formatter required; ensure valid JSON.

**Step 4: Commit**

```bash
git add search-request.schema.json
git commit -m "docs: document from/search_after in search schema"
```

### Task 3: Update OpenAPI spec

**Files:**
- Modify: `openapi.yaml`

**Step 1: Add endpoints**

Add `POST /indexes/{name}/mget` and `POST /indexes/{name}/multi_search` with request/response schemas. Reuse the existing `MgetRequest`, `MgetResponse`, `MultiSearchRequest`, and `MultiSearchResponse` definitions (add if missing).

**Step 2: Extend SearchResult/Hit schemas**

Add:
- `next_search_after` to `SearchResult` (array, nullable).
- `sort_key` to `Hit` (array, nullable) describing its use for `search_after`.

**Step 3: Commit**

```bash
git add openapi.yaml
git commit -m "docs: add mget/multi_search and pagination fields to OpenAPI"
```

### Task 4: Update user docs (README or docs/quickstart)

**Files:**
- Modify: `README.md` or `docs/quickstart.md`

**Step 1: Add examples**

Add concise examples for:
- `/indexes/{name}/mget` request/response.
- `/indexes/{name}/search` with `from`/`size`.
- `/indexes/{name}/search` with `search_after` and `next_search_after`.
- Mention `MAX_PAGE_SIZE = 1000`, `MAX_MGET_IDS = 1024`, and that cursor/search_after/from are mutually exclusive.

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add pagination and mget examples"
```

### Task 5: Verification and wrap-up

**Files:**
- Verify: `search-request.schema.json`, `openapi.yaml`, docs

**Step 1: Run formatting and linting**

```bash
cargo fmt --all
cargo clippy --all --all-features --all-targets -- -D warnings
```
Expected: PASS

**Step 2: Run tests**

```bash
cargo test --all --all-features
```
Expected: PASS

**Step 3: Optional bench (if performance-sensitive changes were made)**

```bash
cargo bench -p searchlite-core
```
Expected: PASS (compile-only if no benches)

**Step 4: Final commit (if needed)**

If any edits remain uncommitted, add and commit with Conventional Commit type.
