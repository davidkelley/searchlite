# Issue #89 Partial Update API Design

## Goal
Add partial document mutation APIs (`set`/`unset`) so clients can update availability-style fields without full reindex. The update path should be consistent with existing WAL/commit semantics, preserve fast fields, and reject unsafe schemas.

## Decisions
- **Implementation strategy:** Read current stored document, apply patch, enqueue a normal add for the same `doc_id` (delete+add behavior on commit). No new WAL entry types.
- **Bulk semantics:** Best-effort. Each item returns status; failures do not abort the entire request.
- **Safety:** Reject updates when the schema has any field that is `indexed` or `fast` but not `stored` (prevents silent data loss when reconstructing docs).
- **Patch order:** Apply `unset` first, then `set`.
- **Paths:** Dotted paths allowed for nested fields. Arrays are not traversed; only object keys are removed/created.

## Core Flow
1. Validate patch: non-empty `set`/`unset`, `id` valid, no `doc_id_field` mutation, and paths resolve to schema fields (including nested leaf paths).
2. Resolve current doc:
   - Check pending ops (latest add wins; pending delete yields 404).
   - Otherwise load from committed segments via reader mget (`return_stored=true`).
3. Apply patch via dot-walk on a JSON object.
4. Validate full document with `Schema::validate_document`.
5. Enqueue add via writer (WAL append + pending op). Commit remains explicit.

## HTTP API
- `POST /indexes/:name/update`
  ```json
  { "id": "urn:123", "set": { "quantity": 10 }, "unset": ["max"] }
  ```
- `POST /indexes/:name/_bulk_update` (NDJSON pairs)
  ```
  {"update":{"_id":"urn:123"}}
  {"set":{"quantity":10},"unset":["max"]}
  ```
Errors: 404 missing doc, 400 validation, 401/403 write key, 500 internal.

## Testing
- Unit tests: dot-path set/unset, nested updates, missing doc, non-stored field rejection.
- Integration: end-to-end update via HTTP, bulk mixed success, nested required field removal.
- Recovery: update enqueued then replayed from WAL should yield correct doc after commit.

## Risks
- **Performance:** Each update reads and rewrites full doc. Acceptable for small documents; document in API docs.
- **Structure conflicts:** Setting a path that crosses non-object values should return 400.
