# Binding lifecycle (FFI + WASM)

## FFI

- `searchlite_add_json` and `searchlite_add_json_batch` queue documents but do not commit. Call `searchlite_commit` after a batch to make changes visible and durable.
- `searchlite_add_json_batch` accepts a JSON object or array of objects; use it to avoid repeated calls when loading bulk data.
- Use `searchlite_search_request` to send a full `SearchRequest` JSON (filters, sort, aggregations, suggest, etc.). `return_stored` defaults to `false`; set it to `true` when you need stored fields in hits.
- `searchlite_search` remains a convenience for simple query-string searches; it also defaults `return_stored` to `false`.
- Write-key variants (`_with_write_key` suffix) are required when the index was created with a write key; omitting the key returns error code `-8`.

### FFI error codes

All FFI functions that return `c_int` use the following codes:

| Code | Meaning |
| --- | --- |
| `>= 0` | Success. For add functions: document count queued. For commit: `0` = success. |
| `-1` | Null handle or null pointer argument. |
| `-2` | Storage or commit failure (I/O error). |
| `-3` | Commit failed to obtain or use the writer (not write-key related). |
| `-4` | Add operation failed to obtain or use the writer (not write-key related). |
| `-5` | Invalid UTF-8 or malformed JSON in input. |
| `-6` | Parsed JSON is not an object (e.g., array or scalar passed to single-doc add). |
| `-8` | Write key missing or invalid UTF-8 in write key argument. |
| `-100` | Panic caught in Rust code. After a panic from a mutating call (`add`, `commit`), reopen the handle to ensure on-disk consistency. |

### FFI search buffer behavior

`searchlite_search` and `searchlite_search_request` write JSON results into a caller-provided buffer (`out_json_buf` / `buf_cap`). If the result exceeds `buf_cap`, the output is silently truncated to `buf_cap - 1` bytes plus a null terminator. Treat a returned byte count equal to `buf_cap - 1` as potentially truncated and retry with a larger buffer.

## WASM

- `add_document` / `add_documents` queue writes; `commit()` persists them and makes them searchable. `flush_storage()` only drains pending storage writes if you need it.
- `delete_document(id)` queues a single delete by doc id; `delete_documents(ids)` accepts a string or array of strings for batch deletes. Both require `commit()` to persist.
- `update_document({ id, set?, unset? })` queues partial updates by doc id; `set` and `unset` follow core patch semantics and require `commit()` to persist.
- `mget({ ids, return_stored? })` returns `{ docs: [...] }` in input order with per-id `found` and optional `_source`.
- `multi_search({ searches, parallel?, max_concurrency? })` returns `{ results: [...] }` preserving request order.
- Maintenance helpers: `compact()` -> `{ compacted }`, `inspect()` -> `{ manifest }` (write-key fields redacted), and `stats()` -> index-level counts and metadata.
- `search(query, limit, returnStored?)` takes an optional third argument; omit it for the default (`false`) or pass `true` to fetch stored fields.
- `search_request` accepts a JSON string; `search_request_value` takes a JS object. Both support the full `SearchRequest` surface with the same `return_stored` default of `false`.
- Package target note: `wasm-pack build --target web` exposes a default init export; the default bundler target auto-initializes and only exposes named exports (for example `Searchlite`).
- Controlled search variants accept cancellation and timeout controls:
  - `search_controlled(query, limit, returnStored?, abortSignal?, timeoutMs?)`
  - `search_request_controlled(json, abortSignal?, timeoutMs?)`
  - `search_request_value_controlled(value, abortSignal?, timeoutMs?)`
  - `search_request_value_async(value, abortSignal?, timeoutMs?)` (Promise-based worker-oriented entrypoint)
- Lifecycle helpers are exposed as static methods: `Searchlite.list_indexes()`, `Searchlite.clear_index(name)`, and `Searchlite.drop_index(name)`.
- Storage helpers: `Searchlite.storage_usage()` reports usage/quota when the browser exposes it; `Searchlite.cleanup_indexes(stale_older_than_ms, dry_run?)` removes stale persisted indexes by age.
- Index cleanup helper: `db.cleanup_orphaned_files(dry_run?)` removes IndexedDB blobs not referenced by the active manifest.
- `Searchlite.plan_migration(name, schemaJson)` reports `missing`, `compatible`, or `rebuild_required` without mutating state.
- `Searchlite.migrate_index(name, schemaJson)` applies migration planning and returns `created`, `compatible`, or `rebuilt`; on rebuild failure it restores the previous snapshot before returning a typed error.
- Reusing a `db_name` reopens an existing index; schema mismatches still return `schema_mismatch` if you call `init` directly.

### WASM error shape

WASM methods return structured errors instead of plain strings:

```json
{
  "type": "invalid_search_request",
  "reason": "..."
}
```

Use `error.type` for programmatic handling and `error.reason` for logging/UI.

`quota_exceeded` is returned when IndexedDB rejects writes due to storage limits; the reason includes recovery guidance (`compact`, stale-index cleanup, or clearing/dropping unused data).
`aborted` is returned when an `AbortSignal` is already aborted (or becomes aborted at a control check).
`timeout` is returned when `timeoutMs` is exceeded at a control check.

### Worker-first demo notes

- `searchlite-wasm/searchlite-worker-client.mjs` provides a worker-first search wrapper over `searchlite-wasm/searchlite-demo-worker.mjs`.
- The client restarts workers after timeout/abort to stop in-flight operations.
- Worker client options validate `timeoutMs` and `delayMs` as non-negative finite numbers and return typed `invalid_timeout` / `invalid_argument` errors when invalid.
- With `indexeddb` storage this preserves state on restart; with `memory` storage it does not, so the demo falls back to main-thread execution.

### WASM threading

Call `await db.init_threads(threads?)` on the index instance (returned from `Searchlite.init(...)`) before any search if you want multi-threaded execution. The optional `threads` parameter defaults to `navigator.hardwareConcurrency`. Threading requires:

- The `threads` crate feature enabled at build time (`--features threads`).
- The page served with COOP/COEP headers (`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`) so that `SharedArrayBuffer` is available.

If the feature is disabled, `init_threads()` returns an error.

### WASM storage mode caveats

- **Do not switch storage modes for the same `db_name`.** If you create an index with `"indexeddb"` and later reopen the same `db_name` with `"memory"`, the previous IndexedDB data is ignored and effectively lost. Going the other direction loads a stale IndexedDB snapshot. Always use a fresh `db_name` when changing storage modes.
- `add_document` / `add_documents` are synchronous (queued in memory). Only `commit()` is async and triggers IndexedDB persistence. Do not drop the instance immediately after adding documents; always `await commit()` first.
