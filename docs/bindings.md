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
| `-3` | Writer creation failed (not write-key related). |
| `-4` | Writer error (generic, non-write-key). |
| `-5` | Invalid UTF-8 or malformed JSON in input. |
| `-6` | Parsed JSON is not an object (e.g., array or scalar passed to single-doc add). |
| `-8` | Write key missing or invalid UTF-8 in write key argument. |
| `-100` | Panic caught in Rust code. After a panic from a mutating call (`add`, `commit`), reopen the handle to ensure on-disk consistency. |

### FFI search buffer behavior

`searchlite_search` and `searchlite_search_request` write JSON results into a caller-provided buffer (`out_json_buf` / `buf_cap`). If the result exceeds `buf_cap`, the output is silently truncated to `buf_cap - 1` bytes plus a null terminator. Always compare the returned byte count against the expected result size to detect truncation.

## WASM

- `add_document` / `add_documents` queue writes; `commit()` persists them and makes them searchable. `flush_storage()` only drains pending storage writes if you need it.
- `search(query, limit, returnStored?)` takes an optional third argument; omit it for the default (`false`) or pass `true` to fetch stored fields.
- `search_request` accepts a JSON string; `search_request_value` takes a JS object. Both support the full `SearchRequest` surface with the same `return_stored` default of `false`.
- Reusing a `db_name` reopens an existing index; schema mismatches return an error (there is no migration path; use a new `db_name` or delete the stored index).

### WASM threading

Call `await Searchlite.init_threads(threads?)` before any search if you want multi-threaded execution. The optional `threads` parameter defaults to `navigator.hardwareConcurrency`. Threading requires:

- The `threads` crate feature enabled at build time (`--features threads`).
- The page served with COOP/COEP headers (`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`) so that `SharedArrayBuffer` is available.

If the feature is disabled, `init_threads()` returns an error.

### WASM storage mode caveats

- **Do not switch storage modes for the same `db_name`.** If you create an index with `"indexeddb"` and later reopen the same `db_name` with `"memory"`, the previous IndexedDB data is ignored and effectively lost. Going the other direction loads a stale IndexedDB snapshot. Always use a fresh `db_name` when changing storage modes.
- `add_document` / `add_documents` are synchronous (queued in memory). Only `commit()` is async and triggers IndexedDB persistence. Do not drop the instance immediately after adding documents; always `await commit()` first.
