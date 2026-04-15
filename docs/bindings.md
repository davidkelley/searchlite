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

`searchlite_search` and `searchlite_search_request` write JSON results into a caller-provided buffer (`out_json_buf` / `buf_cap`) and use the return value to report success, error, or buffer-too-small:

| Return value `N` | Meaning |
| --- | --- |
| `0` | Error (null argument, search failure, or JSON serialization failure). Buffer is untouched. |
| `0 < N <= buf_cap - 1` | Success. `N` bytes of JSON were written, followed by a NUL terminator. Read the result from `out_json_buf`. |
| `N > buf_cap` | Buffer was too small. No JSON was written (when `buf_cap >= 1` the buffer is NUL-terminated at index 0). `N` is the required size including the NUL terminator -- allocate `N` bytes and retry. |
| `-100` | `searchlite_search` only: a Rust panic was caught (`SEARCHLITE_ERR_PANIC`). |

`N == buf_cap` is never returned: success always leaves at least one byte for the NUL terminator, so `N > buf_cap` is an unambiguous "buffer too small" signal even when `buf_cap == 0`.

**C callers, signed/unsigned caveat for `searchlite_search`:** the return type is `ssize_t` but `buf_cap` is `size_t`. A direct `ret > buf_cap` comparison will promote a negative sentinel such as `SEARCHLITE_ERR_PANIC` (`-100`) to a huge unsigned value and misclassify it as "buffer too small". Check `ret <= 0` first (handling errors and panics), then compare `(size_t)ret > buf_cap` only when `ret > 0`. `searchlite_search_request` returns `size_t`, so a plain `ret > buf_cap` check is sufficient.

## WASM

- `add_document` / `add_documents` queue writes; `commit()` persists them and makes them searchable. `flush_storage()` only drains pending storage writes if you need it.
- `search(query, limit, returnStored?)` takes an optional third argument; omit it for the default (`false`) or pass `true` to fetch stored fields.
- `search_request` accepts a JSON string; `search_request_value` takes a JS object. Both support the full `SearchRequest` surface with the same `return_stored` default of `false`.
- Reusing a `db_name` reopens an existing index; schema mismatches return an error (there is no migration path; use a new `db_name` or delete the stored index).

### WASM threading

Call `await db.init_threads(threads?)` on the index instance (returned from `Searchlite.init(...)`) before any search if you want multi-threaded execution. The optional `threads` parameter defaults to `navigator.hardwareConcurrency`. Threading requires:

- The `threads` crate feature enabled at build time (`--features threads`).
- The page served with COOP/COEP headers (`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`) so that `SharedArrayBuffer` is available.

If the feature is disabled, `init_threads()` returns an error.

### WASM storage mode caveats

- **Do not switch storage modes for the same `db_name`.** If you create an index with `"indexeddb"` and later reopen the same `db_name` with `"memory"`, the previous IndexedDB data is ignored and effectively lost. Going the other direction loads a stale IndexedDB snapshot. Always use a fresh `db_name` when changing storage modes.
- `add_document` / `add_documents` are synchronous (queued in memory). Only `commit()` is async and triggers IndexedDB persistence. Do not drop the instance immediately after adding documents; always `await commit()` first.
