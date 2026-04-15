# WASM error reference

Every `searchlite-wasm` method rejects with the same shape:

```json
{
  "type": "quota_exceeded",
  "reason": "indexeddb quota exceeded while committing index data; run compact(), remove stale indexes with Searchlite.cleanup_indexes(...), or clear/drop unused indexes before retrying. detail: ..."
}
```

- `type` is a stable machine-readable code. Use it for `switch`-style dispatch.
- `reason` is a human-readable message suitable for logs and UI. Treat it as
  advisory — do not parse it.

The same shape is used from:

- `Searchlite` static methods (`init`, `list_indexes`, `drop_index`, ...)
- `Searchlite` instance methods (`add_documents`, `commit`, `search_*`, ...)
- `SearchliteWorkerClient` methods — worker errors are normalized to the same
  `{ type, reason }` shape before being rejected on the main thread.

---

## Dispatch pattern

```javascript
try {
  await db.commit();
} catch (err) {
  switch (err.type) {
    case "quota_exceeded":
      await db.compact();
      await Searchlite.cleanup_indexes(30 * 24 * 3600 * 1000);
      break;
    case "aborted":
    case "timeout":
      // operation was cancelled — usually no action needed
      break;
    case "schema_mismatch":
      await Searchlite.migrate_index(name, schemaJson);
      break;
    default:
      console.error("unhandled searchlite error", err.type, err.reason);
      throw err;
  }
}
```

---

## Validation

Errors caused by malformed input before any IndexedDB or search work runs.
Fix the input and retry.

| `type` | When | Recovery |
| --- | --- | --- |
| `invalid_json` | A JS value could not be parsed as JSON / converted to the expected shape (e.g. `add_document(doc)` where `doc` is not an object). | Pass a valid JSON-serializable object. |
| `invalid_schema_json` | `init` / `plan_migration` / `migrate_index` received a `schema_json` string that isn't parseable as a Searchlite schema. | Fix the schema. See [docs/wasm.md § Schema mini-reference](wasm.md#schema-mini-reference) for the shape. |
| `invalid_search_request` | `search_request` / `search_request_value` received a payload that isn't a valid `SearchRequest`. | Fix the request shape. |
| `invalid_update_request` | `update_document` received a payload that isn't `{ id, set?, unset? }`. | Fix the request shape. |
| `invalid_doc_id_batch` | `delete_documents` received a non-array or an array with non-string members. | Pass an array of strings. |
| `invalid_id` | A doc id failed validation (empty, whitespace only, control characters). | Use a non-empty printable string id. |
| `invalid_document` | Document shape didn't match the schema (missing required field, wrong type). | Align the document with the schema. |
| `invalid_cleanup_request` | `cleanup_indexes(stale_older_than_ms, ...)` received a negative / non-finite duration. | Pass a non-negative finite number of milliseconds. |
| `invalid_timeout` | A controlled search or worker-client method received a negative / non-finite `timeoutMs`. | Pass a non-negative finite number. |
| `invalid_argument` | Worker-client method received an invalid option (e.g. negative `delayMs`). | Inspect `err.reason` and fix the argument. |
| `missing_patch` | `update_document` payload had neither `set` nor `unset` fields. | Include at least one `set` or `unset` clause. |
| `reserved_name` | An index operation used the reserved registry db name (`searchlite_registry`). | Pick a different `db_name`. |

---

## Lifecycle & storage

Errors raised while opening, reading, writing, or deleting persisted state.
These are usually transient (closed connection, stale transaction) or
environmental (IndexedDB disabled, storage partitioned).

| `type` | When | Recovery |
| --- | --- | --- |
| `storage_open_error` | `JsStorage::new` failed to open the IndexedDB database (incl. during `init`). | Check that IndexedDB is available and not disabled by privacy mode. Retry. |
| `storage_clear_error` | `clear_index` / internal clear-during-rebuild failed to clear the object store. | Retry. Inspect browser storage quota / permissions. |
| `storage_delete_error` | `drop_index` / `cleanup_indexes` failed to delete the IndexedDB database. | Close other tabs holding the DB open, then retry. |
| `storage_flush_error` | `commit()` / `flush_storage()` failed to drain pending writes. Will appear as `quota_exceeded` when the underlying cause is storage limits. | Inspect `err.reason`; retry after freeing space if quota-related. |
| `storage_read_error` | Reading a persisted blob failed. | Retry. Registry might be corrupted — consider `drop_index` + re-ingest. |
| `storage_write_error` | Writing a persisted blob failed outside the normal flush path. | Retry. See also `quota_exceeded`. |
| `storage_list_error` | Listing stored paths failed (during orphan cleanup / reindex). | Retry. |
| `storage_snapshot_error` | Capturing the pre-rebuild snapshot during `migrate_index` failed. | Retry. The rebuild aborts before touching data; state is unchanged. |
| `storage_cleanup_error` | `cleanup_orphaned_files` failed to delete an unreferenced blob. | Retry. The active manifest's files are preserved. |
| `registry_read_error` | Listing the registry of known indexes failed. | Retry. |
| `registry_write_error` | Writing to the registry during `init` / `migrate_index` failed. | Retry. Does not affect the index's data. |
| `registry_delete_error` | Removing a registry entry during `drop_index` / `cleanup_indexes` failed. | Retry. The data deletion already succeeded; only the registry entry is orphaned. |
| `meta_decode_error` | Parsing `.searchlite_meta.json` failed (likely corruption or an older format). | Clear the index and re-ingest, or open a fresh `db_name`. |
| `meta_encode_error` | Serialising the meta file during `init` / migration failed. | Inspect `err.reason`; typically indicates a logic bug — file an issue. |

---

## Migration

Errors emitted by `plan_migration` and `migrate_index`.

| `type` | When | Recovery |
| --- | --- | --- |
| `schema_mismatch` | `init` was called on an existing index with a different schema. | Call `Searchlite.plan_migration(name, schema)` then `Searchlite.migrate_index(name, schema)`, or pick a fresh `db_name`. |
| `migration_rebuild_failed` | `migrate_index` could not rebuild the index under the new schema. | The previous snapshot has been automatically restored and the old schema is still usable. Inspect `err.reason`, fix the schema, retry. |
| `migration_rollback_failed` | Rolling back after a failed rebuild also failed. **Rare and serious.** | Treat as data loss. The safest recovery is `drop_index(name)` and re-ingest from source. |
| `migration_injected_failure` | A test-only failure injection triggered. | Not emitted in production builds. |
| `schema_serialization_error` | Serialising the schema for storage failed. | Inspect `err.reason`; typically a logic bug — file an issue. |

---

## Search & writer

Errors from the reader/writer surface. These usually indicate a schema/data
mismatch or a broken invariant.

| `type` | When | Recovery |
| --- | --- | --- |
| `reader_open_error` | Could not open a reader (e.g. during `search_*`, `mget`, `multi_search`). | Usually transient; retry. Persistent errors indicate corruption — `drop_index` + re-ingest. |
| `writer_open_error` | Could not open a writer (e.g. during `add_documents`, `update_document`, `delete_*`). | Retry. |
| `index_open_error` | Could not open the on-disk index during `init` / migration. | Inspect `err.reason`; may indicate corruption. |
| `index_create_error` | Could not create a new index on first use. | Inspect `err.reason`. Check schema validity and storage availability. |
| `compact_failed` | `compact()` could not merge segments. | Retry. If it recurs, inspect `stats()` and consider `cleanup_orphaned_files`. |
| `mget_failed` | `mget()` failed after validation. | Retry. If it recurs, corruption is possible — `drop_index` + re-ingest. |
| `multi_search_failed` | `multi_search()` failed after validation. | Retry; inspect individual request shapes. |
| `update_failed` | `update_document()` patch could not be applied (unknown field, illegal path). | Fix the patch. See `err.reason` for the field path. |
| `document_not_found` | `update_document()` targeted a doc id that doesn't exist. | Ensure the doc was previously `add_document`'d and `commit`'d. |
| `vector_fields_unsupported` | A patch referenced a vector field in a build without the `vectors` feature. | Rebuild with `--features vectors` or drop the vector field from the patch. |

---

## Runtime control

Signals from the cancellation / timeout / quota machinery.

| `type` | When | Recovery |
| --- | --- | --- |
| `aborted` | The supplied `AbortSignal` was aborted (either pre-call or during a control check). | No action needed — the caller requested the cancellation. Safe to retry a new search. |
| `timeout` | A `timeoutMs` was supplied and elapsed before the operation completed a control check. | Increase `timeoutMs` or simplify the query. See [docs/wasm.md § Cancellation and timeout](wasm.md#cancellation-and-timeout-example). |
| `quota_exceeded` | IndexedDB rejected a write because storage limits were reached. | In order of preference: `db.compact()`, `Searchlite.cleanup_indexes(ageMs)`, `Searchlite.drop_index(unusedName)`, ask the user to free browser storage. |

---

## Worker & threads

Worker-client and threading failures. The worker client normalizes worker-side
errors to this shape before rejecting the caller's promise.

| `type` | When | Recovery |
| --- | --- | --- |
| `worker_error` | Generic worker runtime error (uncaught throw, unexpected message shape). | Inspect `err.reason` — frequently a JS-side bug. Main-thread fallback is always available. |
| `worker_spawn_error` | `new Worker(url, { type: "module" })` threw (CSP, missing file, classic-only environment). | Check CSP `worker-src`, asset availability, and that module workers are supported. Fall back to main-thread APIs. |
| `worker_module_import_error` | The worker script failed to `import(...)` its module dependencies. | Check bundler output paths and MIME types. Fall back to main-thread APIs. |
| `worker_client_init_error` | `SearchliteWorkerClient` could not construct (e.g. the script URL couldn't be resolved). | Inspect `err.reason`. Fall back to main-thread APIs. |
| `threads_feature_disabled` | `init_threads()` was called but the crate was built without `--features threads`. | Rebuild with `wasm-pack build ... -- --features threads`. |
| `thread_pool_init_error` | `init_threads()` tried to start the rayon pool and failed (missing COOP/COEP, `SharedArrayBuffer` unavailable, already initialised). | Serve with `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`. Call `init_threads` exactly once. |

---

## Generic

Catch-all codes. Rare in practice.

| `type` | When | Recovery |
| --- | --- | --- |
| `internal_error` | A fallback wrapper for errors that don't have a more specific code. | Inspect `err.reason`. If it recurs, file an issue with reproduction steps. |
| `serialization_error` | `serde_wasm_bindgen::to_value` failed to convert a response to a JS value. | Likely a logic bug — file an issue. |

---

## See also

- [docs/wasm.md](wasm.md) — WASM API and build guide.
- [docs/bindings.md](bindings.md) — binding semantics for FFI and WASM.
- [searchlite-wasm/README.md](../searchlite-wasm/README.md) — package-level
  quickstart.
