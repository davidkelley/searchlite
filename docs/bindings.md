# Binding lifecycle (FFI + WASM)

## FFI
- `searchlite_add_json` and `searchlite_add_json_batch` queue documents but do not commit. Call `searchlite_commit` after a batch to make changes visible and durable.
- `searchlite_add_json_batch` accepts a JSON object or array of objects; use it to avoid repeated calls when loading bulk data.
- Use `searchlite_search_request` to send a full `SearchRequest` JSON (filters, sort, aggregations, suggest, etc.). `return_stored` defaults to `false`; set it to `true` when you need stored fields in hits.
- `searchlite_search` remains a convenience for simple query-string searches; it also defaults `return_stored` to `false`.

## WASM
- `add_document` / `add_documents` queue writes; `commit()` persists them and makes them searchable. `flush_storage()` only drains pending storage writes if you need it.
- `search(query, limit, returnStored?)` takes an optional third argument; omit it for the default (`false`) or pass `true` to fetch stored fields.
- `search_request` accepts a JSON string; `search_request_value` takes a JS object. Both support the full `SearchRequest` surface with the same `return_stored` default of `false`.
- Reusing a `db_name` reopens an existing index; schema mismatches still return an error.
