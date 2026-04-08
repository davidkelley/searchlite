# C FFI

The FFI crate exposes Searchlite as a C-compatible shared library (`.dylib` / `.so` / `.dll`)
with a C header. Use it to embed Searchlite in applications written in any language
that can call C functions: Python (via ctypes/cffi), Swift, Go (via cgo), Ruby, Node.js
(via ffi-napi), and more.

**When you'd use the C FFI:**
- Adding local search to a mobile app (Swift/Kotlin calling the C library)
- Embedding search in a Python data pipeline without running a server
- Building language-specific bindings for your team
- Adding search to an existing C/C++ application

---

## Building the library

```bash
# Release build
cargo build -p searchlite-ffi --release --features ffi

# With optional features
cargo build -p searchlite-ffi --release --features "ffi,vectors,zstd"
```

This produces:
- **macOS:** `target/release/libsearchlite_ffi.dylib`
- **Linux:** `target/release/libsearchlite_ffi.so`
- **Windows:** `target/release/searchlite_ffi.dll`
- **C header:** `searchlite-ffi/searchlite.h`

---

## Feature flags

| Flag | Purpose |
|---|---|
| `ffi` | **Required.** Builds the C FFI surface. |
| `vectors` | Enable vector field storage and ANN search. |
| `zstd` | Compress stored fields with Zstandard. |
| `gpu` | Stub GPU reranker hooks. |

---

## Usage pattern

The FFI follows the same lifecycle as the Rust API:

1. **Create or open** an index with a schema
2. **Add documents** via the ingest functions (documents are buffered)
3. **Commit** with `searchlite_commit` to flush writes to a segment
4. **Search** with `searchlite_search_request` for full query support
5. **Compact** periodically to merge segments

Key notes:
- Searches default to `return_stored: false` -- set it explicitly when you need field values.
- All strings are C-compatible null-terminated UTF-8.
- See [bindings.md](bindings.md) for a quick reference of binding behaviors and memory management.
