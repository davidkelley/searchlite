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

---

## A complete C example

This is a minimal but complete C program that opens an index, adds two
documents, commits, and runs a search. Save it as `demo.c`.

> The example below calls `searchlite_index_open(..., create_if_missing=true)`
> which creates a new index with the built-in default schema
> (`Schema::default_text_body()` -- a single text `body` field). The FFI surface
> does not let you *customise* the schema, so if you need keyword fields,
> numeric fields, nested fields, or analyzer configuration, initialise the
> index first with the CLI (`searchlite init /tmp/ffi_idx schema.json`) or the
> HTTP `/init` endpoint, then open it from C with `create_if_missing=false`.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "searchlite.h"

int main(void) {
  // 1. Open (or create) the index. Pass `true` for create_if_missing to auto-
  //    initialize the directory. If you need a custom schema, initialize via
  //    the CLI (`searchlite init ... schema.json`) beforehand.
  IndexHandle* idx = searchlite_index_open("/tmp/ffi_idx", true);
  if (!idx) {
    fprintf(stderr, "failed to open index\n");
    return 1;
  }

  // 2. Queue two documents. The bodies are JSON objects keyed by field names.
  const char* doc1 = "{\"_id\":\"1\",\"body\":\"Rust is fast\"}";
  const char* doc2 = "{\"_id\":\"2\",\"body\":\"SQLite is embedded\"}";
  if (searchlite_add_json(idx, doc1, strlen(doc1)) < 0 ||
      searchlite_add_json(idx, doc2, strlen(doc2)) < 0) {
    fprintf(stderr, "add failed\n");
    searchlite_index_close(idx);
    return 1;
  }

  // 3. Commit to make the docs searchable.
  if (searchlite_commit(idx) < 0) {
    fprintf(stderr, "commit failed\n");
    searchlite_index_close(idx);
    return 1;
  }

  // 4. Run a search. We allocate a buffer for the JSON response; if the
  //    buffer is too small, the return value exceeds `buf_cap` and reports
  //    the required size so we can retry.
  const char* request =
    "{\"query\":\"rust\",\"limit\":5,\"return_stored\":true}";
  char buf[64 * 1024] = {0};
  size_t written = searchlite_search_request(
      idx, request, strlen(request), buf, sizeof(buf));

  if (written == 0) {
    fprintf(stderr, "search failed\n");
  } else if (written > sizeof(buf)) {
    // Buffer was too small. `written` is the required size including the
    // NUL terminator -- allocate and retry.
    char* big = malloc(written);
    size_t retry = searchlite_search_request(
        idx, request, strlen(request), big, written);
    if (retry > 0 && retry <= written) {
      printf("%s\n", big);
    } else {
      fprintf(stderr, "search retry failed\n");
    }
    free(big);
  } else {
    printf("%s\n", buf);
  }

  // 5. Always close the handle.
  searchlite_index_close(idx);
  return 0;
}
```

Compile and run on Linux:

```bash
# Build the shared library + header
cargo build -p searchlite-ffi --release --features ffi

# Compile the C demo against it
cc demo.c \
   -I searchlite-ffi \
   -L target/release -lsearchlite_ffi \
   -o demo
LD_LIBRARY_PATH=target/release ./demo
```

(macOS users swap `LD_LIBRARY_PATH` for `DYLD_LIBRARY_PATH` and link against
`libsearchlite_ffi.dylib` -- the compile line is otherwise identical.)

### Working with a write-key protected index

If the index was created with a write key, every mutating call needs the
matching `_with_write_key` variant:

```c
IndexHandle* idx =
  searchlite_index_open_with_write_key("/tmp/ffi_idx", false, "my-secret");

int rc = searchlite_add_json_with_write_key(idx, doc1, strlen(doc1), "my-secret");
if (rc == SEARCHLITE_ERR_WRITE_KEY) {
  fprintf(stderr, "missing or incorrect write key\n");
}
```

Read calls (`searchlite_search`, `searchlite_search_request`) never take a
write key. See [write-key.md](write-key.md) for how to create a protected index.

### Handling panics across the boundary

A bug in Searchlite could theoretically panic inside a mutating call. If that
happens, the FFI catches the panic and returns `SEARCHLITE_ERR_PANIC` (`-100`).
Treat it as a hard error: close the handle, reopen it, and decide whether to
retry. Never keep using a handle that returned `-100` from an `add`/`commit`
call without reopening -- on-disk state may be mid-write.
