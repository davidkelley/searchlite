# Feature Flags

Searchlite uses Cargo feature flags to keep the default build lean. Optional
capabilities -- vector search, compression, FFI, write-key protection -- are
compiled in only when you need them.

---

## Core flags (`searchlite-core`)

These flags apply to the core library. Other crates forward them to `searchlite-core`.

| Flag | Default | What it enables |
|---|---|---|
| `vectors` | off | Vector field storage and HNSW ANN search. Adds `bytemuck` and `bincode` dependencies. See [vectors.md](vectors.md). |
| `write-key` | off | Argon2-based write-key protection for indexes. Adds `argon2`, `hmac`, `sha2`, and `subtle` dependencies. See [write-key.md](write-key.md). |
| `zstd` | off | Zstandard compression for stored fields. Reduces on-disk size at the cost of CPU during reads/writes. Adds the `zstd` dependency. |
| `gpu` | off | Stub hooks for GPU-accelerated reranking. Currently a no-op placeholder for future development. |
| `ffi` | off | Marks types and functions for C FFI export. Used by the `searchlite-ffi` crate. |
| `browser` | off | Adjustments for browser/WASM environments. Used by `searchlite-wasm`. |
| `tokio-runtime` | off | Sync→async bridge that lets blocking callers drive `aws-sdk-s3` (and any other async `BlobStore` impl) via `runtime::block_on_blob`. Used by `searchlite-s3`, which enables it transitively -- you don't need to set it manually unless you're calling the bridge directly. See [s3.md](s3.md). |

### Usage

```toml
# Just the core library (default, no optional features)
searchlite-core = "0.5"

# With vector search
searchlite-core = { version = "0.5", features = ["vectors"] }

# With vector search and compression
searchlite-core = { version = "0.5", features = ["vectors", "zstd"] }

# With write-key protection
searchlite-core = { version = "0.5", features = ["write-key"] }

# Everything
searchlite-core = { version = "0.5", features = ["vectors", "write-key", "zstd"] }
```

---

## CLI flags (`searchlite-cli`)

The CLI binary can be built with optional capabilities:

| Flag | Default | What it enables |
|---|---|---|
| `s3` | **on** | `searchlite sync` and `s3://bucket/prefix` paths on the `search` / `inspect` `<INDEX>` arg (plus `--index name:s3://bucket/prefix` on `searchlite http`). Pulls in `searchlite-s3` (and `aws-sdk-s3`). Disable with `--no-default-features` for size-sensitive packagers. See [s3.md](s3.md). |
| `vectors` | off | Vector field support in CLI commands |
| `zstd` | off | Zstandard compression for stored fields |
| `gpu` | off | GPU reranker stubs |
| `ffi` | off | FFI exports on the CLI binary |

```bash
# Default build — includes the S3 backend.
cargo build -p searchlite-cli

# Slim build without S3 (for size-sensitive packagers).
cargo build -p searchlite-cli --no-default-features

# Build CLI with vector support on top of the default S3 build.
cargo build -p searchlite-cli --features vectors
```

The CLI always includes `write-key` support (it's a dependency of the core crate
as used by the CLI).

---

## HTTP server flags (`searchlite-http`)

| Flag | Default | What it enables |
|---|---|---|
| `s3` | off (on via the CLI binary) | `--index name:s3://bucket/prefix` mount form for read-only S3-backed indexes. The CLI binary's `s3` feature turns this on transitively. |
| `vectors` | off | Vector field support in HTTP endpoints |
| `zstd` | off | Zstandard compression |
| `gpu` | off | GPU reranker stubs |

---

## FFI flags (`searchlite-ffi`)

| Flag | What it enables |
|---|---|
| `ffi` | **Required.** Builds the C shared library and header. |
| `vectors` | Vector field support in FFI bindings |
| `zstd` | Zstandard compression |
| `gpu` | GPU reranker stubs |

```bash
cargo build -p searchlite-ffi --release --features "ffi,vectors,zstd"
```

The FFI crate always includes `write-key` support.

---

## WASM flags (`searchlite-wasm`)

| Flag | What it enables |
|---|---|
| `vectors` | Vector field support in WASM bindings |
| `threads` | Multi-threaded WASM via `SharedArrayBuffer`. Requires COOP/COEP headers. See [wasm.md](wasm.md). |

```bash
# Standard build
wasm-pack build searchlite-wasm --target web --release

# With vector support
wasm-pack build searchlite-wasm --target web --release -- --features vectors

# With threading
wasm-pack build searchlite-wasm --target web --release -- --features threads
```

---

## Flag interactions

- `vectors` and `zstd` are independent -- enable either or both.
- `write-key` is independent of all other flags.
- `ffi` is only meaningful on the `searchlite-ffi` crate.
- `threads` is only meaningful on the `searchlite-wasm` crate and requires the nightly
  Rust toolchain pinned in `searchlite-wasm/rust-toolchain.toml`.
- `browser` is used internally by `searchlite-wasm` and should not be set manually.
- `gpu` is a placeholder -- it compiles but does not currently accelerate anything.
- `tokio-runtime` is only meaningful when an async `BlobStore` is being driven from sync code; `searchlite-s3` enables it transitively, so most users never need to flip it manually.
- `s3` on the CLI binary turns on the same feature on `searchlite-http` automatically — there is no need to set both. The CLI's `--no-default-features` opts out of both at once.

No feature flags conflict with each other. Any combination is valid.
