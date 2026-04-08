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

| Flag | What it enables |
|---|---|
| `vectors` | Vector field support in CLI commands |
| `zstd` | Zstandard compression for stored fields |
| `gpu` | GPU reranker stubs |
| `ffi` | FFI exports on the CLI binary |

```bash
# Build CLI with vector support
cargo build -p searchlite-cli --features vectors

# Build CLI with all optional features
cargo build -p searchlite-cli --features "vectors,zstd"
```

The CLI always includes `write-key` support (it's a dependency of the core crate
as used by the CLI).

---

## HTTP server flags (`searchlite-http`)

| Flag | What it enables |
|---|---|
| `vectors` | Vector field support in HTTP endpoints |
| `zstd` | Zstandard compression |
| `gpu` | GPU reranker stubs |

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

No feature flags conflict with each other. Any combination is valid.
