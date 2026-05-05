# searchlite-s3

S3-compatible storage for Searchlite. Bake an index locally, sync it to a
bucket, and any number of read-only processes can serve queries straight
out of S3, R2, or MinIO -- no local disk, no replication daemon.

- **Stateless readers**: `open_index_read_only` against a prefix. A
  64 MiB byte-weighted RAM cache fronts bounded postings/docstore range
  reads; whole-file segment artifacts are fetched at open and re-verified
  on each fresh `Index::reader()` under the default `Strict` checksum
  policy. Drop to `ChecksumPolicy::TrustManifest` to skip whole-file
  verification entirely.
- **Atomic publish**: `sync_to_s3` uploads segment artifacts first and
  publishes `MANIFEST.json` last as the visibility fence -- partial syncs
  never leave a servable manifest.
- **Multi-provider**: AWS S3, Cloudflare R2, and MinIO via `aws-sdk-s3`,
  with config presets per backend.
- **Read-only enforcement**: every mutator on an S3-backed `Index`
  (`writer`, `compact`, `merge_segments`) errors at the entry point.
- **Fail-closed sync**: refuses to upload partially-baked indexes
  (pending manifests, non-empty WAL, staging files, legacy v1 manifests).

> **Status:** Experimental. The API is functional but may change before 1.0.

---

## Install

```toml
[dependencies]
searchlite-core = "0.8"
searchlite-s3 = "0.2"
tokio = { version = "1", features = ["full"] }
anyhow = "1"
serde_json = "1"
```

`searchlite-s3` brings `aws-sdk-s3` and the `tokio-runtime` feature on
`searchlite-core` along with it. No additional features to enable.
`anyhow` and `serde_json` are used by the quickstart below.

---

## Quickstart

Bake locally with the regular `searchlite-core` API, then sync and serve:

```rust
use searchlite_core::api::{
    builder::IndexBuilder,
    types::{Document, IndexOptions, Schema, SearchRequest},
};
use searchlite_s3::{open_index_read_only, sync_to_s3, S3Config, S3Credentials};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Bake locally.
    let local_root = PathBuf::from("/tmp/products");
    let opts = IndexOptions {
        path: local_root.clone(),
        create_if_missing: true,
        ..Default::default()
    };
    let index = IndexBuilder::create(&local_root, Schema::default_text_body(), opts)?;
    let mut writer = index.writer()?;
    writer.add_document(&Document {
        fields: [
            ("_id".into(), serde_json::json!("doc-1")),
            ("body".into(), serde_json::json!("Searchlite is a fast embedded search engine.")),
        ].into_iter().collect(),
    })?;
    writer.commit()?;
    drop(index);

    // 2. Publish to S3.
    let s3_config = S3Config {
        region: "us-east-1".into(),
        bucket: "my-search-indexes".into(),
        prefix: Some("products/v1".into()),
        credentials: S3Credentials::LoadFromEnv,
        ..S3Config::aws_default()
    };
    let report = sync_to_s3(&local_root, s3_config.clone()).await?;
    println!("uploaded {} files / {} bytes", report.files, report.bytes);

    // 3. Open read-only from any reader.
    let remote = open_index_read_only(s3_config).await?;
    let reader = remote.reader()?;
    let hits = reader.search(&SearchRequest::new("search engine").with_limit(5))?;
    for hit in hits.hits {
        println!("{} (score: {:.2})", hit.doc_id, hit.score);
    }
    Ok(())
}
```

---

## Provider presets

| Provider | Constructor | Notes |
| --- | --- | --- |
| **AWS S3** | `S3Config::aws_default()` | Virtual-hosted addressing, conditional PUTs on, default credential chain. |
| **Cloudflare R2** | `S3Config::r2_default()` | Conditional PUTs default OFF -- opt in once your account/bucket supports them. |
| **MinIO / LocalStack** | `S3Config { force_path_style: true, .. }` | Path-style addressing; static credentials. |

---

## Platform support

`searchlite-s3` is non-WASM. `aws-sdk-s3` requires the Tokio reactor and
native TLS; neither compiles to `wasm32`. Browser and worker consumers
stay on the IndexedDB path inside [`searchlite-wasm`](../searchlite-wasm/README.md).

If you're driving the async helpers from sync code, wrap them in a
`tokio::runtime::Runtime`:

```rust
let rt = tokio::runtime::Runtime::new()?;
let index = rt.block_on(open_index_read_only(s3_config))?;
```

The bridge is also safe to call from inside an active Tokio runtime of
either flavor: multi-thread runtimes use `block_in_place` to park the
worker, and current-thread runtimes route the future to a
`std::thread::scope`-spawned worker on a global multi-thread bridge
runtime. See [`searchlite_core::runtime`] for the full contract.

---

## Documentation

- Full guide: [docs/s3.md](../docs/s3.md) -- bake-and-sync workflow,
  provider config, key shape, read-only semantics, limits.
- Feature flags: [docs/feature-flags.md](../docs/feature-flags.md) --
  the `tokio-runtime` core feature and how it relates to this crate.
- Rust API surface: [docs/rust-api.md](../docs/rust-api.md) -- everything
  on the read side works the same once you have an `Index`.
