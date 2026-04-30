# In-Memory Indexes

By default, Searchlite writes segments, manifests, and WAL entries to disk. In-memory
mode keeps everything in RAM instead -- same API, same query behavior, zero filesystem
I/O.

---

## Why it exists

Searchlite is an embedded library, and embedded libraries get tested *inside* the
applications that use them. That means your test suite creates, populates, and queries
indexes hundreds or thousands of times. Hitting the filesystem for every test is slow,
flaky (temp directory cleanup, CI permission issues), and unnecessary when all you care
about is "does my query return the right results?"

In-memory mode solves this. A test that takes 200ms against disk runs in under 5ms
in memory. Multiply that by a hundred tests and the difference is the gap between a
test suite developers actually run and one they skip.

---

## When to use it

| Scenario | Use in-memory? |
|---|---|
| **Unit and integration tests** | Yes -- fast, isolated, no cleanup needed |
| **Benchmarks isolating engine perf** | Yes -- removes filesystem noise from measurements |
| **Short-lived CLI tools** | Sometimes -- if you're processing data and don't need persistence |
| **Ephemeral serverless functions** | Sometimes -- if the index is rebuilt on each invocation |
| **Application with persistent data** | No -- use `StorageType::Filesystem` |
| **Anything that must survive a restart** | No -- in-memory indexes are gone when the process exits |

---

## Usage

The only change is setting `storage: StorageType::InMemory` in your `IndexOptions`.
Everything else -- schema definition, document writes, queries, aggregations,
highlighting -- works identically.

### Rust API

```rust
use searchlite_core::api::{
    builder::IndexBuilder,
    types::{IndexOptions, Schema, SearchRequest, StorageType},
};
use std::path::{Path, PathBuf};

let schema = Schema::default_text_body();

let opts = IndexOptions {
    path: PathBuf::from("unused"), // path is ignored for in-memory storage
    create_if_missing: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::InMemory,
    ..Default::default()
};

let index = IndexBuilder::create(Path::new("unused"), schema, opts)?;

// From here on, the API is identical to filesystem indexes
let mut writer = index.writer()?;
writer.add_document(&doc)?;
writer.commit()?;

let reader = index.reader()?;
let results = reader.search(&SearchRequest::new("my query").with_limit(5))?;
```

The `path` field is required by `IndexOptions` but ignored when storage is in-memory.
Use any placeholder value.

### In tests

A common pattern is a helper function that creates a fresh in-memory index for each test:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn test_index(schema: Schema) -> Index {
        let opts = IndexOptions {
            path: PathBuf::from("test"),
            create_if_missing: true,
            bm25_k1: 0.9,
            bm25_b: 0.4,
            storage: StorageType::InMemory,
            ..Default::default()
        };
        IndexBuilder::create(Path::new("test"), schema, opts).unwrap()
    }

    #[test]
    fn search_finds_matching_documents() {
        let index = test_index(Schema::default_text_body());
        let mut writer = index.writer().unwrap();
        writer.add_document(&Document {
            fields: [
                ("_id".into(), json!("1")),
                ("body".into(), json!("Rust is a systems programming language")),
            ].into_iter().collect(),
        }).unwrap();
        writer.commit().unwrap();

        let reader = index.reader().unwrap();
        let results = reader.search(
            &SearchRequest::new("rust").with_limit(10)
        ).unwrap();

        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, "1");
    }
}
```

No temp directories to create or clean up. No filesystem permissions to worry about.
Each test gets a completely isolated index that vanishes when the test ends.

---

## What's the same

Everything. In-memory indexes support the full feature set:

- All query types (bool, phrase, fuzzy, prefix, wildcard, function scores, etc.)
- Filters, aggregations, highlighting, collapsing
- WAL-based writes with checkpoint/rollback
- Compaction and segment merging
- Multi-search and mget
- Vector search (with the `vectors` feature flag)

The search results are identical for the same data and queries regardless of storage
backend.

## What's different

| | Filesystem | In-Memory |
|---|---|---|
| **Persistence** | Survives process restart | Gone when process exits |
| **Crash safety** | WAL replay recovers uncommitted data | N/A (no crash recovery needed) |
| **fsync** | Segments and manifest are fsync'd | Skipped (no I/O to sync) |
| **Memory usage** | OS page cache manages memory pressure | Entire index lives in process memory |
| **Capacity** | Limited by disk | Limited by available RAM |
| **Performance** | Filesystem I/O on writes; mmap'd reads | Pure memory operations |

For small-to-medium indexes (up to a few hundred thousand documents), the performance
difference on reads is minimal because the OS page cache keeps hot segments in memory
anyway. The biggest win is on writes -- no fsync overhead means commits are
significantly faster.

For large indexes, be mindful of memory. A 100K-document index with stored fields
typically uses 50-200 MB of RAM depending on document size. This is fine for tests
but would consume real application memory in production.
