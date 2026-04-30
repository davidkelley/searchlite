# Rust API Guide

This guide covers the full Searchlite Rust API for developers embedding search
directly in their applications. It goes beyond the README quick-start to cover
the complete write lifecycle, search features, and index management.

```toml
[dependencies]
searchlite-core = "0.5"
serde_json = "1"
```

---

## Creating an index

Every index starts with a schema and options. The schema defines your fields; the
options control storage, BM25 tuning, and feature flags.

```rust
use searchlite_core::api::{
    builder::IndexBuilder,
    types::{IndexOptions, KeywordField, NumericField, Schema, StorageType},
};
use std::path::PathBuf;

let mut schema = Schema::default_text_body(); // "body" text field with default analyzer
schema.keyword_fields.push(KeywordField {
    name: "category".into(), stored: true, indexed: true, fast: true,
});
schema.numeric_fields.push(NumericField {
    name: "price".into(), i64: true, fast: true, stored: true,
});

let path = PathBuf::from("/tmp/products");
let opts = IndexOptions {
    path: path.clone(),
    create_if_missing: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    ..Default::default()
};

let index = IndexBuilder::create(&path, schema, opts)?;
```

To reopen an existing index later:

```rust
let index = Index::open(IndexOptions {
    path: PathBuf::from("/tmp/products"),
    create_if_missing: false,
    ..// same options as above
})?;
```

### In-memory indexes

For tests, benchmarks, or ephemeral workloads, use in-memory storage. The API is
identical -- the only difference is that nothing is written to disk:

```rust
let opts = IndexOptions {
    path: PathBuf::from("unused"),  // path is ignored for in-memory
    storage: StorageType::InMemory,
    create_if_missing: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    ..Default::default()
};

let index = IndexBuilder::create(Path::new("unused"), schema, opts)?;
// Use exactly like a filesystem index -- just no persistence across restarts
```

---

## Writing documents

The `IndexWriter` handles all write operations. Documents are buffered in a WAL
(write-ahead log) until you call `commit()`, which flushes them into an immutable
segment and makes them visible to readers.

```rust
let mut writer = index.writer()?;
```

### Adding documents

```rust
use searchlite_core::api::types::Document;

let doc = Document {
    fields: [
        ("_id".into(), serde_json::json!("product-1")),
        ("body".into(), serde_json::json!("Wireless Bluetooth headphones with noise cancelling")),
        ("category".into(), serde_json::json!("electronics")),
        ("price".into(), serde_json::json!(4999)),
    ].into_iter().collect(),
};

writer.add_document(&doc)?;
```

Every document must include the `_id` field (or whatever `searchlite:docIdField` your
schema specifies). Adding a document with an existing ID is an upsert -- the old version is
replaced on commit.

### Batch adds (atomic)

`add_documents_batch` adds all documents atomically -- if any document fails validation,
none are added:

```rust
let docs = vec![product_a, product_b, product_c];
let count = writer.add_documents_batch(&docs)?;
// count == 3; all or nothing
```

This is faster than individual `add_document` calls for bulk loading.

### Deleting documents

```rust
// Single delete
writer.delete_document("product-old")?;

// Batch delete
writer.delete_documents(&[
    "product-discontinued-1".into(),
    "product-discontinued-2".into(),
])?;
```

Deletes are applied on commit. Deleted documents are hidden from search immediately
after commit and physically removed on the next compaction.

### Partial updates (patching)

Update specific fields without re-indexing the entire document:

```rust
use std::collections::BTreeMap;

let mut set = BTreeMap::new();
set.insert("price".to_string(), serde_json::json!(3999));  // new price

let unset = vec!["sale_label".to_string()];  // remove this field

writer.apply_patch("product-1", &set, &unset)?;
```

The patch reads the existing document, applies the changes, and queues the result
as an upsert. Vector fields cannot be patched -- you need to re-add the full document.

### Committing

```rust
writer.commit()?;
// All buffered adds, deletes, and patches are now visible to readers
```

**Commit with automatic merging:** If your application makes many small commits,
segments can accumulate. `commit_with_merge` runs a tiered merge pass after committing
to keep the segment count healthy:

```rust
writer.commit_with_merge(true)?;
```

### Checkpoint and rollback

For complex write workflows where you need to roll back on failure:

```rust
// Save current position
let checkpoint = writer.checkpoint()?;

// Try a complex operation
match perform_complex_import(&mut writer) {
    Ok(_) => writer.commit()?,
    Err(_) => {
        // Undo everything since the checkpoint
        writer.rollback_to(checkpoint)?;
    }
}
```

To discard **all** uncommitted changes:

```rust
writer.rollback()?;
```

---

## Searching

The `IndexReader` provides read-only access to the committed state of the index.
Multiple readers can search concurrently.

```rust
let reader = index.reader()?;
```

### Basic search with the builder API

The `SearchRequest::new()` constructor and `with_*` builder methods make common
searches concise:

```rust
use searchlite_core::api::types::SearchRequest;
use searchlite_core::api::Filter;

let results = reader.search(
    &SearchRequest::new("wireless headphones")
        .with_limit(10)
        .with_filter(Filter::KeywordEq {
            field: "category".into(),
            value: "electronics".into(),
        })
        .with_return_stored(true)
        .with_highlight_field("body"),
)?;

for hit in &results.hits {
    println!("{} (score: {:.2})", hit.doc_id, hit.score);
    if let Some(snippet) = &hit.snippet {
        println!("  {}", snippet);
    }
}
```

### Available builder methods

The builder only covers the most common fields; for anything else, construct
the `SearchRequest` directly and override fields. This is how you set the
tuning knobs from [queries.md#request-level-tuning-knobs](queries.md#request-level-tuning-knobs)
(`track_total_hits`, `bmw_block_size`, `candidate_size`, `return_hits`,
`execution`) as well as the debugging flags from
[queries.md#debugging-aids](queries.md#debugging-aids) (`explain`, `profile`).

| Method | Purpose |
|---|---|
| `SearchRequest::new(query)` | Create with a query (string or `QueryNode`) and sensible defaults |
| `.with_limit(n)` | Max hits to return (default: 10) |
| `.with_from(n)` | Skip the first N results (offset pagination) |
| `.with_filter(filter)` | Add a post-query filter |
| `.with_return_stored(bool)` | Include stored field values in results |
| `.with_highlight_field(field)` | Single-field highlighting (returns `snippet`) |
| `.with_highlight(request)` | Multi-field highlighting (returns `highlights` map) |
| `.with_aggs(map)` | Add aggregations |
| `.with_fuzzy(options)` | Enable typo-tolerant matching |
| `.with_sort(specs)` | Custom sort order |

For fields without a dedicated builder method, set them after construction:

```rust
let mut req = SearchRequest::new("rust search")
    .with_limit(20)
    .with_return_stored(true);
req.explain = true;                                    // per-hit score breakdown
req.profile = true;                                    // execution profile
req.track_total_hits = Some(true);                     // exact total
req.execution = ExecutionStrategy::Bmw;                // BMW pruning
req.collapse = Some(CollapseRequest { /* ... */ });    // field collapsing
req.rescore  = Some(RescoreRequest  { /* ... */ });    // two-pass re-ranking
```

### Understanding search results

```rust
pub struct SearchResult {
    pub total_hits_estimate: u64,  // approximate total matches
    pub total_groups: Option<u64>, // distinct groups (when using collapse)
    pub hits: Vec<Hit>,
    pub next_cursor: Option<String>,        // for cursor pagination
    pub next_search_after: Option<Vec<_>>,  // for search_after pagination
    pub aggregations: BTreeMap<String, AggregationResponse>,
    pub suggest: BTreeMap<String, SuggestResult>,
    pub profile: Option<ProfileResult>,     // when profile: true
}

pub struct Hit {
    pub doc_id: String,
    pub score: f32,                           // BM25 relevance score
    pub vector_score: Option<f32>,            // set during hybrid vector search
    pub sort_key: Option<Vec<serde_json::Value>>,  // values of sort fields
    pub fields: Option<serde_json::Value>,    // stored fields (when return_stored: true)
    pub snippet: Option<String>,              // single-field highlight
    pub highlights: Option<BTreeMap<String, Vec<String>>>, // multi-field highlights
    pub explanation: Option<HitExplanation>,  // when explain: true
    pub inner_hits: Option<Vec<Hit>>,         // when using collapse with inner_hits
}
```

### Pagination: cursor, search_after, offset

Searchlite supports three mutually-exclusive pagination modes. Pick the one
that matches how your UI lets users move through results. The first mode
(offset) uses `from`/`limit` directly; the other two use response tokens
exposed on `SearchResult` (`next_cursor` / `next_search_after`).

```rust
// Offset: classic "page 1, 2, 3" navigation (from + limit <= 1000)
let page2 = reader.search(
    &SearchRequest::new("rust").with_from(10).with_limit(10),
)?;

// Cursor: "load more" / infinite scroll -- opaque token
let first = reader.search(&SearchRequest::new("rust").with_limit(10))?;
if let Some(tok) = first.next_cursor.clone() {
    let mut req = SearchRequest::new("rust").with_limit(10);
    req.cursor = Some(tok);
    let next = reader.search(&req)?;
}

// search_after: unbounded, needs a stable sort on at least one field
let mut req = SearchRequest::new("rust")
    .with_limit(10)
    .with_sort(vec![
        SortSpec { field: "year".into(), order: Some(SortOrder::Desc) },
        SortSpec { field: "_id".into(),  order: Some(SortOrder::Asc)  },
    ]);
let first = reader.search(&req)?;
if let Some(after) = first.next_search_after.clone() {
    req.search_after = Some(after);
    let next = reader.search(&req)?;
}
```

Offset pagination is capped at `from + limit <= 1000`. For anything beyond
that, use `search_after` (unbounded, requires a stable sort) or `cursor`
(opaque hex token bounded to ~50K advance; works with or without a sort
clause but can become stale across commits).

### Fetching documents by ID

When you know the document IDs (e.g., from a cache or external system):

```rust
let docs = reader.mget(
    &["product-1".into(), "product-2".into(), "product-42".into()],
    true, // return_stored
)?;

for doc in &docs {
    if doc.found {
        println!("{}: {:?}", doc.doc_id, doc._source);
    } else {
        println!("{}: not found", doc.doc_id);
    }
}
```

Supports up to 1,024 IDs per call.

### Multi-search

Execute multiple search requests in a single call. Useful when a page needs several
independent queries (e.g., main results + related articles + trending topics):

```rust
let results = reader.multi_search(&[
    SearchRequest::new("wireless headphones").with_limit(10),
    SearchRequest::new("best sellers").with_limit(5),
    SearchRequest::new("new arrivals")
        .with_limit(5)
        .with_sort(vec![SortSpec { field: "created_at".into(), order: Some(SortOrder::Desc) }]),
])?;

let main_results = &results[0];
let best_sellers = &results[1];
let new_arrivals = &results[2];
```

Requests are executed sequentially. The advantage over separate `search()` calls is a
single reader snapshot -- all results reflect the same index state.

### Debugging with explain and profile

When relevance doesn't look right, enable `explain` to see score breakdowns:

```rust
let mut req = SearchRequest::new("rust search");
req.explain = true;
req.profile = true;

let results = reader.search(&req)?;

for hit in &results.hits {
    if let Some(explanation) = &hit.explanation {
        println!("{}: {:#?}", hit.doc_id, explanation);
    }
}

if let Some(profile) = &results.profile {
    println!("Scored {} docs, examined {} candidates",
        profile.execution.scored_docs,
        profile.execution.candidates_examined);
}
```

---

## Index management

### Compaction

Over time, commits create many small segments. Compaction merges them all into one,
improving search performance and reclaiming space from deleted documents:

```rust
index.compact()?;
```

**When to compact:**
- After a bulk import (many commits created many segments)
- Periodically in a maintenance window
- When `inspect` shows many small segments

### Selective merge

For more control, merge specific segments instead of compacting everything:

```rust
// Get current segment IDs from the manifest
let manifest = index.manifest();
let small_segments: Vec<String> = manifest.segments
    .iter()
    .filter(|s| s.doc_count < 10_000)
    .map(|s| s.id.clone())
    .collect();

if small_segments.len() > 1 {
    index.merge_segments(&small_segments, None)?;
}
```

### Inspecting the index

```rust
let manifest = index.manifest();
println!("Segments: {}", manifest.segments.len());
for seg in &manifest.segments {
    println!("  {} — {} docs ({} deleted)",
        seg.id, seg.doc_count, seg.deleted_docs.len());
}
```

---

## Concurrency model

Searchlite uses a **single-writer, multi-reader** model:

- **One writer at a time.** `index.writer()` acquires an exclusive lock. A second call
  blocks until the first writer is dropped or committed.
- **Many readers concurrently.** `index.reader()` creates a lightweight snapshot that
  can be used from any thread. Readers see the index state as of their creation and are
  not affected by ongoing writes.
- **Readers are cheap.** Create a new reader after each commit to see the latest data,
  or reuse a reader for consistent point-in-time snapshots.

```rust
use std::thread;

let index = /* ... */;

// Writer thread
let idx = index.clone(); // Index is Arc-wrapped internally
thread::spawn(move || {
    let mut writer = idx.writer().unwrap();
    // ... add documents ...
    writer.commit().unwrap();
});

// Reader threads (can run concurrently with writer)
for _ in 0..4 {
    let idx = index.clone();
    thread::spawn(move || {
        let reader = idx.reader().unwrap();
        let results = reader.search(&SearchRequest::new("query")).unwrap();
        // ... use results ...
    });
}
```

---

## Error handling

All API methods return `anyhow::Result<T>`. Common error scenarios:

| Situation | Error |
|---|---|
| Index directory doesn't exist | `"index not found"` or `"manifest not found"` |
| Document missing `_id` field | `"document is missing doc_id field '_id'"` |
| Unknown field in document | `"unknown field <name>"` |
| Schema mismatch on open | `"schema mismatch"` |
| Write key required but not provided | `"this index requires a write key"` |
| Patch on nonexistent document | `PatchError::DocumentNotFound` |
| Patch on vector field | `PatchError::VectorFieldsUnsupported` |
| Aggregation on non-fast field | `AggregationError::MissingFastField` |
| mget with > 1024 IDs | `"mget ids length N exceeds max supported 1024"` |
| from + limit > 1000 | `"from + limit exceeds max"` |
