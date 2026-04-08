# Benchmarks

How fast is Searchlite? These benchmarks measure the operations that matter most:
how quickly you can index documents, how fast searches return, how aggregations
perform at scale, and how long maintenance operations take.

The numbers below give a practical ballpark. Your results will vary with hardware,
data shape, and query complexity -- but the relative performance across operations
should be consistent.

**Run them on your own machine:**

```bash
cargo bench -p searchlite-core --bench scale
```

## Environment

| | |
|---|---|
| **CPU** | Apple M3 Max |
| **RAM** | 36 GB |
| **OS** | macOS (Darwin 25.3.0, arm64) |
| **Rust** | 1.92.0 |
| **Date** | 2026-04-08 |

## Results

All benchmarks use **in-memory storage** (`StorageType::InMemory`) to isolate
engine performance from filesystem I/O. Execution strategy is **WAND** (exact
top-K pruning). Times are Criterion medians with the reported confidence
interval.

### Indexing throughput

| Benchmark | Documents | Segments | Median | Range |
|---|---|---|---|---|
| index_10k | 10,000 | 1 | **205.95 ms** | [204.62 ms, 207.21 ms] |
| index_100k | 100,000 | 1 | **4.21 s** | [4.00 s, 4.58 s] |

Indexing includes document parsing, analysis (default tokenizer), inverted index
construction, fast field encoding, and segment flush. The 100K benchmark
produces roughly **23,700 docs/sec** sustained throughput.

### Search latency

All search benchmarks query a **single-segment 100K-document index**.

| Benchmark | Query | Top-K | Median | Range |
|---|---|---|---|---|
| search_single_term | `"search"` | 10 | **4.26 ms** | [4.22 ms, 4.31 ms] |
| search_bool_and | `"search engine"` | 10 | **6.25 ms** | [6.18 ms, 6.33 ms] |
| search_filtered | `"search"` + keyword filter | 10 | **4.42 ms** | [4.38 ms, 4.47 ms] |
| search_top100 | `"search"` | 100 | **4.35 ms** | [4.33 ms, 4.38 ms] |

Filter overhead is minimal (~0.16 ms) thanks to fast-field evaluation. Retrieving
100 results instead of 10 adds negligible cost because WAND pruning dominates
the work.

### Aggregation latency

Aggregations run over **all matched documents** (not just top-K hits) on a 100K-document index.

| Benchmark | Type | Field | Median | Range |
|---|---|---|---|---|
| agg_terms | Terms (size=50) | category | **8.29 ms** | [8.21 ms, 8.38 ms] |
| agg_histogram | Histogram (interval=500) | price | **7.93 ms** | [7.89 ms, 7.97 ms] |

### Maintenance

| Benchmark | Operation | Median | Range |
|---|---|---|---|
| commit_100k | Add 1,000 docs + commit to a 100K index | **47.48 ms** | [46.14 ms, 48.75 ms] |
| compact_100k | Full compaction of 10 segments (100K docs) | **1.61 s** | [1.58 s, 1.63 s] |

Incremental commits are fast (~47 ms for 1,000 documents) because only the new
segment is written. Full compaction rewrites all segments into one and is
proportional to total index size.

## Methodology

- **Framework:** [Criterion.rs](https://github.com/bheisler/criterion.rs) 0.5
- **Storage:** `StorageType::InMemory` (no filesystem I/O)
- **Data generator:** `benches/datagen.rs` produces synthetic documents with 7 fields
  (title, body, category, price, rating, created_at, _id) using Zipfian category
  distribution and a 229-word vocabulary
- **Index shape:** single segment for search/aggregation benchmarks, 10 segments
  for maintenance benchmarks
- **Execution:** WAND pruning (default), BM25 with k1=0.9, b=0.4
- **Sample sizes:** 10 for heavy operations (indexing, maintenance), 20 for latency
  benchmarks (search, aggregations)

Additional benchmark suites (`benches/aggs.rs`, `benches/end_to_end.rs`) cover
aggregation-specific and integration scenarios at smaller scale.
