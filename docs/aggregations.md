# Aggregations

Aggregations compute summary statistics and group-by breakdowns across your search
results. They power the analytics features users expect in modern search: faceted
navigation sidebars, price range histograms, tag clouds, time-series charts, and
statistical dashboards.

**When you'd use aggregations:**
- An e-commerce site shows a sidebar with category counts, price ranges, and brand filters
- An analytics dashboard displays a histogram of response times over the last 24 hours
- A content platform shows trending tags and author activity

Aggregations run over **all matched documents** (not just the top-K results you display).
Set `limit: 0` to skip hit ranking entirely and return only aggregations -- useful for
pure analytics queries.

Aggregation fields must have `"fast": true` in the [schema](schema.md).

---

## Types of aggregations

### Bucket aggregations (group documents)

Bucket aggregations split documents into groups. Each bucket has a count and can
contain nested sub-aggregations.

| Type | What it does | Example use case |
|---|---|---|
| `terms` | Groups by unique keyword values | Category sidebar: "Electronics (42), Books (28), Clothing (15)" |
| `significant_terms` | Finds unusually frequent terms vs. background | "What topics are trending in today's articles?" |
| `rare_terms` | Finds low-frequency values | "Which languages have only 1-2 articles?" |
| `histogram` | Groups numeric values into fixed-width buckets | Price range chart: "$0-50 (120), $50-100 (85), ..." |
| `date_histogram` | Groups dates into calendar or fixed intervals | Daily article count over the last month |
| `range` / `date_range` | Custom range buckets | "Under $25 / $25-$100 / Over $100" |
| `filter` | A single bucket matching a filter | "How many results are in English?" |
| `nested` | Scopes sub-aggregations to nested objects | "Top tags per metadata key" (preserving key/value binding) |
| `composite` | Paginated multi-source grouping | Iterate through all (language, year) combinations |

### Metric aggregations (compute statistics)

Metric aggregations compute values within each bucket (or across all results).

| Type | What it does | Example use case |
|---|---|---|
| `stats` / `extended_stats` | min, max, avg, sum, count (+ variance, std_dev) | "Average price of products in this category" |
| `value_count` | Count of non-null values | "How many products have a rating?" |
| `cardinality` | Approximate distinct count | "How many unique authors?" |
| `percentiles` | Value at given percentile ranks | "P50 and P99 response times" |
| `percentile_ranks` | Percentile rank of given values | "What percent of prices are below $50?" |
| `top_hits` | Actual documents per bucket | "Show the top 3 articles per category" |

### Pipeline aggregations (post-process buckets)

Pipeline aggregations run after bucket/metric aggregations and transform the results.

| Type | What it does | Example use case |
|---|---|---|
| `bucket_sort` | Reorder or truncate buckets | "Top 5 categories by average price" |
| `avg_bucket` / `sum_bucket` | Aggregate across buckets | "Average of per-category averages" |
| `derivative` | Rate of change between buckets | "Is daily traffic increasing or decreasing?" |
| `moving_avg` | Smoothed average over a window | "7-day moving average of sales" |
| `bucket_script` | Arithmetic over bucket values | "Revenue per customer = total_revenue / customer_count" |

---

## Field requirements

- **`terms` / `significant_terms` / `rare_terms`**: fast keyword field
- **`histogram` / `date_histogram` / `range` / `percentiles`**: fast numeric field
- **`cardinality`**: fast keyword or numeric field
- **`top_hits`**: no field requirement (returns stored fields)

---

## Basic example: faceted navigation

An e-commerce product search with category counts and price stats:

```json
{
  "query": "wireless headphones",
  "limit": 10,
  "aggs": {
    "categories": { "type": "terms", "field": "category", "size": 10 },
    "price_ranges": {
      "type": "histogram", "field": "price_cents", "interval": 5000,
      "min_doc_count": 0
    },
    "price_stats": { "type": "stats", "field": "price_cents" }
  }
}
```

The response includes both the top 10 search hits AND the aggregation results, so
you can render a search results page with a filter sidebar in a single request.

---

## Nested aggregations

Use `nested` aggregations when you need to respect object boundaries. For example,
products with metadata key/value pairs where "Category: Electronics" and "Color: Black"
must stay separate from "Category: Black" combinations:

```json
{
  "metadata_facets": {
    "type": "nested",
    "path": "metadata",
    "aggs": {
      "by_key": {
        "type": "terms", "field": "key", "size": 10,
        "aggs": {
          "by_value": { "type": "terms", "field": "value", "size": 10 }
        }
      }
    }
  }
}
```

---

## Composite aggregations (pagination)

When you have too many groups to return at once, `composite` lets you page through
them deterministically:

```json
{
  "by_lang_year": {
    "type": "composite",
    "size": 100,
    "sources": [
      { "type": "terms", "name": "lang", "field": "lang" },
      { "type": "histogram", "name": "year", "field": "year", "interval": 1 }
    ]
  }
}
```

The response includes `after_key` -- pass it back as `after` in the next request
to get the next page of groups.

---

## Pipeline example: trend analysis

Combine a date histogram with derivative and moving average to build a time-series
dashboard:

```json
{
  "daily_latency": {
    "type": "date_histogram",
    "field": "timestamp_ms",
    "fixed_interval": "1d",
    "aggs": {
      "latency": { "type": "stats", "field": "latency_ms" },
      "trend": {
        "type": "derivative", "buckets_path": "latency.avg",
        "gap_policy": "skip", "unit": 86400000
      },
      "smooth": {
        "type": "moving_avg", "buckets_path": "latency.avg",
        "window": 7, "predict": 1
      }
    }
  }
}
```

This gives you daily average latency, the day-over-day change (derivative), and a
7-day smoothed trend line -- all in one query.

---

## Sampling

For very large result sets, aggregations can be sampled to trade precision for speed:

```json
{
  "sampled_categories": {
    "type": "terms", "field": "category", "size": 20,
    "sampling": { "probability": 0.1, "seed": 42 }
  }
}
```

Responses include `sampled: true` when sampling is active. Counts become approximate
but ordering remains deterministic for a fixed seed.

---

## General notes

- Aggregations run over **all matched documents**, not just top-K. Use `limit: 0` to
  skip hit ranking when you only need aggregations.
- Bucket aggregations can contain nested sub-aggregations (metrics or other buckets).
- Pipeline aggregations reference other aggregations via dot-separated `buckets_path`
  values like `"by_tag.score_stats.avg"`.
- Nested aggregation cost is query-time only: index write performance is unaffected.
  Keep nested arrays bounded and add outer filters to limit the scope.
