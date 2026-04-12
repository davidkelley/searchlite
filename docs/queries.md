# Queries

Queries are how you find documents. Searchlite provides a structured query DSL that
ranges from simple text search ("find products matching 'wireless mouse'") to
complex relevance tuning ("boost newer articles, prefer title matches over body
matches, and re-rank the top 50 with phrase proximity").

Every search request has a `query` field. The simplest form is a plain string:

```json
{ "query": "rust search engine" }
```

For more control, use a structured query node with `"type"`:

```json
{ "query": { "type": "query_string", "query": "rust search engine", "fields": ["title", "body"] } }
```

The `query_string` node supports field-scoped terms (`title:rust`), phrases in quotes
(`"exact phrase"`), and negation with a leading dash (`-excluded`).

---

## Query types at a glance

| Type | Purpose | Example use case |
|---|---|---|
| `match_all` | Match every document in the index | Pure aggregation requests, "browse all" UIs |
| `query_string` | Free-text search across fields | Search bar in any application |
| `term` | Exact match on a single analyzed term | Internal lookups by known token |
| `prefix` | Match terms starting with a prefix | Autocomplete in a search box |
| `wildcard` | Pattern matching with `*` and `?` | Searching file names, product codes |
| `regex` | Regular expression matching | Advanced pattern search for power users |
| `phrase` | Ordered multi-word match with optional slop | "Find documents where 'machine learning' appears as a phrase" |
| `multi_match` | Search across multiple fields with boosting | Title matches ranked higher than body |
| `dis_max` | Best-of-N subqueries with tie breaking | Combine several relevance strategies |
| `bool` | Combine must/should/must_not/filter clauses | Complex search pages with required + optional criteria |
| `constant_score` | Fixed score for filter-only queries | "All English documents score equally" |
| `function_score` | Customize scoring with functions | Boost popular or recent results |
| `rank_feature` | Boost by a numeric field value | Sort-by-popularity without losing relevance |
| `script_score` | Arithmetic expressions over fields | Custom ranking formulas |
| `vector` ⚙️ | ANN search over an embedding field (requires `vectors` feature) | Semantic search / "find similar" |

---

## Simple queries, step by step

If you're new to search engines, start here. Every example below produces a
complete request body you can paste into `/indexes/{name}/search`.

### `match_all` — "every document"

Useful when you only care about aggregations, or when you need a paginated
browse page with no search term entered yet:

```json
{
  "query": { "type": "match_all" },
  "limit": 20,
  "return_stored": true
}
```

### `query_string` — "the usual search bar"

The input in most UIs. It tokenises the query with the target field's analyzer,
supports field-scoped terms (`title:rust`), quoted phrases (`"machine learning"`),
and negation (`-draft`):

```json
{
  "query": {
    "type": "query_string",
    "query": "\"machine learning\" -draft title:rust",
    "fields": ["title", "body"]
  },
  "limit": 10
}
```

### `term` — "exact analyzed match on one field"

Unlike `query_string`, `term` runs a single analyzed token straight against the
inverted index. Use it when you already know the token you want (from a filter,
an autocomplete suggestion, a link click, etc.):

```json
{ "query": { "type": "term", "field": "title", "value": "rust" } }
```

The value is still passed through the field's analyzer -- so for most text
fields `"Rust"` and `"rust"` are equivalent.

### `bool` — "combine clauses"

`bool` is the workhorse of non-trivial search. Each slot has a defined role:

- **`must`** -- clauses that *must* match and contribute to the score
- **`should`** -- optional boosts; non-matching documents are still returned
- **`must_not`** -- exclusions; non-scoring
- **`filter`** -- *must* match, but does not affect scores. Unlike `must`/`should`/`must_not`, each entry here is a `Filter` variant (`KeywordEq`, `I64Range`, `Nested`, …), not a query node.

```json
{
  "query": {
    "type": "bool",
    "must":     [ { "type": "query_string", "query": "laptop" } ],
    "should":   [ { "type": "term",         "field": "brand", "value": "framework" } ],
    "must_not": [ { "type": "term",         "field": "status", "value": "archived" } ],
    "filter":   [
      { "KeywordEq": { "field": "in_stock", "value": "true" } }
    ]
  }
}
```

---

## Prefix, wildcard, and regex queries

These queries expand against the term dictionary, letting users search with partial
or pattern-based input.

**Prefix** -- ideal for autocomplete. As a user types "pro" in a search bar, a prefix
query finds all documents with terms starting with "pro" (product, programming, prometheus):

```json
{ "query": { "type": "prefix", "field": "title", "value": "pro", "max_expansions": 50 } }
```

**Wildcard** -- match patterns with `*` (any characters) and `?` (single character).
Useful for searching product codes or structured identifiers:

```json
{ "query": { "type": "wildcard", "field": "sku", "value": "WDG-*-BLK", "max_expansions": 100 } }
```

**Regex** -- full regular expression matching for power-user search interfaces:

```json
{ "query": { "type": "regex", "field": "title", "value": "r(ust|uby)", "max_expansions": 100 } }
```

Each query type analyzes the input with the field's search analyzer, expands against
the segment term dictionary (capped by `max_expansions` per segment to bound cost),
and ORs the resulting terms for BM25 scoring. Use `boost` on any node to influence
its weight relative to other clauses.

---

## Suggestions

Suggestions power search-as-you-type dropdowns. Instead of returning full search
results, they return **term completions** ranked by frequency -- perfect for showing
a list of suggestions as the user types.

For example, in a recipe app, typing "ch" in the search bar could suggest
"chicken", "cheese", "chocolate", "cherry":

```json
{
  "query": { "type": "match_all" },
  "limit": 0,
  "suggest": {
    "ingredient_suggest": {
      "type": "completion",
      "field": "ingredients",
      "prefix": "ch",
      "size": 5,
      "fuzzy": {
        "max_edits": 1,
        "prefix_length": 1,
        "max_expansions": 20,
        "min_length": 2
      }
    }
  }
}
```

The `fuzzy` option makes suggestions typo-tolerant -- typing "chiken" still suggests
"chicken". The response returns suggestions ranked by term frequency:

```json
{
  "suggest": {
    "ingredient_suggest": {
      "options": [
        { "text": "chicken", "score": 42.0, "doc_freq": 128 },
        { "text": "cheese", "score": 38.0, "doc_freq": 95 }
      ]
    }
  }
}
```

---

## Multi-field relevance queries

Most applications have multiple searchable fields (title, body, tags, description).
Multi-field queries let you search across all of them with control over how scores
are combined.

**`multi_match`** -- the workhorse for multi-field search. In a documentation site,
you might want title matches to count double:

```json
{
  "query": {
    "type": "multi_match",
    "query": "rust search",
    "match_type": "best_fields",
    "fields": [{ "field": "title", "boost": 2.0 }, { "field": "body" }],
    "operator": "or",
    "tie_breaker": 0.2,
    "minimum_should_match": "75%"
  },
  "limit": 10
}
```

Match types:
- **`best_fields`** (default) -- takes the highest-scoring field and blends the rest
  via `tie_breaker`. Best when each field is an independent unit (title vs body).
- **`most_fields`** -- sums scores across all fields. Best when the same content is
  analyzed multiple ways (e.g., with and without stemming).
- **`cross_fields`** -- treats all fields as one blended field. Best when you don't
  know which field a term belongs to (e.g., a person's first + last name).

**`dis_max`** -- picks the best-scoring subquery and blends the rest. Useful when
you want to try several relevance strategies and take the best result:

```json
{
  "query": {
    "type": "dis_max",
    "tie_breaker": 0.4,
    "queries": [
      { "type": "term", "field": "title", "value": "rust" },
      { "type": "term", "field": "body", "value": "rust" }
    ]
  }
}
```

**`phrase` with slop** -- match multi-word phrases with some flexibility. A slop of 1
means one word can appear between the terms. Useful for natural language where word
order matters but isn't rigid ("search engine" also matches "search optimization engine"):

```json
{
  "query": {
    "type": "phrase",
    "field": "body",
    "terms": ["rust", "search"],
    "slop": 1
  }
}
```

---

## Custom scoring and reranking

Default BM25 scoring ranks documents by text relevance alone. Custom scoring lets you
blend in business signals -- popularity, recency, user ratings -- so that results
reflect both relevance and real-world quality.

### `function_score`

Combine multiple scoring functions with the base BM25 score. In a news app, you might
boost English articles, favor recent content with a time decay, and factor in popularity:

```json
{
  "query": {
    "type": "function_score",
    "query": { "type": "match_all" },
    "functions": [
      {
        "type": "weight",
        "weight": 2.0,
        "filter": { "KeywordEq": { "field": "lang", "value": "en" } }
      },
      {
        "type": "decay",
        "field": "age_days",
        "origin": 0,
        "scale": 30,
        "offset": 0,
        "decay": 0.5,
        "function": "linear"
      },
      {
        "type": "field_value_factor",
        "field": "popularity",
        "factor": 0.25,
        "modifier": "log1p",
        "missing": 0.0
      }
    ],
    "score_mode": "sum",
    "boost_mode": "sum",
    "max_boost": 5.0,
    "min_score": 0.5
  },
  "limit": 10
}
```

Function types:
- **`weight`** -- multiply by a constant (optionally filtered). "English docs score 2x."
- **`field_value_factor`** -- incorporate a numeric field. "More popular items rank higher."
- **`decay`** -- distance-based scoring (exp, gauss, linear). "Prefer items near price $50" or "prefer recent articles."

### `constant_score`

Returns a fixed score for all matching documents. Useful when you want filter-only
results where relevance ranking doesn't matter (e.g., "show all English documents"):

```json
{
  "query": {
    "type": "constant_score",
    "filter": { "KeywordEq": { "field": "lang", "value": "en" } },
    "boost": 2.5
  }
}
```

### Rescoring

Re-rank the top results with a more expensive query. This is a two-pass strategy:
the first pass quickly finds the best ~50 candidates with BM25, then the second pass
re-ranks them with phrase proximity. This gives you phrase-quality results at BM25 speed:

```json
{
  "query": { "type": "query_string", "query": "rust search" },
  "limit": 10,
  "rescore": {
    "window_size": 50,
    "query": {
      "type": "phrase",
      "field": "body",
      "terms": ["rust", "search"],
      "slop": 1
    },
    "score_mode": "total"
  }
}
```

---

## Request-level tuning knobs

Beyond the query itself, a `SearchRequest` exposes a handful of fields that
tune how the search engine runs. The most common are:

| Field | Default | What it does |
|---|---|---|
| `limit` | `10` | Max hits returned (alias: `size`). Set to `0` to skip hit ranking and only run aggregations. |
| `from` | `0` | Offset-pagination skip count; `from + limit` must not exceed `1000`. |
| `return_hits` | `true` | When `false`, omits the `hits` array from the response. Perfect for pure aggregation requests where you want the facets but not the documents. |
| `return_stored` | `false` | When `true`, each hit includes the stored fields (title, body, price, …). Off by default so responses stay compact. |
| `track_total_hits` | `false` | When `true`, Searchlite counts every matching document, even with WAND pruning enabled. Set it when you need exact total counts (e.g., "Page 1 of 37"). Leave it off for infinite-scroll / "load more" UIs. |
| `execution` | `"wand"` | Scoring strategy -- `"wand"`, `"bmw"`, or `"bm25"`. See [Query execution modes in the CLI guide](cli.md#query-execution-modes). |
| `bmw_block_size` | engine default | Advanced: override the per-block posting size used by the `bmw` execution strategy. Only relevant when `"execution": "bmw"`. |
| `candidate_size` | `limit` | Oversampling pool before re-ranking or collapsing (also used by hybrid vector search). Increase for better recall at the cost of latency. |

Example of "pure analytics" mode -- no hits, just facets:

```json
{
  "query": { "type": "match_all" },
  "limit": 0,
  "return_hits": false,
  "aggs": {
    "tags":     { "type": "terms", "field": "tag", "size": 10 },
    "by_year":  { "type": "histogram", "field": "year", "interval": 1 }
  }
}
```

## Debugging aids

When results don't look right, these flags help you understand what's happening:

- **`explain: true`** -- returns a per-hit score breakdown showing which terms matched,
  their individual contributions, and any function score or rescore adjustments.
  Invaluable for tuning relevance.
- **`profile: true`** -- attaches execution stats (`candidates_examined`, `scored_docs`,
  postings advances) and timing buckets (`search_ms`, `rescore_ms`). Useful for
  identifying slow queries.
- **`track_total_hits: true`** -- when the `total_hits_estimate` you're seeing
  doesn't match what you expected, this disables WAND's pruning shortcuts and
  computes the exact total. Use it only while diagnosing -- it's measurably slower
  on large indexes.

All three flags are off by default to avoid overhead in production.
