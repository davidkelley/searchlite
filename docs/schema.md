# Schema and Documents

A schema defines the shape of your index: which fields exist, how text is analyzed,
and which fields support filtering and aggregations. You write it once when you create
an index, and every document you add is validated against it.

Schemas are standard **JSON Schema 2020-12** documents annotated with `searchlite:`
vocabulary keywords. Think of it as a document shape definition with search
annotations layered on top. The same file both validates your documents and
configures how Searchlite indexes them.

A blog might define a `title` (text, searchable), `body` (text, searchable,
highlighted), `author` (keyword, filterable), and `published_at` (numeric, sortable).
An e-commerce catalog might add `price` (numeric, fast for range filters) and
`category` (keyword, fast for faceted navigation).

## Example schema

Here is an abbreviated version of the recipes example schema:

```json
{
  "$schema": "https://searchlite.dev/draft/2025/schema",
  "type": "object",
  "searchlite:docIdField": "doc_id",
  "properties": {
    "title": { "type": "string" },
    "description": { "type": "string" },
    "cuisine": { "type": "string", "searchlite:kind": "keyword" },
    "difficulty": { "type": "string", "searchlite:kind": "keyword" },
    "prep_time_minutes": { "type": "integer", "searchlite:stored": true },
    "servings": { "type": "integer", "searchlite:stored": true },
    "ingredients": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "item": { "type": "string", "searchlite:kind": "keyword" },
          "quantity": { "type": "number", "searchlite:stored": true },
          "unit": { "type": "string", "searchlite:kind": "keyword" }
        }
      }
    }
  }
}
```

This single file does double duty. As JSON Schema, it validates that every document
has a `title` string, an `ingredients` array of objects, and so on. The `searchlite:`
keywords tell the engine that `cuisine` is a keyword (exact-match, filterable),
`prep_time_minutes` should be stored for retrieval, and `ingredients` is a nested
field with its own sub-fields.

## Type inference

Searchlite infers the field type from the JSON Schema `type` value:

| JSON Schema `type` | Searchlite field type | Notes |
|---|---|---|
| `"string"` | text | Full-text analyzed. Override with `searchlite:kind: "keyword"` for exact match. |
| `"string"` + `searchlite:kind: "keyword"` | keyword | Unanalyzed, exact-match values. |
| `"integer"` | numeric (i64) | Signed 64-bit integer. |
| `"number"` | numeric (f64) | 64-bit floating point. |
| `"object"` with `properties` | nested | Single nested object with typed sub-fields. |
| `"array"` with `items.type: "object"` | nested | Array of nested objects (most common form). |
| `"array"` with `items.type: "number"` + `searchlite:vector` | vector | Dense embedding vector. |
| `["string", "null"]` | nullable text | Any base type can be made nullable this way. |

## `searchlite:` keyword reference

All configuration beyond standard JSON Schema uses `searchlite:` prefixed keywords.
Any unknown `searchlite:` keyword on a property is rejected at parse time.

### Root-level keywords

These appear on the root schema object, alongside `type` and `properties`:

| Keyword | Type | Default | Description |
|---|---|---|---|
| `searchlite:docIdField` | string | `"_id"` | Name of the string primary key field expected on every document. |
| `searchlite:analyzers` | array | `[]` | Custom analyzer definitions (see [Analyzers](#analyzers)). |

### Property-level keywords

These appear on individual property definitions inside `properties`:

| Keyword | Type | Default | Applies to | Description |
|---|---|---|---|---|
| `searchlite:kind` | `"keyword"` | -- | string fields | Marks a string as a keyword (exact-match) instead of text (analyzed). |
| `searchlite:stored` | boolean | text: `true`, keyword: `true`, numeric: `false` | text, keyword, numeric | Save the raw value in the doc store for retrieval. |
| `searchlite:indexed` | boolean | text: `true`, keyword: `true` | text, keyword | Add the field to the inverted index so it can be searched. |
| `searchlite:fast` | boolean | keyword: `true`, numeric: `true` | keyword, numeric | Build a columnar store for fast filtering, sorting, and aggregations. |
| `searchlite:analyzer` | string | `"default"` | text | Analyzer applied at index time (see [Analyzers](#analyzers)). |
| `searchlite:searchAnalyzer` | string | same as analyzer | text | Separate analyzer applied at query time. |
| `searchlite:searchAsYouType` | `{minGram, maxGram}` | -- | text | Enable automatic edge n-gram indexing (see [Search-as-you-type](#search-as-you-type)). |
| `searchlite:nullable` | boolean | `false` | all | Allow the field to be absent from documents. |
| `searchlite:vector` | `{dim, metric, hnsw?}` | -- | array-of-number | Configure dense vector storage (see [Vector fields](#vector-fields)). |

## Field types

### Text fields

Text fields are analyzed (tokenized, lowercased, stemmed) and support full-text search.
When a user searches for "programming languages", a text field will match documents
containing "language", "programming", or related stems.

```json
{
  "body": { "type": "string" },
  "title": { "type": "string", "searchlite:analyzer": "english" }
}
```

A bare `{"type": "string"}` creates a text field with all defaults: stored, indexed,
using the `default` analyzer. Add `searchlite:` keywords only when you need to
override a default:

- **`searchlite:stored`** (default: `true`) -- the original text is saved and can be
  returned in search results (useful for displaying snippets, titles, or descriptions
  in your UI).
- **`searchlite:indexed`** (default: `true`) -- the field is tokenized and added to
  the inverted index so it can be searched.
- **`searchlite:analyzer`** (default: `"default"`) -- controls how text is broken into
  tokens (see [Analyzers](#analyzers) below).
- **`searchlite:searchAnalyzer`** -- optional separate analyzer used at query time.
  Useful when the index analyzer produces edge n-grams for autocomplete but you want
  the search analyzer to match full words.

### Keyword fields

Keyword fields store exact, unanalyzed values. Use them for categorical data that you
filter or aggregate on: language codes, product categories, tags, status labels, user IDs.

```json
{
  "category": { "type": "string", "searchlite:kind": "keyword" },
  "source": { "type": "string", "searchlite:kind": "keyword", "searchlite:fast": false }
}
```

The `searchlite:kind: "keyword"` annotation is what distinguishes a keyword field from
a text field. Default keyword settings are stored, indexed, and fast -- all `true`.
Override only what you need:

- **`searchlite:fast`** (default: `true`) -- builds a columnar store for the field.
  Required for filters, sorting, and aggregations. This is what makes
  `KeywordEq { field: "category", value: "electronics" }` fast even over millions
  of documents.
- **`searchlite:stored`** (default: `true`) -- save the raw value for retrieval.
- **`searchlite:indexed`** (default: `true`) -- add to the inverted index for term
  matching.

### Numeric fields

Numeric fields store integer or floating-point values. Use them for prices, ratings,
timestamps, counters, or any value you want to filter by range or aggregate with stats.

```json
{
  "year": { "type": "integer" },
  "price": { "type": "number", "searchlite:stored": true },
  "rating": { "type": "number", "searchlite:stored": true }
}
```

Use `"type": "integer"` for i64 (signed 64-bit integer) or `"type": "number"` for f64
(floating point). Numeric fields are fast by default (`searchlite:fast: true`), which
is required for range filters (`I64Range`, `F64Range`), sorting, and numeric
aggregations like `stats`, `histogram`, and `percentiles`. Unlike text and keyword
fields, numeric fields are **not** stored by default -- add `searchlite:stored: true`
if you need to retrieve the value in search results.

### Nested fields

Nested fields model arrays of objects where each object's fields must be queried
together. For example, a product with multiple reviews where you need to filter by
"reviews where user=alice AND rating >= 4" (not "any review by alice" AND "any
review with rating >= 4").

The most common form uses an array of objects:

```json
{
  "ingredients": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "item": { "type": "string", "searchlite:kind": "keyword" },
        "quantity": { "type": "number", "searchlite:stored": true },
        "unit": { "type": "string", "searchlite:kind": "keyword" }
      }
    }
  }
}
```

You can also use a plain object for a single nested level:

```json
{
  "metadata": {
    "type": "object",
    "properties": {
      "author": { "type": "string", "searchlite:kind": "keyword" },
      "version": { "type": "integer" }
    }
  }
}
```

Nested objects are flattened into dotted field names internally (e.g.,
`ingredients.item`, `metadata.author`). Properties inside nested objects follow the
same type inference and `searchlite:` keyword rules as top-level properties -- you
can nest text, keyword, numeric, and even deeper object fields.

See [filters.md](filters.md) for nested filter examples.

### Vector fields

Vector fields store numeric embeddings for approximate nearest neighbor (ANN) search.
They require the `vectors` feature flag. See [vectors.md](vectors.md) for search
usage.

```json
{
  "embedding": {
    "type": "array",
    "items": { "type": "number" },
    "searchlite:vector": { "dim": 384, "metric": "Cosine" }
  }
}
```

The `searchlite:vector` object is what distinguishes a vector field from a nested
array. It requires:

- **`dim`** -- embedding dimension (must match your model output).
- **`metric`** -- `"Cosine"` (similarity, best for normalized embeddings) or `"L2"`
  (Euclidean distance, best for unnormalized embeddings).
- **`hnsw`** -- optional HNSW tuning parameters (see [HNSW tuning](#hnsw-tuning)
  below).

## Analyzers

Analyzers control how text is processed before being indexed and searched. Choosing
the right analyzer determines whether a search for "running" matches a document
containing "ran".

Define custom analyzers with the `searchlite:analyzers` array at the schema root:

```json
{
  "$schema": "https://searchlite.dev/draft/2025/schema",
  "type": "object",
  "searchlite:analyzers": [
    {
      "name": "english",
      "tokenizer": "default",
      "filters": [{ "stopwords": "en" }, { "stemmer": "english" }]
    },
    {
      "name": "autocomplete",
      "tokenizer": "default",
      "filters": [{ "edge_ngram": { "min": 1, "max": 10 } }]
    }
  ],
  "properties": {
    "body": { "type": "string", "searchlite:analyzer": "english" },
    "title": {
      "type": "string",
      "searchlite:analyzer": "autocomplete",
      "searchlite:searchAnalyzer": "english"
    }
  }
}
```

If you omit `searchlite:analyzers`, Searchlite uses its built-in `default` analyzer
(ASCII lowercase + alphanumeric tokenization). Reference a custom analyzer by name
via `searchlite:analyzer` on any text field.

### Available tokenizers

| Tokenizer | Behavior | When to use |
|---|---|---|
| `default` | ASCII lowercase, splits on non-alphanumeric | General-purpose English text |
| `unicode` | NFKC normalization, case-folded words | Multilingual content, accented characters |
| `whitespace` | Splits only on whitespace, no case folding | Log messages, identifiers, code |

### Available token filters

| Filter | Behavior | When to use |
|---|---|---|
| `lowercase` | Lowercases all tokens | Case-insensitive search |
| `stopwords` | Removes common words ("the", "is", "at") | Reduce noise in relevance scoring |
| `stemmer` | Reduces words to their root ("running" -> "run") | Match word variants in English |
| `synonyms` | Expands terms (`from`/`to` lists at the same position) | "laptop" also matches "notebook" |
| `edge_ngram` | Generates prefixes of each token ("rust" -> "r", "ru", "rus", "rust") | Autocomplete / search-as-you-type |

### Example: English analyzer for a blog

```json
{
  "name": "english",
  "tokenizer": "default",
  "filters": [{ "stopwords": "en" }, { "stemmer": "english" }]
}
```

Searching for "programming languages" will match documents containing "program",
"language", "programmed", etc.

### Example: Autocomplete analyzer for a search bar

```json
{
  "name": "autocomplete",
  "tokenizer": "default",
  "filters": [{ "edge_ngram": { "min": 1, "max": 10 } }]
}
```

Use this as the index analyzer with a separate `searchlite:searchAnalyzer` so that
typing "pro" in a search bar matches "programming", "production", "prometheus".

## Defaults

When `searchlite:` keywords are omitted from a property, Searchlite applies
sensible defaults. You only need to specify non-default values, which keeps schemas
concise.

| Field type | `stored` | `indexed` | `fast` | `analyzer` | `nullable` |
|---|---|---|---|---|---|
| text | `true` | `true` | -- | `"default"` | `false` |
| keyword | `true` | `true` | `true` | -- | `false` |
| numeric | `false` | -- | `true` | -- | `false` |

This means `{"type": "string"}` is equivalent to writing:

```json
{
  "type": "string",
  "searchlite:stored": true,
  "searchlite:indexed": true,
  "searchlite:analyzer": "default",
  "searchlite:nullable": false
}
```

And `{"type": "string", "searchlite:kind": "keyword"}` is equivalent to:

```json
{
  "type": "string",
  "searchlite:kind": "keyword",
  "searchlite:stored": true,
  "searchlite:indexed": true,
  "searchlite:fast": true,
  "searchlite:nullable": false
}
```

When serializing a schema back to JSON, Searchlite omits keywords that match their
default values. Only emit non-default values in your own schemas to keep them
readable.

## Nullable fields

By default, every field defined in the schema is **required** -- Searchlite rejects
documents that omit a defined field. There are two ways to make a field optional.

### Option 1: JSON Schema type array

Use the standard JSON Schema nullable pattern:

```json
{
  "subtitle": { "type": ["string", "null"] },
  "sale_price": { "type": ["integer", "null"] },
  "brand": { "type": ["string", "null"], "searchlite:kind": "keyword" }
}
```

### Option 2: `searchlite:nullable`

Use the `searchlite:nullable` keyword directly:

```json
{
  "subtitle": { "type": "string", "searchlite:nullable": true },
  "sale_price": { "type": "integer", "searchlite:nullable": true }
}
```

Both approaches are equivalent. If either one signals nullable, the field is optional.

With these definitions, a document like
`{"_id": "1", "title": "Widget", "price": 999, "category": "gadgets"}` is valid
even though it omits `subtitle`, `sale_price`, and `brand`.

**Behavior of nullable fields:**

- Nullable text fields: documents without the field are simply not indexed for that
  field. Searches against the field will not match those documents.
- Nullable keyword/numeric fields: missing values produce no fast-field entry.
  Filters and aggregations skip documents without a value.
- Nullable nested fields: documents without the nested array are valid. Nested
  filters and aggregations simply do not match those documents.

## Vector fields

Vector fields store numeric embeddings for approximate nearest neighbor (ANN) search.
They are declared as a JSON Schema array of numbers with the `searchlite:vector`
annotation:

```json
{
  "embedding": {
    "type": "array",
    "items": { "type": "number" },
    "searchlite:vector": { "dim": 384, "metric": "Cosine" }
  }
}
```

- **`dim`** -- embedding dimension (must match your model output).
- **`metric`** -- `"Cosine"` (similarity, best for normalized embeddings) or `"L2"`
  (Euclidean distance, best for unnormalized embeddings).

### HNSW tuning

Each vector field uses an HNSW index for fast approximate search. You can tune the
index parameters for your recall/speed tradeoff:

```json
{
  "embedding": {
    "type": "array",
    "items": { "type": "number" },
    "searchlite:vector": {
      "dim": 384,
      "metric": "Cosine",
      "hnsw": { "m": 16, "ef_construction": 64 }
    }
  }
}
```

| Parameter | Default | Effect |
|---|---|---|
| `m` | 16 | Max edges per node. Higher = better recall, more memory. |
| `ef_construction` | 64 | Beam width during index building. Higher = better graph quality, slower indexing. |

The defaults work well for most workloads (up to ~1M vectors). Increase `m` to 32 and
`ef_construction` to 128 for high-recall requirements on larger datasets. At query
time, you can further tune recall with the `ef_search` and `candidate_size` parameters
on the search request (see [vectors.md](vectors.md)).

## Search-as-you-type

For building autocomplete UIs, text fields can opt into automatic edge n-gram
indexing with `searchlite:searchAsYouType`. This means typing just a few characters
in a search box will immediately match full words.

```json
{
  "title": {
    "type": "string",
    "searchlite:searchAsYouType": { "minGram": 1, "maxGram": 10 }
  }
}
```

With this configuration, searching for `"ru"` matches documents with `"rustacean"`,
`"ruby"`, or `"runtime"` in the title -- perfect for powering a live search dropdown.

| Parameter | Default | Description |
|---|---|---|
| `minGram` | 1 | Minimum prefix length to generate. |
| `maxGram` | 15 | Maximum prefix length to generate. |

You can also build this manually by defining an analyzer with an `edge_ngram` filter
as the index analyzer and a separate `searchlite:searchAnalyzer` for query time. The
`searchlite:searchAsYouType` keyword is a shorthand that does this for you.

## Document ID

Every document must include a string primary key under the field named by
`searchlite:docIdField` (defaults to `"_id"`). This ID is stored automatically and
used for upserts, deletes, and multi-get lookups. Do not list it in your `properties`.

```json
{
  "$schema": "https://searchlite.dev/draft/2025/schema",
  "type": "object",
  "searchlite:docIdField": "doc_id",
  "properties": {
    "title": { "type": "string" }
  }
}
```

With this schema, documents must include a `doc_id` field:

```json
{"doc_id": "product-42", "title": "Wireless Mouse"}
```

If you omit `searchlite:docIdField`, the default is `"_id"`:

```json
{"_id": "product-42", "title": "Wireless Mouse"}
```
