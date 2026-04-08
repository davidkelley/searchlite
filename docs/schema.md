# Schema and Documents

A schema defines the shape of your index: which fields exist, how text is analyzed,
and which fields support filtering and aggregations. You write it once when you create
an index, and every document you add is validated against it.

Think of it like a database table definition, but tuned for search. A blog might define
a `title` (text, searchable), `body` (text, searchable, highlighted), `author` (keyword,
filterable), and `published_at` (numeric, sortable). An e-commerce catalog might add
`price` (numeric, fast for range filters) and `category` (keyword, fast for faceted
navigation).

## Example schema

```json
{
  "doc_id_field": "_id",
  "analyzers": [
    {
      "name": "english",
      "tokenizer": "default",
      "filters": [{ "stopwords": "en" }, { "stemmer": "english" }]
    },
    {
      "name": "title_prefix",
      "tokenizer": "default",
      "filters": [{ "edge_ngram": { "min": 1, "max": 5 } }]
    }
  ],
  "text_fields": [
    { "name": "body", "analyzer": "english", "stored": true, "indexed": true },
    {
      "name": "title",
      "analyzer": "title_prefix",
      "search_analyzer": "english",
      "stored": true,
      "indexed": true
    }
  ],
  "keyword_fields": [
    { "name": "lang", "stored": true, "indexed": true, "fast": true }
  ],
  "numeric_fields": [{ "name": "year", "i64": true, "fast": true }],
  "nested_fields": [
    {
      "name": "comment",
      "fields": [
        {
          "type": "keyword",
          "name": "author",
          "stored": true,
          "indexed": true,
          "fast": true
        }
      ]
    }
  ],
  "vector_fields": []
}
```

## Field types

### Text fields

Text fields are analyzed (tokenized, lowercased, stemmed) and support full-text search.
When a user searches for "programming languages", a text field will match documents
containing "language", "programming", or related stems.

```json
{ "name": "body", "analyzer": "english", "stored": true, "indexed": true }
```

- **`stored: true`** -- the original text is saved and can be returned in search results
  (useful for displaying snippets, titles, or descriptions in your UI).
- **`indexed: true`** -- the field is tokenized and added to the inverted index so it
  can be searched.
- **`analyzer`** -- controls how text is broken into tokens (see Analyzers below).
- **`search_analyzer`** -- optional separate analyzer used at query time. Useful when
  the index analyzer produces edge n-grams for autocomplete but you want the search
  analyzer to match full words.
- **`nullable: true`** -- allows the field to be omitted from documents. See
  [Nullable fields](#nullable-fields) below.

### Keyword fields

Keyword fields store exact, unanalyzed values. Use them for categorical data that you
filter or aggregate on: language codes, product categories, tags, status labels, user IDs.

```json
{ "name": "category", "stored": true, "indexed": true, "fast": true }
```

- **`fast: true`** -- builds a columnar store (like a database column index) for the
  field. Required for filters, sorting, and aggregations. This is what makes
  `KeywordEq { field: "category", value: "electronics" }` fast even over millions
  of documents.
- **`nullable: true`** -- allows the field to be omitted from documents. See
  [Nullable fields](#nullable-fields) below.

### Numeric fields

Numeric fields store integer (`i64`) or floating-point (`f64`) values. Use them for
prices, ratings, timestamps, counters, or any value you want to filter by range or
aggregate with stats.

```json
{ "name": "price", "i64": true, "fast": true, "stored": true }
{ "name": "rating", "i64": false, "fast": true, "stored": true }
```

Setting `i64: false` stores the value as `f64` (floating point). The `fast` flag
is required for range filters (`I64Range`, `F64Range`), sorting, and numeric
aggregations like `stats`, `histogram`, and `percentiles`. As with other field types,
set `nullable: true` if the field may be absent from some documents.

### Nested fields

Nested fields model arrays of objects where each object's fields must be queried
together. For example, a product with multiple reviews where you need to filter by
"reviews where user=alice AND rating >= 4" (not "any review by alice" AND "any
review with rating >= 4").

Each nested field contains an array of property definitions. Properties can be keyword,
numeric, text, or object (for deeper nesting):

```json
{
  "name": "review",
  "fields": [
    { "type": "keyword", "name": "author", "fast": true, "stored": true, "indexed": true },
    { "type": "numeric", "name": "rating", "i64": true, "fast": true, "stored": true },
    { "type": "text", "name": "comment", "analyzer": "default", "stored": true, "indexed": true },
    {
      "type": "object", "name": "reply",
      "fields": [
        { "type": "keyword", "name": "tag", "fast": true, "stored": true, "indexed": true }
      ]
    }
  ]
}
```

Nested objects are flattened into dotted field names internally (e.g., `review.author`,
`review.reply.tag`). The `object` type creates another level of nesting for
hierarchical data like threaded comments.

Set `"nullable": true` on the nested field itself to allow documents without the
nested array.

See [filters.md](filters.md) for nested filter examples.

## Analyzers

Analyzers control how text is processed before being indexed and searched. Choosing
the right analyzer determines whether a search for "running" matches a document
containing "ran".

If you omit `analyzers`, Searchlite uses its built-in `default` analyzer (ASCII
lowercase + alphanumeric tokenization).

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

Use this as the index analyzer with a standard `search_analyzer` so that typing "pro"
in a search bar matches "programming", "production", "prometheus".

## Field storage and fast fields

These two flags control what you can do with a field at query time:

- **`stored: true`** -- the raw value is saved in the docstore. Enable this on fields
  you want to return in search results (titles, descriptions, prices). Without it, the
  field is searchable but its original value is not returned.
- **`fast: true`** -- builds a columnar store for the field. Enable this on fields you
  want to filter, sort, or aggregate on. Fast fields are memory-mapped for zero-copy
  access, making filter evaluation fast even at scale.

Nested objects are flattened into dotted field names (e.g., `comment.author`). You can
filter on the dotted path directly, or wrap the clause in a `Nested` filter to enforce
per-object binding (see [filters](filters.md)).

## Nullable fields

By default, every field defined in the schema is **required** -- Searchlite rejects
documents that omit a defined field. Set `nullable: true` on a field to make it
optional.

This is common in real-world data. Not every product has a sale price. Not every
article has a subtitle. Not every user has a bio.

```json
{
  "text_fields": [
    { "name": "title", "analyzer": "default", "stored": true, "indexed": true },
    { "name": "subtitle", "analyzer": "default", "stored": true, "indexed": true, "nullable": true }
  ],
  "numeric_fields": [
    { "name": "price", "i64": true, "fast": true, "stored": true },
    { "name": "sale_price", "i64": true, "fast": true, "stored": true, "nullable": true }
  ],
  "keyword_fields": [
    { "name": "category", "stored": true, "indexed": true, "fast": true },
    { "name": "brand", "stored": true, "indexed": true, "fast": true, "nullable": true }
  ]
}
```

With this schema, a document like `{"_id": "1", "title": "Widget", "price": 999, "category": "gadgets"}`
is valid even though it omits `subtitle`, `sale_price`, and `brand`.

**Behavior of nullable fields:**
- Nullable text fields: documents without the field are simply not indexed for that field.
  Searches against the field won't match those documents.
- Nullable keyword/numeric fields: missing values produce no fast-field entry.
  Filters and aggregations skip documents without a value.
- Nullable nested fields: documents without the nested array are valid.
  Nested filters and aggregations simply don't match those documents.

## Vector fields

Vector fields store numeric embeddings for approximate nearest neighbor (ANN) search.
They require the `vectors` feature flag. See [vectors.md](vectors.md) for search
usage.

```json
{
  "vector_fields": [
    { "name": "embedding", "dim": 384, "metric": "Cosine" }
  ]
}
```

- **`dim`** -- embedding dimension (must match your model output)
- **`metric`** -- `Cosine` (similarity, best for normalized embeddings) or `L2`
  (Euclidean distance, best for unnormalized embeddings)

### HNSW tuning

Each vector field uses an HNSW index for fast approximate search. You can tune the
index parameters for your recall/speed tradeoff:

```json
{
  "vector_fields": [
    {
      "name": "embedding", "dim": 384, "metric": "Cosine",
      "hnsw": { "m": 16, "ef_construction": 64 }
    }
  ]
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

## Document ID

Every document must include a string primary key under the field named by `doc_id_field`
(defaults to `_id`). This ID is stored automatically and used for upserts, deletes,
and multi-get lookups. Don't list it in your field definitions.

```json
{"_id": "product-42", "title": "Wireless Mouse", "price": 2999, "category": "electronics"}
```

## Search-as-you-type

For building autocomplete UIs, text fields can opt into automatic edge n-gram
indexing. This means typing just a few characters in a search box will immediately
match full words.

```json
{
  "name": "title",
  "analyzer": "default",
  "stored": true,
  "indexed": true,
  "search_as_you_type": { "min_gram": 1, "max_gram": 10 }
}
```

With this configuration, searching for `"ru"` matches documents with `"rustacean"`,
`"ruby"`, or `"runtime"` in the title -- perfect for powering a live search dropdown.

You can also build this manually by defining an analyzer with an `edge_ngram` filter
as the index analyzer and a normal analyzer as the search analyzer. The built-in
`search_as_you_type` option is a shorthand that does this for you.
