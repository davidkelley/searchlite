# searchlite-js

A fast full-text search engine for Node.js with two index backends:

- **`EmbeddedIndex`** — native Rust bindings via [napi-rs](https://napi.rs). No external services, no network calls, no setup.
- **`RemoteIndex`** — HTTP client for a remote [searchlite-http](../searchlite-http) server. Query indexes that live on another machine.

Both implement the same async `SearchIndex` interface, so you can swap backends without changing application code.

```javascript
const { EmbeddedIndex } = require('searchlite-js');

const index = new EmbeddedIndex('./my-index', {
  schema: { title: 'text', body: 'text', tag: 'keyword' },
});

await index.add({ _id: '1', title: 'Getting Started', body: 'Hello, world!', tag: 'intro' });
await index.add({ _id: '2', title: 'Advanced Search', body: 'Filters, facets, and more', tag: 'guide' });
await index.commit();

const results = await index.search('hello');
console.log(results.hits[0].docId); // "1"

await index.close();
```

Or connect to a remote server:

```javascript
const { RemoteIndex } = require('searchlite-js');

const index = new RemoteIndex('http://localhost:8080', 'my-index');
const results = await index.search('hello');
```

## Installation

```bash
npm install searchlite-js
```

Prebuilt binaries are available for:

| Platform | Architectures |
|----------|---------------|
| macOS    | x64, arm64    |
| Linux    | x64, arm64    |
| Windows  | x64           |

## Quick Start

### 1. Define a Schema

Every index has a schema that describes what fields your documents contain. Use shorthand strings for common configurations:

```javascript
const index = new EmbeddedIndex('./products', {
  schema: {
    name: 'text',       // full-text searchable, stored
    description: 'text',
    brand: 'keyword',   // exact match, filterable, fast
    price: 'float',     // numeric, fast field for range filters
    year: 'integer',    // integer, fast field
  },
});
```

Or use detailed definitions when you need more control:

```javascript
const index = new EmbeddedIndex('./products', {
  schema: {
    name: { type: 'text', stored: true, indexed: true, analyzer: 'default' },
    brand: { type: 'keyword', stored: true, fast: true },
    price: { type: 'float', stored: true, fast: true },
  },
});
```

### 2. Add Documents

```javascript
await index.add({
  _id: 'product-1',
  name: 'Wireless Headphones',
  description: 'Noise-cancelling over-ear headphones',
  brand: 'AudioCo',
  price: 79.99,
  year: 2024,
});

// Or add many at once
const count = await index.addMany([
  { _id: 'product-2', name: 'USB Microphone', brand: 'SoundPro', price: 49.99, year: 2024 },
  { _id: 'product-3', name: 'Webcam HD', brand: 'VisionTech', price: 39.99, year: 2023 },
]);
console.log(count); // 2
```

### 3. Commit

Documents are queued in memory until you commit. This makes bulk indexing fast — commit once after adding a batch.

```javascript
await index.commit();
// Now documents are searchable and durable on disk
```

### 4. Search

```javascript
const results = await index.search('headphones');
console.log(results.totalHits);           // 1
console.log(results.hits[0].docId);       // "product-1"
console.log(results.hits[0].score);       // BM25 relevance score
```

## Choosing an Index Type

| | `EmbeddedIndex` | `RemoteIndex` |
|---|---|---|
| **Use when** | Search runs in-process | Index lives on another server |
| **Latency** | Microseconds (native) | Network round-trip |
| **Write support** | Full (add, commit, compact) | Full (via HTTP API) |
| **Dependencies** | Native binary (.node) | `fetch` (Node 18+) |
| **Constructor** | `new EmbeddedIndex(path, opts?)` | `new RemoteIndex(baseUrl, indexName, opts?)` |

Both implement the `SearchIndex` interface — all methods return Promises.

## API Reference

### `new EmbeddedIndex(path, options?)`

Opens or creates an index at the given filesystem path.

```javascript
// Create a new index (schema required)
const index = new EmbeddedIndex('./my-index', { schema: { title: 'text' } });

// Open an existing index
const index = new EmbeddedIndex('./my-index');

// With a write key for access control
const index = new EmbeddedIndex('./my-index', {
  schema: { title: 'text' },
  writeKey: 'my-secret-key',
});
```

**Behavior:**

| Schema provided? | Index exists? | Result |
|---|---|---|
| Yes | No | Creates the index |
| Yes | Yes | Opens and validates schema matches |
| No | Yes | Opens the index |
| No | No | Throws an error |

If you provide a schema when reopening an existing index, it's validated against the on-disk schema. A mismatch throws an error — this prevents accidentally writing documents with the wrong field types.

### `index.add(doc)`

Queues a single document for indexing. The `_id` field is used as the document identifier.

```javascript
index.add({ _id: 'doc-1', title: 'Hello World', body: 'My first document' });
```

### `index.addMany(docs)`

Queues multiple documents. Returns the number of documents queued.

```javascript
const count = index.addMany([
  { _id: 'doc-1', title: 'First' },
  { _id: 'doc-2', title: 'Second' },
]);
// count === 2
```

### `index.commit()`

Makes all queued documents durable and searchable. Until you call `commit()`, added documents won't appear in search results.

### `index.search(query)`

Search with a simple string or a full request object.

**String query** — searches across all indexed text fields:

```javascript
const results = index.search('wireless headphones');
```

**Request object** — full control over search behavior:

```javascript
const results = index.search({
  query: 'wireless',
  limit: 20,
  returnStored: true,
  filter: { KeywordEq: { field: 'brand', value: 'AudioCo' } },
});
```

Returns a `SearchResult`:

```javascript
{
  totalHits: 42,            // estimated total matching documents
  hits: [
    {
      docId: 'product-1',   // document ID
      score: 1.23,          // BM25 relevance score
      fields: { ... },      // stored fields (if returnStored: true)
      highlights: { ... },  // highlighted snippets (if requested)
    },
  ],
  nextCursor: '...',        // for pagination (if more results exist)
  aggregations: { ... },    // aggregation results (if requested)
}
```

### `index.compact()`

Merges index segments for better read performance. Call this periodically after many commits.

### `index.close()`

Closes the index and releases native resources. Any subsequent method calls will throw.

### `new RemoteIndex(baseUrl, indexName, options?)`

Connects to a remote searchlite-http server.

```javascript
const { RemoteIndex } = require('searchlite-js');

// Basic connection
const index = new RemoteIndex('http://localhost:8080', 'products');

// With write key for protected indexes
const index = new RemoteIndex('http://localhost:8080', 'products', {
  writeKey: 'my-secret-key',
});

// With custom fetch (for testing or custom transports)
const index = new RemoteIndex('http://localhost:8080', 'products', {
  fetch: myCustomFetch,
});
```

`RemoteIndex` implements the same `SearchIndex` interface as `EmbeddedIndex` — all methods (`add`, `addMany`, `commit`, `compact`, `search`, `close`) work identically. The `close()` method is a no-op since HTTP connections are stateless.

Internally, methods map to [searchlite-http](../searchlite-http) endpoints:

| Method | HTTP Endpoint |
|--------|--------------|
| `add(doc)` | `POST /indexes/:name/bulk` |
| `addMany(docs)` | `POST /indexes/:name/bulk` |
| `commit()` | `POST /indexes/:name/commit` |
| `compact()` | `POST /indexes/:name/compact` |
| `search(query)` | `POST /indexes/:name/search` |

## Schema

### Field Types

| Type | Description | Defaults |
|------|-------------|----------|
| `'text'` | Full-text searchable with BM25 scoring | `stored: true, indexed: true, analyzer: 'default'` |
| `'keyword'` | Exact-match filtering and aggregations | `stored: true, indexed: true, fast: true` |
| `'integer'` | 64-bit integer, range filters | `fast: true, stored: false` |
| `'float'` | 64-bit float, range filters | `fast: true, stored: false` |

### Detailed Field Options

Override any default with the detailed syntax:

```javascript
{
  title: { type: 'text', stored: true, indexed: true, analyzer: 'default' },
  body: { type: 'text', stored: false, indexed: true },  // indexed but not stored
  status: { type: 'keyword', fast: true, stored: false }, // filterable but not returned
  count: { type: 'integer', stored: true, fast: true },   // stored and fast
}
```

**Text field options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `stored` | boolean | `true` | Include in stored fields for retrieval |
| `indexed` | boolean | `true` | Include in full-text index |
| `analyzer` | string | `'default'` | Text analysis pipeline |
| `nullable` | boolean | `false` | Allow null values |

**Keyword field options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `stored` | boolean | `true` | Include in stored fields |
| `indexed` | boolean | `true` | Include in term index |
| `fast` | boolean | `true` | Enable fast-field for filtering and aggregations |
| `nullable` | boolean | `false` | Allow null values |

**Numeric field options (`integer` / `float`):**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `stored` | boolean | `false` | Include in stored fields |
| `fast` | boolean | `true` | Enable fast-field for range filters |
| `nullable` | boolean | `false` | Allow null values |

## Search Options

The `search()` method accepts a request object with these fields:

```javascript
index.search({
  // Required
  query: 'search terms',             // string or structured query object

  // Pagination
  limit: 10,                         // max results to return (default: 10, max: 10000)
  from: 0,                           // offset for pagination
  cursor: '...',                     // cursor from previous result's nextCursor
  searchAfter: [...],                // keyset pagination values

  // Field control
  returnStored: false,               // include stored fields in hits
  returnHits: true,                  // include hits array in response

  // Scoring
  execution: 'wand',                 // scoring algorithm: 'wand', 'bmw', or 'bm25'
  trackTotalHits: true,              // count all matches (slower but accurate)
  explain: false,                    // include score explanations in hits
  profile: false,                    // include query execution profile

  // Filtering & sorting
  filter: { ... },                   // pre-scoring filter (see Filters)
  sort: [{ price: 'asc' }],         // sort by field values

  // Features
  fuzzy: { maxEdits: 1 },            // fuzzy matching
  highlightField: 'body',            // field to highlight
  aggs: { ... },                     // aggregations (see Aggregations)
  collapse: { field: 'brand' },      // deduplicate by field
});
```

## Filters

Filters narrow results without affecting relevance scores. They use PascalCase variant names.

### Keyword Filters

```javascript
// Exact match
{ KeywordEq: { field: 'status', value: 'active' } }

// Match any value in a set
{ KeywordIn: { field: 'color', values: ['red', 'blue', 'green'] } }
```

### Range Filters

```javascript
// Integer range
{ I64Range: { field: 'year', min: 2020, max: 2024 } }

// Float range
{ F64Range: { field: 'price', min: 10.0, max: 99.99 } }
```

### Boolean Combinations

```javascript
// AND — all conditions must match
{ And: [
  { KeywordEq: { field: 'brand', value: 'AudioCo' } },
  { I64Range: { field: 'year', min: 2023, max: 2025 } },
]}

// OR — any condition matches
{ Or: [
  { KeywordEq: { field: 'brand', value: 'AudioCo' } },
  { KeywordEq: { field: 'brand', value: 'SoundPro' } },
]}

// NOT — exclude matches
{ Not: { KeywordEq: { field: 'status', value: 'discontinued' } } }
```

### Nested Filters

For nested document fields:

```javascript
{ Nested: {
  path: 'variants',
  filter: { KeywordEq: { field: 'variants.color', value: 'red' } },
}}
```

## Aggregations

Compute facets and statistics alongside search results. Aggregation types use snake_case `type` fields.

### Terms Aggregation

Count documents by keyword field values:

```javascript
const results = index.search({
  query: 'headphones',
  aggs: {
    brands: {
      type: 'terms',
      field: 'brand',
      size: 10,
    },
  },
});

console.log(results.aggregations.brands);
// { buckets: [{ key: 'AudioCo', doc_count: 5 }, { key: 'SoundPro', doc_count: 3 }] }
```

### Stats Aggregation

Get min, max, average, sum, and count for a numeric field:

```javascript
const results = index.search({
  query: 'headphones',
  aggs: {
    priceStats: {
      type: 'stats',
      field: 'price',
    },
  },
});
```

### Range Aggregation

Bucket documents into numeric ranges:

```javascript
const results = index.search({
  query: '*',
  aggs: {
    priceRanges: {
      type: 'range',
      field: 'price',
      keyed: false,
      ranges: [
        { key: 'cheap', to: 50 },
        { key: 'mid', from: 50, to: 100 },
        { key: 'premium', from: 100 },
      ],
    },
  },
});
```

### Histogram Aggregation

Fixed-width numeric buckets:

```javascript
const results = index.search({
  query: '*',
  aggs: {
    priceHist: {
      type: 'histogram',
      field: 'price',
      interval: 25,
    },
  },
});
```

### Nested Aggregations

Aggregations can be nested to create multi-level facets:

```javascript
const results = index.search({
  query: '*',
  aggs: {
    byBrand: {
      type: 'terms',
      field: 'brand',
      aggs: {
        avgPrice: {
          type: 'stats',
          field: 'price',
        },
      },
    },
  },
});
```

## Structured Queries

For advanced search, pass a structured query object instead of a string. Query types use a snake_case `type` field.

### Multi-Match

Search across multiple fields with boosting:

```javascript
index.search({
  query: {
    type: 'multi_match',
    query: 'wireless noise cancelling',
    fields: [
      { field: 'name', boost: 2.0 },
      { field: 'description' },
    ],
    fuzziness: 'AUTO',
  },
});
```

### Boolean Queries

Combine multiple query clauses:

```javascript
index.search({
  query: {
    type: 'bool',
    must: [
      { type: 'query_string', query: 'headphones' },
    ],
    should: [
      { type: 'term', field: 'brand', value: 'AudioCo' },
    ],
    must_not: [
      { type: 'term', field: 'status', value: 'discontinued' },
    ],
  },
});
```

### Phrase Matching

Match exact phrases with optional slop (word distance):

```javascript
index.search({
  query: {
    type: 'phrase',
    field: 'description',
    terms: ['noise', 'cancelling'],
    slop: 1,
  },
});
```

### Prefix, Wildcard, and Regex

```javascript
// Prefix
index.search({ query: { type: 'prefix', field: 'name', value: 'wire' } });

// Wildcard (? = single char, * = any chars)
index.search({ query: { type: 'wildcard', field: 'name', value: 'head*' } });

// Regex
index.search({ query: { type: 'regex', field: 'name', value: 'head(phone|set)s?' } });
```

## Sorting

Sort results by field values instead of relevance:

```javascript
// Simple ascending
index.search({ query: 'headphones', sort: [{ price: 'asc' }] });

// Multiple sort fields
index.search({
  query: 'headphones',
  sort: [
    { year: 'desc' },
    { price: 'asc' },
  ],
});
```

## Pagination

### Cursor-Based (recommended)

Use cursors for efficient deep pagination:

```javascript
// First page
const page1 = index.search({ query: 'headphones', limit: 10 });

// Next page
const page2 = index.search({
  query: 'headphones',
  limit: 10,
  cursor: page1.nextCursor,
});
```

### Offset-Based

Use `from` for simple offset pagination (less efficient for deep pages):

```javascript
const page3 = index.search({ query: 'headphones', limit: 10, from: 20 });
```

## Fuzzy Search

Allow typos and misspellings:

```javascript
index.search({
  query: 'headphoens',  // typo
  fuzzy: {
    maxEdits: 2,         // allow up to 2 character edits
    prefixLength: 2,     // first 2 chars must match exactly
  },
});
```

## Highlighting

There are two highlighting modes:

**Simple** — use `highlightField` for a quick snippet from a single field:

```javascript
const results = index.search({
  query: 'wireless',
  highlightField: 'description',
});

console.log(results.hits[0].snippet);
// "... <em>Wireless</em> noise-cancelling headphones ..."
```

**Multi-field** — use `highlight` for full control over multiple fields with custom tags:

```javascript
const results = index.search({
  query: 'wireless',
  highlight: {
    fields: {
      name: { pre_tag: '<b>', post_tag: '</b>', fragment_size: 64, number_of_fragments: 1 },
      description: { pre_tag: '<em>', post_tag: '</em>', fragment_size: 160, number_of_fragments: 2 },
    },
  },
});

console.log(results.hits[0].highlights);
// { name: ['<b>Wireless</b> Headphones'], description: ['<em>Wireless</em> noise-cancelling...'] }
```

## Result Collapsing

Deduplicate results by a keyword field (e.g., show one result per brand):

```javascript
const results = index.search({
  query: 'headphones',
  collapse: { field: 'brand' },
});
```

## Write Key Protection

Protect an index with a write key to prevent unauthorized modifications:

```javascript
// Create with write key
const index = new EmbeddedIndex('./protected', {
  schema: { title: 'text' },
  writeKey: 'my-secret',
});

// Reopen — must provide the same write key to write
const index = new EmbeddedIndex('./protected', { writeKey: 'my-secret' });
```

## Error Handling

All errors throw synchronously. Common error scenarios:

```javascript
// Missing index without schema
try {
  new EmbeddedIndex('./nonexistent');
} catch (e) {
  // "index does not exist; provide a schema to create it"
}

// Schema mismatch on reopen
try {
  new EmbeddedIndex('./existing', { schema: { different: 'text' } });
} catch (e) {
  // "schema mismatch: provided schema does not match existing index"
}

// Operations on closed index
const index = new EmbeddedIndex('./my-index');
index.close();
try {
  index.search('hello');
} catch (e) {
  // "index is closed"
}

// Invalid document shape
try {
  index.add('not an object');
} catch (e) {
  // ZodError — documents must be plain objects
}
```

## Complete Example

```javascript
const { EmbeddedIndex } = require('searchlite-js');

// Create an index for a recipe database
const index = new EmbeddedIndex('./recipes', {
  schema: {
    title: 'text',
    ingredients: 'text',
    cuisine: 'keyword',
    prepTime: 'integer',
    rating: 'float',
  },
});

// Index some recipes
index.addMany([
  {
    _id: 'pad-thai',
    title: 'Classic Pad Thai',
    ingredients: 'rice noodles, shrimp, peanuts, bean sprouts, lime',
    cuisine: 'thai',
    prepTime: 30,
    rating: 4.8,
  },
  {
    _id: 'carbonara',
    title: 'Spaghetti Carbonara',
    ingredients: 'spaghetti, eggs, pecorino, guanciale, black pepper',
    cuisine: 'italian',
    prepTime: 25,
    rating: 4.6,
  },
  {
    _id: 'tacos',
    title: 'Fish Tacos',
    ingredients: 'cod, tortillas, cabbage, lime, chipotle mayo',
    cuisine: 'mexican',
    prepTime: 20,
    rating: 4.5,
  },
]);
index.commit();

// Simple text search
const results = index.search('noodles');
console.log(`Found ${results.totalHits} recipes`);

// Search with filter and aggregation
const filtered = index.search({
  query: 'lime',
  filter: { I64Range: { field: 'prepTime', min: 0, max: 30 } },
  returnStored: true,
  aggs: {
    byCuisine: { type: 'terms', field: 'cuisine' },
  },
});

for (const hit of filtered.hits) {
  console.log(`${hit.docId}: score ${hit.score.toFixed(2)}`);
}
console.log('Cuisines:', filtered.aggregations.byCuisine);

index.close();
```

## TypeScript

Full type definitions are included. Import and use with full IntelliSense:

```typescript
import { Index, SearchResult, SearchRequest, SchemaDefinition } from 'searchlite-js';

const schema: SchemaDefinition = {
  title: 'text',
  tag: 'keyword',
};

const index = new EmbeddedIndex('./my-index', { schema });
index.add({ _id: '1', title: 'Hello', tag: 'greeting' });
index.commit();

const results: SearchResult = index.search({ query: 'hello', returnStored: true });
```

## License

MIT
