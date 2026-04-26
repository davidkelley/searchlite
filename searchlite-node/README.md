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

Or pass a full JSON Schema with `searchlite:` vocabulary keywords for complete control:

```javascript
const index = new EmbeddedIndex('./products', {
  schema: {
    type: 'object',
    properties: {
      name: { type: 'string' },
      brand: { type: 'string', 'searchlite:kind': 'keyword' },
      price: { type: 'number', 'searchlite:stored': true },
    },
  },
});
```

Or, if you prefer a single schema for indexing **and** runtime validation **and** TypeScript types,
use the Zod-native authoring path:

```typescript
import { z } from 'zod';
import { EmbeddedIndex, sl } from 'searchlite-js';

const ProductSchema = sl.index(
  z.object({
    id: z.string().uuid(),              // auto-promoted to keyword
    name: z.string(),                    // text (full-text)
    brand: sl.keyword(),                 // keyword (exact match, fast)
    price: sl.float({ stored: true }),
    year: sl.integer({ stored: true }),
  }),
  { docIdField: 'id' },
);

type Product = z.infer<typeof ProductSchema>;

const index = new EmbeddedIndex<Product>('./products', { schema: ProductSchema });
await index.add({
  id: '550e8400-e29b-41d4-a716-446655440000',
  name: 'Wireless Headphones',
  brand: 'AudioCo',
  price: 79.99,
  year: 2024,
});
await index.commit();

const result = await index.search('wireless');
// result.hits[0].fields is typed as Product
```

With the Zod path:
- `add()` / `addMany()` validate documents against the Zod schema before indexing.
- `search()` results are validated and typed against the same schema — no need to pass it again.
- `z.infer<typeof Schema>` gives you the document type for free.

> **All three forms** (shorthand, raw JSON Schema, Zod) compile to the same internal representation and are fully interchangeable. Choose based on ergonomics:
> - Shorthand — fastest to author for simple flat schemas
> - Raw JSON Schema — interop with non-TS clients (sharable as a `schema.json` file)
> - Zod — single source of truth for indexing + validation + TS types; supports nested objects and arrays natively
>
> See [`docs/zod-guide.md`](../docs/zod-guide.md) for the full Zod walkthrough, the type-mapping rules, and migration recipes.

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
| **Dependencies** | Native binary (.node) | `fetch` (Node 20+) |
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

Schemas define the fields in your documents. You can use either the shorthand format
(Node.js only) or the full JSON Schema format with `searchlite:` vocabulary keywords.

### Shorthand Format (Node.js only)

The shorthand maps field names to type strings. `expandSchema()` converts this to
JSON Schema internally.

| Shorthand | JSON Schema Output | Description |
|-----------|-------------------|-------------|
| `'text'` | `{ type: "string" }` | Full-text searchable with BM25 scoring |
| `'keyword'` | `{ type: "string", "searchlite:kind": "keyword" }` | Exact-match filtering and aggregations |
| `'integer'` | `{ type: "integer" }` | 64-bit integer, range filters |
| `'float'` | `{ type: "number" }` | 64-bit float, range filters |

### JSON Schema Format

The canonical format uses standard JSON Schema types with `searchlite:` vocabulary
keywords for search-engine-specific behavior. This format is shared across all
searchlite clients.

```javascript
{
  type: 'object',
  properties: {
    title: { type: 'string' },                                          // text (default)
    body: { type: 'string', 'searchlite:stored': false },               // text, not stored
    status: { type: 'string', 'searchlite:kind': 'keyword' },           // keyword
    count: { type: 'integer', 'searchlite:stored': true },              // integer, stored
    price: { type: 'number' },                                          // float
    notes: { type: ['string', 'null'] },                                // nullable text
  },
}
```

### `searchlite:` Vocabulary Keywords

These keywords extend standard JSON Schema to control search-engine behavior.

**For text fields** (`{ type: "string" }` without `searchlite:kind`):

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `searchlite:stored` | boolean | `true` | Include in stored fields for retrieval |
| `searchlite:indexed` | boolean | `true` | Include in full-text index |
| `searchlite:analyzer` | string | `"default"` | Text analysis pipeline |

**For keyword fields** (`{ type: "string", "searchlite:kind": "keyword" }`):

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `searchlite:stored` | boolean | `true` | Include in stored fields |
| `searchlite:indexed` | boolean | `true` | Include in term index |
| `searchlite:fast` | boolean | `true` | Enable fast-field for filtering and aggregations |

**For numeric fields** (`{ type: "integer" }` or `{ type: "number" }`):

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `searchlite:stored` | boolean | `false` | Include in stored fields |
| `searchlite:fast` | boolean | `true` | Enable fast-field for range filters |

**Nullable fields:** Use a JSON Schema type array to allow null values, e.g.
`{ type: ["string", "null"] }` or `{ type: ["integer", "null"] }`.

**Index-level keywords** (on the root schema object):

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `searchlite:docIdField` | string | `"_id"` | Name of the document ID field |
| `searchlite:analyzers` | array | -- | Custom analyzer definitions |

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

Constructor errors throw synchronously. All other methods return rejected Promises on failure:

```javascript
// Constructor errors throw synchronously
try {
  new EmbeddedIndex('./nonexistent');
} catch (e) {
  // "index does not exist; provide a schema to create it"
}

// Async method errors — use try/await
const index = new EmbeddedIndex('./my-index', { schema: { body: 'text' } });

try {
  await index.add('not an object');
} catch (e) {
  // validation error — documents must be plain objects
}

// Operations on closed index
await index.close();
try {
  await index.search('hello');
} catch (e) {
  // "index is closed"
}

// RemoteIndex HTTP errors
const remote = new RemoteIndex('http://localhost:8080', 'missing');
try {
  await remote.search('hello');
} catch (e) {
  // "searchlite search failed (404): index 'missing' does not exist"
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

## Typed Search with Zod

Pass a [Zod](https://zod.dev) schema as the first argument to `search()` and every hit's `fields` property is validated and fully typed at compile time. No more `as any` casts or manual null checks on search results.

When you pass a schema, `returnStored` is set automatically — you don't need to specify it.

### Basic typed search

```typescript
import { EmbeddedIndex } from 'searchlite-js';
import { z } from 'zod';

const index = new EmbeddedIndex('./products', {
  schema: { name: 'text', brand: 'keyword', price: 'float' },
});

// ... add documents and commit ...

// Define the shape you expect back
const ProductFields = z.object({
  name: z.string(),
  brand: z.string(),
  price: z.number(),
});

// Pass the schema as the first argument — results are typed
const results = await index.search(ProductFields, 'wireless headphones');

for (const hit of results.hits) {
  // hit.fields is { name: string; brand: string; price: number } — fully typed
  console.log(`${hit.fields.name} by ${hit.fields.brand} — $${hit.fields.price}`);
}
```

### Typed search with filters and aggregations

The schema works with structured queries too:

```typescript
const BrandPrice = z.object({
  brand: z.string(),
  price: z.number(),
});

const results = await index.search(BrandPrice, {
  query: 'headphones',
  filter: { F64Range: { field: 'price', min: 20, max: 100 } },
  sort: [{ price: 'asc' }],
  aggs: {
    brands: { type: 'terms', field: 'brand', size: 5 },
  },
});

// Fields are typed, aggregations are available alongside
for (const hit of results.hits) {
  console.log(`${hit.fields.brand}: $${hit.fields.price.toFixed(2)}`);
}
console.log('Top brands:', results.aggregations?.brands);
```

### Transforms and defaults

Zod transforms and defaults run during validation, so you can reshape data as it comes out of the index:

```typescript
const Product = z.object({
  name: z.string().transform((s) => s.toUpperCase()),
  price: z.number(),
  currency: z.string().default('USD'),
});

const results = await index.search(Product, 'headphones');

// Transforms are applied — name is uppercased, currency defaults to "USD"
console.log(results.hits[0].fields.name);     // "WIRELESS HEADPHONES"
console.log(results.hits[0].fields.currency);  // "USD"
```

### Optional and partial fields

Use `.optional()` for fields that may not be stored on every document:

```typescript
const FlexProduct = z.object({
  name: z.string(),
  brand: z.string().optional(),
  price: z.number().optional(),
});

const results = await index.search(FlexProduct, 'headphones');
// hit.fields.brand is string | undefined — TypeScript knows
```

### Validation errors

If a document's stored fields don't match your schema, `search()` throws with a clear error that includes the document ID and field path:

```typescript
const StrictFields = z.object({
  name: z.string(),
  price: z.number(), // will fail if price is missing or wrong type
});

try {
  await index.search(StrictFields, 'headphones');
} catch (e) {
  // Error: 'Invalid fields on hit 0 (docId: "product-1"):\n Required at "price"'
}
```

### End-to-end example: typed product search

```typescript
import { EmbeddedIndex } from 'searchlite-js';
import { z } from 'zod';

// Schema for the index
const index = new EmbeddedIndex('./shop', {
  schema: {
    name: 'text',
    description: 'text',
    category: 'keyword',
    price: 'float',
    inStock: 'integer',
  },
});

// Index a product catalog
await index.addMany([
  { _id: 'sku-1', name: 'Wireless Earbuds', description: 'Bluetooth 5.3 with ANC', category: 'audio', price: 59.99, inStock: 142 },
  { _id: 'sku-2', name: 'Studio Headphones', description: 'Over-ear open-back for mixing', category: 'audio', price: 199.99, inStock: 38 },
  { _id: 'sku-3', name: 'USB Microphone', description: 'Condenser mic for podcasting', category: 'microphones', price: 89.99, inStock: 67 },
  { _id: 'sku-4', name: 'Webcam 4K', description: 'Ultra HD with autofocus', category: 'video', price: 129.99, inStock: 0 },
]);
await index.commit();

// Zod schema for validated results
const CatalogItem = z.object({
  name: z.string(),
  category: z.string(),
  price: z.number(),
  inStock: z.number().transform((n) => n > 0),  // transform count to boolean
});

type CatalogItem = z.infer<typeof CatalogItem>;
// { name: string; category: string; price: number; inStock: boolean }

// Typed search with filter and facets
const results = await index.search(CatalogItem, {
  query: { type: 'multi_match', query: 'audio', fields: [{ field: 'name' }, { field: 'description', boost: 0.5 }] },
  filter: { F64Range: { field: 'price', min: 0, max: 150 } },
  aggs: {
    categories: { type: 'terms', field: 'category' },
    priceStats: { type: 'stats', field: 'price' },
  },
  highlightField: 'description',
});

for (const hit of results.hits) {
  const { name, category, price, inStock } = hit.fields;
  const badge = inStock ? 'In Stock' : 'Out of Stock';
  console.log(`[${badge}] ${name} (${category}) — $${price}`);
  if (hit.snippet) console.log(`  ${hit.snippet}`);
}

await index.close();
```

## TypeScript

Full type definitions are included. Import and use with full IntelliSense:

```typescript
import { EmbeddedIndex, RemoteIndex, SchemaDefinition } from 'searchlite-js';
import type { SearchIndex, SearchResult, TypedSearchResult } from 'searchlite-js';
import { z } from 'zod';

// Both index types implement the same SearchIndex interface
const schema: SchemaDefinition = { title: 'text', tag: 'keyword' };
const embedded = new EmbeddedIndex('./my-index', { schema });
await embedded.add({ _id: '1', title: 'Hello', tag: 'greeting' });
await embedded.commit();

// Untyped search — returns SearchResult with optional fields
const basic: SearchResult = await embedded.search({ query: 'hello', returnStored: true });

// Typed search — returns TypedSearchResult<T> with validated fields
const Fields = z.object({ title: z.string(), tag: z.string() });
const typed: TypedSearchResult<z.infer<typeof Fields>> = await embedded.search(Fields, 'hello');
// typed.hits[0].fields.title — string, guaranteed

// RemoteIndex works the same way
const remote = new RemoteIndex('http://localhost:8080', 'my-index');
const remoteTyped = await remote.search(Fields, 'hello');
```

## License

MIT
