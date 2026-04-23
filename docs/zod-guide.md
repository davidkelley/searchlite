# Zod-native schema authoring

A walkthrough for defining a searchlite index with a [Zod](https://zod.dev)
schema. Before you read this, skim [`schema.md`](./schema.md) for the
underlying model — searchlite's native engine speaks JSON Schema with
`searchlite:` vocabulary keywords, and the Zod authoring path compiles to
exactly that. Nothing changes on the engine side; the Zod path is a
TypeScript-first authoring surface.

## Why Zod

Without Zod you maintain two schemas for any TypeScript project with typed
search:

1. The **index schema** — either shorthand `{title: "text"}` or a raw JSON
   Schema file. Consumed by the engine.
2. A **Zod validator** passed per-call to `search()` — so the returned fields
   can be typed and validated.

These have to be kept in lock-step by hand. Add a field in one place, remember
to add it in the other, hope no one forgets.

With Zod-native authoring, one schema does three jobs:

- The native engine gets its index definition (compiled to JSON Schema).
- Every document is validated against the same schema on `add()` / `addMany()`.
- Every search result is validated against the same schema on `search()` —
  and the return type is inferred automatically.

`z.infer<typeof Schema>` gives you the document type; you type the class
generic (`new EmbeddedIndex<Product>(...)`) and the full pipeline
(`add`, `addMany`, `search.fields`) is typed end-to-end.

## Getting started

Install:

```bash
npm install searchlite-js zod
```

Here is a complete, runnable example — copy this into a `.ts` file and run
with `tsx`:

```typescript
import { z } from "zod";
import { sl } from "searchlite-js";

// 1. Define your schema once.
const BlogSchema = sl.index(
  z.object({
    id:       z.string().uuid(),          // auto-promoted to keyword
    title:    z.string(),                  // text (full-text search)
    slug:     sl.keyword(),               // keyword (exact match)
    status:   z.enum(["draft", "published", "archived"]),
    views:    sl.integer(),               // stored by default in Zod path
    authorId: z.string().uuid(),
  }),
  { docIdField: "id" },
);

// 2. Create a typed index — T is inferred from the schema.
const index = sl.embedded("./data/blog", BlogSchema);

// 3. Insert — typed AND validated (wrong types throw immediately).
await index.add({
  id: "550e8400-e29b-41d4-a716-446655440000",
  title: "Hello world",
  slug: "hello-world",
  status: "published",
  views: 0,
  authorId: "550e8400-e29b-41d4-a716-446655440001",
});
await index.commit();

// 4. Search — hit.fields is auto-typed and auto-validated.
const result = await index.search("hello");
console.log(result.hits[0].fields?.title); // "Hello world"

await index.close();
```

The `sl.embedded()` factory infers the document type from the schema — no
`<User>` annotation, no `z.infer<typeof ...>`, no options-bag ceremony. For a
remote index, use `sl.remote(url, name, schema)` instead.

## Primitives

### Strings: text vs keyword

`z.string()` defaults to **text** (analyzed, full-text searchable):

```typescript
z.object({
  body: z.string(),   // text field with default analyzer
})
```

String refinements that represent identifiers auto-promote to **keyword**
(exact match, fast) — because `.uuid()`, `.cuid()`, `.cuid2()`, `.ulid()`,
`.nanoid()`, `.email()`, and `.url()` values are never useful as full-text:

```typescript
z.object({
  id:    z.string().uuid(),     // -> keyword (auto)
  email: z.string().email(),     // -> keyword (auto)
  href:  z.string().url(),       // -> keyword (auto)
})
```

Override with the explicit helper when you do want a refinement to stay as
text (e.g., to partially-match email local parts):

```typescript
z.object({
  bio: sl.text(z.string().email()),  // -> text (explicit wins)
})
```

For the exact-match form, use `sl.keyword()`:

```typescript
z.object({
  category: sl.keyword(),                     // simple
  tag:      sl.keyword({ fast: false }),      // opt out of fast columnar store
  label:    z.enum(["new", "sale", "featured"]),  // enums are keyword by default
})
```

### Numbers: integer vs float

`z.number()` is `float` by default. Use `.int()` (or `sl.integer()`) for
integers:

```typescript
z.object({
  price: z.number(),               // float, stored by default
  year:  z.number().int(),         // integer, stored by default
  age:   sl.integer(),             // integer (equivalent; also attaches metadata)
  score: sl.float({ stored: false }),  // opt out of stored if you only need fast
})
```

In the Zod path, numeric fields default to `stored: true` and `fast: true`
— so any field you declare in your schema will round-trip through search
results. This differs from the shorthand path (where numerics default to
`stored: false`) because Zod users who explicitly declare a field have a
strong expectation it will appear in `hit.fields`. Opt out with
`sl.integer({ stored: false })` or `sl.float({ stored: false })` if you
only need the fast columnar store for filtering/aggregations.

### Literals and enums

String literals and enums become keyword fields:

```typescript
z.object({
  version: z.literal("v1"),
  status:  z.enum(["draft", "published", "archived"]),
})
```

Numeric literals become `integer` or `float` based on value:

```typescript
z.object({
  legacy_version: z.literal(1),      // integer
  pi_approx:      z.literal(3.14),   // float
})
```

Boolean literals are rejected — the core engine has no boolean kind. Use an
enum instead:

```typescript
z.object({
  // z.literal(true)                       // ✗ error
  is_public: z.enum(["true", "false"]),     // ✓ keyword
})
```

## Nested objects and arrays

Unlike the flat shorthand format, the Zod path supports arbitrary nesting:

```typescript
const ProductSchema = sl.index(
  z.object({
    id: z.string().uuid(),
    name: z.string(),
    meta: z.object({
      sku:    sl.keyword(),
      weight: sl.float({ stored: true }),
    }),
    variants: z.array(
      z.object({
        color: sl.keyword(),
        price: sl.float({ stored: true }),
      }),
    ),
  }),
  { docIdField: "id" },
);
```

Rules:
- `z.object({...})` compiles to a nested object field.
- `z.array(z.object({...}))` compiles to a multi-valued nested field.
- `z.array(<primitive>)` is **not supported** — the core engine requires
  nested arrays to contain objects (or numbers for vectors). Wrap primitives
  in a named object key: `z.array(z.object({ value: sl.keyword() }))`.

## Vectors

Dense embedding vectors use `sl.vector()`:

```typescript
const DocSchema = sl.index(
  z.object({
    id:        z.string().uuid(),
    body:      z.string(),
    embedding: sl.vector({ dim: 768, metric: "Cosine" }),
  }),
);
```

The runtime Zod type is `z.ZodArray<z.ZodNumber>` with a length constraint
equal to `dim`, so malformed vectors are caught on `add()`.

Options:
- `dim` (required) — vector dimensionality.
- `metric` (required) — `"Cosine"` or `"L2"`.
- `hnsw?` — optional HNSW index config: `{ m?: number, efConstruction?: number }`.

## Optional, nullable, default

These wrap any inner type:

```typescript
z.object({
  subtitle:    z.string().optional(),         // may be omitted
  deleted_at:  z.string().nullable(),         // may be null
  visibility:  z.enum(["public", "private"]).default("private"),
})
```

- `.optional()` makes the field absence-tolerant (the field may be missing on
  insert).
- `.nullable()` emits `type: [T, "null"]` — the field must be present but may
  be null.
- `.default(v)` applies the default during Zod validation; the index field is
  unchanged.

## Validation behavior

### On insert

When the index was constructed with a Zod schema, `add()` and `addMany()`
validate each document against that schema **before** it hits the native
engine:

```typescript
await index.add({
  id: "not-a-uuid",   // throws ZodError
  title: "hi",
  ...
});
// Invalid `id`: expected uuid string
```

For `addMany`, the error message includes the index of the failing document:

```
Invalid documents[2]:
✖ Invalid input: expected uuid string
  → at id
```

If the index has no Zod schema (shorthand / raw JSON Schema paths), the
existing basic validation (`DocumentSchema` — must be a non-null object) is
applied instead.

### On search

When the index has a Zod schema, search results are validated automatically.
You do NOT need to pass the schema again:

```typescript
const r = await index.search("hello");
// r.hits[0].fields is validated against BlogSchema and typed as BlogPost
```

If the search result fails validation (e.g., a schema was changed without
reindexing), the error names the failing hit:

```
Invalid fields on hit 0 (docId: "abc"):
✖ Invalid input: expected string
  → at title
```

### Per-call schema override

Passing a schema to `search()` explicitly takes precedence over the stored
one for that call — useful for projecting to a subset schema:

```typescript
const Subset = z.object({ title: z.string(), slug: z.string() });
const r = await index.search(Subset, "hello");
// r.hits[0].fields: { title: string, slug: string }
```

## Migrating

### From the shorthand format

```typescript
// Before:
new EmbeddedIndex("./idx", {
  schema: { title: "text", tag: "keyword", year: "integer" },
});

// After:
import { z } from "zod";
import { sl } from "searchlite-js";

const Schema = sl.index(z.object({
  title: z.string(),
  tag:   sl.keyword(),
  year:  sl.integer(),
}));

new EmbeddedIndex("./idx", { schema: Schema });
```

The compiled native behavior is identical. Round-trip parity is verified by
`test/zod-roundtrip.test.mjs` for every rule pair.

### From raw JSON Schema

```typescript
// Before — schema.json lives on disk:
new EmbeddedIndex("./idx", { schema: require("./schema.json") });

// After — equivalent Zod schema:
const Schema = sl.index(
  z.object({
    title: z.string(),
    meta:  z.object({
      sku:    sl.keyword(),
      weight: sl.float({ stored: true }),
    }),
    items: z.array(z.object({
      name:     z.string(),
      quantity: sl.integer({ stored: true }),
    })),
  }),
  { docIdField: "doc_id" },
);

new EmbeddedIndex("./idx", { schema: Schema });
```

You can migrate incrementally — the two paths are interchangeable and can
coexist in a codebase.

### Keeping the old schema available

`compileZodSchema()` is exported, so you can convert a Zod schema into the
same JSON Schema that the raw path would accept:

```typescript
import { compileZodSchema } from "searchlite-js";
import { writeFileSync } from "node:fs";

const Schema = sl.index(z.object({ ... }));
writeFileSync("schema.json", JSON.stringify(compileZodSchema(Schema), null, 2));
```

Useful for sharing a schema with non-TS consumers (the HTTP server, CLI, WASM
runtime) or for committing a checked-in canonical version alongside your
source.

## Performance notes

### `addMany` throughput

Zod validation is per-document. For very large bulk inserts (tens of thousands
of documents per call), Zod parse cost may dominate. If you hit this wall:

- Pre-validate off the hot path and insert in trusted chunks.
- For a given insert batch, parse once and reuse a `.safeParse()` result.
- If you know the data is clean (e.g., it came from another system that
  already validated), use the shorthand / raw JSON Schema path to skip the
  Zod validation step.

A dedicated `skipValidation` option is not currently exposed; track requests
for one in `searchlite/issues`.

### Compile cost

The Zod → JSON Schema compile happens **once** at `new EmbeddedIndex(...)`
construction. Subsequent operations don't re-compile. You can cache the
compiled output for multiple constructions:

```typescript
const compiled = compileZodSchema(Schema);
// Reuse across many index constructions:
const a = new EmbeddedIndex("./a", { schema: compiled });
```

But in practice construction is rare enough (once per process, usually) that
compile cost is negligible.

## Advanced: custom registry metadata

Under the hood, `sl.*` helpers attach metadata via Zod's v4 metadata registry.
You can attach the same metadata manually — useful for programmatic schema
construction:

```typescript
import { SearchliteFieldRegistry } from "searchlite-js";

const s = z.string().register(SearchliteFieldRegistry, {
  kind: "keyword",
  fast: false,
});
// Equivalent to: sl.keyword({ fast: false })
```

The registry is also the only way to attach a `.meta({...})` entry the
compiler reads (see `schema.md` → metadata override precedence).

## Troubleshooting

**"schema must be wrapped with `sl.index(...)`"** — you passed a bare
`z.object()` to the constructor. Wrap it: `sl.index(z.object({...}))`.

**"field `X.Y.Z` — unsupported Zod type ZodUnion"** — searchlite has no
kind for the construct you used. Read the error's suggestion; common fix is
lifting the discriminator to the parent object.

**"Invalid documents[N]: ..."** — `addMany` validation caught a bad document.
The path (e.g., `authorId`) points at the field that failed.

**"Invalid fields on hit 0 (docId: ...)"** — a search result's fields don't
match the schema. Usually means the stored data is from a previous
(incompatible) schema version. Either reindex or use a per-call override
schema that tolerates the legacy shape.

**Types work in my IDE but `tsc` fails** — make sure Zod is at version
`>=4.3.6 <5`. The compiler depends on `_def.type` discriminators which moved
between minor versions.

## See also

- [`docs/schema.md`](./schema.md) — the canonical JSON Schema format and
  `searchlite:` vocabulary reference.
- [`examples/zod-blog/`](../examples/zod-blog/) — minimal runnable example.
- [`examples/zod-products/`](../examples/zod-products/) — nested variants and
  vectors.
- [`examples/zod-events/`](../examples/zod-events/) — epoch-ms timestamps.
