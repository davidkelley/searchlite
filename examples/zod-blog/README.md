# Zod-native blog example

A minimal runnable example of using Searchlite with a single Zod schema for
indexing, validation, and typed search results.

## Highlights

- One schema (`BlogSchema`) validates documents on insert and types search
  results — no per-call schema needed.
- `z.infer<typeof BlogSchema>` gives the `BlogPost` type for free.
- Auto-promotion: `z.string().uuid()` indexes as `keyword` automatically.
- Explicit `sl.keyword()` for the `slug` field to opt into exact-match.
- Nested arrays of objects for `tags`.

## Run

```bash
npm install
npm run example
```

You should see:

```
Search "searchlite" → 1 hit(s):
  • Getting started with Searchlite (slug=getting-started, views=42)

Published posts mentioning "zod" → 1:
  • Zod-native schemas

Invalid doc rejected at add-time: Invalid document:
```

## What to look at in the code

- [`index.ts`](./index.ts) — the whole thing, annotated.
- The schema uses `sl.index(...)` at the root; that's how searchlite
  recognizes a Zod-authored index.
- The class generic `EmbeddedIndex<BlogPost>` flows through to
  `add(doc: BlogPost)` and `search()` returns `SearchResult<BlogPost>`.

See [`docs/zod-guide.md`](../../docs/zod-guide.md) for a full walkthrough.
