# Zod-native event log example

Demonstrates the **"no date kind"** pattern. Searchlite-core has no native
date type — `z.date()` hard-errors at compile time. This example shows the
recommended workaround: model timestamps as epoch-ms integers
(`z.number().int()`) and convert to/from `Date` at your application boundary.

## Why no date kind?

The core engine stores integers, floats, strings, and nested objects. Dates
are not a first-class kind because:

- Range filters on integers (`I64Range`) are fast, exact, and work
  identically for milliseconds-since-epoch.
- Multiple timestamp representations (ISO string vs unix seconds vs
  milliseconds) are best normalized at the application layer where the
  tradeoffs are clearer.

If the compiler encountered `z.date()` it would happily accept `Date`
objects at the Zod layer but produce incorrect query behavior at the
engine layer (you can't do a range filter on a `Date` object). The hard
error is a feature — it forces you to choose a concrete representation.

## Highlights

- Uses `z.number().int()` (via `sl.integer()`) with a branded type
  (`EpochMs`) for type-safe timestamp handling.
- Range filter (`I64Range`) for "last N seconds" queries — works because
  timestamps are plain integers.
- Helper functions `toEpoch(Date)` / `fromEpoch(ms)` bridge to the JS Date
  API at the application boundary.

## Run

```bash
npm install
npm run example
```

Expected output:

```
Events matching "timeout": 1
  • [error] error at <timestamp>: Payment gateway timeout

Events in the last 45s: 2
  • <timestamp> — purchase (info)
  • <timestamp> — error (error)

Zod caught bad timestamp: Invalid document:
```

## What to look at in the code

- `EpochMs = sl.integer({ stored: true }).brand<"EpochMs">()` — branded
  integer so you can't accidentally confuse `occurredAt` with arbitrary
  numbers. The brand is transparent to the compiler (it compiles as a
  regular integer field).
- Inserts convert `Date → epoch-ms` via `toEpoch`. Searches convert back.
- `I64Range` filters work directly on the integer field.

See [`docs/zod-guide.md`](../../docs/zod-guide.md) for the full walkthrough
and a side-by-side rule table showing which Zod types are supported /
unsupported.
