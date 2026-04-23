# Zod-native product catalog example

A more advanced Zod-native example showing how to index a product catalog
with nested metadata and multi-valued variants — all defined in one Zod
schema.

## Highlights

- **Nested objects** (`meta: z.object({...})`) for structured metadata.
- **Arrays of objects** (`variants: z.array(z.object({...}))`) for multi-
  valued nested fields.
- Numeric fields explicitly opt into `stored: true` so they're returned in
  `hit.fields`.
- `z.boolean()` would not compile — we model booleans as `z.enum(["yes","no"])`
  because searchlite-core has no native boolean kind (the compiler points
  this out with a remediation hint).

> Dense embedding vectors (`sl.vector(...)`) require the `vectors` feature
> flag at compile time — see `docs/vectors.md`. This example uses the
> default build so it runs out of the box. Adding a vector field is a
> one-line change once the feature is enabled.

## Run

```bash
npm install
npm run example
```

Expected output:

```
Products matching "headphones": 1
  • Wireless Noise-Cancelling Headphones — $149.99 (weight 0.28kg)

AudioCo products: 1
  • Wireless Noise-Cancelling Headphones

Products under $100: 1
  • USB Condenser Microphone ($89.00)

Rejected by Zod (type error):
Invalid document:
✖ Invalid input: expected number, received string
```

## What to look at in the code

- The single `ProductSchema` drives the native index, Zod runtime validation,
  and the `Product` TypeScript type (`z.infer<typeof ProductSchema>`).
- `stored: true` on numerics ensures `price`, `weightKg`, and variant prices
  are returned in search results (numerics default to `stored: false` to keep
  the doc store small).
- `inStock: z.enum(["yes","no"])` demonstrates the boolean workaround.

See [`docs/zod-guide.md`](../../docs/zod-guide.md) for the full walkthrough.
