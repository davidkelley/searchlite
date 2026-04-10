# Filters

Filters narrow search results without affecting relevance scores. While queries answer
"how well does this document match?", filters answer "should this document be included
at all?" -- and they're fast because they operate on columnar fast fields rather than
the inverted index.

**When to use filters:**
- An e-commerce site lets users narrow by category, price range, and brand
- A job board filters by location, salary range, and employment type
- A content platform restricts results to a specific language or publication date

Filters require the field to have `"fast": true` in the [schema](schema.md).

---

## Basic keyword equality

Match documents where a keyword field has an exact value. Use this for category
dropdowns, language selectors, or status filters.

*"Show only English-language articles:"*

```json
{ "filter": { "KeywordEq": { "field": "lang", "value": "en" } } }
```

## Keyword membership (`IN`)

Match documents where a keyword field equals any of several values. Use this for
multi-select filters.

*"Show articles in English or French:"*

```json
{ "filter": { "KeywordIn": { "field": "lang", "values": ["en", "fr"] } } }
```

## Numeric ranges

Filter by integer or floating-point ranges. Ranges are inclusive on both ends.

*"Products between $20 and $100 with a rating of 4.0 or above:"*

```json
{
  "filter": {
    "And": [
      { "I64Range": { "field": "price_cents", "min": 2000, "max": 10000 } },
      { "F64Range": { "field": "rating", "min": 4.0, "max": 5.0 } }
    ]
  }
}
```

## Boolean combinators

Combine filters with `And`, `Or`, and `Not` to build complex conditions.

*"Electronics under $50, OR anything on sale:"*

```json
{
  "filter": {
    "Or": [
      {
        "And": [
          { "KeywordEq": { "field": "category", "value": "electronics" } },
          { "I64Range": { "field": "price_cents", "min": 0, "max": 5000 } }
        ]
      },
      { "KeywordEq": { "field": "on_sale", "value": "true" } }
    ]
  }
}
```

*"Everything except archived items:"*

```json
{ "filter": { "Not": { "KeywordEq": { "field": "status", "value": "archived" } } } }
```

## Multi-valued fields

If a document has an array value (e.g., `tags: ["rust", "search"]`) and the field
is a fast keyword field, any single value can satisfy the filter.

*"Articles tagged with 'rust':"*

```json
{ "filter": { "KeywordEq": { "field": "tags", "value": "rust" } } }
```

This matches even if the document has other tags too.

---

## Nested object filters

Nested filters are for arrays of objects where you need to match conditions **within
the same object**. Without nested filters, conditions are evaluated across all objects
independently, which can produce false matches.

### Single-level nesting

Consider a product with multiple reviews. You want products where **a single review**
was written by "alice" AND rated 5 stars -- not just any product reviewed by alice
plus any product with a 5-star review.

Schema excerpt:

```json
{
  "review": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "author": { "type": "string", "searchlite:kind": "keyword" },
        "rating": { "type": "integer" }
      }
    }
  }
}
```

*"Products with a review by alice that has a rating of exactly 5:"*

```json
{
  "filter": {
    "And": [
      { "Nested": { "path": "review", "filter": { "KeywordEq": { "field": "author", "value": "alice" } } } },
      { "Nested": { "path": "review", "filter": { "I64Range": { "field": "rating", "min": 5, "max": 5 } } } }
    ]
  }
}
```

Both conditions must match **the same** review object.

### Deeply nested hierarchy

Nested filters compose for multi-level hierarchies. A blog post has comments, and
each comment can have replies.

*"Posts where bob commented and his comment has a reply tagged 'helpful':"*

```json
{
  "filter": {
    "And": [
      {
        "Nested": {
          "path": "comment",
          "filter": { "KeywordEq": { "field": "author", "value": "bob" } }
        }
      },
      {
        "Nested": {
          "path": "comment",
          "filter": {
            "Nested": {
              "path": "reply",
              "filter": { "KeywordEq": { "field": "tag", "value": "helpful" } }
            }
          }
        }
      }
    ]
  }
}
```

The inner `Nested` is scoped to replies belonging to the same comment that matched
the outer `Nested`.

### Mixed top-level and nested filters

You can freely combine top-level field filters with nested filters.

*"Articles from 2020-2025 that have a comment by alice:"*

```json
{
  "filter": {
    "And": [
      { "I64Range": { "field": "year", "min": 2020, "max": 2025 } },
      {
        "Nested": {
          "path": "comment",
          "filter": { "KeywordEq": { "field": "author", "value": "alice" } }
        }
      }
    ]
  }
}
```

---

## Tips

- Mark every field you want to filter or sort on with `"fast": true` in the schema.
- For nested filters, wrap each condition in its own `Nested` block to enforce per-object binding.
- Stored nested fields preserve their original structure in results; unstored fields are omitted.
- Filters are applied after scoring, so they don't affect relevance rankings -- they only include/exclude documents.
