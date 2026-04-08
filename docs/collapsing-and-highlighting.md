# Field Collapsing and Highlighting

## Field collapsing

Field collapsing groups search results by a keyword field and returns only the top
hit from each group. This is essential when your index contains many documents from
the same source and you want to show diverse results.

**When you'd use collapsing:**
- A news aggregator shows one article per publisher instead of 10 from the same outlet
- A job board shows one listing per company per search
- A marketplace shows one product per seller so no single seller dominates the page

```json
{
  "query": "rust systems programming",
  "collapse": {
    "field": "author",
    "inner_hits": {
      "size": 3,
      "sort": [{ "field": "_score", "order": "desc" }]
    }
  },
  "sort": [
    { "field": "published_at", "order": "desc" },
    { "field": "_score", "order": "desc" }
  ],
  "limit": 10,
  "return_stored": true
}
```

This returns the 10 most recent articles, one per author. Each hit includes up to 3
`inner_hits` -- the other top articles by that same author, so users can expand
"more from this author" in the UI.

The response includes:
- **`total_groups`** -- the number of distinct authors that matched
- **`total_hits_estimate`** -- the total number of matching documents (before collapsing)

The `collapse` field must be a fast keyword field.

---

## Highlighting

Highlighting adds visual emphasis to the matching words in search results, helping
users quickly see **why** a result matched their query. It's the bold text you see
in Google search results or the highlighted snippets in documentation search.

### Single-field highlighting (simple)

The quickest way to add highlights -- returns a single snippet for one field:

```json
{
  "query": "rust search engine",
  "highlight_field": "body",
  "return_stored": true
}
```

Each hit includes a `snippet` field with the matching fragment:
```
"snippet": "...a fast <em>search</em> <em>engine</em> written in <em>Rust</em>..."
```

### Multi-field highlighting (full control)

For richer UIs, configure highlighting per field with custom HTML tags and fragment sizes:

```json
{
  "query": "rust search engine",
  "highlight": {
    "fields": {
      "title": {
        "pre_tag": "<b>",
        "post_tag": "</b>",
        "fragment_size": 80,
        "number_of_fragments": 1
      },
      "body": {
        "pre_tag": "<mark>",
        "post_tag": "</mark>",
        "fragment_size": 160,
        "number_of_fragments": 3
      }
    }
  },
  "return_stored": true
}
```

Each hit includes a `highlights` map:

```json
{
  "highlights": {
    "title": ["<b>Rust</b> <b>Search</b> <b>Engine</b> Guide"],
    "body": [
      "...building a fast <mark>search</mark> <mark>engine</mark> in <mark>Rust</mark>...",
      "...the <mark>Rust</mark> ecosystem provides excellent tooling for <mark>search</mark>..."
    ]
  }
}
```

### How highlighting works

- Uses the field's search analyzer (including synonyms and edge n-grams) to detect matches
- Is phrase-aware -- a phrase query highlights the complete phrase, not individual words
- Centers fragments around the first match in the field
- `fragment_size` controls how many characters surround each match
- `number_of_fragments` controls how many separate snippets are returned

### When to use each approach

| Approach | When to use |
|---|---|
| `highlight_field` (string) | Simple search UIs with one content field |
| `highlight` (object) | Rich UIs with title + body + description, or when you need custom HTML tags per field |

Both can be used in the same request -- `highlight_field` produces a `snippet` field,
while `highlight` produces a `highlights` map. The `highlight_field` approach is the
legacy shorthand and is always available for backward compatibility.
