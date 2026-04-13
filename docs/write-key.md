# Write Keys

Write keys protect an index from unauthorized modifications. When a write key is set
during index creation, every subsequent write operation (commit, compact, merge) must
provide the same key, or the operation is rejected.

**When you'd use write keys:**
- Running the HTTP server on a shared network where you want to prevent accidental
  writes from misconfigured clients
- Multi-tenant deployments where different services manage different indexes
- Adding a safety layer against scripts or tools accidentally modifying production data

Write keys are **not** a substitute for network-level access control. They protect
against accidental misuse, not determined attackers. Always front the HTTP service
with a proxy for real authentication.

---

## Enabling the feature

Write keys require the `write-key` feature flag at compile time:

```toml
[dependencies]
searchlite-core = { version = "0.5", features = ["write-key"] }
```

The CLI and FFI crates enable this feature automatically. If you're using
`searchlite-core` directly, you need to opt in.

---

## Creating an index with a write key

### Rust API

```rust
let index = Index::create_with_write_key(
    &path,
    schema,
    opts,
    Some("my-secret-key"),
)?;
```

The key is hashed with Argon2id and the hash is stored in the manifest. The plaintext
key is never persisted.

### HTTP API

```bash
curl -XPOST http://localhost:8080/indexes/secure/init \
  -H 'Content-Type: application/json' \
  -H 'X-Searchlite-Write-Key: my-secret-key' \
  --data-binary @schema.json
```

### CLI

Every write-capable CLI command accepts `--write-key <KEY>`:

```bash
# Create a protected index
searchlite init /tmp/secure-idx schema.json --write-key "my-secret-key"

# Every write command from here on must pass the same key
searchlite add    /tmp/secure-idx docs.jsonl --write-key "my-secret-key"
searchlite commit /tmp/secure-idx            --write-key "my-secret-key"
searchlite delete /tmp/secure-idx ids.txt    --write-key "my-secret-key"
searchlite compact /tmp/secure-idx           --write-key "my-secret-key"

# Read commands never need it
searchlite search  /tmp/secure-idx -q "example"
searchlite inspect /tmp/secure-idx
```

Supplying the wrong key (or forgetting it) fails with an authorization error
and no changes are applied.

---

## Using a write-key protected index

Once set, the key must be provided for all write operations:

### Rust API

```rust
// Writer
let mut writer = index.writer_with_key(Some("my-secret-key"))?;
writer.add_document(&doc)?;
writer.commit()?;

// Compact
index.compact_with_key(Some("my-secret-key"))?;

// Merge specific segments
index.merge_segments(&segment_ids, Some("my-secret-key"))?;
```

Calling `index.writer()` (without a key) on a key-protected index returns an error.

### HTTP API

Include the `X-Searchlite-Write-Key` header on write endpoints:

```bash
curl -XPOST http://localhost:8080/indexes/secure/add \
  -H 'X-Searchlite-Write-Key: my-secret-key' \
  -H 'Content-Type: application/x-ndjson' \
  --data-binary @docs.ndjson

curl -XPOST http://localhost:8080/indexes/secure/commit \
  -H 'X-Searchlite-Write-Key: my-secret-key'
```

**Read operations (search, mget, inspect, stats) never require a write key.**

---

## How it works

1. On index creation, the write key is hashed using **Argon2id** (a memory-hard KDF
   designed to resist brute-force attacks). The salt, hash, and KDF parameters are
   stored in the manifest.

2. An **HMAC-SHA256 binding** is computed from the key and the index UUID. This binding
   is stored alongside segments and WAL entries to detect tampering.

3. On every write operation, the provided key is verified against the stored hash
   using constant-time comparison (preventing timing attacks). The binding is also
   verified to ensure the operation targets the correct index.

## Tuning Argon2 parameters

The default Argon2 memory cost is 64 MiB, which takes ~100ms on modern hardware.
For environments where this is too slow (high-throughput ingest) or too fast
(security-critical deployments), override via environment variable:

```bash
# Lower memory cost for faster key derivation (less secure)
SEARCHLITE_WRITE_KEY_M_COST_KIB=16384 searchlite init ...

# Higher memory cost for stronger key derivation
SEARCHLITE_WRITE_KEY_M_COST_KIB=262144 searchlite init ...
```

This only affects key derivation during index creation. The parameters are stored in
the manifest and used for verification thereafter.
