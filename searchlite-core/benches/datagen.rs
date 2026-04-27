use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use searchlite_core::api::types::{Document, KeywordField, NumericField, Schema, TextField};

/// Fixed vocabulary of ~200 common English words for generating realistic text.
const VOCABULARY: &[&str] = &[
  "the",
  "quick",
  "brown",
  "fox",
  "jumps",
  "over",
  "lazy",
  "dog",
  "search",
  "engine",
  "index",
  "query",
  "document",
  "field",
  "term",
  "score",
  "result",
  "filter",
  "aggregate",
  "bucket",
  "range",
  "histogram",
  "keyword",
  "text",
  "numeric",
  "nested",
  "boolean",
  "match",
  "phrase",
  "prefix",
  "wildcard",
  "fuzzy",
  "highlight",
  "suggest",
  "complete",
  "analysis",
  "tokenize",
  "stem",
  "normalize",
  "segment",
  "commit",
  "merge",
  "compact",
  "reader",
  "writer",
  "schema",
  "mapping",
  "setting",
  "cluster",
  "node",
  "shard",
  "replica",
  "primary",
  "backup",
  "restore",
  "snapshot",
  "monitor",
  "health",
  "status",
  "performance",
  "benchmark",
  "throughput",
  "latency",
  "memory",
  "disk",
  "network",
  "compute",
  "storage",
  "cache",
  "buffer",
  "queue",
  "stream",
  "batch",
  "bulk",
  "scroll",
  "page",
  "sort",
  "order",
  "ascending",
  "descend",
  "limit",
  "offset",
  "cursor",
  "token",
  "session",
  "context",
  "scope",
  "global",
  "local",
  "remote",
  "distributed",
  "parallel",
  "concurrent",
  "async",
  "sync",
  "block",
  "release",
  "acquire",
  "lock",
  "atomic",
  "fence",
  "barrier",
  "channel",
  "message",
  "signal",
  "event",
  "handler",
  "callback",
  "closure",
  "iterator",
  "collect",
  "map",
  "reduce",
  "fold",
  "scan",
  "chain",
  "zip",
  "enumerate",
  "take",
  "skip",
  "window",
  "chunk",
  "split",
  "join",
  "format",
  "parse",
  "serialize",
  "encode",
  "decode",
  "compress",
  "extract",
  "transform",
  "convert",
  "validate",
  "verify",
  "check",
  "test",
  "assert",
  "expect",
  "require",
  "ensure",
  "guarantee",
  "promise",
  "future",
  "await",
  "spawn",
  "task",
  "thread",
  "process",
  "system",
  "kernel",
  "driver",
  "module",
  "package",
  "crate",
  "library",
  "framework",
  "platform",
  "layer",
  "stack",
  "heap",
  "pool",
  "arena",
  "allocator",
  "garbage",
  "collect",
  "reference",
  "pointer",
  "slice",
  "array",
  "vector",
  "list",
  "tree",
  "graph",
  "table",
  "entry",
  "record",
  "row",
  "column",
  "cell",
  "value",
  "data",
  "information",
  "knowledge",
  "insight",
  "report",
  "summary",
  "detail",
  "overview",
  "review",
  "update",
  "create",
  "delete",
  "modify",
  "insert",
  "remove",
  "append",
  "prepend",
  "replace",
  "swap",
  "rotate",
  "shuffle",
  "sample",
  "random",
  "seed",
  "generate",
  "produce",
  "consume",
  "publish",
  "subscribe",
  "notify",
  "observe",
  "watch",
  "listen",
  "respond",
  "accept",
  "reject",
  "approve",
  "deny",
  "grant",
  "revoke",
  "allow",
];

/// Fixed set of ~50 categories for keyword field (Zipfian distribution).
const CATEGORIES: &[&str] = &[
  "electronics",
  "books",
  "clothing",
  "home",
  "sports",
  "toys",
  "automotive",
  "health",
  "beauty",
  "grocery",
  "garden",
  "tools",
  "music",
  "movies",
  "software",
  "hardware",
  "furniture",
  "jewelry",
  "shoes",
  "kitchen",
  "office",
  "outdoors",
  "pets",
  "baby",
  "crafts",
  "industrial",
  "luggage",
  "watches",
  "cameras",
  "computers",
  "phones",
  "tablets",
  "gaming",
  "fitness",
  "cycling",
  "running",
  "swimming",
  "hiking",
  "camping",
  "fishing",
  "hunting",
  "cooking",
  "baking",
  "brewing",
  "sewing",
  "painting",
  "drawing",
  "writing",
  "reading",
  "travel",
];

/// Pick a category index using a Zipfian-like distribution.
/// Probability proportional to 1/rank (rank = index + 1).
fn zipfian_pick(rng: &mut impl Rng, n: usize) -> usize {
  if n == 0 {
    return 0;
  }
  let harmonic: f64 = (1..=n).map(|r| 1.0 / r as f64).sum();
  let u: f64 = rng.random::<f64>() * harmonic;
  let mut cumulative = 0.0;
  for rank in 1..=n {
    cumulative += 1.0 / rank as f64;
    if u <= cumulative {
      return rank - 1;
    }
  }
  n - 1
}

/// Generate a random text string with `min_words..=max_words` words from the vocabulary.
fn random_text(rng: &mut impl Rng, min_words: usize, max_words: usize) -> String {
  let count = rng.random_range(min_words..=max_words);
  let mut words = Vec::with_capacity(count);
  for _ in 0..count {
    let idx = rng.random_range(0..VOCABULARY.len());
    words.push(VOCABULARY[idx]);
  }
  words.join(" ")
}

/// Generate a random ISO 8601 date string within the last 2 years.
fn random_date(rng: &mut impl Rng) -> String {
  // Span: ~730 days in seconds
  let seconds_in_2_years: u64 = 2 * 365 * 24 * 3600;
  let offset = rng.random_range(0..seconds_in_2_years);
  // Base: 2024-01-01T00:00:00Z
  let base_epoch: u64 = 1_704_067_200;
  let ts = base_epoch + offset;
  // Convert epoch to ISO date string manually (avoid heavy deps)
  let days_since_epoch = ts / 86400;
  let time_of_day = ts % 86400;
  let hours = time_of_day / 3600;
  let minutes = (time_of_day % 3600) / 60;
  let seconds = time_of_day % 60;

  // Simple date calculation from days since 1970-01-01
  let (year, month, day) = days_to_ymd(days_since_epoch);
  format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
  // Algorithm from http://howardhinnant.github.io/date_algorithms.html
  let z = days + 719468;
  let era = z / 146097;
  let doe = z - era * 146097;
  let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  let y = yoe + era * 400;
  let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  let mp = (5 * doy + 2) / 153;
  let d = doy - (153 * mp + 2) / 5 + 1;
  let m = if mp < 10 { mp + 3 } else { mp - 9 };
  let y = if m <= 2 { y + 1 } else { y };
  (y, m, d)
}

/// Generate `count` synthetic documents using a deterministic RNG seeded with `seed`.
pub fn generate_docs(count: usize, seed: u64) -> Vec<Document> {
  let mut rng = StdRng::seed_from_u64(seed);
  let mut docs = Vec::with_capacity(count);

  for i in 0..count {
    let title = random_text(&mut rng, 3, 8);
    let body = random_text(&mut rng, 20, 100);
    let category_idx = zipfian_pick(&mut rng, CATEGORIES.len());
    let category = CATEGORIES[category_idx];
    let price = rng.random_range(1..=10000i64);
    let rating = 1.0 + rng.random::<f64>() * 4.0; // 1.0..5.0
    let created_at = random_date(&mut rng);

    let mut fields = std::collections::BTreeMap::new();
    fields.insert("_id".to_string(), serde_json::json!(format!("doc-{i}")));
    fields.insert("title".to_string(), serde_json::json!(title));
    fields.insert("body".to_string(), serde_json::json!(body));
    fields.insert("category".to_string(), serde_json::json!(category));
    fields.insert("price".to_string(), serde_json::json!(price));
    fields.insert("rating".to_string(), serde_json::json!(rating));
    fields.insert("created_at".to_string(), serde_json::json!(created_at));

    docs.push(Document { fields });
  }

  docs
}

/// Build a schema matching the synthetic documents.
pub fn generate_schema() -> Schema {
  Schema {
    doc_id_field: "_id".to_string(),
    analyzers: Vec::new(),
    text_fields: vec![
      TextField {
        name: "title".to_string(),
        analyzer: "default".to_string(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: false,
        search_as_you_type: None,
      },
      TextField {
        name: "body".to_string(),
        analyzer: "default".to_string(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: false,
        search_as_you_type: None,
      },
    ],
    keyword_fields: vec![
      KeywordField {
        name: "category".to_string(),
        stored: true,
        indexed: true,
        fast: true,
        nullable: false,
      },
      KeywordField {
        name: "created_at".to_string(),
        stored: true,
        indexed: false,
        fast: false,
        nullable: false,
      },
    ],
    numeric_fields: vec![
      NumericField {
        name: "price".to_string(),
        i64: true,
        fast: true,
        stored: true,
        nullable: false,
      },
      NumericField {
        name: "rating".to_string(),
        i64: false,
        fast: true,
        stored: true,
        nullable: false,
      },
    ],
    nested_fields: Vec::new(),
    #[cfg(feature = "vectors")]
    vector_fields: Vec::new(),
  }
}

/// Sample a few query terms from the vocabulary for use in benchmark queries.
pub fn sample_query_terms(rng: &mut impl Rng) -> Vec<String> {
  let count = rng.random_range(2..=4);
  let mut terms = Vec::with_capacity(count);
  for _ in 0..count {
    let idx = rng.random_range(0..VOCABULARY.len());
    terms.push(VOCABULARY[idx].to_string());
  }
  terms
}
