import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { afterEach, describe, expect, it } from "vitest";
import { Index } from "../dist/index.js";

let cleanup = [];

function tempDir() {
	const dir = mkdtempSync(join(tmpdir(), "searchlite-test-"));
	cleanup.push(dir);
	return dir;
}

function createIndex(schema = { body: "text" }) {
	return new Index(join(tempDir(), "idx"), { schema });
}

afterEach(() => {
	for (const dir of cleanup) {
		rmSync(dir, { recursive: true, force: true });
	}
	cleanup = [];
});

// =============================================================================
// Constructor
// =============================================================================

describe("constructor", () => {
	it("creates index with shorthand schema", () => {
		const idx = createIndex({ title: "text", body: "text" });
		expect(idx).toBeTruthy();
		idx.close();
	});

	it("creates index with detailed schema", () => {
		const idx = createIndex({
			title: { type: "text", stored: true, analyzer: "default" },
			tag: { type: "keyword", fast: true },
			year: { type: "integer" },
		});
		expect(idx).toBeTruthy();
		idx.close();
	});

	it("opens existing index without schema", () => {
		const path = join(tempDir(), "idx");

		const idx1 = new Index(path, { schema: { body: "text" } });
		idx1.add({ _id: "1", body: "hello" });
		idx1.commit();
		idx1.close();

		const idx2 = new Index(path);
		expect(idx2.search("hello").totalHits).toBe(1);
		idx2.close();
	});

	it("throws on missing index without schema", () => {
		expect(() => new Index(join(tempDir(), "nonexistent"))).toThrowError(
			/provide a schema to create it/,
		);
	});

	it("throws on empty path", () => {
		expect(() => new Index("")).toThrowError(/non-empty string/);
	});

	it("throws on invalid options", () => {
		expect(() => new Index("/tmp/test", { bogus: true })).toThrowError(/Unrecognized key/);
	});
});

// =============================================================================
// Schema validation on reopen
// =============================================================================

describe("schema validation on reopen", () => {
	it("succeeds with matching schema", () => {
		const path = join(tempDir(), "idx");
		const schema = { title: "text", tag: "keyword" };

		const idx1 = new Index(path, { schema });
		idx1.close();

		const idx2 = new Index(path, { schema });
		expect(idx2).toBeTruthy();
		idx2.close();
	});

	it("throws on mismatched schema", () => {
		const path = join(tempDir(), "idx");

		const idx1 = new Index(path, { schema: { title: "text" } });
		idx1.close();

		expect(() => new Index(path, { schema: { title: "text", extra: "keyword" } })).toThrowError(
			/schema mismatch/,
		);
	});
});

// =============================================================================
// Add and commit
// =============================================================================

describe("add and commit", () => {
	it("roundtrip: add, commit, search", () => {
		const idx = createIndex();
		idx.add({ _id: "1", body: "the quick brown fox" });
		idx.commit();

		const result = idx.search("quick");
		expect(result.totalHits).toBe(1);
		expect(result.hits).toHaveLength(1);
		expect(result.hits[0].docId).toBe("1");
		expect(result.hits[0].score).toBeGreaterThan(0);
		idx.close();
	});

	it("addMany queues multiple documents", () => {
		const idx = createIndex();
		const count = idx.addMany([
			{ _id: "1", body: "hello world" },
			{ _id: "2", body: "hello node" },
			{ _id: "3", body: "hello rust" },
		]);
		expect(count).toBe(3);
		idx.commit();
		expect(idx.search("hello").totalHits).toBe(3);
		idx.close();
	});

	it("documents not searchable before commit", () => {
		const idx = createIndex();
		idx.add({ _id: "1", body: "uncommitted data" });
		expect(idx.search("uncommitted").totalHits).toBe(0);
		idx.commit();
		expect(idx.search("uncommitted").totalHits).toBe(1);
		idx.close();
	});

	it("requires _id field on documents", () => {
		const idx = createIndex();
		expect(() => idx.add({ body: "no id document" })).toThrow();
		idx.close();
	});

	it("handles unicode content", () => {
		const idx = createIndex();
		idx.add({
			_id: "1",
			body: "caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u00fc\u00f1\u00ee\u00e7\u00f6d\u00e9",
		});
		idx.commit();
		expect(idx.search("caf\u00e9").totalHits).toBe(1);
		idx.close();
	});

	it("handles very long field values", () => {
		const idx = createIndex();
		const longText = "word ".repeat(10000);
		idx.add({ _id: "1", body: longText });
		idx.commit();
		expect(idx.search("word").totalHits).toBe(1);
		idx.close();
	});
});

// =============================================================================
// Search — camelCase and options
// =============================================================================

describe("search options", () => {
	it("accepts camelCase request options", () => {
		const idx = createIndex();
		idx.addMany([
			{ _id: "1", body: "alpha beta gamma" },
			{ _id: "2", body: "alpha delta" },
			{ _id: "3", body: "epsilon" },
		]);
		idx.commit();

		const result = idx.search({ query: "alpha", limit: 1, returnStored: false });
		expect(result.hits).toHaveLength(1);
		expect(result.totalHits).toBeGreaterThanOrEqual(2);
		idx.close();
	});

	it("returnStored returns document fields", () => {
		const idx = createIndex();
		idx.add({ _id: "doc1", body: "stored field test" });
		idx.commit();

		const without = idx.search({ query: "stored", returnStored: false });
		expect(without.hits[0].fields).toBeNull();

		const with_ = idx.search({ query: "stored", returnStored: true });
		expect(with_.hits[0].fields).toBeTruthy();
		idx.close();
	});

	it("limit restricts result count", () => {
		const idx = createIndex();
		for (let i = 0; i < 20; i++) {
			idx.add({ _id: `${i}`, body: "common term here" });
		}
		idx.commit();

		const result = idx.search({ query: "common", limit: 5 });
		expect(result.hits).toHaveLength(5);
		expect(result.totalHits).toBe(20);
		idx.close();
	});

	it("from offsets results", () => {
		const idx = createIndex();
		for (let i = 0; i < 10; i++) {
			idx.add({ _id: `${i}`, body: "pagination test" });
		}
		idx.commit();

		const page1 = idx.search({ query: "pagination", limit: 3, from: 0 });
		const page2 = idx.search({ query: "pagination", limit: 3, from: 3 });
		expect(page1.hits).toHaveLength(3);
		expect(page2.hits).toHaveLength(3);
		// Pages should contain different documents
		const ids1 = page1.hits.map((h) => h.docId);
		const ids2 = page2.hits.map((h) => h.docId);
		expect(ids1).not.toEqual(ids2);
		idx.close();
	});

	it("cursor-based pagination", () => {
		const idx = createIndex();
		for (let i = 0; i < 10; i++) {
			idx.add({ _id: `${i}`, body: "cursor test document" });
		}
		idx.commit();

		const page1 = idx.search({ query: "cursor", limit: 3 });
		expect(page1.hits).toHaveLength(3);
		expect(page1.nextCursor).toBeTruthy();

		const page2 = idx.search({ query: "cursor", limit: 3, cursor: page1.nextCursor });
		expect(page2.hits).toHaveLength(3);
		expect(page2.hits[0].docId).not.toBe(page1.hits[0].docId);
		idx.close();
	});
});

// =============================================================================
// Structured queries
// =============================================================================

describe("structured queries", () => {
	function indexedDocs() {
		const idx = createIndex({
			title: "text",
			body: "text",
			tag: "keyword",
		});
		idx.addMany([
			{ _id: "1", title: "Rust Programming", body: "Systems language with safety", tag: "tech" },
			{ _id: "2", title: "Node.js Guide", body: "JavaScript runtime for servers", tag: "tech" },
			{
				_id: "3",
				title: "Cooking Basics",
				body: "Learn to cook simple meals",
				tag: "food",
			},
		]);
		idx.commit();
		return idx;
	}

	it("match_all returns all documents", () => {
		const idx = indexedDocs();
		const result = idx.search({ query: { type: "match_all" } });
		expect(result.totalHits).toBe(3);
		idx.close();
	});

	it("filter on keyword field", () => {
		const idx = indexedDocs();
		const result = idx.search({
			query: { type: "match_all" },
			filter: { KeywordEq: { field: "tag", value: "food" } },
		});
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe("3");
		idx.close();
	});

	it("bool query combines must and must_not", () => {
		const idx = indexedDocs();
		const result = idx.search({
			query: {
				type: "bool",
				must: [{ type: "match_all" }],
				must_not: [{ type: "query_string", query: "cooking" }],
			},
		});
		expect(result.totalHits).toBe(2);
		idx.close();
	});

	it("phrase query matches exact phrases", () => {
		const idx = indexedDocs();
		const result = idx.search({
			query: { type: "phrase", field: "body", terms: ["systems", "language"] },
		});
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe("1");
		idx.close();
	});

	it("prefix query matches prefix", () => {
		const idx = indexedDocs();
		const result = idx.search({
			query: { type: "prefix", field: "title", value: "rust" },
		});
		expect(result.totalHits).toBe(1);
		idx.close();
	});

	it("wildcard query matches pattern", () => {
		const idx = indexedDocs();
		const result = idx.search({
			query: { type: "wildcard", field: "title", value: "*guide*" },
		});
		expect(result.totalHits).toBeGreaterThanOrEqual(1);
		idx.close();
	});
});

// =============================================================================
// Filters
// =============================================================================

describe("filters", () => {
	it("KeywordEq filters by exact value", () => {
		const idx = createIndex({ title: "text", category: "keyword" });
		idx.add({ _id: "1", title: "apple pie recipe", category: "food" });
		idx.add({ _id: "2", title: "apple iphone review", category: "tech" });
		idx.commit();

		const result = idx.search({
			query: "apple",
			filter: { KeywordEq: { field: "category", value: "food" } },
		});
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe("1");
		idx.close();
	});

	it("KeywordIn filters by set membership", () => {
		const idx = createIndex({ body: "text", tag: "keyword" });
		idx.addMany([
			{ _id: "1", body: "doc one", tag: "a" },
			{ _id: "2", body: "doc two", tag: "b" },
			{ _id: "3", body: "doc three", tag: "c" },
		]);
		idx.commit();

		const result = idx.search({
			query: "doc",
			filter: { KeywordIn: { field: "tag", values: ["a", "c"] } },
		});
		expect(result.totalHits).toBe(2);
		idx.close();
	});

	it("I64Range filters by integer range", () => {
		const idx = createIndex({ body: "text", year: "integer" });
		idx.addMany([
			{ _id: "1", body: "old doc", year: 2020 },
			{ _id: "2", body: "mid doc", year: 2022 },
			{ _id: "3", body: "new doc", year: 2024 },
		]);
		idx.commit();

		const result = idx.search({
			query: "doc",
			filter: { I64Range: { field: "year", min: 2021, max: 2023 } },
		});
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe("2");
		idx.close();
	});

	it("And combines filters", () => {
		const idx = createIndex({ body: "text", tag: "keyword", year: "integer" });
		idx.addMany([
			{ _id: "1", body: "doc", tag: "a", year: 2020 },
			{ _id: "2", body: "doc", tag: "a", year: 2024 },
			{ _id: "3", body: "doc", tag: "b", year: 2024 },
		]);
		idx.commit();

		const result = idx.search({
			query: "doc",
			filter: {
				And: [
					{ KeywordEq: { field: "tag", value: "a" } },
					{ I64Range: { field: "year", min: 2023, max: 2025 } },
				],
			},
		});
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe("2");
		idx.close();
	});

	it("Not negates a filter", () => {
		const idx = createIndex({ body: "text", tag: "keyword" });
		idx.addMany([
			{ _id: "1", body: "doc", tag: "keep" },
			{ _id: "2", body: "doc", tag: "remove" },
		]);
		idx.commit();

		const result = idx.search({
			query: "doc",
			filter: { Not: { KeywordEq: { field: "tag", value: "remove" } } },
		});
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe("1");
		idx.close();
	});
});

// =============================================================================
// Aggregations
// =============================================================================

describe("aggregations", () => {
	it("terms aggregation returns buckets", () => {
		const idx = createIndex({ body: "text", tag: "keyword" });
		idx.addMany([
			{ _id: "1", body: "doc", tag: "a" },
			{ _id: "2", body: "doc", tag: "a" },
			{ _id: "3", body: "doc", tag: "b" },
		]);
		idx.commit();

		const result = idx.search({
			query: "doc",
			aggs: { tags: { type: "terms", field: "tag" } },
		});
		expect(result.aggregations.tags).toBeTruthy();
		const buckets = result.aggregations.tags.buckets;
		expect(buckets).toBeTruthy();
		expect(buckets.length).toBeGreaterThanOrEqual(2);
		idx.close();
	});

	it("stats aggregation returns statistics", () => {
		const idx = createIndex({ body: "text", price: "float" });
		idx.addMany([
			{ _id: "1", body: "item", price: 10.0 },
			{ _id: "2", body: "item", price: 20.0 },
			{ _id: "3", body: "item", price: 30.0 },
		]);
		idx.commit();

		const result = idx.search({
			query: "item",
			aggs: { priceStats: { type: "stats", field: "price" } },
		});
		const stats = result.aggregations.priceStats;
		expect(stats).toBeTruthy();
		expect(stats.count).toBe(3);
		expect(stats.min).toBe(10.0);
		expect(stats.max).toBe(30.0);
		idx.close();
	});

	it("nested aggregation with sub-aggs", () => {
		const idx = createIndex({ body: "text", tag: "keyword", price: "float" });
		idx.addMany([
			{ _id: "1", body: "doc", tag: "a", price: 10 },
			{ _id: "2", body: "doc", tag: "a", price: 20 },
			{ _id: "3", body: "doc", tag: "b", price: 30 },
		]);
		idx.commit();

		const result = idx.search({
			query: "doc",
			aggs: {
				byTag: {
					type: "terms",
					field: "tag",
					aggs: { avg_price: { type: "stats", field: "price" } },
				},
			},
		});
		const byTag = result.aggregations.byTag;
		expect(byTag).toBeTruthy();
		expect(byTag.buckets.length).toBeGreaterThanOrEqual(2);
		// Each bucket should have the sub-aggregation (key matches agg name)
		const firstBucket = byTag.buckets[0];
		const subAggKeys = Object.keys(firstBucket).filter((k) => k !== "key" && k !== "doc_count");
		expect(subAggKeys.length).toBeGreaterThanOrEqual(1);
		idx.close();
	});
});

// =============================================================================
// Fuzzy search
// =============================================================================

describe("fuzzy search", () => {
	it("finds documents despite typos", () => {
		const idx = createIndex();
		idx.add({ _id: "1", body: "headphones wireless bluetooth" });
		idx.commit();

		const result = idx.search({
			query: "headphoens",
			fuzzy: { maxEdits: 2, prefixLength: 2 },
		});
		expect(result.totalHits).toBe(1);
		idx.close();
	});
});

// =============================================================================
// Highlighting
// =============================================================================

describe("highlighting", () => {
	it("highlightField returns snippet", () => {
		const idx = createIndex();
		idx.add({ _id: "1", body: "the quick brown fox jumps over the lazy dog" });
		idx.commit();

		const result = idx.search({ query: "quick", highlightField: "body" });
		expect(result.hits[0].snippet).toBeTruthy();
		idx.close();
	});

	it("highlight request returns highlights map", () => {
		const idx = createIndex();
		idx.add({ _id: "1", body: "the quick brown fox jumps over the lazy dog" });
		idx.commit();

		const result = idx.search({
			query: "quick",
			highlight: {
				fields: {
					body: { pre_tag: "<em>", post_tag: "</em>", fragment_size: 64, number_of_fragments: 1 },
				},
			},
		});
		expect(result.hits[0].highlights).toBeTruthy();
		expect(result.hits[0].highlights.body).toBeTruthy();
		expect(result.hits[0].highlights.body[0]).toContain("<em>");
		idx.close();
	});
});

// =============================================================================
// Result collapsing
// =============================================================================

describe("result collapsing", () => {
	it("deduplicates by keyword field", () => {
		const idx = createIndex({ body: "text", brand: "keyword" });
		idx.addMany([
			{ _id: "1", body: "headphones model a", brand: "acme" },
			{ _id: "2", body: "headphones model b", brand: "acme" },
			{ _id: "3", body: "headphones model c", brand: "other" },
		]);
		idx.commit();

		const result = idx.search({
			query: "headphones",
			collapse: { field: "brand" },
		});
		// Should collapse the two acme results into one
		expect(result.hits).toHaveLength(2);
		idx.close();
	});
});

// =============================================================================
// Compact
// =============================================================================

describe("compact", () => {
	it("succeeds after multiple commits", () => {
		const idx = createIndex();
		idx.add({ _id: "1", body: "first" });
		idx.commit();
		idx.add({ _id: "2", body: "second" });
		idx.commit();
		expect(() => idx.compact()).not.toThrow();
		idx.close();
	});
});

// =============================================================================
// Resource lifecycle
// =============================================================================

describe("resource lifecycle", () => {
	it("operations after close throw", () => {
		const idx = createIndex();
		idx.close();
		expect(() => idx.add({ _id: "1", body: "test" })).toThrowError(/index is closed/);
		expect(() => idx.search("test")).toThrowError(/index is closed/);
		expect(() => idx.commit()).toThrowError(/index is closed/);
	});

	it("close is idempotent", () => {
		const idx = createIndex();
		idx.close();
		expect(() => idx.close()).not.toThrow();
	});

	it("Symbol.dispose calls close", () => {
		const idx = createIndex();
		idx[Symbol.dispose]();
		expect(() => idx.search("test")).toThrowError(/index is closed/);
	});
});

// =============================================================================
// Write key
// =============================================================================

describe("write key", { timeout: 30_000 }, () => {
	it("create with writeKey, reopen with writeKey succeeds", () => {
		const path = join(tempDir(), "idx");
		const idx1 = new Index(path, { schema: { body: "text" }, writeKey: "secret" });
		idx1.add({ _id: "1", body: "protected data" });
		idx1.commit();
		idx1.close();

		const idx2 = new Index(path, { writeKey: "secret" });
		const result = idx2.search("protected");
		expect(result.totalHits).toBe(1);
		idx2.close();
	});

	it("create with writeKey, write without key fails", () => {
		const path = join(tempDir(), "idx");
		const idx1 = new Index(path, { schema: { body: "text" }, writeKey: "secret" });
		idx1.close();

		const idx2 = new Index(path);
		expect(() => {
			idx2.add({ _id: "1", body: "should fail" });
		}).toThrow();
		idx2.close();
	});
});

// =============================================================================
// Validation
// =============================================================================

describe("validation", () => {
	it("add rejects non-object documents", () => {
		const idx = createIndex();
		expect(() => idx.add("not an object")).toThrow();
		expect(() => idx.add(42)).toThrow();
		expect(() => idx.add(null)).toThrow();
		idx.close();
	});

	it("addMany rejects invalid input", () => {
		const idx = createIndex();
		expect(() => idx.addMany("not valid")).toThrow();
		expect(() => idx.addMany(42)).toThrow();
		idx.close();
	});

	it("search rejects invalid query types", () => {
		const idx = createIndex();
		idx.commit();
		expect(() => idx.search(42)).toThrow();
		expect(() => idx.search(null)).toThrow();
		expect(() => idx.search(true)).toThrow();
		idx.close();
	});

	it("rejects invalid schema type", () => {
		expect(() => createIndex({ title: "invalid_type" })).toThrowError(/unknown field type/);
	});
});

// =============================================================================
// Result shape
// =============================================================================

describe("result shape", () => {
	it("has expected camelCase shape", () => {
		const idx = createIndex();
		idx.add({ _id: "1", body: "shape test" });
		idx.commit();

		const result = idx.search("shape");
		expect(typeof result.totalHits).toBe("number");
		expect(Array.isArray(result.hits)).toBe(true);
		expect(typeof result.hits[0].docId).toBe("string");
		expect(typeof result.hits[0].score).toBe("number");
		expect(typeof result.aggregations).toBe("object");
		expect(typeof result.suggest).toBe("object");
		idx.close();
	});

	it("numeric fields stored and retrieved correctly", () => {
		const idx = createIndex({ body: "text", count: { type: "integer", stored: true } });
		idx.add({ _id: "1", body: "numeric test", count: 42 });
		idx.commit();

		const result = idx.search({ query: "numeric", returnStored: true });
		expect(result.hits[0].fields).toBeTruthy();
		idx.close();
	});
});

// =============================================================================
// Typed search (Zod schema)
// =============================================================================

describe("typed search", () => {
	const BodySchema = z.object({
		body: z.string(),
	});

	it("validates and returns typed fields with request object", () => {
		const idx = createIndex();
		idx.add({ _id: "1", body: "hello world" });
		idx.commit();

		const result = idx.search(BodySchema, { query: "hello" });
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].fields).toEqual({ body: "hello world" });
		idx.close();
	});

	it("validates and returns typed fields with string query", () => {
		const idx = createIndex();
		idx.add({ _id: "1", body: "hello world" });
		idx.commit();

		const result = idx.search(BodySchema, "hello");
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].fields).toEqual({ body: "hello world" });
		idx.close();
	});

	it("auto-sets returnStored when schema is provided", () => {
		const idx = createIndex();
		idx.add({ _id: "1", body: "auto stored" });
		idx.commit();

		// No need to manually set returnStored
		const result = idx.search(BodySchema, { query: "auto" });
		expect(result.hits[0].fields).toBeTruthy();
		expect(result.hits[0].fields.body).toBe("auto stored");
		idx.close();
	});

	it("throws with context when fields do not match schema", () => {
		const idx = createIndex();
		idx.add({ _id: "doc-42", body: "mismatch test" });
		idx.commit();

		const StrictSchema = z.object({
			body: z.string(),
			missing: z.number(),
		});

		expect(() => idx.search(StrictSchema, "mismatch")).toThrowError(/hit 0.*docId.*doc-42/s);
		idx.close();
	});

	it("succeeds with empty results", () => {
		const idx = createIndex();
		idx.commit();

		const result = idx.search(BodySchema, "nonexistent_xyz");
		expect(result.hits).toHaveLength(0);
		idx.close();
	});

	it("works with optional fields in schema", () => {
		const idx = createIndex({ body: "text", tag: "keyword" });
		idx.add({ _id: "1", body: "optional test" });
		idx.commit();

		const FlexSchema = z.object({
			body: z.string(),
			tag: z.string().optional(),
		});

		const result = idx.search(FlexSchema, "optional");
		expect(result.hits[0].fields.body).toBe("optional test");
		idx.close();
	});

	it("works with multiple hits", () => {
		const idx = createIndex();
		idx.addMany([
			{ _id: "1", body: "alpha beta" },
			{ _id: "2", body: "alpha gamma" },
		]);
		idx.commit();

		const result = idx.search(BodySchema, "alpha");
		expect(result.totalHits).toBe(2);
		for (const hit of result.hits) {
			expect(hit.fields.body).toBeTruthy();
		}
		idx.close();
	});

	it("applies Zod transforms to fields", () => {
		const idx = createIndex();
		idx.add({ _id: "1", body: "transform test" });
		idx.commit();

		const UpperSchema = z.object({
			body: z.string().transform((s) => s.toUpperCase()),
		});

		const result = idx.search(UpperSchema, "transform");
		expect(result.hits[0].fields.body).toBe("TRANSFORM TEST");
		idx.close();
	});

	it("applies Zod defaults for missing fields", () => {
		const idx = createIndex();
		idx.add({ _id: "1", body: "defaults test" });
		idx.commit();

		const WithDefault = z.object({
			body: z.string(),
			extra: z.string().default("fallback"),
		});

		const result = idx.search(WithDefault, "defaults");
		expect(result.hits[0].fields.body).toBe("defaults test");
		expect(result.hits[0].fields.extra).toBe("fallback");
		idx.close();
	});

	it("strips unknown keys by default (z.object behavior)", () => {
		const idx = createIndex({ body: "text", tag: "keyword" });
		idx.add({ _id: "1", body: "strip test", tag: "hello" });
		idx.commit();

		const PartialSchema = z.object({
			body: z.string(),
		});

		const result = idx.search(PartialSchema, "strip");
		expect(result.hits[0].fields.body).toBe("strip test");
		// tag was stored but not in the schema — stripped by Zod
		expect("tag" in result.hits[0].fields).toBe(false);
		idx.close();
	});

	it("preserves unknown keys with passthrough", () => {
		const idx = createIndex({ body: "text", tag: "keyword" });
		idx.add({ _id: "1", body: "passthrough test", tag: "kept" });
		idx.commit();

		const PassthroughSchema = z.object({
			body: z.string(),
		}).passthrough();

		const result = idx.search(PassthroughSchema, "passthrough");
		expect(result.hits[0].fields.body).toBe("passthrough test");
		// tag is preserved because of passthrough
		expect(result.hits[0].fields.tag).toBe("kept");
		idx.close();
	});
});
