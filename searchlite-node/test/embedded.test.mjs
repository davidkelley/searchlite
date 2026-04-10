import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { z } from "zod";
import { EmbeddedIndex } from "../dist/index.js";

let cleanup = [];

function tempDir() {
	const dir = mkdtempSync(join(tmpdir(), "searchlite-test-"));
	cleanup.push(dir);
	return dir;
}

function createIndex(schema = { body: "text" }) {
	return new EmbeddedIndex(join(tempDir(), "idx"), { schema });
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
	it("creates index with shorthand schema", async () => {
		const idx = createIndex({ title: "text", body: "text" });
		expect(idx).toBeTruthy();
		await idx.close();
	});

	it("creates index with detailed schema", async () => {
		const idx = createIndex({
			title: { type: "text", stored: true, analyzer: "default" },
			tag: { type: "keyword", fast: true },
			year: { type: "integer" },
		});
		expect(idx).toBeTruthy();
		await idx.close();
	});

	it("opens existing index without schema", async () => {
		const path = join(tempDir(), "idx");

		const idx1 = new EmbeddedIndex(path, { schema: { body: "text" } });
		await idx1.add({ _id: "1", body: "hello" });
		await idx1.commit();
		await idx1.close();

		const idx2 = new EmbeddedIndex(path);
		expect((await idx2.search("hello")).totalHits).toBe(1);
		await idx2.close();
	});

	it("throws on missing index without schema", () => {
		expect(() => new EmbeddedIndex(join(tempDir(), "nonexistent"))).toThrowError(
			/provide a schema to create it/,
		);
	});

	it("throws on empty path", () => {
		expect(() => new EmbeddedIndex("")).toThrowError(/non-empty string/);
	});

	it("throws on invalid options", () => {
		expect(() => new EmbeddedIndex("/tmp/test", { bogus: true })).toThrowError(/Unrecognized key/);
	});
});

// =============================================================================
// Schema validation on reopen
// =============================================================================

describe("schema validation on reopen", () => {
	it("succeeds with matching schema", async () => {
		const path = join(tempDir(), "idx");
		const schema = { title: "text", tag: "keyword" };

		const idx1 = new EmbeddedIndex(path, { schema });
		await idx1.close();

		const idx2 = new EmbeddedIndex(path, { schema });
		expect(idx2).toBeTruthy();
		await idx2.close();
	});

	it("throws on mismatched schema", async () => {
		const path = join(tempDir(), "idx");

		const idx1 = new EmbeddedIndex(path, { schema: { title: "text" } });
		await idx1.close();

		expect(
			() => new EmbeddedIndex(path, { schema: { title: "text", extra: "keyword" } }),
		).toThrowError(/schema mismatch/);
	});
});

// =============================================================================
// Add and commit
// =============================================================================

describe("add and commit", () => {
	it("roundtrip: add, commit, search", async () => {
		const idx = createIndex();
		await idx.add({ _id: "1", body: "the quick brown fox" });
		await idx.commit();

		const result = await idx.search("quick");
		expect(result.totalHits).toBe(1);
		expect(result.hits).toHaveLength(1);
		expect(result.hits[0].docId).toBe("1");
		expect(result.hits[0].score).toBeGreaterThan(0);
		await idx.close();
	});

	it("addMany queues multiple documents", async () => {
		const idx = createIndex();
		const count = await idx.addMany([
			{ _id: "1", body: "hello world" },
			{ _id: "2", body: "hello node" },
			{ _id: "3", body: "hello rust" },
		]);
		expect(count).toBe(3);
		await idx.commit();
		expect((await idx.search("hello")).totalHits).toBe(3);
		await idx.close();
	});

	it("documents not searchable before commit", async () => {
		const idx = createIndex();
		await idx.add({ _id: "1", body: "uncommitted data" });
		expect((await idx.search("uncommitted")).totalHits).toBe(0);
		await idx.commit();
		expect((await idx.search("uncommitted")).totalHits).toBe(1);
		await idx.close();
	});

	it("requires _id field on documents", async () => {
		const idx = createIndex();
		await expect(idx.add({ body: "no id document" })).rejects.toThrow();
		await idx.close();
	});

	it("handles unicode content", async () => {
		const idx = createIndex();
		await idx.add({
			_id: "1",
			body: "caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u00fc\u00f1\u00ee\u00e7\u00f6d\u00e9",
		});
		await idx.commit();
		expect((await idx.search("caf\u00e9")).totalHits).toBe(1);
		await idx.close();
	});

	it("handles very long field values", async () => {
		const idx = createIndex();
		const longText = "word ".repeat(10000);
		await idx.add({ _id: "1", body: longText });
		await idx.commit();
		expect((await idx.search("word")).totalHits).toBe(1);
		await idx.close();
	});
});

// =============================================================================
// Search — camelCase and options
// =============================================================================

describe("search options", () => {
	it("accepts camelCase request options", async () => {
		const idx = createIndex();
		await idx.addMany([
			{ _id: "1", body: "alpha beta gamma" },
			{ _id: "2", body: "alpha delta" },
			{ _id: "3", body: "epsilon" },
		]);
		await idx.commit();

		const result = await idx.search({ query: "alpha", limit: 1, returnStored: false });
		expect(result.hits).toHaveLength(1);
		expect(result.totalHits).toBeGreaterThanOrEqual(2);
		await idx.close();
	});

	it("returnStored returns document fields", async () => {
		const idx = createIndex();
		await idx.add({ _id: "doc1", body: "stored field test" });
		await idx.commit();

		const without = await idx.search({ query: "stored", returnStored: false });
		expect(without.hits[0].fields).toBeNull();

		const with_ = await idx.search({ query: "stored", returnStored: true });
		expect(with_.hits[0].fields).toBeTruthy();
		await idx.close();
	});

	it("limit restricts result count", async () => {
		const idx = createIndex();
		for (let i = 0; i < 20; i++) {
			await idx.add({ _id: `${i}`, body: "common term here" });
		}
		await idx.commit();

		const result = await idx.search({ query: "common", limit: 5 });
		expect(result.hits).toHaveLength(5);
		expect(result.totalHits).toBe(20);
		await idx.close();
	});

	it("from offsets results", async () => {
		const idx = createIndex();
		for (let i = 0; i < 10; i++) {
			await idx.add({ _id: `${i}`, body: "pagination test" });
		}
		await idx.commit();

		const page1 = await idx.search({ query: "pagination", limit: 3, from: 0 });
		const page2 = await idx.search({ query: "pagination", limit: 3, from: 3 });
		expect(page1.hits).toHaveLength(3);
		expect(page2.hits).toHaveLength(3);
		const ids1 = page1.hits.map((h) => h.docId);
		const ids2 = page2.hits.map((h) => h.docId);
		expect(ids1).not.toEqual(ids2);
		await idx.close();
	});

	it("cursor-based pagination", async () => {
		const idx = createIndex();
		for (let i = 0; i < 10; i++) {
			await idx.add({ _id: `${i}`, body: "cursor test document" });
		}
		await idx.commit();

		const page1 = await idx.search({ query: "cursor", limit: 3 });
		expect(page1.hits).toHaveLength(3);
		expect(page1.nextCursor).toBeTruthy();

		const page2 = await idx.search({ query: "cursor", limit: 3, cursor: page1.nextCursor });
		expect(page2.hits).toHaveLength(3);
		expect(page2.hits[0].docId).not.toBe(page1.hits[0].docId);
		await idx.close();
	});
});

// =============================================================================
// Structured queries
// =============================================================================

describe("structured queries", () => {
	async function indexedDocs() {
		const idx = createIndex({
			title: "text",
			body: "text",
			tag: "keyword",
		});
		await idx.addMany([
			{ _id: "1", title: "Rust Programming", body: "Systems language with safety", tag: "tech" },
			{ _id: "2", title: "Node.js Guide", body: "JavaScript runtime for servers", tag: "tech" },
			{
				_id: "3",
				title: "Cooking Basics",
				body: "Learn to cook simple meals",
				tag: "food",
			},
		]);
		await idx.commit();
		return idx;
	}

	it("match_all returns all documents", async () => {
		const idx = await indexedDocs();
		const result = await idx.search({ query: { type: "match_all" } });
		expect(result.totalHits).toBe(3);
		await idx.close();
	});

	it("filter on keyword field", async () => {
		const idx = await indexedDocs();
		const result = await idx.search({
			query: { type: "match_all" },
			filter: { KeywordEq: { field: "tag", value: "food" } },
		});
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe("3");
		await idx.close();
	});

	it("bool query combines must and must_not", async () => {
		const idx = await indexedDocs();
		const result = await idx.search({
			query: {
				type: "bool",
				must: [{ type: "match_all" }],
				must_not: [{ type: "query_string", query: "cooking" }],
			},
		});
		expect(result.totalHits).toBe(2);
		await idx.close();
	});

	it("phrase query matches exact phrases", async () => {
		const idx = await indexedDocs();
		const result = await idx.search({
			query: { type: "phrase", field: "body", terms: ["systems", "language"] },
		});
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe("1");
		await idx.close();
	});

	it("prefix query matches prefix", async () => {
		const idx = await indexedDocs();
		const result = await idx.search({
			query: { type: "prefix", field: "title", value: "rust" },
		});
		expect(result.totalHits).toBe(1);
		await idx.close();
	});

	it("wildcard query matches pattern", async () => {
		const idx = await indexedDocs();
		const result = await idx.search({
			query: { type: "wildcard", field: "title", value: "*guide*" },
		});
		expect(result.totalHits).toBeGreaterThanOrEqual(1);
		await idx.close();
	});
});

// =============================================================================
// Filters
// =============================================================================

describe("filters", () => {
	it("KeywordEq filters by exact value", async () => {
		const idx = createIndex({ title: "text", category: "keyword" });
		await idx.add({ _id: "1", title: "apple pie recipe", category: "food" });
		await idx.add({ _id: "2", title: "apple iphone review", category: "tech" });
		await idx.commit();

		const result = await idx.search({
			query: "apple",
			filter: { KeywordEq: { field: "category", value: "food" } },
		});
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe("1");
		await idx.close();
	});

	it("KeywordIn filters by set membership", async () => {
		const idx = createIndex({ body: "text", tag: "keyword" });
		await idx.addMany([
			{ _id: "1", body: "doc one", tag: "a" },
			{ _id: "2", body: "doc two", tag: "b" },
			{ _id: "3", body: "doc three", tag: "c" },
		]);
		await idx.commit();

		const result = await idx.search({
			query: "doc",
			filter: { KeywordIn: { field: "tag", values: ["a", "c"] } },
		});
		expect(result.totalHits).toBe(2);
		await idx.close();
	});

	it("I64Range filters by integer range", async () => {
		const idx = createIndex({ body: "text", year: "integer" });
		await idx.addMany([
			{ _id: "1", body: "old doc", year: 2020 },
			{ _id: "2", body: "mid doc", year: 2022 },
			{ _id: "3", body: "new doc", year: 2024 },
		]);
		await idx.commit();

		const result = await idx.search({
			query: "doc",
			filter: { I64Range: { field: "year", min: 2021, max: 2023 } },
		});
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe("2");
		await idx.close();
	});

	it("And combines filters", async () => {
		const idx = createIndex({ body: "text", tag: "keyword", year: "integer" });
		await idx.addMany([
			{ _id: "1", body: "doc", tag: "a", year: 2020 },
			{ _id: "2", body: "doc", tag: "a", year: 2024 },
			{ _id: "3", body: "doc", tag: "b", year: 2024 },
		]);
		await idx.commit();

		const result = await idx.search({
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
		await idx.close();
	});

	it("Not negates a filter", async () => {
		const idx = createIndex({ body: "text", tag: "keyword" });
		await idx.addMany([
			{ _id: "1", body: "doc", tag: "keep" },
			{ _id: "2", body: "doc", tag: "remove" },
		]);
		await idx.commit();

		const result = await idx.search({
			query: "doc",
			filter: { Not: { KeywordEq: { field: "tag", value: "remove" } } },
		});
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe("1");
		await idx.close();
	});
});

// =============================================================================
// Aggregations
// =============================================================================

describe("aggregations", () => {
	it("terms aggregation returns buckets", async () => {
		const idx = createIndex({ body: "text", tag: "keyword" });
		await idx.addMany([
			{ _id: "1", body: "doc", tag: "a" },
			{ _id: "2", body: "doc", tag: "a" },
			{ _id: "3", body: "doc", tag: "b" },
		]);
		await idx.commit();

		const result = await idx.search({
			query: "doc",
			aggs: { tags: { type: "terms", field: "tag" } },
		});
		expect(result.aggregations.tags).toBeTruthy();
		const buckets = result.aggregations.tags.buckets;
		expect(buckets).toBeTruthy();
		expect(buckets.length).toBeGreaterThanOrEqual(2);
		await idx.close();
	});

	it("stats aggregation returns statistics", async () => {
		const idx = createIndex({ body: "text", price: "float" });
		await idx.addMany([
			{ _id: "1", body: "item", price: 10.0 },
			{ _id: "2", body: "item", price: 20.0 },
			{ _id: "3", body: "item", price: 30.0 },
		]);
		await idx.commit();

		const result = await idx.search({
			query: "item",
			aggs: { priceStats: { type: "stats", field: "price" } },
		});
		const stats = result.aggregations.priceStats;
		expect(stats).toBeTruthy();
		expect(stats.count).toBe(3);
		expect(stats.min).toBe(10.0);
		expect(stats.max).toBe(30.0);
		await idx.close();
	});

	it("nested aggregation with sub-aggs", async () => {
		const idx = createIndex({ body: "text", tag: "keyword", price: "float" });
		await idx.addMany([
			{ _id: "1", body: "doc", tag: "a", price: 10 },
			{ _id: "2", body: "doc", tag: "a", price: 20 },
			{ _id: "3", body: "doc", tag: "b", price: 30 },
		]);
		await idx.commit();

		const result = await idx.search({
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
		const firstBucket = byTag.buckets[0];
		const subAggKeys = Object.keys(firstBucket).filter((k) => k !== "key" && k !== "doc_count");
		expect(subAggKeys.length).toBeGreaterThanOrEqual(1);
		await idx.close();
	});
});

// =============================================================================
// Fuzzy search
// =============================================================================

describe("fuzzy search", () => {
	it("finds documents despite typos", async () => {
		const idx = createIndex();
		await idx.add({ _id: "1", body: "headphones wireless bluetooth" });
		await idx.commit();

		const result = await idx.search({
			query: "headphoens",
			fuzzy: { maxEdits: 2, prefixLength: 2 },
		});
		expect(result.totalHits).toBe(1);
		await idx.close();
	});
});

// =============================================================================
// Highlighting
// =============================================================================

describe("highlighting", () => {
	it("highlightField returns snippet", async () => {
		const idx = createIndex();
		await idx.add({ _id: "1", body: "the quick brown fox jumps over the lazy dog" });
		await idx.commit();

		const result = await idx.search({ query: "quick", highlightField: "body" });
		expect(result.hits[0].snippet).toBeTruthy();
		await idx.close();
	});

	it("highlight request returns highlights map", async () => {
		const idx = createIndex();
		await idx.add({ _id: "1", body: "the quick brown fox jumps over the lazy dog" });
		await idx.commit();

		const result = await idx.search({
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
		await idx.close();
	});
});

// =============================================================================
// Result collapsing
// =============================================================================

describe("result collapsing", () => {
	it("deduplicates by keyword field", async () => {
		const idx = createIndex({ body: "text", brand: "keyword" });
		await idx.addMany([
			{ _id: "1", body: "headphones model a", brand: "acme" },
			{ _id: "2", body: "headphones model b", brand: "acme" },
			{ _id: "3", body: "headphones model c", brand: "other" },
		]);
		await idx.commit();

		const result = await idx.search({
			query: "headphones",
			collapse: { field: "brand" },
		});
		expect(result.hits).toHaveLength(2);
		await idx.close();
	});
});

// =============================================================================
// Compact
// =============================================================================

describe("compact", () => {
	it("succeeds after multiple commits", async () => {
		const idx = createIndex();
		await idx.add({ _id: "1", body: "first" });
		await idx.commit();
		await idx.add({ _id: "2", body: "second" });
		await idx.commit();
		await idx.compact();
		await idx.close();
	});
});

// =============================================================================
// Resource lifecycle
// =============================================================================

describe("resource lifecycle", () => {
	it("operations after close reject", async () => {
		const idx = createIndex();
		await idx.close();
		await expect(idx.add({ _id: "1", body: "test" })).rejects.toThrowError(/index is closed/);
		await expect(idx.search("test")).rejects.toThrowError(/index is closed/);
		await expect(idx.commit()).rejects.toThrowError(/index is closed/);
	});

	it("close is idempotent", async () => {
		const idx = createIndex();
		await idx.close();
		await idx.close();
	});

	it("Symbol.dispose calls close", async () => {
		const idx = createIndex();
		idx[Symbol.dispose]();
		await expect(idx.search("test")).rejects.toThrowError(/index is closed/);
	});
});

// =============================================================================
// Write key
// =============================================================================

describe("write key", { timeout: 30_000 }, () => {
	it("create with writeKey, reopen with writeKey succeeds", async () => {
		const path = join(tempDir(), "idx");
		const idx1 = new EmbeddedIndex(path, { schema: { body: "text" }, writeKey: "secret" });
		await idx1.add({ _id: "1", body: "protected data" });
		await idx1.commit();
		await idx1.close();

		const idx2 = new EmbeddedIndex(path, { writeKey: "secret" });
		const result = await idx2.search("protected");
		expect(result.totalHits).toBe(1);
		await idx2.close();
	});

	it("create with writeKey, write without key fails", async () => {
		const path = join(tempDir(), "idx");
		const idx1 = new EmbeddedIndex(path, { schema: { body: "text" }, writeKey: "secret" });
		await idx1.close();

		const idx2 = new EmbeddedIndex(path);
		await expect(idx2.add({ _id: "1", body: "should fail" })).rejects.toThrow();
		await idx2.close();
	});
});

// =============================================================================
// Validation
// =============================================================================

describe("validation", () => {
	it("add rejects non-object documents", async () => {
		const idx = createIndex();
		await expect(idx.add("not an object")).rejects.toThrow();
		await expect(idx.add(42)).rejects.toThrow();
		await expect(idx.add(null)).rejects.toThrow();
		await idx.close();
	});

	it("addMany rejects invalid input", async () => {
		const idx = createIndex();
		await expect(idx.addMany("not valid")).rejects.toThrow();
		await expect(idx.addMany(42)).rejects.toThrow();
		await idx.close();
	});

	it("search rejects invalid query types", async () => {
		const idx = createIndex();
		await idx.commit();
		await expect(idx.search(42)).rejects.toThrow();
		await expect(idx.search(null)).rejects.toThrow();
		await expect(idx.search(true)).rejects.toThrow();
		await idx.close();
	});

	it("rejects invalid schema type", () => {
		expect(() => createIndex({ title: "invalid_type" })).toThrowError(/unknown field type/);
	});
});

// =============================================================================
// Result shape
// =============================================================================

describe("result shape", () => {
	it("has expected camelCase shape", async () => {
		const idx = createIndex();
		await idx.add({ _id: "1", body: "shape test" });
		await idx.commit();

		const result = await idx.search("shape");
		expect(typeof result.totalHits).toBe("number");
		expect(Array.isArray(result.hits)).toBe(true);
		expect(typeof result.hits[0].docId).toBe("string");
		expect(typeof result.hits[0].score).toBe("number");
		expect(typeof result.aggregations).toBe("object");
		expect(typeof result.suggest).toBe("object");
		await idx.close();
	});

	it("numeric fields stored and retrieved correctly", async () => {
		const idx = createIndex({ body: "text", count: { type: "integer", stored: true } });
		await idx.add({ _id: "1", body: "numeric test", count: 42 });
		await idx.commit();

		const result = await idx.search({ query: "numeric", returnStored: true });
		expect(result.hits[0].fields).toBeTruthy();
		await idx.close();
	});
});

// =============================================================================
// Typed search (Zod schema)
// =============================================================================

describe("typed search", () => {
	const BodySchema = z.object({
		body: z.string(),
	});

	it("validates and returns typed fields with request object", async () => {
		const idx = createIndex();
		await idx.add({ _id: "1", body: "hello world" });
		await idx.commit();

		const result = await idx.search(BodySchema, { query: "hello" });
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].fields).toEqual({ body: "hello world" });
		await idx.close();
	});

	it("validates and returns typed fields with string query", async () => {
		const idx = createIndex();
		await idx.add({ _id: "1", body: "hello world" });
		await idx.commit();

		const result = await idx.search(BodySchema, "hello");
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].fields).toEqual({ body: "hello world" });
		await idx.close();
	});

	it("auto-sets returnStored when schema is provided", async () => {
		const idx = createIndex();
		await idx.add({ _id: "1", body: "auto stored" });
		await idx.commit();

		const result = await idx.search(BodySchema, { query: "auto" });
		expect(result.hits[0].fields).toBeTruthy();
		expect(result.hits[0].fields.body).toBe("auto stored");
		await idx.close();
	});

	it("throws with context when fields do not match schema", async () => {
		const idx = createIndex();
		await idx.add({ _id: "doc-42", body: "mismatch test" });
		await idx.commit();

		const StrictSchema = z.object({
			body: z.string(),
			missing: z.number(),
		});

		await expect(idx.search(StrictSchema, "mismatch")).rejects.toThrowError(
			/hit 0.*docId.*doc-42/s,
		);
		await idx.close();
	});

	it("succeeds with empty results", async () => {
		const idx = createIndex();
		await idx.commit();

		const result = await idx.search(BodySchema, "nonexistent_xyz");
		expect(result.hits).toHaveLength(0);
		await idx.close();
	});

	it("works with optional fields in schema", async () => {
		const idx = createIndex({ body: "text", tag: "keyword" });
		await idx.add({ _id: "1", body: "optional test" });
		await idx.commit();

		const FlexSchema = z.object({
			body: z.string(),
			tag: z.string().optional(),
		});

		const result = await idx.search(FlexSchema, "optional");
		expect(result.hits[0].fields.body).toBe("optional test");
		await idx.close();
	});

	it("works with multiple hits", async () => {
		const idx = createIndex();
		await idx.addMany([
			{ _id: "1", body: "alpha beta" },
			{ _id: "2", body: "alpha gamma" },
		]);
		await idx.commit();

		const result = await idx.search(BodySchema, "alpha");
		expect(result.totalHits).toBe(2);
		for (const hit of result.hits) {
			expect(hit.fields.body).toBeTruthy();
		}
		await idx.close();
	});

	it("applies Zod transforms to fields", async () => {
		const idx = createIndex();
		await idx.add({ _id: "1", body: "transform test" });
		await idx.commit();

		const UpperSchema = z.object({
			body: z.string().transform((s) => s.toUpperCase()),
		});

		const result = await idx.search(UpperSchema, "transform");
		expect(result.hits[0].fields.body).toBe("TRANSFORM TEST");
		await idx.close();
	});

	it("applies Zod defaults for missing fields", async () => {
		const idx = createIndex();
		await idx.add({ _id: "1", body: "defaults test" });
		await idx.commit();

		const WithDefault = z.object({
			body: z.string(),
			extra: z.string().default("fallback"),
		});

		const result = await idx.search(WithDefault, "defaults");
		expect(result.hits[0].fields.body).toBe("defaults test");
		expect(result.hits[0].fields.extra).toBe("fallback");
		await idx.close();
	});

	it("strips unknown keys by default (z.object behavior)", async () => {
		const idx = createIndex({ body: "text", tag: "keyword" });
		await idx.add({ _id: "1", body: "strip test", tag: "hello" });
		await idx.commit();

		const PartialSchema = z.object({
			body: z.string(),
		});

		const result = await idx.search(PartialSchema, "strip");
		expect(result.hits[0].fields.body).toBe("strip test");
		expect("tag" in result.hits[0].fields).toBe(false);
		await idx.close();
	});

	it("preserves unknown keys with passthrough", async () => {
		const idx = createIndex({ body: "text", tag: "keyword" });
		await idx.add({ _id: "1", body: "passthrough test", tag: "kept" });
		await idx.commit();

		const PassthroughSchema = z
			.object({
				body: z.string(),
			})
			.passthrough();

		const result = await idx.search(PassthroughSchema, "passthrough");
		expect(result.hits[0].fields.body).toBe("passthrough test");
		expect(result.hits[0].fields.tag).toBe("kept");
		await idx.close();
	});
});
