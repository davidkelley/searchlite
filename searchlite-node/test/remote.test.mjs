import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { RemoteIndex } from "../dist/index.js";

// --- Mock fetch helper ---

function mockFetch(responses) {
	const calls = [];
	let callIndex = 0;
	const fn = vi.fn(async (url, init) => {
		calls.push({
			url,
			method: init?.method,
			headers: init?.headers,
			body: init?.body ? JSON.parse(init.body) : undefined,
		});
		const response =
			typeof responses === "function"
				? responses(callIndex)
				: Array.isArray(responses)
					? responses[callIndex]
					: responses;
		callIndex++;
		return {
			ok: response.ok ?? true,
			status: response.status ?? 200,
			statusText: response.statusText ?? "OK",
			json: async () => response.body,
		};
	});
	fn._calls = calls;
	return fn;
}

const SEARCH_RESULT = {
	total_hits_estimate: 2,
	hits: [
		{ doc_id: "1", score: 1.5, fields: { body: "hello world" } },
		{ doc_id: "2", score: 0.8, fields: null },
	],
	aggregations: {},
};

// =============================================================================
// Constructor
// =============================================================================

describe("constructor", () => {
	it("requires non-empty baseUrl", () => {
		expect(() => new RemoteIndex("", "idx")).toThrowError(/baseUrl.*non-empty/);
	});

	it("requires non-empty indexName", () => {
		expect(() => new RemoteIndex("http://localhost:8080", "")).toThrowError(/indexName.*non-empty/);
	});

	it("strips trailing slashes from baseUrl", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://localhost:8080///", "my-index", { fetch });
		await idx.search("test");
		expect(fetch._calls[0].url).toBe("http://localhost:8080/indexes/my-index/search");
	});
});

// =============================================================================
// URL construction
// =============================================================================

describe("URL construction", () => {
	it("constructs correct search URL", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://host:9200", "products", { fetch });
		await idx.search("test");
		expect(fetch._calls[0].url).toBe("http://host:9200/indexes/products/search");
	});

	it("constructs correct bulk URL", async () => {
		const fetch = mockFetch({ body: { queued: 1 } });
		const idx = new RemoteIndex("http://host:9200", "products", { fetch });
		await idx.add({ _id: "1", body: "test" });
		expect(fetch._calls[0].url).toBe("http://host:9200/indexes/products/bulk");
	});

	it("constructs correct commit URL", async () => {
		const fetch = mockFetch({ body: {} });
		const idx = new RemoteIndex("http://host:9200", "products", { fetch });
		await idx.commit();
		expect(fetch._calls[0].url).toBe("http://host:9200/indexes/products/commit");
	});

	it("constructs correct compact URL", async () => {
		const fetch = mockFetch({ body: {} });
		const idx = new RemoteIndex("http://host:9200", "products", { fetch });
		await idx.compact();
		expect(fetch._calls[0].url).toBe("http://host:9200/indexes/products/compact");
	});

	it("encodes index name in URL", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://host:9200", "my index", { fetch });
		await idx.search("test");
		expect(fetch._calls[0].url).toBe("http://host:9200/indexes/my%20index/search");
	});
});

// =============================================================================
// Request formatting
// =============================================================================

describe("request formatting", () => {
	it("sends POST with JSON content type", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.search("hello");
		expect(fetch._calls[0].method).toBe("POST");
		expect(fetch._calls[0].headers["Content-Type"]).toBe("application/json");
	});

	it("sends X-Write-Key header when configured", async () => {
		const fetch = mockFetch({ body: { queued: 1 } });
		const idx = new RemoteIndex("http://host:9200", "idx", { writeKey: "secret", fetch });
		await idx.add({ _id: "1", body: "test" });
		expect(fetch._calls[0].headers["X-Write-Key"]).toBe("secret");
	});

	it("omits X-Write-Key header when not configured", async () => {
		const fetch = mockFetch({ body: { queued: 1 } });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.add({ _id: "1", body: "test" });
		expect(fetch._calls[0].headers["X-Write-Key"]).toBeUndefined();
	});

	it("converts camelCase search request to snake_case", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.search({ query: "hello", returnStored: true, trackTotalHits: true });
		const body = fetch._calls[0].body;
		expect(body.return_stored).toBe(true);
		expect(body.track_total_hits).toBe(true);
		expect(body.returnStored).toBeUndefined();
	});

	// Regression: candidateSize and bmwBlockSize appeared in the snake-case
	// mapping but were undeclared in SearchRequestSchema, so Zod's default
	// strip mode silently dropped them before they could be mapped. Users
	// could not tune these from the Node client. These tests lock in the
	// round-trip to the server.
	it("forwards candidateSize as candidate_size", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.search({ query: "hello", candidateSize: 250 });
		const body = fetch._calls[0].body;
		expect(body.candidate_size).toBe(250);
		expect(body.candidateSize).toBeUndefined();
	});

	it("forwards bmwBlockSize as bmw_block_size", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.search({ query: "hello", bmwBlockSize: 32 });
		const body = fetch._calls[0].body;
		expect(body.bmw_block_size).toBe(32);
		expect(body.bmwBlockSize).toBeUndefined();
	});

	// Regression: SortSpecSchema previously accepted shorthand forms that
	// the Rust `SortSpec` struct rejects, and rejected the canonical
	// `{field, order}` shape that actually works on the wire. These tests
	// assert that every accepted form lands on the server as the canonical
	// shape defined by `search-request.schema.json`.
	it("sends canonical sort {field, order} unchanged", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.search({
			query: "hello",
			sort: [{ field: "price", order: "asc" }],
		});
		expect(fetch._calls[0].body.sort).toEqual([{ field: "price", order: "asc" }]);
	});

	it("normalizes string sort shorthand to canonical", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.search({ query: "hello", sort: ["price"] });
		expect(fetch._calls[0].body.sort).toEqual([{ field: "price" }]);
	});

	it("normalizes {field: 'asc'} sort shorthand to canonical", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.search({ query: "hello", sort: [{ price: "desc" }] });
		expect(fetch._calls[0].body.sort).toEqual([{ field: "price", order: "desc" }]);
	});

	it("normalizes {field: {order}} sort shorthand to canonical", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.search({ query: "hello", sort: [{ price: { order: "asc" } }] });
		expect(fetch._calls[0].body.sort).toEqual([{ field: "price", order: "asc" }]);
	});

	it("normalizes mixed sort forms within a single request", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.search({
			query: "hello",
			sort: [{ field: "price", order: "desc" }, "title", { year: "asc" }],
		});
		expect(fetch._calls[0].body.sort).toEqual([
			{ field: "price", order: "desc" },
			{ field: "title" },
			{ field: "year", order: "asc" },
		]);
	});

	it("wraps single doc in bulk format for add()", async () => {
		const fetch = mockFetch({ body: { queued: 1 } });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.add({ _id: "doc1", body: "test" });
		expect(fetch._calls[0].body).toEqual({ docs: [{ _id: "doc1", body: "test" }] });
	});

	it("wraps array in bulk format for addMany()", async () => {
		const fetch = mockFetch({ body: { queued: 2 } });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		const count = await idx.addMany([
			{ _id: "1", body: "one" },
			{ _id: "2", body: "two" },
		]);
		expect(count).toBe(2);
		expect(fetch._calls[0].body.docs).toHaveLength(2);
	});

	it("wraps single object in array for addMany()", async () => {
		const fetch = mockFetch({ body: { queued: 1 } });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		const count = await idx.addMany({ _id: "1", body: "one" });
		expect(count).toBe(1);
		expect(fetch._calls[0].body.docs).toHaveLength(1);
	});

	it("sends simple string query as search request", async () => {
		const fetch = mockFetch({ body: SEARCH_RESULT });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.search("hello");
		expect(fetch._calls[0].body).toEqual({ query: "hello", limit: 10 });
	});
});

// =============================================================================
// Response transformation
// =============================================================================

describe("response transformation", () => {
	it("transforms snake_case response to camelCase", async () => {
		const fetch = mockFetch({
			body: {
				total_hits_estimate: 1,
				hits: [{ doc_id: "abc", score: 2.5, sort_key: [1], vector_score: 0.9 }],
				next_cursor: "cur123",
				next_search_after: [42],
				aggregations: { tags: { buckets: [] } },
				suggest: { didYouMean: {} },
			},
		});
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		const result = await idx.search("test");

		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe("abc");
		expect(result.hits[0].score).toBe(2.5);
		expect(result.hits[0].sortKey).toEqual([1]);
		expect(result.hits[0].vectorScore).toBe(0.9);
		expect(result.nextCursor).toBe("cur123");
		expect(result.nextSearchAfter).toEqual([42]);
		expect(result.aggregations.tags).toBeTruthy();
		expect(result.suggest.didYouMean).toBeTruthy();
	});

	it("defaults aggregations and suggest to empty objects", async () => {
		const fetch = mockFetch({
			body: { total_hits_estimate: 0, hits: [] },
		});
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		const result = await idx.search("test");

		expect(result.aggregations).toEqual({});
		expect(result.suggest).toEqual({});
	});

	it("transforms inner_hits recursively", async () => {
		const fetch = mockFetch({
			body: {
				total_hits_estimate: 1,
				hits: [
					{
						doc_id: "parent",
						score: 1.0,
						inner_hits: [{ doc_id: "child", score: 0.5 }],
					},
				],
			},
		});
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		const result = await idx.search("test");

		expect(result.hits[0].innerHits).toHaveLength(1);
		expect(result.hits[0].innerHits[0].docId).toBe("child");
	});
});

// =============================================================================
// Error handling
// =============================================================================

describe("error handling", () => {
	it("throws on HTTP error with reason from response body", async () => {
		const fetch = mockFetch({
			ok: false,
			status: 404,
			statusText: "Not Found",
			body: { error: { type: "index_missing", reason: "index 'missing' does not exist" } },
		});
		const idx = new RemoteIndex("http://host:9200", "missing", { fetch });
		await expect(idx.search("test")).rejects.toThrowError(/index 'missing' does not exist/);
	});

	it("falls back to statusText when body parse fails", async () => {
		const fn = vi.fn(async () => ({
			ok: false,
			status: 500,
			statusText: "Internal Server Error",
			json: async () => {
				throw new Error("invalid json");
			},
		}));
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch: fn });
		await expect(idx.search("test")).rejects.toThrowError(/Internal Server Error/);
	});

	it("includes status code in error message", async () => {
		const fetch = mockFetch({
			ok: false,
			status: 401,
			statusText: "Unauthorized",
			body: { error: { type: "write_key_required", reason: "write key required" } },
		});
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await expect(idx.commit()).rejects.toThrowError(/401/);
	});
});

// =============================================================================
// Validation
// =============================================================================

describe("validation", () => {
	it("rejects invalid documents", async () => {
		const fetch = mockFetch({ body: {} });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await expect(idx.add("not an object")).rejects.toThrow();
		await expect(idx.add(null)).rejects.toThrow();
	});

	it("rejects invalid addMany input", async () => {
		const fetch = mockFetch({ body: {} });
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await expect(idx.addMany("not valid")).rejects.toThrow();
		await expect(idx.addMany(42)).rejects.toThrow();
	});
});

// =============================================================================
// Typed search
// =============================================================================

describe("typed search", () => {
	const BodySchema = z.object({
		body: z.string(),
	});

	it("validates fields against Zod schema", async () => {
		const fetch = mockFetch({
			body: {
				total_hits_estimate: 1,
				hits: [{ doc_id: "1", score: 1.0, fields: { body: "hello world" } }],
			},
		});
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		const result = await idx.search(BodySchema, "hello");
		expect(result.hits[0].fields).toEqual({ body: "hello world" });
	});

	it("auto-sets returnStored when schema is provided", async () => {
		const fetch = mockFetch({
			body: {
				total_hits_estimate: 1,
				hits: [{ doc_id: "1", score: 1.0, fields: { body: "test" } }],
			},
		});
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		await idx.search(BodySchema, { query: "test" });
		expect(fetch._calls[0].body.return_stored).toBe(true);
	});

	it("throws when fields do not match schema", async () => {
		const fetch = mockFetch({
			body: {
				total_hits_estimate: 1,
				hits: [{ doc_id: "doc-42", score: 1.0, fields: { wrong: 123 } }],
			},
		});
		const idx = new RemoteIndex("http://host:9200", "idx", { fetch });
		const StrictSchema = z.object({ body: z.string() });
		await expect(idx.search(StrictSchema, "test")).rejects.toThrowError(/hit 0.*docId.*doc-42/s);
	});
});

// =============================================================================
// close() is a no-op
// =============================================================================

describe("close", () => {
	it("resolves without error", async () => {
		const idx = new RemoteIndex("http://host:9200", "idx");
		await idx.close();
	});

	it("can be called multiple times", async () => {
		const idx = new RemoteIndex("http://host:9200", "idx");
		await idx.close();
		await idx.close();
	});
});
