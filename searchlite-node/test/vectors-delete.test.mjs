import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { EmbeddedIndex } from "../dist/index.js";

// Regression guards for Stage 1 of the searchlite-memory work:
//   1. `vectors` is on by default, so a `searchlite:vector` schema is accepted
//      and both the query-node and typed `vectorQuery` forms run.
//   2. `delete` / `deleteMany` are bound and remove documents (commit-on-delete).

let cleanup = [];

function tempDir() {
	const dir = mkdtempSync(join(tmpdir(), "searchlite-vecdel-"));
	cleanup.push(dir);
	return dir;
}

afterEach(() => {
	for (const dir of cleanup) {
		rmSync(dir, { recursive: true, force: true });
	}
	cleanup = [];
});

// Raw JSON Schema (the "already JSON Schema" passthrough branch) with a 4-D
// cosine vector field — the shorthand schema form does not support vectors.
// `_id` is the implicit doc-id field and must NOT be redeclared in properties.
const vectorSchema = {
	type: "object",
	properties: {
		body: { type: "string" },
		embedding: {
			type: "array",
			items: { type: "number" },
			"searchlite:vector": { dim: 4, metric: "Cosine" },
		},
	},
};

async function seedVectors() {
	const idx = new EmbeddedIndex(join(tempDir(), "idx"), { schema: vectorSchema });
	await idx.addMany([
		{ _id: "a", body: "alpha", embedding: [1, 0, 0, 0] },
		{ _id: "b", body: "bravo", embedding: [0, 1, 0, 0] },
		{ _id: "c", body: "charlie", embedding: [0, 0, 1, 0] },
	]);
	await idx.commit();
	return idx;
}

describe("vectors feature (default-enabled binding)", () => {
	it("accepts a searchlite:vector schema without throwing", async () => {
		const idx = new EmbeddedIndex(join(tempDir(), "idx"), { schema: vectorSchema });
		expect(idx).toBeTruthy();
		await idx.close();
	});

	it("ranks by similarity via the query-node form and returns vectorScore", async () => {
		const idx = await seedVectors();
		const res = await idx.search({
			query: { type: "vector", field: "embedding", vector: [0.9, 0.1, 0, 0], k: 3, alpha: 0.0 },
			returnStored: true,
			limit: 3,
		});
		expect(res.hits.length).toBeGreaterThan(0);
		expect(res.hits[0].docId).toBe("a");
		expect(typeof res.hits[0].vectorScore).toBe("number");
		await idx.close();
	});

	it("supports the typed top-level vectorQuery passthrough (camelCase → snake_case)", async () => {
		const idx = await seedVectors();
		const res = await idx.search({
			query: { type: "match_all" },
			vectorQuery: {
				field: "embedding",
				vector: [0, 0.95, 0.05, 0],
				k: 3,
				alpha: 0.0,
				efSearch: 32,
				candidateSize: 16,
			},
			returnStored: true,
			limit: 3,
		});
		expect(res.hits[0].docId).toBe("b");
		expect(typeof res.hits[0].vectorScore).toBe("number");
		await idx.close();
	});
});

describe("delete / deleteMany binding", () => {
	it("delete removes a single doc and commits", async () => {
		const idx = new EmbeddedIndex(join(tempDir(), "idx"), { schema: { body: "text" } });
		await idx.addMany([
			{ _id: "1", body: "keep this" },
			{ _id: "2", body: "remove me" },
		]);
		await idx.commit();
		expect((await idx.search("remove")).totalHits).toBe(1);

		await idx.delete("2");
		expect((await idx.search("remove")).totalHits).toBe(0);
		expect((await idx.search("keep")).totalHits).toBe(1);
		await idx.close();
	});

	it("deleteMany removes several docs and returns the submitted count", async () => {
		const idx = new EmbeddedIndex(join(tempDir(), "idx"), { schema: { body: "text" } });
		await idx.addMany([
			{ _id: "1", body: "one" },
			{ _id: "2", body: "two" },
			{ _id: "3", body: "three" },
		]);
		await idx.commit();

		const n = await idx.deleteMany(["1", "3"]);
		expect(n).toBe(2);
		expect((await idx.search("one")).totalHits).toBe(0);
		expect((await idx.search("two")).totalHits).toBe(1);
		expect((await idx.search("three")).totalHits).toBe(0);
		await idx.close();
	});

	it("deleteMany([]) is a no-op returning 0", async () => {
		const idx = new EmbeddedIndex(join(tempDir(), "idx"), { schema: { body: "text" } });
		await idx.add({ _id: "1", body: "x" });
		await idx.commit();
		expect(await idx.deleteMany([])).toBe(0);
		expect((await idx.search("x")).totalHits).toBe(1);
		await idx.close();
	});

	it("rejects invalid delete arguments", async () => {
		const idx = new EmbeddedIndex(join(tempDir(), "idx"), { schema: { body: "text" } });
		await expect(idx.delete("")).rejects.toThrow();
		await expect(idx.deleteMany(["ok", ""])).rejects.toThrow();
		await idx.close();
	});
});
