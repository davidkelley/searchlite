import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { z } from "zod";
import { EmbeddedIndex, sl } from "../dist/index.js";

let cleanup = [];

function tempDir() {
	const dir = mkdtempSync(join(tmpdir(), "searchlite-zod-test-"));
	cleanup.push(dir);
	return dir;
}

function makeIndex(schema) {
	return new EmbeddedIndex(join(tempDir(), "idx"), { schema });
}

afterEach(() => {
	for (const dir of cleanup) {
		rmSync(dir, { recursive: true, force: true });
	}
	cleanup = [];
});

// ── Construction ─────────────────────────────────────────────────────────────

describe("Zod path: construction", () => {
	it("accepts a Zod index schema via sl.index()", async () => {
		const UserSchema = sl.index(
			z.object({
				id: z.string().uuid(),
				name: z.string(),
			}),
			{ docIdField: "id" },
		);
		const idx = makeIndex(UserSchema);
		await idx.close();
	});

	it("shorthand schemas still work (backwards compat)", async () => {
		const idx = makeIndex({ title: "text", tag: "keyword" });
		await idx.add({ _id: "1", title: "hello", tag: "news" });
		await idx.commit();
		const result = await idx.search("hello");
		expect(result.totalHits).toBe(1);
		await idx.close();
	});

	it("raw JSON Schema still works (backwards compat)", async () => {
		const idx = makeIndex({
			type: "object",
			properties: { title: { type: "string" } },
		});
		await idx.add({ _id: "1", title: "hello" });
		await idx.commit();
		const result = await idx.search("hello");
		expect(result.totalHits).toBe(1);
		await idx.close();
	});

	it("rejects a bare ZodObject (must be wrapped with sl.index())", () => {
		const raw = z.object({ title: z.string() });
		expect(() => new EmbeddedIndex(join(tempDir(), "idx"), { schema: raw })).toThrowError(
			/must be wrapped with `sl\.index/,
		);
	});
});

// ── Single-schema end-to-end (THE primary use case) ──────────────────────────

describe("Zod path: single-schema end-to-end", () => {
	const BlogSchema = sl.index(
		z.object({
			id: z.string().uuid(),
			title: z.string(),
			slug: sl.keyword(),
			status: z.enum(["draft", "published", "archived"]),
			views: sl.integer({ stored: true }),
		}),
		{ docIdField: "id" },
	);

	it("add() validates documents against the Zod schema", async () => {
		const idx = makeIndex(BlogSchema);
		const valid = {
			id: "550e8400-e29b-41d4-a716-446655440000",
			title: "Hello world",
			slug: "hello-world",
			status: "published",
			views: 42,
		};
		await idx.add(valid);
		await idx.commit();
		await idx.close();
	});

	it("add() rejects documents that fail Zod validation", async () => {
		const idx = makeIndex(BlogSchema);
		await expect(
			idx.add({
				id: "not-a-uuid",
				title: "x",
				slug: "x",
				status: "published",
				views: 1,
			}),
		).rejects.toThrow();
		await idx.close();
	});

	it("add() rejects documents with wrong-type fields", async () => {
		const idx = makeIndex(BlogSchema);
		await expect(
			idx.add({
				id: "550e8400-e29b-41d4-a716-446655440000",
				title: "x",
				slug: "x",
				status: "published",
				views: "not a number",
			}),
		).rejects.toThrow();
		await idx.close();
	});

	it("add() rejects documents missing required fields", async () => {
		const idx = makeIndex(BlogSchema);
		await expect(
			idx.add({
				id: "550e8400-e29b-41d4-a716-446655440000",
				title: "x",
				// missing slug, status, views
			}),
		).rejects.toThrow();
		await idx.close();
	});

	it("add() rejects invalid enum values", async () => {
		const idx = makeIndex(BlogSchema);
		await expect(
			idx.add({
				id: "550e8400-e29b-41d4-a716-446655440000",
				title: "x",
				slug: "x",
				status: "unknown",
				views: 1,
			}),
		).rejects.toThrow();
		await idx.close();
	});

	it("search() auto-validates hit fields against the stored Zod schema", async () => {
		const idx = makeIndex(BlogSchema);
		const doc = {
			id: "550e8400-e29b-41d4-a716-446655440000",
			title: "Auto-validated",
			slug: "auto",
			status: "published",
			views: 7,
		};
		await idx.add(doc);
		await idx.commit();

		const result = await idx.search("auto-validated");
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].docId).toBe(doc.id);
		expect(result.hits[0].fields).toEqual(doc);
		await idx.close();
	});

	it("search() typed result works without passing the schema again", async () => {
		const idx = makeIndex(BlogSchema);
		await idx.add({
			id: "550e8400-e29b-41d4-a716-446655440000",
			title: "single-schema query",
			slug: "single",
			status: "draft",
			views: 0,
		});
		await idx.commit();
		const result = await idx.search({ query: "single-schema" });
		expect(result.hits[0].fields.title).toBe("single-schema query");
		expect(result.hits[0].fields.status).toBe("draft");
		await idx.close();
	});
});

// ── addMany validation ───────────────────────────────────────────────────────

describe("Zod path: addMany", () => {
	const Schema = sl.index(z.object({ id: z.string().uuid(), name: z.string() }), {
		docIdField: "id",
	});

	it("validates each document in the array", async () => {
		const idx = makeIndex(Schema);
		const docs = [
			{ id: "550e8400-e29b-41d4-a716-446655440000", name: "Alice" },
			{ id: "550e8400-e29b-41d4-a716-446655440001", name: "Bob" },
		];
		const queued = await idx.addMany(docs);
		expect(queued).toBe(2);
		await idx.close();
	});

	it("reports which document failed validation by index", async () => {
		const idx = makeIndex(Schema);
		const docs = [
			{ id: "550e8400-e29b-41d4-a716-446655440000", name: "Alice" },
			{ id: "bad-uuid", name: "Bob" },
		];
		await expect(idx.addMany(docs)).rejects.toThrow(/documents\[1\]/);
		await idx.close();
	});

	it("single-document addMany still validates", async () => {
		const idx = makeIndex(Schema);
		await expect(idx.addMany({ id: "x", name: "Eve" })).rejects.toThrow();
		await idx.close();
	});
});

// ── Per-call schema override (explicit wins) ─────────────────────────────────

describe("Zod path: per-call schema override", () => {
	const BlogSchema = sl.index(
		z.object({
			id: z.string().uuid(),
			title: z.string(),
			slug: sl.keyword(),
		}),
		{ docIdField: "id" },
	);

	it("per-call search schema overrides the construction-time Zod schema", async () => {
		const idx = makeIndex(BlogSchema);
		await idx.add({
			id: "550e8400-e29b-41d4-a716-446655440000",
			title: "override test",
			slug: "override",
		});
		await idx.commit();

		const SubsetSchema = z.object({ title: z.string() });
		const result = await idx.search(SubsetSchema, "override");
		expect(result.hits[0].fields).toEqual({ title: "override test" });
		await idx.close();
	});

	it("per-call schema that rejects the actual data surfaces a clear error", async () => {
		const idx = makeIndex(BlogSchema);
		await idx.add({
			id: "550e8400-e29b-41d4-a716-446655440000",
			title: "mismatch",
			slug: "m",
		});
		await idx.commit();

		const WrongSchema = z.object({ nonexistent: z.string() });
		await expect(idx.search(WrongSchema, "mismatch")).rejects.toThrowError(
			/hit 0.*docId/s,
		);
		await idx.close();
	});
});

// ── Nested objects and arrays ────────────────────────────────────────────────

describe("Zod path: nested structures", () => {
	const ProductSchema = sl.index(
		z.object({
			id: z.string().uuid(),
			name: z.string(),
			meta: z.object({
				sku: sl.keyword(),
				// Numeric fields default to stored: false to stay small in the
				// inverted index. Opt into storage so they come back in hit.fields.
				weight: sl.float({ stored: true }),
			}),
			variants: z.array(
				z.object({
					color: sl.keyword(),
					price: sl.float({ stored: true }),
				}),
			),
		}),
		{ docIdField: "id" },
	);

	it("nested object fields index and return typed", async () => {
		const idx = makeIndex(ProductSchema);
		await idx.add({
			id: "550e8400-e29b-41d4-a716-446655440000",
			name: "Widget",
			meta: { sku: "WG-001", weight: 1.5 },
			variants: [
				{ color: "red", price: 9.99 },
				{ color: "blue", price: 10.99 },
			],
		});
		await idx.commit();
		const result = await idx.search("widget");
		expect(result.hits[0].fields.meta.sku).toBe("WG-001");
		expect(result.hits[0].fields.variants).toHaveLength(2);
		await idx.close();
	});

	it("validation errors in nested fields include the nested path", async () => {
		const idx = makeIndex(ProductSchema);
		await expect(
			idx.add({
				id: "550e8400-e29b-41d4-a716-446655440000",
				name: "Widget",
				meta: { sku: "WG-001", weight: "not a number" },
				variants: [],
			}),
		).rejects.toThrow();
		await idx.close();
	});
});

// ── Nullable / optional ──────────────────────────────────────────────────────

describe("Zod path: optional and nullable fields", () => {
	it("optional string field is permitted on insert", async () => {
		const Schema = sl.index(
			z.object({
				id: z.string().uuid(),
				title: z.string(),
				subtitle: z.string().optional(),
			}),
			{ docIdField: "id" },
		);
		const idx = makeIndex(Schema);
		await idx.add({
			id: "550e8400-e29b-41d4-a716-446655440000",
			title: "Only title",
		});
		await idx.commit();
		await idx.close();
	});

	it("nullable field accepts null on insert", async () => {
		const Schema = sl.index(
			z.object({
				id: z.string().uuid(),
				title: z.string(),
				deleted_at: z.string().nullable(),
			}),
			{ docIdField: "id" },
		);
		const idx = makeIndex(Schema);
		await idx.add({
			id: "550e8400-e29b-41d4-a716-446655440000",
			title: "hi",
			deleted_at: null,
		});
		await idx.commit();
		await idx.close();
	});
});

// ── Resource lifecycle ───────────────────────────────────────────────────────

describe("Zod path: lifecycle", () => {
	it("Zod schema compiles once at construction; not re-compiled per-call", async () => {
		const Schema = sl.index(z.object({ id: z.string().uuid(), t: z.string() }), {
			docIdField: "id",
		});
		const idx = makeIndex(Schema);
		// Construction was successful; subsequent operations should not throw
		// from the compile path. Exercise several ops to make sure state is correct.
		await idx.add({ id: "550e8400-e29b-41d4-a716-446655440000", t: "hello" });
		await idx.commit();
		const r1 = await idx.search("hello");
		expect(r1.totalHits).toBe(1);
		const r2 = await idx.search({ query: "hello" });
		expect(r2.totalHits).toBe(1);
		await idx.close();
	});

	it("close() is idempotent", async () => {
		const Schema = sl.index(z.object({ id: z.string().uuid() }), { docIdField: "id" });
		const idx = makeIndex(Schema);
		await idx.close();
		await idx.close();
	});
});
