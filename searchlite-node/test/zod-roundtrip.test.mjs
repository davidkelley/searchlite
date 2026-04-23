import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { z } from "zod";
import { EmbeddedIndex, compileZodSchema, expandSchema, sl } from "../dist/index.js";

// Round-trip parity tests.
//
// The whole value proposition of keeping three authoring paths (shorthand,
// raw JSON Schema, Zod) is that they produce IDENTICAL native behavior for
// equivalent logical schemas. These tests:
//
//  1. Define the same logical schema three ways.
//  2. Compare compiled JSON outputs structurally.
//  3. Exercise the native engine end-to-end with each variant and assert the
//     search results match.

let cleanup = [];

function tempDir() {
	const dir = mkdtempSync(join(tmpdir(), "searchlite-roundtrip-"));
	cleanup.push(dir);
	return dir;
}

afterEach(() => {
	for (const dir of cleanup) {
		rmSync(dir, { recursive: true, force: true });
	}
	cleanup = [];
});

// ── Helper: compile each form to JSON Schema output ──────────────────────────

function compileAll({ shorthand, raw, zod }) {
	return {
		shorthand: expandSchema(shorthand),
		raw: expandSchema(raw),
		zod: compileZodSchema(zod),
	};
}

// ── Case 1: single text field ────────────────────────────────────────────────

describe("round-trip: single text field", () => {
	const out = compileAll({
		shorthand: { title: "text" },
		raw: {
			type: "object",
			properties: { title: { type: "string" } },
		},
		zod: sl.index(z.object({ title: z.string() })),
	});

	it("JSON outputs are structurally equal", () => {
		expect(out.shorthand).toEqual(out.raw);
		expect(out.zod).toEqual(out.shorthand);
	});

	it("native engine produces identical search behavior", async () => {
		const results = [];
		for (const schema of [out.shorthand, out.raw, out.zod]) {
			const idx = new EmbeddedIndex(join(tempDir(), "idx"), { schema });
			await idx.add({ _id: "1", title: "hello world" });
			await idx.commit();
			const r = await idx.search("hello");
			results.push({ total: r.totalHits, firstDoc: r.hits[0]?.docId });
			await idx.close();
		}
		expect(results[0]).toEqual(results[1]);
		expect(results[1]).toEqual(results[2]);
		expect(results[0].total).toBe(1);
	});
});

// ── Case 2: mixed keyword + integer ──────────────────────────────────────────

describe("round-trip: keyword + integer", () => {
	const out = compileAll({
		shorthand: { tag: "keyword", year: "integer" },
		raw: {
			type: "object",
			properties: {
				tag: { type: "string", "searchlite:kind": "keyword" },
				year: { type: "integer" },
			},
		},
		zod: sl.index(
			z.object({
				tag: sl.keyword(),
				year: sl.integer(),
			}),
		),
	});

	it("keyword fields are structurally equal across all paths", () => {
		expect(out.zod.properties.tag).toEqual(out.shorthand.properties.tag);
	});

	it("Zod integer adds stored: true (intentional DX divergence)", () => {
		expect(out.zod.properties.year).toEqual({ type: "integer", "searchlite:stored": true });
		expect(out.shorthand.properties.year).toEqual({ type: "integer" });
	});
});

// ── Case 3: custom analyzer on text ──────────────────────────────────────────

describe("round-trip: text with custom analyzer", () => {
	const out = compileAll({
		shorthand: { body: { type: "text", analyzer: "english" } },
		raw: {
			type: "object",
			properties: {
				body: { type: "string", "searchlite:analyzer": "english" },
			},
		},
		zod: sl.index(z.object({ body: sl.text({ analyzer: "english" }) })),
	});

	it("JSON outputs are structurally equal", () => {
		expect(out.shorthand).toEqual(out.raw);
		expect(out.zod).toEqual(out.shorthand);
	});
});

// ── Case 4: keyword with fast: false ─────────────────────────────────────────

describe("round-trip: keyword with fast: false", () => {
	const out = compileAll({
		shorthand: { t: { type: "keyword", fast: false } },
		raw: {
			type: "object",
			properties: {
				t: { type: "string", "searchlite:kind": "keyword", "searchlite:fast": false },
			},
		},
		zod: sl.index(z.object({ t: sl.keyword({ fast: false }) })),
	});

	it("JSON outputs are structurally equal", () => {
		expect(out.shorthand).toEqual(out.raw);
		expect(out.zod).toEqual(out.shorthand);
	});
});

// ── Case 5: integer with stored: true ────────────────────────────────────────

describe("round-trip: integer with stored: true", () => {
	const out = compileAll({
		shorthand: { year: { type: "integer", stored: true } },
		raw: {
			type: "object",
			properties: {
				year: { type: "integer", "searchlite:stored": true },
			},
		},
		zod: sl.index(z.object({ year: sl.integer({ stored: true }) })),
	});

	it("JSON outputs are structurally equal", () => {
		expect(out.shorthand).toEqual(out.raw);
		expect(out.zod).toEqual(out.shorthand);
	});
});

// ── Case 6: float field with fast: false ─────────────────────────────────────

describe("round-trip: float with fast: false", () => {
	const out = compileAll({
		shorthand: { r: { type: "float", fast: false } },
		raw: {
			type: "object",
			properties: {
				r: { type: "number", "searchlite:fast": false },
			},
		},
		zod: sl.index(z.object({ r: sl.float({ fast: false }) })),
	});

	it("shorthand and raw are equal", () => {
		expect(out.shorthand).toEqual(out.raw);
	});

	it("Zod adds stored: true (intentional DX divergence on numerics)", () => {
		expect(out.zod.properties.r["searchlite:fast"]).toBe(false);
		expect(out.zod.properties.r["searchlite:stored"]).toBe(true);
	});
});

// ── Case 7: nullable text field ──────────────────────────────────────────────

describe("round-trip: nullable text", () => {
	const out = compileAll({
		shorthand: { t: { type: "text", nullable: true } },
		raw: {
			type: "object",
			properties: { t: { type: ["string", "null"] } },
		},
		zod: sl.index(z.object({ t: z.string().nullable() })),
	});

	it("JSON outputs are structurally equal", () => {
		expect(out.shorthand).toEqual(out.raw);
		expect(out.zod).toEqual(out.shorthand);
	});
});

// ── Case 8: custom docIdField (separate from properties) ─────────────────────

describe("round-trip: custom docIdField", () => {
	const out = compileAll({
		shorthand: {
			doc_id_field: "urn",
			title: "text",
			tag: "keyword",
		},
		raw: {
			type: "object",
			"searchlite:docIdField": "urn",
			properties: {
				title: { type: "string" },
				tag: { type: "string", "searchlite:kind": "keyword" },
			},
		},
		zod: sl.index(
			z.object({
				title: z.string(),
				tag: sl.keyword(),
			}),
			{ docIdField: "urn" },
		),
	});

	it("JSON outputs are structurally equal", () => {
		expect(out.shorthand).toEqual(out.raw);
		expect(out.zod).toEqual(out.shorthand);
	});

	it("native engine uses urn as the docIdField for each variant", async () => {
		for (const schema of [out.shorthand, out.raw, out.zod]) {
			const idx = new EmbeddedIndex(join(tempDir(), "idx"), { schema });
			await idx.add({ urn: "my-doc-1", title: "hello", tag: "n" });
			await idx.commit();
			const r = await idx.search("hello");
			expect(r.hits[0].docId).toBe("my-doc-1");
			await idx.close();
		}
	});
});

// ── Case 9: nested raw JSON vs nested Zod ────────────────────────────────────
//
// Shorthand cannot express nesting, so this case only compares raw vs Zod.

describe("round-trip: nested object", () => {
	const rawSchema = {
		type: "object",
		properties: {
			meta: {
				type: "object",
				properties: {
					sku: { type: "string", "searchlite:kind": "keyword" },
				},
			},
		},
	};
	const zodSchema = sl.index(
		z.object({
			meta: z.object({ sku: sl.keyword() }),
		}),
	);

	it("raw and Zod outputs are structurally equal", () => {
		expect(compileZodSchema(zodSchema)).toEqual(expandSchema(rawSchema));
	});
});

// ── Case 10: array-of-object nested ──────────────────────────────────────────

describe("round-trip: array of objects", () => {
	const rawSchema = {
		type: "object",
		properties: {
			items: {
				type: "array",
				items: {
					type: "object",
					properties: {
						name: { type: "string" },
					},
				},
			},
		},
	};
	const zodSchema = sl.index(
		z.object({
			items: z.array(z.object({ name: z.string() })),
		}),
	);

	it("raw and Zod outputs are structurally equal", () => {
		expect(compileZodSchema(zodSchema)).toEqual(expandSchema(rawSchema));
	});

	it("native engine accepts nested schemas from both paths without error", async () => {
		// Verify that the compiled outputs are accepted by the Rust side of both
		// paths. Nested search semantics (traversal into array-of-object fields)
		// require query-time `Nested` filters, which are covered by the
		// dedicated integration tests in `test/zod-embedded.test.mjs` — here we
		// just confirm schema equivalence makes it through construction + commit.
		for (const schema of [expandSchema(rawSchema), compileZodSchema(zodSchema)]) {
			const idx = new EmbeddedIndex(join(tempDir(), "idx"), { schema });
			await idx.add({
				_id: "1",
				items: [{ name: "alpha" }, { name: "bravo" }],
			});
			await idx.commit();
			await idx.close();
		}
	});
});

// ── Case 11: UUID field (Zod auto-promotes to keyword) ───────────────────────
//
// The Zod path auto-promotes `z.string().uuid()` to keyword. The shorthand
// has no equivalent refinement, so the comparison is: explicit keyword
// shorthand ≡ Zod with uuid auto-promoted.

describe("round-trip: uuid auto-promotes to keyword", () => {
	it("shorthand keyword ≡ Zod uuid", () => {
		const shorthand = expandSchema({ id: "keyword" });
		// The Zod compiler omits the id field from properties because it matches
		// docIdField; to compare compiled properties, use a non-id field name.
		const zod = compileZodSchema(sl.index(z.object({ ref: z.string().uuid() })));
		const shorthandRef = expandSchema({ ref: "keyword" });
		expect(zod).toEqual(shorthandRef);
	});
});

// ── Case 12: all three paths produce identical hit ordering ──────────────────

describe("round-trip: search result ordering parity", () => {
	const shorthand = { title: "text", tag: "keyword" };
	const raw = {
		type: "object",
		properties: {
			title: { type: "string" },
			tag: { type: "string", "searchlite:kind": "keyword" },
		},
	};
	const zod = sl.index(
		z.object({
			title: z.string(),
			tag: sl.keyword(),
		}),
	);

	it("top-1 hit ID is identical across all three paths", async () => {
		const docs = [
			{ _id: "doc-a", title: "lorem ipsum dolor sit amet", tag: "test" },
			{ _id: "doc-b", title: "lorem", tag: "prod" },
			{ _id: "doc-c", title: "amet", tag: "test" },
		];

		const results = [];
		for (const schema of [shorthand, raw, zod]) {
			const idx = new EmbeddedIndex(join(tempDir(), "idx"), { schema });
			for (const d of docs) await idx.add(d);
			await idx.commit();
			const r = await idx.search("lorem");
			results.push(r.hits.map((h) => h.docId));
			await idx.close();
		}
		expect(results[0]).toEqual(results[1]);
		expect(results[1]).toEqual(results[2]);
	});
});
