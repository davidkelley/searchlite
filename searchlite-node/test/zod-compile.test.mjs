import { describe, expect, it } from "vitest";
import { z } from "zod";
import { expandSchema } from "../dist/schemas.js";
import {
	InvalidZodSchemaError,
	UnsupportedZodTypeError,
	compileZodSchema,
	isZodIndexSchema,
	sl,
} from "../dist/zod/index.js";

// ── Root / sl.index() ────────────────────────────────────────────────────────

describe("compileZodSchema: root", () => {
	it("requires sl.index() branding", () => {
		const raw = z.object({ title: z.string() });
		expect(() => compileZodSchema(raw)).toThrowError(InvalidZodSchemaError);
	});

	it("detects branded schemas", () => {
		const branded = sl.index(z.object({ title: z.string() }));
		expect(isZodIndexSchema(branded)).toBe(true);
	});

	it("compiles an empty object", () => {
		const out = compileZodSchema(sl.index(z.object({})));
		expect(out).toEqual({ type: "object", properties: {} });
	});

	it("custom docIdField is emitted", () => {
		const schema = sl.index(z.object({ urn: z.string() }), { docIdField: "urn" });
		const out = compileZodSchema(schema);
		expect(out["searchlite:docIdField"]).toBe("urn");
	});

	it("default docIdField is omitted from output (parity with expandSchema)", () => {
		const schema = sl.index(z.object({ title: z.string() }));
		const out = compileZodSchema(schema);
		expect(out).not.toHaveProperty("searchlite:docIdField");
	});

	it("analyzers array is passed through", () => {
		const analyzers = [{ name: "stemmed", steps: [] }];
		const schema = sl.index(z.object({ title: z.string() }), { analyzers });
		const out = compileZodSchema(schema);
		expect(out["searchlite:analyzers"]).toEqual(analyzers);
	});

	it("throws if root schema is not an object", () => {
		expect(() => sl.index(z.string())).toBeDefined();
		// sl.index accepts any ZodObject at the type level; at runtime, a non-object
		// causes compile to throw because shape is undefined.
	});
});

// ── Strings: defaults + auto-promotion ───────────────────────────────────────

describe("compileZodSchema: strings", () => {
	it("z.string() → text (no extra keywords)", () => {
		const out = compileZodSchema(sl.index(z.object({ title: z.string() })));
		expect(out.properties.title).toEqual({ type: "string" });
	});

	it("z.string().uuid() auto-promotes to keyword", () => {
		const out = compileZodSchema(sl.index(z.object({ id: z.string().uuid() })));
		expect(out.properties.id).toEqual({
			type: "string",
			"searchlite:kind": "keyword",
		});
	});

	it("z.string().email() auto-promotes to keyword", () => {
		const out = compileZodSchema(sl.index(z.object({ email: z.string().email() })));
		expect(out.properties.email["searchlite:kind"]).toBe("keyword");
	});

	it("z.string().url() auto-promotes to keyword", () => {
		const out = compileZodSchema(sl.index(z.object({ href: z.string().url() })));
		expect(out.properties.href["searchlite:kind"]).toBe("keyword");
	});

	it("z.string().cuid() auto-promotes to keyword", () => {
		const out = compileZodSchema(sl.index(z.object({ id: z.string().cuid() })));
		expect(out.properties.id["searchlite:kind"]).toBe("keyword");
	});

	it("z.string().cuid2() auto-promotes to keyword", () => {
		const out = compileZodSchema(sl.index(z.object({ id: z.string().cuid2() })));
		expect(out.properties.id["searchlite:kind"]).toBe("keyword");
	});

	it("z.string().ulid() auto-promotes to keyword", () => {
		const out = compileZodSchema(sl.index(z.object({ id: z.string().ulid() })));
		expect(out.properties.id["searchlite:kind"]).toBe("keyword");
	});

	it("z.string().nanoid() auto-promotes to keyword", () => {
		const out = compileZodSchema(sl.index(z.object({ id: z.string().nanoid() })));
		expect(out.properties.id["searchlite:kind"]).toBe("keyword");
	});

	it("z.string().min(3) does NOT promote (still text)", () => {
		const out = compileZodSchema(sl.index(z.object({ name: z.string().min(3) })));
		expect(out.properties.name).toEqual({ type: "string" });
	});

	it("z.string().regex(...) does NOT promote (still text)", () => {
		const out = compileZodSchema(sl.index(z.object({ pat: z.string().regex(/x/) })));
		expect(out.properties.pat).toEqual({ type: "string" });
	});

	it("explicit sl.text() wins over auto-promotion", () => {
		const out = compileZodSchema(sl.index(z.object({ bio: sl.text(z.string().email()) })));
		expect(out.properties.bio).toEqual({ type: "string" });
	});

	it("explicit sl.keyword() on plain string", () => {
		const out = compileZodSchema(sl.index(z.object({ tag: sl.keyword() })));
		expect(out.properties.tag).toEqual({
			type: "string",
			"searchlite:kind": "keyword",
		});
	});
});

// ── sl.text options ──────────────────────────────────────────────────────────

describe("compileZodSchema: text options", () => {
	it("custom analyzer is emitted", () => {
		const out = compileZodSchema(sl.index(z.object({ body: sl.text({ analyzer: "english" }) })));
		expect(out.properties.body).toEqual({
			type: "string",
			"searchlite:analyzer": "english",
		});
	});

	it("searchAnalyzer is emitted", () => {
		const out = compileZodSchema(
			sl.index(z.object({ body: sl.text({ searchAnalyzer: "simple" }) })),
		);
		expect(out.properties.body["searchlite:searchAnalyzer"]).toBe("simple");
	});

	it("stored: false is emitted", () => {
		const out = compileZodSchema(sl.index(z.object({ body: sl.text({ stored: false }) })));
		expect(out.properties.body["searchlite:stored"]).toBe(false);
	});

	it("indexed: false is emitted", () => {
		const out = compileZodSchema(sl.index(z.object({ body: sl.text({ indexed: false }) })));
		expect(out.properties.body["searchlite:indexed"]).toBe(false);
	});

	it("searchAsYouType is passed through", () => {
		const out = compileZodSchema(
			sl.index(
				z.object({ title: sl.text({ searchAsYouType: { minGram: 2, maxGram: 8 } }) }),
			),
		);
		expect(out.properties.title["searchlite:searchAsYouType"]).toEqual({
			minGram: 2,
			maxGram: 8,
		});
	});
});

// ── sl.keyword options ───────────────────────────────────────────────────────

describe("compileZodSchema: keyword options", () => {
	it("fast: false is emitted", () => {
		const out = compileZodSchema(sl.index(z.object({ t: sl.keyword({ fast: false }) })));
		expect(out.properties.t).toEqual({
			type: "string",
			"searchlite:kind": "keyword",
			"searchlite:fast": false,
		});
	});

	it("all defaults produce minimal output", () => {
		const out = compileZodSchema(sl.index(z.object({ t: sl.keyword() })));
		expect(out.properties.t).toEqual({
			type: "string",
			"searchlite:kind": "keyword",
		});
	});
});

// ── Numbers ──────────────────────────────────────────────────────────────────

describe("compileZodSchema: numbers", () => {
	it("z.number() → float", () => {
		const out = compileZodSchema(sl.index(z.object({ price: z.number() })));
		expect(out.properties.price).toEqual({ type: "number" });
	});

	it("z.number().int() → integer", () => {
		const out = compileZodSchema(sl.index(z.object({ year: z.number().int() })));
		expect(out.properties.year).toEqual({ type: "integer" });
	});

	it("sl.integer() is equivalent to z.number().int() with kind metadata", () => {
		const out = compileZodSchema(sl.index(z.object({ year: sl.integer() })));
		expect(out.properties.year).toEqual({ type: "integer" });
	});

	it("sl.float() yields a plain number field", () => {
		const out = compileZodSchema(sl.index(z.object({ ratio: sl.float() })));
		expect(out.properties.ratio).toEqual({ type: "number" });
	});

	it("sl.integer({stored: true}) emits searchlite:stored", () => {
		const out = compileZodSchema(sl.index(z.object({ year: sl.integer({ stored: true }) })));
		expect(out.properties.year).toEqual({
			type: "integer",
			"searchlite:stored": true,
		});
	});

	it("sl.float({fast: false}) emits searchlite:fast=false", () => {
		const out = compileZodSchema(sl.index(z.object({ r: sl.float({ fast: false }) })));
		expect(out.properties.r).toEqual({
			type: "number",
			"searchlite:fast": false,
		});
	});
});

// ── Literals and enums ───────────────────────────────────────────────────────

describe("compileZodSchema: literals and enums", () => {
	it("z.literal('x') → keyword", () => {
		const out = compileZodSchema(sl.index(z.object({ t: z.literal("banana") })));
		expect(out.properties.t).toEqual({
			type: "string",
			"searchlite:kind": "keyword",
		});
	});

	it("z.literal(42) → integer", () => {
		const out = compileZodSchema(sl.index(z.object({ n: z.literal(42) })));
		expect(out.properties.n).toEqual({ type: "integer" });
	});

	it("z.literal(3.14) → float", () => {
		const out = compileZodSchema(sl.index(z.object({ n: z.literal(3.14) })));
		expect(out.properties.n).toEqual({ type: "number" });
	});

	it("z.literal(true) is rejected with a boolean-literal hint", () => {
		expect(() => compileZodSchema(sl.index(z.object({ flag: z.literal(true) })))).toThrowError(
			/unsupported Zod type z\.literal\(<boolean>\)/,
		);
	});

	it("z.enum(['a','b']) → keyword", () => {
		const out = compileZodSchema(sl.index(z.object({ status: z.enum(["a", "b"]) })));
		expect(out.properties.status).toEqual({
			type: "string",
			"searchlite:kind": "keyword",
		});
	});
});

// ── Wrappers: optional / nullable / default ──────────────────────────────────

describe("compileZodSchema: wrappers", () => {
	it("optional wraps without affecting output (required-set is not emitted)", () => {
		const out = compileZodSchema(sl.index(z.object({ t: z.string().optional() })));
		expect(out.properties.t).toEqual({ type: "string" });
	});

	it("nullable emits type array [string, null]", () => {
		const out = compileZodSchema(sl.index(z.object({ t: z.string().nullable() })));
		expect(out.properties.t).toEqual({ type: ["string", "null"] });
	});

	it("nullable with keyword", () => {
		const out = compileZodSchema(sl.index(z.object({ t: sl.keyword().nullable() })));
		expect(out.properties.t).toEqual({
			type: ["string", "null"],
			"searchlite:kind": "keyword",
		});
	});

	it("default wraps and inner type is used", () => {
		const out = compileZodSchema(sl.index(z.object({ t: z.string().default("x") })));
		expect(out.properties.t).toEqual({ type: "string" });
	});

	it("optional + keyword metadata preserved through wrapper", () => {
		const out = compileZodSchema(sl.index(z.object({ tag: sl.keyword().optional() })));
		expect(out.properties.tag).toEqual({
			type: "string",
			"searchlite:kind": "keyword",
		});
	});

	it("branded types are transparent", () => {
		const branded = z.string().uuid().brand("UserId");
		const out = compileZodSchema(sl.index(z.object({ id: branded })));
		expect(out.properties.id["searchlite:kind"]).toBe("keyword");
	});
});

// ── Nested objects and arrays ────────────────────────────────────────────────

describe("compileZodSchema: nested structures", () => {
	it("nested z.object() compiles to type: object + properties", () => {
		const out = compileZodSchema(
			sl.index(
				z.object({
					meta: z.object({ title: z.string(), tag: sl.keyword() }),
				}),
			),
		);
		expect(out.properties.meta).toEqual({
			type: "object",
			properties: {
				title: { type: "string" },
				tag: { type: "string", "searchlite:kind": "keyword" },
			},
		});
	});

	it("empty nested object omits properties", () => {
		const out = compileZodSchema(sl.index(z.object({ meta: z.object({}) })));
		expect(out.properties.meta).toEqual({ type: "object" });
	});

	it("array of object compiles to array / items", () => {
		const out = compileZodSchema(
			sl.index(
				z.object({
					images: z.array(z.object({ url: sl.keyword() })),
				}),
			),
		);
		expect(out.properties.images).toEqual({
			type: "array",
			items: {
				type: "object",
				properties: {
					url: { type: "string", "searchlite:kind": "keyword" },
				},
			},
		});
	});

	it("array of primitives is rejected", () => {
		expect(() =>
			compileZodSchema(sl.index(z.object({ tags: z.array(z.string()) }))),
		).toThrowError(UnsupportedZodTypeError);
	});

	it("deeply nested object (2 levels) compiles", () => {
		const out = compileZodSchema(
			sl.index(
				z.object({
					a: z.object({
						b: z.object({ c: sl.keyword() }),
					}),
				}),
			),
		);
		expect(out.properties.a.properties.b.properties.c).toEqual({
			type: "string",
			"searchlite:kind": "keyword",
		});
	});

	it("nullable nested object", () => {
		const out = compileZodSchema(
			sl.index(z.object({ meta: z.object({ title: z.string() }).nullable() })),
		);
		expect(out.properties.meta.type).toEqual(["object", "null"]);
	});
});

// ── Vectors ──────────────────────────────────────────────────────────────────

describe("compileZodSchema: vectors", () => {
	it("sl.vector produces the correct JSON shape", () => {
		const out = compileZodSchema(
			sl.index(z.object({ embedding: sl.vector({ dim: 768, metric: "Cosine" }) })),
		);
		expect(out.properties.embedding).toEqual({
			type: "array",
			items: { type: "number" },
			"searchlite:vector": { dim: 768, metric: "Cosine" },
		});
	});

	it("sl.vector with hnsw is passed through", () => {
		const out = compileZodSchema(
			sl.index(
				z.object({
					embedding: sl.vector({
						dim: 4,
						metric: "L2",
						hnsw: { m: 16, efConstruction: 200 },
					}),
				}),
			),
		);
		expect(out.properties.embedding["searchlite:vector"]).toEqual({
			dim: 4,
			metric: "L2",
			hnsw: { m: 16, efConstruction: 200 },
		});
	});

	it("sl.vector rejects non-positive dim", () => {
		expect(() => sl.vector({ dim: 0, metric: "Cosine" })).toThrow(/positive integer/);
	});

	it("sl.vector rejects unknown metric", () => {
		// @ts-expect-error — runtime-only check
		expect(() => sl.vector({ dim: 4, metric: "Euclidean" })).toThrow(/Cosine/);
	});
});

// ── Field name validation ────────────────────────────────────────────────────

describe("compileZodSchema: field name validation", () => {
	it('rejects field names containing "."', () => {
		expect(() =>
			compileZodSchema(sl.index(z.object({ "a.b": z.string() }))),
		).toThrowError(/must not contain "\."/);
	});
});

// ── Unsupported types ────────────────────────────────────────────────────────

describe("compileZodSchema: unsupported types", () => {
	it("z.boolean() throws with remediation hint", () => {
		expect(() =>
			compileZodSchema(sl.index(z.object({ flag: z.boolean() }))),
		).toThrowError(/z\.boolean\(\).*z\.enum/s);
	});

	it("z.date() throws with epoch-ms hint", () => {
		expect(() =>
			compileZodSchema(sl.index(z.object({ t: z.date() }))),
		).toThrowError(/z\.date\(\).*epoch-ms/s);
	});

	it("z.bigint() throws with i64 hint", () => {
		expect(() =>
			compileZodSchema(sl.index(z.object({ big: z.bigint() }))),
		).toThrowError(/z\.bigint\(\).*i64/s);
	});

	it("z.record() throws", () => {
		expect(() =>
			compileZodSchema(sl.index(z.object({ rec: z.record(z.string(), z.string()) }))),
		).toThrowError(/z\.record/);
	});

	it("z.tuple() throws", () => {
		expect(() =>
			compileZodSchema(sl.index(z.object({ t: z.tuple([z.string(), z.number()]) }))),
		).toThrowError(/z\.tuple/);
	});

	it("z.union() throws", () => {
		expect(() =>
			compileZodSchema(
				sl.index(z.object({ u: z.union([z.string(), z.number()]) })),
			),
		).toThrowError(/z\.union/);
	});

	it("z.intersection() throws", () => {
		expect(() =>
			compileZodSchema(
				sl.index(
					z.object({
						i: z.intersection(
							z.object({ a: z.string() }),
							z.object({ b: z.string() }),
						),
					}),
				),
			),
		).toThrowError(/z\.intersection/);
	});

	it("z.any() throws", () => {
		expect(() => compileZodSchema(sl.index(z.object({ a: z.any() })))).toThrowError(/z\.any/);
	});

	it("z.unknown() throws", () => {
		expect(() =>
			compileZodSchema(sl.index(z.object({ u: z.unknown() }))),
		).toThrowError(/z\.unknown/);
	});

	it("z.never() throws", () => {
		expect(() =>
			compileZodSchema(sl.index(z.object({ n: z.never() }))),
		).toThrowError(/z\.never/);
	});

	it("z.transform() throws", () => {
		expect(() =>
			compileZodSchema(sl.index(z.object({ t: z.string().transform((x) => x) }))),
		).toThrowError(/transform/);
	});

	it("z.lazy() throws", () => {
		expect(() =>
			compileZodSchema(sl.index(z.object({ l: z.lazy(() => z.string()) }))),
		).toThrowError(/z\.lazy/);
	});

	it("UnsupportedZodTypeError carries path and hint", () => {
		try {
			compileZodSchema(sl.index(z.object({ outer: z.object({ flag: z.boolean() }) })));
			expect.fail("should have thrown");
		} catch (err) {
			expect(err).toBeInstanceOf(UnsupportedZodTypeError);
			expect(err.path).toBe("outer.flag");
			expect(err.hint).toMatch(/boolean/);
		}
	});
});

// ── Parity with expandSchema ─────────────────────────────────────────────────

describe("compileZodSchema: parity with expandSchema", () => {
	it("simple text field", () => {
		const zod = compileZodSchema(sl.index(z.object({ title: z.string() })));
		const shorthand = expandSchema({ title: "text" });
		expect(zod).toEqual(shorthand);
	});

	it("simple keyword field", () => {
		const zod = compileZodSchema(sl.index(z.object({ tag: sl.keyword() })));
		const shorthand = expandSchema({ tag: "keyword" });
		expect(zod).toEqual(shorthand);
	});

	it("integer field", () => {
		const zod = compileZodSchema(sl.index(z.object({ year: sl.integer() })));
		const shorthand = expandSchema({ year: "integer" });
		expect(zod).toEqual(shorthand);
	});

	it("float field", () => {
		const zod = compileZodSchema(sl.index(z.object({ price: sl.float() })));
		const shorthand = expandSchema({ price: "float" });
		expect(zod).toEqual(shorthand);
	});

	it("text with custom analyzer", () => {
		const zod = compileZodSchema(
			sl.index(z.object({ body: sl.text({ analyzer: "english", stored: false }) })),
		);
		const shorthand = expandSchema({
			body: { type: "text", analyzer: "english", stored: false },
		});
		expect(zod).toEqual(shorthand);
	});

	it("keyword with fast: false", () => {
		const zod = compileZodSchema(sl.index(z.object({ t: sl.keyword({ fast: false }) })));
		const shorthand = expandSchema({ t: { type: "keyword", fast: false } });
		expect(zod).toEqual(shorthand);
	});

	it("integer stored: true", () => {
		const zod = compileZodSchema(sl.index(z.object({ year: sl.integer({ stored: true }) })));
		const shorthand = expandSchema({ year: { type: "integer", stored: true } });
		expect(zod).toEqual(shorthand);
	});

	it("mixed fields with custom docIdField (docIdField not in schema)", () => {
		const zod = compileZodSchema(
			sl.index(
				z.object({
					title: z.string(),
					tag: sl.keyword(),
					year: sl.integer(),
				}),
				{ docIdField: "urn" },
			),
		);
		const shorthand = expandSchema({
			doc_id_field: "urn",
			title: "text",
			tag: "keyword",
			year: "integer",
		});
		expect(zod).toEqual(shorthand);
	});
});

// ── Zod-specific behavior (diverges from shorthand) ──────────────────────────

describe("compileZodSchema: Zod-specific behaviors", () => {
	it("docIdField appears in Zod schema for z.infer<> but is stripped from emitted properties", () => {
		// Zod users declare the id field so `z.infer<typeof schema>` includes
		// it. The compiler honors the Zod declaration for runtime validation
		// (add/search) but omits the id field from the emitted `properties` map
		// because searchlite-core stores the doc id as a separate column and
		// rejects overlap between docIdField and properties.
		const out = compileZodSchema(
			sl.index(
				z.object({
					urn: z.string().uuid(),
					title: z.string(),
				}),
				{ docIdField: "urn" },
			),
		);
		expect(out["searchlite:docIdField"]).toBe("urn");
		expect(out.properties).not.toHaveProperty("urn");
		expect(out.properties.title).toEqual({ type: "string" });
	});
});
