import { describe, expect, it } from "vitest";
import { expandSchema, SearchRequestSchema } from "../dist/schemas.js";

describe("expandSchema", () => {
	describe("shorthand strings", () => {
		it("expands 'text' with defaults", () => {
			const schema = expandSchema({ title: "text" });
			expect(schema.type).toBe("object");
			expect(schema.properties.title).toEqual({ type: "string" });
		});

		it("expands 'keyword' with defaults", () => {
			const schema = expandSchema({ tag: "keyword" });
			expect(schema.properties.tag).toEqual({
				type: "string",
				"searchlite:kind": "keyword",
			});
		});

		it("expands 'integer' with defaults", () => {
			const schema = expandSchema({ year: "integer" });
			expect(schema.properties.year).toEqual({ type: "integer" });
		});

		it("expands 'float' with defaults", () => {
			const schema = expandSchema({ price: "float" });
			expect(schema.properties.price).toEqual({ type: "number" });
		});
	});

	describe("detailed objects", () => {
		it("overrides text field defaults", () => {
			const schema = expandSchema({
				body: { type: "text", stored: false, analyzer: "english" },
			});
			expect(schema.properties.body).toEqual({
				type: "string",
				"searchlite:analyzer": "english",
				"searchlite:stored": false,
			});
		});

		it("overrides keyword field defaults", () => {
			const schema = expandSchema({
				status: { type: "keyword", stored: false, fast: false },
			});
			expect(schema.properties.status).toEqual({
				type: "string",
				"searchlite:kind": "keyword",
				"searchlite:stored": false,
				"searchlite:fast": false,
			});
		});

		it("overrides numeric field defaults", () => {
			const schema = expandSchema({
				count: { type: "integer", stored: true },
			});
			expect(schema.properties.count).toEqual({
				type: "integer",
				"searchlite:stored": true,
			});
		});

		it("sets nullable on any field type", () => {
			const schema = expandSchema({
				notes: { type: "text", nullable: true },
			});
			expect(schema.properties.notes.type).toEqual(["string", "null"]);
		});
	});

	describe("mixed shorthand and detailed", () => {
		it("groups fields by type correctly", () => {
			const schema = expandSchema({
				title: "text",
				body: { type: "text", analyzer: "english" },
				tag: "keyword",
				year: "integer",
				price: "float",
			});
			expect(Object.keys(schema.properties)).toHaveLength(5);
			expect(schema.properties.title.type).toBe("string");
			expect(schema.properties.body["searchlite:analyzer"]).toBe("english");
			expect(schema.properties.tag["searchlite:kind"]).toBe("keyword");
			expect(schema.properties.year.type).toBe("integer");
			expect(schema.properties.price.type).toBe("number");
		});
	});

	describe("metadata fields", () => {
		it("omits searchlite:docIdField when default _id", () => {
			const schema = expandSchema({ title: "text" });
			expect(schema["searchlite:docIdField"]).toBeUndefined();
		});

		it("allows doc_id_field override", () => {
			const schema = expandSchema({ doc_id_field: "uuid", title: "text" });
			expect(schema["searchlite:docIdField"]).toBe("uuid");
		});

		it("passes through analyzers", () => {
			const schema = expandSchema({ analyzers: [{ name: "custom" }], title: "text" });
			expect(schema["searchlite:analyzers"]).toEqual([{ name: "custom" }]);
		});

		it("omits analyzers when not provided", () => {
			const schema = expandSchema({ title: "text" });
			expect(schema["searchlite:analyzers"]).toBeUndefined();
		});
	});

	describe("JSON Schema pass-through", () => {
		it("returns JSON Schema format unchanged", () => {
			const jsonSchema = {
				type: "object",
				properties: {
					body: { type: "string" },
				},
			};
			const result = expandSchema(jsonSchema);
			expect(result).toBe(jsonSchema);
		});

		it("passes through $schema-prefixed input", () => {
			const jsonSchema = {
				$schema: "https://searchlite.dev/draft/2025/schema",
				type: "object",
				properties: {},
			};
			const result = expandSchema(jsonSchema);
			expect(result).toBe(jsonSchema);
		});
	});

	describe("validation", () => {
		it("rejects unknown field type", () => {
			expect(() => expandSchema({ title: "invalid" })).toThrowError(/unknown field type/);
		});

		it("rejects empty field name", () => {
			expect(() => expandSchema({ "": "text" })).toThrowError(/field name must not be empty/);
		});

		it("rejects field name with dot", () => {
			expect(() => expandSchema({ "nested.field": "text" })).toThrowError(/must not contain "\."/);
		});

		it("rejects field name matching doc_id_field", () => {
			expect(() => expandSchema({ _id: "text" })).toThrowError(/conflicts with doc_id_field/);
		});

		it("rejects field name matching custom doc_id_field", () => {
			expect(() => expandSchema({ doc_id_field: "uuid", uuid: "keyword" })).toThrowError(
				/conflicts with doc_id_field/,
			);
		});

		it("rejects non-object input", () => {
			expect(() => expandSchema(null)).toThrowError(/schema must be a plain object/);
		});

		it("rejects array input", () => {
			expect(() => expandSchema([])).toThrowError(/schema must be a plain object/);
		});

		it("rejects old-format schemas", () => {
			expect(() =>
				expandSchema({ text_fields: [], keyword_fields: [], numeric_fields: [] }),
			).toThrowError(/legacy field-array schema/);
		});

		it("rejects old-format schemas with only keyword_fields", () => {
			expect(() => expandSchema({ keyword_fields: [] })).toThrowError(/legacy field-array schema/);
		});

		it("rejects old-format schemas with only numeric_fields", () => {
			expect(() => expandSchema({ numeric_fields: [] })).toThrowError(/legacy field-array schema/);
		});

		it("rejects JSON Schema input without type=object", () => {
			expect(() => expandSchema({ properties: { x: { type: "string" } } })).toThrowError(
				/type: "object"/,
			);
		});

		it("rejects JSON Schema input with non-object properties", () => {
			expect(() => expandSchema({ type: "object", properties: "not-an-object" })).toThrowError(
				/properties.*plain object/,
			);
		});

		it("rejects non-string doc_id_field", () => {
			expect(() => expandSchema({ doc_id_field: 123, title: "text" })).toThrowError(
				/doc_id_field must be a non-empty string/,
			);
		});

		it("rejects empty doc_id_field", () => {
			expect(() => expandSchema({ doc_id_field: "", title: "text" })).toThrowError(
				/doc_id_field must be a non-empty string/,
			);
		});
	});
});

// =============================================================================
// SearchRequestSchema — field coverage
// =============================================================================

describe("SearchRequestSchema", () => {
	describe("candidateSize and bmwBlockSize", () => {
		// Regression: both fields appeared in `REQUEST_KEY_MAP` but were
		// undeclared in the schema, so the default strip-mode z.object() silently
		// dropped user-supplied values before they could reach the mapper. Callers
		// could not tune the WAND candidate pool or BMW block size from Node at
		// all — the mapping entries were dead code.
		it("preserves candidateSize through validation", () => {
			const parsed = SearchRequestSchema.parse({ query: "x", candidateSize: 250 });
			expect(parsed.candidateSize).toBe(250);
		});

		it("preserves bmwBlockSize through validation", () => {
			const parsed = SearchRequestSchema.parse({ query: "x", bmwBlockSize: 32 });
			expect(parsed.bmwBlockSize).toBe(32);
		});

		it("rejects non-positive candidateSize", () => {
			expect(() => SearchRequestSchema.parse({ query: "x", candidateSize: 0 })).toThrow();
			expect(() => SearchRequestSchema.parse({ query: "x", candidateSize: -1 })).toThrow();
		});

		it("rejects non-integer candidateSize", () => {
			expect(() => SearchRequestSchema.parse({ query: "x", candidateSize: 1.5 })).toThrow();
		});

		it("rejects non-positive bmwBlockSize", () => {
			expect(() => SearchRequestSchema.parse({ query: "x", bmwBlockSize: 0 })).toThrow();
		});

		// The wire schema declares `candidate_size` and `bmw_block_size` as
		// `["integer", "null"]`. Some callers round-trip payloads where the
		// override has been cleared by sending `null`; the Rust `Option<usize>`
		// accepts that too, so the camelCase fields must as well.
		it("accepts explicit null for candidateSize and bmwBlockSize", () => {
			const parsed = SearchRequestSchema.parse({
				query: "x",
				candidateSize: null,
				bmwBlockSize: null,
			});
			expect(parsed.candidateSize).toBeNull();
			expect(parsed.bmwBlockSize).toBeNull();
		});
	});
});

// =============================================================================
// SearchRequestSchema.sort — normalization to canonical {field, order?}
// =============================================================================
//
// Regression: before this fix, `SortSpecSchema` accepted three shorthand
// forms that all failed at the Rust `SortSpec = {field: String, order: Option}`
// deserialization step, and rejected the canonical form that would have
// worked. Sorting was effectively broken from the Node client. The fix
// normalizes every accepted shape to the canonical wire format defined by
// `search-request.schema.json`'s `sort_spec`.

describe("SearchRequestSchema.sort", () => {
	it("accepts the canonical {field, order} form", () => {
		const parsed = SearchRequestSchema.parse({
			query: "x",
			sort: [{ field: "price", order: "asc" }],
		});
		expect(parsed.sort).toEqual([{ field: "price", order: "asc" }]);
	});

	it("accepts canonical {field} without order", () => {
		const parsed = SearchRequestSchema.parse({ query: "x", sort: [{ field: "price" }] });
		expect(parsed.sort).toEqual([{ field: "price" }]);
	});

	it("normalizes string shorthand to canonical", () => {
		const parsed = SearchRequestSchema.parse({ query: "x", sort: ["price"] });
		expect(parsed.sort).toEqual([{ field: "price" }]);
	});

	it("normalizes single-key {field: 'asc'} shorthand to canonical", () => {
		const parsed = SearchRequestSchema.parse({ query: "x", sort: [{ price: "asc" }] });
		expect(parsed.sort).toEqual([{ field: "price", order: "asc" }]);
	});

	// The shorthand-order variant is tried before the canonical variant in
	// the union so that `{field: "asc"}` reads as "sort by the field named
	// `field`, ascending" — the natural shorthand interpretation — rather
	// than as canonical `{field: "asc"}` which would mean "sort by a field
	// literally named 'asc'". Users with a field whose name happens to
	// equal "asc"/"desc" must use the explicit canonical form (with an
	// `order`) to disambiguate; this is documented intentional behavior.
	it("treats {field: 'asc'} as shorthand for sort by 'field' ascending", () => {
		const parsed = SearchRequestSchema.parse({ query: "x", sort: [{ field: "asc" }] });
		expect(parsed.sort).toEqual([{ field: "field", order: "asc" }]);
	});

	it("treats {field: 'desc'} as shorthand for sort by 'field' descending", () => {
		const parsed = SearchRequestSchema.parse({ query: "x", sort: [{ field: "desc" }] });
		expect(parsed.sort).toEqual([{ field: "field", order: "desc" }]);
	});

	it("falls through to canonical when the value is not asc/desc", () => {
		// `{field: "name"}` cannot match the shorthand-order variant
		// because "name" isn't a valid order, so it falls through to the
		// canonical variant and means "sort by the field named 'name', no
		// explicit order".
		const parsed = SearchRequestSchema.parse({ query: "x", sort: [{ field: "name" }] });
		expect(parsed.sort).toEqual([{ field: "name" }]);
	});

	it("normalizes single-key {field: {order}} shorthand to canonical", () => {
		const parsed = SearchRequestSchema.parse({
			query: "x",
			sort: [{ price: { order: "desc" } }],
		});
		expect(parsed.sort).toEqual([{ field: "price", order: "desc" }]);
	});

	it("normalizes mixed forms within a single sort array", () => {
		const parsed = SearchRequestSchema.parse({
			query: "x",
			sort: [
				{ field: "price", order: "desc" },
				"title",
				{ year: "asc" },
				{ author: { order: "desc" } },
			],
		});
		expect(parsed.sort).toEqual([
			{ field: "price", order: "desc" },
			{ field: "title" },
			{ field: "year", order: "asc" },
			{ field: "author", order: "desc" },
		]);
	});

	it("rejects multi-key shorthand records", () => {
		// `{a: "asc", b: "desc"}` is ambiguous — two fields in one sort slot. The
		// canonical form would be `[{field: "a", order: "asc"}, {field: "b", order: "desc"}]`.
		expect(() =>
			SearchRequestSchema.parse({ query: "x", sort: [{ a: "asc", b: "desc" }] }),
		).toThrow();
	});

	it("rejects invalid order values", () => {
		expect(() => SearchRequestSchema.parse({ query: "x", sort: [{ price: "up" }] })).toThrow();
		expect(() =>
			SearchRequestSchema.parse({ query: "x", sort: [{ field: "price", order: "up" }] }),
		).toThrow();
	});

	it("rejects non-string sort entries", () => {
		expect(() => SearchRequestSchema.parse({ query: "x", sort: [42] })).toThrow();
		expect(() => SearchRequestSchema.parse({ query: "x", sort: [null] })).toThrow();
	});
});
