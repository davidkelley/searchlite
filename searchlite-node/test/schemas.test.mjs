import { describe, expect, it } from "vitest";
import { expandSchema } from "../dist/schemas.js";

describe("expandSchema", () => {
	describe("shorthand strings", () => {
		it("expands 'text' with defaults", () => {
			const schema = expandSchema({ title: "text" });
			expect(schema.text_fields).toEqual([
				{ name: "title", analyzer: "default", stored: true, indexed: true, nullable: false },
			]);
			expect(schema.keyword_fields).toEqual([]);
			expect(schema.numeric_fields).toEqual([]);
		});

		it("expands 'keyword' with defaults", () => {
			const schema = expandSchema({ tag: "keyword" });
			expect(schema.keyword_fields).toEqual([
				{ name: "tag", stored: true, indexed: true, fast: true, nullable: false },
			]);
		});

		it("expands 'integer' with defaults", () => {
			const schema = expandSchema({ year: "integer" });
			expect(schema.numeric_fields).toEqual([
				{ name: "year", i64: true, fast: true, stored: false, nullable: false },
			]);
		});

		it("expands 'float' with defaults", () => {
			const schema = expandSchema({ price: "float" });
			expect(schema.numeric_fields).toEqual([
				{ name: "price", i64: false, fast: true, stored: false, nullable: false },
			]);
		});
	});

	describe("detailed objects", () => {
		it("overrides text field defaults", () => {
			const schema = expandSchema({
				body: { type: "text", stored: false, analyzer: "english" },
			});
			expect(schema.text_fields[0]).toEqual({
				name: "body",
				analyzer: "english",
				stored: false,
				indexed: true,
				nullable: false,
			});
		});

		it("overrides keyword field defaults", () => {
			const schema = expandSchema({
				status: { type: "keyword", stored: false, fast: false },
			});
			expect(schema.keyword_fields[0]).toEqual({
				name: "status",
				stored: false,
				indexed: true,
				fast: false,
				nullable: false,
			});
		});

		it("overrides numeric field defaults", () => {
			const schema = expandSchema({
				count: { type: "integer", stored: true },
			});
			expect(schema.numeric_fields[0]).toEqual({
				name: "count",
				i64: true,
				fast: true,
				stored: true,
				nullable: false,
			});
		});

		it("sets nullable on any field type", () => {
			const schema = expandSchema({
				notes: { type: "text", nullable: true },
			});
			expect(schema.text_fields[0].nullable).toBe(true);
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
			expect(schema.text_fields).toHaveLength(2);
			expect(schema.keyword_fields).toHaveLength(1);
			expect(schema.numeric_fields).toHaveLength(2);
		});
	});

	describe("metadata fields", () => {
		it("sets doc_id_field to _id by default", () => {
			const schema = expandSchema({ title: "text" });
			expect(schema.doc_id_field).toBe("_id");
		});

		it("allows doc_id_field override", () => {
			const schema = expandSchema({ doc_id_field: "uuid", title: "text" });
			expect(schema.doc_id_field).toBe("uuid");
		});

		it("passes through analyzers", () => {
			const schema = expandSchema({ analyzers: [{ name: "custom" }], title: "text" });
			expect(schema.analyzers).toEqual([{ name: "custom" }]);
		});

		it("defaults analyzers to empty array", () => {
			const schema = expandSchema({ title: "text" });
			expect(schema.analyzers).toEqual([]);
		});

		it("defaults nested_fields to empty array", () => {
			const schema = expandSchema({ title: "text" });
			expect(schema.nested_fields).toEqual([]);
		});
	});

	describe("core format pass-through", () => {
		it("returns core format unchanged", () => {
			const core = {
				text_fields: [
					{ name: "body", analyzer: "default", stored: true, indexed: true, nullable: false },
				],
				keyword_fields: [],
				numeric_fields: [],
			};
			const result = expandSchema(core);
			expect(result).toBe(core);
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
			expect(() => expandSchema(null)).toThrowError(/schema must be an object/);
		});

		it("rejects non-string doc_id_field", () => {
			expect(() => expandSchema({ doc_id_field: 123, title: "text" })).toThrowError(
				/doc_id_field must be a string/,
			);
		});
	});
});
