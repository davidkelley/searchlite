import { z } from "zod";
import { InvalidZodSchemaError, UnsupportedZodTypeError } from "./errors";
import {
	type SearchliteFieldMetadata,
	SearchliteFieldRegistry,
	SearchliteIndexRegistry,
} from "./registries";
import {
	type InferredKind,
	type WrapperState,
	getDefType,
	inferKind,
	resolveFieldMetadata,
	wrapperState,
} from "./rules";

// ── Public types ─────────────────────────────────────────────────────────────

/**
 * JSON Schema output shape, matching `expandSchema()`'s `JsonSchemaOutput`.
 * Kept structurally identical so round-trip tests can diff outputs.
 */
export interface JsonSchemaOutput {
	type: "object";
	"searchlite:docIdField"?: string;
	"searchlite:analyzers"?: unknown[];
	properties: Record<string, Record<string, unknown>>;
	[key: string]: unknown;
}

/** Brand marker on the Zod object returned by `sl.index(...)`. */
declare const __searchliteIndexBrand: unique symbol;

/**
 * A Zod object schema that has been tagged as a searchlite index root via
 * `sl.index(...)`. The brand is a phantom type marker — at runtime, it's the
 * original `z.ZodObject` with metadata attached to `SearchliteIndexRegistry`.
 */
export type ZodIndexSchema<TSchema extends z.ZodObject<z.ZodRawShape> = z.ZodObject<z.ZodRawShape>> =
	TSchema & { readonly [__searchliteIndexBrand]: "searchlite:index" };

/**
 * Runtime predicate: returns true when the passed value has been registered as
 * a searchlite index root (via `sl.index(...)`).
 */
export function isZodIndexSchema(value: unknown): value is ZodIndexSchema {
	if (!value || typeof value !== "object") return false;
	if (getDefType(value) !== "object") return false;
	return SearchliteIndexRegistry.get(value as never) !== undefined;
}

// ── Entry point ──────────────────────────────────────────────────────────────

/**
 * Compile a branded `ZodIndexSchema` into the `JsonSchemaOutput` that the
 * native searchlite binding accepts. Output shape is structurally identical
 * to `expandSchema()` for equivalent logical schemas.
 */
export function compileZodSchema(schema: ZodIndexSchema): JsonSchemaOutput {
	if (!isZodIndexSchema(schema)) {
		throw new InvalidZodSchemaError({
			path: "",
			message: "schema must be wrapped with `sl.index(...)` before compiling",
		});
	}

	const indexMeta = SearchliteIndexRegistry.get(schema as never) ?? {};
	const docIdField = indexMeta.docIdField ?? "_id";

	const shape = getShape(schema);
	if (!shape) {
		throw new InvalidZodSchemaError({
			path: "",
			message: "sl.index() requires a z.object({...}) at the root",
		});
	}

	const properties: Record<string, Record<string, unknown>> = {};

	for (const [name, fieldSchema] of Object.entries(shape)) {
		validateFieldName(name, docIdField);
		// The docIdField is stored as a separate column by the Rust engine and
		// must NOT appear in `properties` (searchlite-core rejects the overlap
		// in manifest validation). Zod users typically declare the id field in
		// their z.object({...}) so `z.infer<>` includes it; we honor that by
		// keeping it in the Zod schema (which still validates the value on
		// insert) but omitting it from the emitted properties map.
		if (name === docIdField) continue;
		properties[name] = emitField(fieldSchema, name);
	}

	const output: JsonSchemaOutput = {
		type: "object",
		properties,
	};

	if (docIdField !== "_id") {
		output["searchlite:docIdField"] = docIdField;
	}
	if (indexMeta.analyzers && indexMeta.analyzers.length > 0) {
		output["searchlite:analyzers"] = indexMeta.analyzers;
	}

	return output;
}

// ── Field name validation ────────────────────────────────────────────────────
//
// Diverges from `expandSchema` in one way: we allow the docIdField to appear
// as a property in the Zod schema. Zod users naturally declare their id field
// (`id: z.string().uuid()`) so that `z.infer<>` carries it; rejecting it would
// force them to keep the shape out-of-sync with `docIdField`. The Rust side
// accepts schemas where docIdField is also a regular property, so emitting
// both is safe.

function validateFieldName(name: string, _docIdField: string): void {
	if (name.length === 0) {
		throw new InvalidZodSchemaError({
			path: name,
			message: "field name must not be empty",
		});
	}
	if (name.includes(".")) {
		throw new InvalidZodSchemaError({
			path: name,
			message: `field name "${name}" must not contain "." (use nested fields instead)`,
		});
	}
}

// ── Field emitter ────────────────────────────────────────────────────────────

function emitField(schema: unknown, path: string): Record<string, unknown> {
	const state = wrapperState(schema);
	const nullable = state.nullable;
	const inner = state.inner;

	const meta = resolveFieldMetadata(
		schema,
		SearchliteFieldRegistry,
		z.globalRegistry as { get(schema: never): Record<string, unknown> | undefined },
	);

	// Vector is an explicit opt-in: only when kind === "vector" in metadata.
	if (meta.kind === "vector") {
		return emitVector(inner, path, nullable, meta);
	}

	const innerType = getDefType(inner);

	// Nested object and arrays route to their own emitters.
	if (innerType === "object") {
		return emitObject(inner, path, nullable);
	}
	if (innerType === "array") {
		return emitArray(inner, path, nullable);
	}

	// Primitive leaves: compute effective kind (explicit > inferred).
	const effectiveKind = meta.kind ?? inferKind(inner);
	if (!effectiveKind) {
		rejectUnsupported(inner, path);
	}

	switch (effectiveKind) {
		case "text":
			return emitText(inner, nullable, meta);
		case "keyword":
			return emitKeyword(inner, nullable, meta);
		case "integer":
			return emitInteger(inner, nullable, meta);
		case "float":
			return emitFloat(inner, nullable, meta);
		default:
			rejectUnsupported(inner, path);
	}
}

// ── Leaf emitters ────────────────────────────────────────────────────────────

function emitText(
	inner: unknown,
	nullable: boolean,
	meta: SearchliteFieldMetadata,
): Record<string, unknown> {
	// Text accepts ZodString (default), ZodLiteral<string> (rare), ZodEnum (but
	// enums infer to keyword already; reach here only via explicit override).
	// Validate: the inner must resolve to a string-compatible JSON type.
	ensureStringCompatible(inner);

	const prop: Record<string, unknown> = {
		type: nullable ? ["string", "null"] : "string",
	};
	const analyzer = meta.analyzer ?? "default";
	if (analyzer !== "default") prop["searchlite:analyzer"] = analyzer;
	if (meta.searchAnalyzer) prop["searchlite:searchAnalyzer"] = meta.searchAnalyzer;
	if (meta.stored === false) prop["searchlite:stored"] = false;
	if (meta.indexed === false) prop["searchlite:indexed"] = false;
	if (meta.searchAsYouType) {
		prop["searchlite:searchAsYouType"] = { ...meta.searchAsYouType };
	}
	return prop;
}

function emitKeyword(
	inner: unknown,
	nullable: boolean,
	meta: SearchliteFieldMetadata,
): Record<string, unknown> {
	ensureStringCompatible(inner);

	const prop: Record<string, unknown> = {
		type: nullable ? ["string", "null"] : "string",
		"searchlite:kind": "keyword",
	};
	if (meta.stored === false) prop["searchlite:stored"] = false;
	if (meta.indexed === false) prop["searchlite:indexed"] = false;
	if (meta.fast === false) prop["searchlite:fast"] = false;
	return prop;
}

function emitInteger(
	inner: unknown,
	nullable: boolean,
	meta: SearchliteFieldMetadata,
): Record<string, unknown> {
	ensureNumericCompatible(inner, "integer");

	const prop: Record<string, unknown> = {
		type: nullable ? ["integer", "null"] : "integer",
	};
	if (meta.fast === false) prop["searchlite:fast"] = false;
	if (meta.stored === true) prop["searchlite:stored"] = true;
	return prop;
}

function emitFloat(
	inner: unknown,
	nullable: boolean,
	meta: SearchliteFieldMetadata,
): Record<string, unknown> {
	ensureNumericCompatible(inner, "float");

	const prop: Record<string, unknown> = {
		type: nullable ? ["number", "null"] : "number",
	};
	if (meta.fast === false) prop["searchlite:fast"] = false;
	if (meta.stored === true) prop["searchlite:stored"] = true;
	return prop;
}

// ── Complex emitters ─────────────────────────────────────────────────────────

function emitObject(
	inner: unknown,
	path: string,
	nullable: boolean,
): Record<string, unknown> {
	const shape = getShape(inner);
	if (!shape) {
		throw new InvalidZodSchemaError({
			path,
			message: "expected an object shape on z.object(...)",
		});
	}

	const properties: Record<string, Record<string, unknown>> = {};
	for (const [name, childSchema] of Object.entries(shape)) {
		const childPath = path ? `${path}.${name}` : name;
		properties[name] = emitField(childSchema, childPath);
	}

	const prop: Record<string, unknown> = {
		type: nullable ? ["object", "null"] : "object",
	};
	if (Object.keys(properties).length > 0) {
		prop.properties = properties;
	}
	return prop;
}

function emitArray(
	inner: unknown,
	path: string,
	nullable: boolean,
): Record<string, unknown> {
	const element = getArrayElement(inner);
	if (!element) {
		throw new InvalidZodSchemaError({
			path,
			message: "z.array(...) requires an element schema",
		});
	}

	const elementType = getDefType(element);
	if (elementType !== "object") {
		// Plain-primitive arrays aren't supported by searchlite-core (other than
		// `z.array(z.number())` as a vector, which goes through `emitVector`).
		throw new UnsupportedZodTypeError({
			path,
			zodType: `z.array(z.${elementType ?? "unknown"}())`,
			hint:
				"searchlite arrays must contain objects (nested multi-valued fields) " +
				"or numbers annotated with `sl.vector({...})`. For a single-valued " +
				"list of primitives, store as a nested object with named fields, or " +
				"join to a string.",
		});
	}

	const itemsPath = `${path}.items`;
	const items: Record<string, unknown> = {
		type: "object",
	};
	const childShape = getShape(element);
	if (childShape) {
		const properties: Record<string, Record<string, unknown>> = {};
		for (const [name, childSchema] of Object.entries(childShape)) {
			properties[name] = emitField(childSchema, `${itemsPath}.${name}`);
		}
		if (Object.keys(properties).length > 0) {
			items.properties = properties;
		}
	}

	const prop: Record<string, unknown> = {
		type: nullable ? ["array", "null"] : "array",
		items,
	};
	return prop;
}

function emitVector(
	inner: unknown,
	path: string,
	nullable: boolean,
	meta: SearchliteFieldMetadata,
): Record<string, unknown> {
	if (meta.dim == null || meta.metric == null) {
		throw new InvalidZodSchemaError({
			path,
			message: "vector fields require `dim` and `metric` metadata (use `sl.vector({...})`)",
		});
	}
	const innerType = getDefType(inner);
	if (innerType !== "array") {
		throw new InvalidZodSchemaError({
			path,
			message: "vector fields must wrap `z.array(z.number())` (use `sl.vector({...})`)",
		});
	}
	const element = getArrayElement(inner);
	if (getDefType(element) !== "number") {
		throw new InvalidZodSchemaError({
			path,
			message: "vector elements must be `z.number()` (use `sl.vector({...})`)",
		});
	}

	const vectorConfig: Record<string, unknown> = {
		dim: meta.dim,
		metric: meta.metric,
	};
	if (meta.hnsw && Object.keys(meta.hnsw).length > 0) {
		vectorConfig.hnsw = meta.hnsw;
	}

	const prop: Record<string, unknown> = {
		type: nullable ? ["array", "null"] : "array",
		items: { type: "number" },
		"searchlite:vector": vectorConfig,
	};
	return prop;
}

// ── Helpers: shape/element extraction ────────────────────────────────────────

function getShape(schema: unknown): Record<string, unknown> | undefined {
	if (!schema || typeof schema !== "object") return undefined;
	const def = (schema as { _def?: { shape?: Record<string, unknown> } })._def;
	return def?.shape;
}

function getArrayElement(schema: unknown): unknown {
	if (!schema || typeof schema !== "object") return undefined;
	const def = (schema as { _def?: { element?: unknown } })._def;
	return def?.element;
}

// ── Compatibility guards ─────────────────────────────────────────────────────

function ensureStringCompatible(inner: unknown): void {
	const t = getDefType(inner);
	if (t === "string" || t === "enum") return;
	if (t === "literal") {
		const values = (inner as { _def?: { values?: unknown[] } })._def?.values ?? [];
		if (values.every((v) => typeof v === "string")) return;
	}
	throw new InvalidZodSchemaError({
		path: "",
		message: `cannot emit string-kind field from non-string Zod type ${t ?? "unknown"}`,
	});
}

function ensureNumericCompatible(inner: unknown, target: "integer" | "float"): void {
	const t = getDefType(inner);
	if (t === "number") return;
	if (t === "literal") {
		const values = (inner as { _def?: { values?: unknown[] } })._def?.values ?? [];
		if (values.every((v) => typeof v === "number")) {
			if (target === "integer" && !values.every((v) => Number.isInteger(v as number))) {
				throw new InvalidZodSchemaError({
					path: "",
					message: "integer field cannot be a non-integer literal",
				});
			}
			return;
		}
	}
	throw new InvalidZodSchemaError({
		path: "",
		message: `cannot emit numeric-kind field from non-numeric Zod type ${t ?? "unknown"}`,
	});
}

// ── Unsupported type rejection ───────────────────────────────────────────────

const UNSUPPORTED_HINTS: Record<string, string> = {
	boolean:
		"searchlite-core has no boolean kind. Use `z.enum(['true','false'])` with `sl.keyword()`, or model as an integer (0/1).",
	date:
		"searchlite-core has no date kind. Use `z.number().int()` for epoch-ms and convert at your application boundary.",
	bigint:
		"searchlite-core has no 128-bit integer kind. Use `z.number().int()` (if values fit in i64) or store as a keyword string.",
	record:
		"dynamic keys can't map to typed columns. Lift known keys to a `z.object({...})` with explicit property names.",
	tuple:
		"heterogeneous arrays are unsupported. Use `z.array(z.object({...}))` with a discriminator field, or model as a fixed object.",
	union:
		"no core kind can represent a union of different types. Lift the discriminator to the parent object.",
	intersection:
		"intersections can't be reconciled to a single index field. Merge at the `z.object({...})` level instead.",
	lazy:
		"recursive Zod schemas are not supported in v1. Flatten the structure or materialize a fixed depth.",
	pipe:
		"shape-changing `.transform()` / `.pipe()` / `.preprocess()` breaks the document ↔ index mapping. Remove the transform or use a separate validator.",
	any: "index field kind must be known. Provide a concrete Zod type.",
	unknown: "index field kind must be known. Provide a concrete Zod type.",
	never: "a `z.never()` field can never be satisfied. Remove the field or use a concrete type.",
	map: "Map fields are not supported. Convert to a `z.object({...})` with known keys or a nested array.",
	set: "Set fields are not supported. Convert to `z.array(z.object({...}))` or a keyword array.",
	function:
		"function-valued fields cannot be indexed. Remove the field or compute it at search time.",
	promise:
		"promise-valued fields cannot be indexed. Resolve values before indexing.",
	void: "void-typed fields can't be indexed. Remove the field.",
	null: "null-only fields have no kind. Wrap with `.nullable()` around a concrete type.",
	undefined:
		"undefined-only fields have no kind. Wrap with `.optional()` around a concrete type.",
	boolean_literal:
		"boolean literals aren't a searchlite kind. Use `z.enum(['true','false'])` with `sl.keyword()`.",
};

function rejectUnsupported(inner: unknown, path: string): never {
	const t = getDefType(inner) ?? "unknown";

	// Special case: boolean literal goes through `literal` type but the value is
	// boolean. Give a tailored hint.
	if (t === "literal") {
		const values = (inner as { _def?: { values?: unknown[] } })._def?.values ?? [];
		if (values.every((v) => typeof v === "boolean")) {
			throw new UnsupportedZodTypeError({
				path,
				zodType: "z.literal(<boolean>)",
				hint: UNSUPPORTED_HINTS.boolean_literal,
			});
		}
	}

	const hint = UNSUPPORTED_HINTS[t] ?? `the Zod type \`${t}\` cannot be mapped to a searchlite field.`;
	throw new UnsupportedZodTypeError({
		path,
		zodType: zodTypeName(t),
		hint,
	});
}

function zodTypeName(defType: string): string {
	switch (defType) {
		case "pipe":
			return "z.transform() / z.pipe()";
		case "record":
			return "z.record(...)";
		case "tuple":
			return "z.tuple(...)";
		case "union":
			return "z.union(...)";
		case "intersection":
			return "z.intersection(...)";
		case "lazy":
			return "z.lazy(...)";
		case "any":
			return "z.any()";
		case "unknown":
			return "z.unknown()";
		case "never":
			return "z.never()";
		case "map":
			return "z.map(...)";
		case "set":
			return "z.set(...)";
		case "date":
			return "z.date()";
		case "bigint":
			return "z.bigint()";
		case "boolean":
			return "z.boolean()";
		case "function":
			return "z.function(...)";
		case "promise":
			return "z.promise(...)";
		case "null":
			return "z.null()";
		case "undefined":
			return "z.undefined()";
		case "void":
			return "z.void()";
		default:
			return `<Zod type: ${defType}>`;
	}
}

// Keep `WrapperState` and `InferredKind` re-exports so consumers of this module
// don't need to reach into `./rules`.
export type { WrapperState, InferredKind };
