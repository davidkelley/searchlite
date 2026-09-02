import { z } from "zod";

// --- Schema shorthand expansion ---

const FIELD_DEFAULTS = {
	text: { stored: true, indexed: true, analyzer: "default", nullable: false },
	keyword: { stored: true, indexed: true, fast: true, nullable: false },
	integer: { fast: true, stored: false, nullable: false },
	float: { fast: true, stored: false, nullable: false },
} as const;

type FieldType = keyof typeof FIELD_DEFAULTS;

interface FieldDef {
	type: FieldType;
	stored?: boolean;
	indexed?: boolean;
	fast?: boolean;
	analyzer?: string;
	nullable?: boolean;
}

/** JSON Schema output with `searchlite:` vocabulary keywords. */
interface JsonSchemaOutput {
	$schema?: string;
	type: "object";
	"searchlite:docIdField"?: string;
	"searchlite:analyzers"?: unknown[];
	properties: Record<string, Record<string, unknown>>;
	[key: string]: unknown;
}

export function expandSchema(input: Record<string, unknown>): JsonSchemaOutput {
	if (!input || typeof input !== "object" || Array.isArray(input)) {
		throw new Error("schema must be a plain object");
	}

	// Already in JSON Schema format (has `properties` or `$schema`)
	if ("properties" in input || "$schema" in input) {
		if (input.type !== "object") {
			throw new Error('JSON Schema input must have `type: "object"` at the root');
		}
		if (
			typeof input.properties !== "object" ||
			input.properties === null ||
			Array.isArray(input.properties)
		) {
			throw new Error("JSON Schema input must have `properties` as a plain object");
		}
		return input as unknown as JsonSchemaOutput;
	}

	// Reject old-format schemas (any of the three legacy field arrays)
	if (
		Array.isArray(input.text_fields) ||
		Array.isArray(input.keyword_fields) ||
		Array.isArray(input.numeric_fields)
	) {
		throw new Error(
			"legacy field-array schema format (text_fields/keyword_fields/numeric_fields) is no longer supported. " +
				"Use JSON Schema with `searchlite:` vocabulary keywords.",
		);
	}

	let docIdField = "_id";
	if ("doc_id_field" in input && input.doc_id_field !== undefined) {
		if (typeof input.doc_id_field !== "string" || input.doc_id_field.length === 0) {
			throw new Error("doc_id_field must be a non-empty string");
		}
		docIdField = input.doc_id_field;
	}
	const properties: Record<string, Record<string, unknown>> = {};

	for (const [name, def] of Object.entries(input)) {
		if (name === "doc_id_field" || name === "analyzers") continue;

		if (name.length === 0) {
			throw new Error("field name must not be empty");
		}
		if (name.includes(".")) {
			throw new Error(`field name "${name}" must not contain "." (use nested fields instead)`);
		}
		if (name === docIdField) {
			throw new Error(`field name "${name}" conflicts with doc_id_field "${docIdField}"`);
		}

		const fieldDef: FieldDef =
			typeof def === "string" ? { type: def as FieldType } : { ...(def as FieldDef) };
		const type = fieldDef.type;

		if (!(type in FIELD_DEFAULTS)) {
			throw new Error(
				`unknown field type "${type}" for "${name}"; expected text, keyword, integer, or float`,
			);
		}

		const prop: Record<string, unknown> = {};

		if (type === "text") {
			const defaults = FIELD_DEFAULTS.text;
			prop.type = fieldDef.nullable ? ["string", "null"] : "string";
			const analyzer = fieldDef.analyzer ?? defaults.analyzer;
			if (analyzer !== "default") prop["searchlite:analyzer"] = analyzer;
			if ((fieldDef.stored ?? defaults.stored) !== true) prop["searchlite:stored"] = false;
			if ((fieldDef.indexed ?? defaults.indexed) !== true) prop["searchlite:indexed"] = false;
		} else if (type === "keyword") {
			const defaults = FIELD_DEFAULTS.keyword;
			prop.type = fieldDef.nullable ? ["string", "null"] : "string";
			prop["searchlite:kind"] = "keyword";
			if ((fieldDef.stored ?? defaults.stored) !== true) prop["searchlite:stored"] = false;
			if ((fieldDef.indexed ?? defaults.indexed) !== true) prop["searchlite:indexed"] = false;
			if ((fieldDef.fast ?? defaults.fast) !== true) prop["searchlite:fast"] = false;
		} else if (type === "integer") {
			const defaults = FIELD_DEFAULTS.integer;
			prop.type = fieldDef.nullable ? ["integer", "null"] : "integer";
			if ((fieldDef.fast ?? defaults.fast) !== true) prop["searchlite:fast"] = false;
			if ((fieldDef.stored ?? defaults.stored) !== false) prop["searchlite:stored"] = true;
		} else {
			// float
			const defaults = FIELD_DEFAULTS.float;
			prop.type = fieldDef.nullable ? ["number", "null"] : "number";
			if ((fieldDef.fast ?? defaults.fast) !== true) prop["searchlite:fast"] = false;
			if ((fieldDef.stored ?? defaults.stored) !== false) prop["searchlite:stored"] = true;
		}

		properties[name] = prop;
	}

	const result: JsonSchemaOutput = {
		type: "object",
		properties,
	};

	if (docIdField !== "_id") {
		result["searchlite:docIdField"] = docIdField;
	}

	if (input.analyzers) {
		result["searchlite:analyzers"] = input.analyzers as unknown[];
	}

	return result;
}

// --- Input schemas (camelCase) ---

// The `schema` field accepts three shapes:
//   1. Flat shorthand (`Record<string, FieldDefinition>`)
//   2. Raw JSON Schema (`{ type: "object", properties: {...} }`)
//   3. A Zod-authored index schema (branded via `sl.index(...)`)
// Shape (3) is a ZodObject instance, which would be rejected by
// `z.record(z.string(), z.unknown())`, so we accept any object here and let
// the constructors dispatch on the concrete shape at runtime.
export const OpenOptionsSchema = z
	.object({
		writeKey: z.string().optional(),
		schema: z.unknown().optional(),
	})
	.strict()
	.optional();

export const DocumentSchema = z
	.record(z.string(), z.unknown())
	.refine((val) => typeof val === "object" && val !== null && !Array.isArray(val), {
		message: "document must be a plain object",
	});

export const DocumentsSchema = z.union([DocumentSchema, z.array(DocumentSchema)]);

const FilterSchema: z.ZodType = z.union([
	z.object({ KeywordEq: z.object({ field: z.string(), value: z.string() }) }),
	z.object({
		KeywordIn: z.object({ field: z.string(), values: z.array(z.string()) }),
	}),
	z.object({
		I64Range: z.object({
			field: z.string(),
			min: z.number().int(),
			max: z.number().int(),
		}),
	}),
	z.object({
		F64Range: z.object({
			field: z.string(),
			min: z.number(),
			max: z.number(),
		}),
	}),
	z.object({
		Nested: z.object({
			path: z.string(),
			filter: z.lazy(() => FilterSchema),
		}),
	}),
	z.object({ And: z.array(z.lazy(() => FilterSchema)) }),
	z.object({ Or: z.array(z.lazy(() => FilterSchema)) }),
	z.object({ Not: z.lazy(() => FilterSchema) }),
]);

// The canonical wire format (see `search-request.schema.json`'s `sort_spec`)
// is `{field: string, order?: "asc" | "desc"}`. The three shorthand forms
// are normalized to the canonical shape by the `SortSpec*` schema
// `.transform()`s below — `requestToSnake` only remaps top-level request
// keys and does not touch `sort` — so users can pick whichever style reads
// best without hitting a deserialization failure on the Rust side.
const SortOrderEnum = z.enum(["asc", "desc"]);

const SortSpecCanonical = z.object({
	field: z.string(),
	order: SortOrderEnum.optional(),
});

const SortSpecShorthandString = z.string().transform((field) => ({ field }));

const SortSpecShorthandOrder = z
	.record(z.string(), SortOrderEnum)
	.refine((rec) => Object.keys(rec).length === 1, {
		message: "sort shorthand must have exactly one field key",
	})
	.transform((rec) => {
		const [field] = Object.keys(rec);
		return { field, order: rec[field] };
	});

const SortSpecShorthandNested = z
	.record(z.string(), z.object({ order: SortOrderEnum }))
	.refine((rec) => Object.keys(rec).length === 1, {
		message: "sort shorthand must have exactly one field key",
	})
	.transform((rec) => {
		const [field] = Object.keys(rec);
		return { field, order: rec[field].order };
	});

// Variant order matters because the canonical and shorthand-order forms
// share the input shape `{<key>: <string>}`. We try the shorthand variants
// first so an input like `{field: "asc"}` is interpreted as "sort by the
// field named 'field', ascending" — the natural shorthand reading — rather
// than as canonical "sort by the field named 'asc' with no order". The
// shorthand variants only match single-key records whose value is `asc` /
// `desc` (or `{order}`), so any record that doesn't fit the shorthand
// pattern (e.g. `{field: "price", order: "asc"}` or `{field: "name"}`)
// cleanly falls through to the canonical variant.
const SortSpecSchema = z.union([
	SortSpecShorthandOrder,
	SortSpecShorthandNested,
	SortSpecCanonical,
	SortSpecShorthandString,
]);

// Structured vector query (camelCase). Mapped to the Rust `VectorQuery`
// (snake_case fields) by `requestToSnake`'s nested transform. The whole
// `vectorQuery` envelope was previously stripped by Zod (unknown key), so
// hybrid/vector search via the typed request path silently no-op'd.
const VectorQuerySchema = z.object({
	field: z.string(),
	vector: z.array(z.number()),
	k: z.number().int().positive().optional(),
	// Blend factor: 1.0 = pure BM25, 0.0 = pure vector. Constrained to [0,1]
	// to reject invalid values before they reach the native layer.
	alpha: z.number().min(0).max(1).optional(),
	efSearch: z.number().int().positive().optional(),
	candidateSize: z.number().int().positive().optional(),
	boost: z.number().optional(),
});

export const SearchRequestSchema = z.object({
	query: z.union([z.string(), z.record(z.string(), z.unknown())]),
	fields: z.array(z.string()).optional(),
	filter: FilterSchema.optional(),
	limit: z.number().int().positive().max(10000).optional(),
	from: z.number().int().nonnegative().optional(),
	returnHits: z.boolean().optional(),
	// Both fields are nullable in the wire schema (`["integer", "null"]`,
	// see `search-request.schema.json`). `.nullish()` accepts both an
	// explicit `null` (to clear an override in callers that round-trip
	// payloads) and `undefined` (the field absent), matching the Rust
	// `Option<usize>` semantics.
	candidateSize: z.number().int().positive().nullish(),
	bmwBlockSize: z.number().int().positive().nullish(),
	sort: z.array(SortSpecSchema).optional(),
	cursor: z.string().optional(),
	searchAfter: z.array(z.unknown()).optional(),
	execution: z.enum(["wand", "bmw", "bm25"]).optional(),
	fuzzy: z
		.object({
			maxEdits: z.number().int().optional(),
			prefixLength: z.number().int().optional(),
		})
		.optional(),
	trackTotalHits: z.boolean().optional(),
	returnStored: z.boolean().optional(),
	highlightField: z.string().optional(),
	highlight: z.record(z.string(), z.unknown()).optional(),
	collapse: z.record(z.string(), z.unknown()).optional(),
	aggs: z.record(z.string(), z.unknown()).optional(),
	suggest: z.record(z.string(), z.unknown()).optional(),
	rescore: z.record(z.string(), z.unknown()).optional(),
	explain: z.boolean().optional(),
	profile: z.boolean().optional(),
	// Vector / hybrid search (requires the `vectors` feature in the native
	// binding, which is on by default). `vectorQuery` is the structured form;
	// the tuple shorthand `["field",[...],alpha]` is also accepted by the
	// engine if passed through `query`.
	vectorQuery: VectorQuerySchema.optional(),
	vectorFilter: FilterSchema.optional(),
	maxGlobalVectorCandidates: z.number().int().positive().nullish(),
});

// --- Output schemas (camelCase) ---

const HitSchema: z.ZodType = z.object({
	docId: z.string(),
	score: z.number(),
	vectorScore: z.number().optional(),
	sortKey: z.array(z.unknown()).optional(),
	fields: z.unknown().optional().nullable(),
	snippet: z.string().optional().nullable(),
	explanation: z.unknown().optional(),
	highlights: z.record(z.string(), z.array(z.string())).optional(),
	innerHits: z.array(z.lazy(() => HitSchema)).optional(),
});

export const SearchResultSchema = z.object({
	totalHits: z.number().int(),
	totalGroups: z.number().int().optional(),
	hits: z.array(HitSchema),
	nextCursor: z.string().optional(),
	nextSearchAfter: z.array(z.unknown()).optional(),
	aggregations: z.record(z.string(), z.unknown()).optional().default({}),
	suggest: z.record(z.string(), z.unknown()).optional().default({}),
	profile: z.unknown().optional(),
});

// --- Inferred types ---

export type OpenOptions = z.infer<typeof OpenOptionsSchema>;
export type SearchRequest = z.input<typeof SearchRequestSchema>;

// --- Explicit generic interfaces (replaces z.infer which resolved to `unknown`) ---

export interface Hit<TFields = unknown> {
	docId: string;
	score: number;
	vectorScore?: number;
	sortKey?: unknown[];
	fields?: TFields | null;
	snippet?: string | null;
	explanation?: unknown;
	highlights?: Record<string, string[]>;
	innerHits?: Hit<TFields>[];
}

export interface SearchResult<TFields = unknown> {
	totalHits: number;
	totalGroups?: number;
	hits: Hit<TFields>[];
	nextCursor?: string;
	nextSearchAfter?: unknown[];
	aggregations: Record<string, unknown>;
	suggest: Record<string, unknown>;
	profile?: unknown;
}

// --- Typed variants returned when a Zod schema is provided to search() ---
// `fields` is guaranteed non-optional because the schema validates it.

export interface TypedHit<TFields> extends Omit<Hit<TFields>, "fields" | "innerHits"> {
	fields: TFields;
	innerHits?: TypedHit<TFields>[];
}

export interface TypedSearchResult<TFields> extends Omit<SearchResult<TFields>, "hits"> {
	hits: TypedHit<TFields>[];
}

export type FieldShorthand = "text" | "keyword" | "integer" | "float";

export type TextFieldDef = {
	type: "text";
	stored?: boolean;
	indexed?: boolean;
	analyzer?: string;
	nullable?: boolean;
};

export type KeywordFieldDef = {
	type: "keyword";
	stored?: boolean;
	indexed?: boolean;
	fast?: boolean;
	nullable?: boolean;
};

export type NumericFieldDef = {
	type: "integer" | "float";
	stored?: boolean;
	fast?: boolean;
	nullable?: boolean;
};

export type FieldDefinition = FieldShorthand | TextFieldDef | KeywordFieldDef | NumericFieldDef;

export type SchemaDefinition = Record<string, FieldDefinition>;
