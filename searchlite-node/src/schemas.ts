import { z } from "zod";

// --- Schema shorthand expansion ---

const FIELD_DEFAULTS = {
	text: { stored: true, indexed: true, analyzer: "default", nullable: false },
	keyword: { stored: true, indexed: true, fast: true, nullable: false },
	integer: { i64: true, fast: true, stored: false, nullable: false },
	float: { i64: false, fast: true, stored: false, nullable: false },
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

interface CoreSchema {
	doc_id_field: string;
	analyzers: unknown[];
	text_fields: Array<{
		name: string;
		analyzer: string;
		stored: boolean;
		indexed: boolean;
		nullable: boolean;
	}>;
	keyword_fields: Array<{
		name: string;
		stored: boolean;
		indexed: boolean;
		fast: boolean;
		nullable: boolean;
	}>;
	numeric_fields: Array<{
		name: string;
		i64: boolean;
		fast: boolean;
		stored: boolean;
		nullable: boolean;
	}>;
	nested_fields: unknown[];
}

export function expandSchema(input: Record<string, unknown>): CoreSchema {
	if (!input || typeof input !== "object") {
		throw new Error("schema must be an object");
	}

	// Already in core format (has text_fields array)
	if (Array.isArray((input as Record<string, unknown>).text_fields)) {
		return input as unknown as CoreSchema;
	}

	if (input.doc_id_field !== undefined && typeof input.doc_id_field !== "string") {
		throw new Error("doc_id_field must be a string");
	}
	const docIdField = (input.doc_id_field as string) ?? "_id";
	const textFields: CoreSchema["text_fields"] = [];
	const keywordFields: CoreSchema["keyword_fields"] = [];
	const numericFields: CoreSchema["numeric_fields"] = [];

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

		const defaults = FIELD_DEFAULTS[type];

		if (type === "text") {
			textFields.push({
				name,
				analyzer: fieldDef.analyzer ?? (defaults as (typeof FIELD_DEFAULTS)["text"]).analyzer,
				stored: fieldDef.stored ?? defaults.stored,
				indexed: fieldDef.indexed ?? (defaults as (typeof FIELD_DEFAULTS)["text"]).indexed,
				nullable: fieldDef.nullable ?? defaults.nullable,
			});
		} else if (type === "keyword") {
			keywordFields.push({
				name,
				stored: fieldDef.stored ?? defaults.stored,
				indexed: fieldDef.indexed ?? (defaults as (typeof FIELD_DEFAULTS)["keyword"]).indexed,
				fast: fieldDef.fast ?? (defaults as (typeof FIELD_DEFAULTS)["keyword"]).fast,
				nullable: fieldDef.nullable ?? defaults.nullable,
			});
		} else {
			// integer or float
			const numDefaults = defaults as (typeof FIELD_DEFAULTS)["integer"];
			numericFields.push({
				name,
				i64: numDefaults.i64,
				fast: fieldDef.fast ?? numDefaults.fast,
				stored: fieldDef.stored ?? numDefaults.stored,
				nullable: fieldDef.nullable ?? numDefaults.nullable,
			});
		}
	}

	return {
		doc_id_field: docIdField,
		analyzers: (input.analyzers as unknown[]) ?? [],
		text_fields: textFields,
		keyword_fields: keywordFields,
		numeric_fields: numericFields,
		nested_fields: [],
	};
}

// --- Input schemas (camelCase) ---

export const OpenOptionsSchema = z
	.object({
		writeKey: z.string().optional(),
		schema: z.record(z.string(), z.unknown()).optional(),
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

const SortSpecSchema = z.object({
	field: z.string(),
	order: z.enum(["asc", "desc"]).optional(),
});

export const SearchRequestSchema = z.object({
	query: z.union([z.string(), z.record(z.string(), z.unknown())]),
	fields: z.array(z.string()).optional(),
	filter: FilterSchema.optional(),
	limit: z.number().int().positive().max(10000).optional(),
	from: z.number().int().nonnegative().optional(),
	returnHits: z.boolean().optional(),
	sort: z.array(SortSpecSchema).optional(),
	cursor: z.string().optional(),
	searchAfter: z.array(z.unknown()).optional(),
	candidateSize: z.number().int().positive().optional(),
	bmwBlockSize: z.number().int().positive().optional(),
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
export type SearchResult = z.infer<typeof SearchResultSchema>;
export type Hit = z.infer<typeof HitSchema>;

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

export type SchemaDefinition = Record<string, FieldDefinition> & {
	doc_id_field?: string;
	analyzers?: unknown[];
};
