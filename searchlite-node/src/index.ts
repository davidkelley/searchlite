export type { SearchIndex } from "./search-index";
export { EmbeddedIndex } from "./embedded";
export { RemoteIndex } from "./remote";
export type { RemoteIndexOptions } from "./remote";

export type {
	FieldDefinition,
	FieldShorthand,
	Hit,
	KeywordFieldDef,
	NumericFieldDef,
	OpenOptions,
	SchemaDefinition,
	SearchRequest,
	SearchResult,
	TextFieldDef,
	TypedHit,
	TypedSearchResult,
} from "./schemas";

export { expandSchema } from "./schemas";

// --- Zod-native authoring ---
//
// Define your index once with Zod and the same schema will validate documents
// on insert, drive the native index definition, and type-check search results.

export { sl } from "./zod/helpers";
export type {
	TextOpts,
	KeywordOpts,
	NumericOpts,
	VectorOpts,
	IndexOpts,
} from "./zod/helpers";
export {
	compileZodSchema,
	deriveResponseSchema,
	isZodIndexSchema,
	type ZodCompiledJsonSchema,
	type ZodIndexSchema,
} from "./zod";
export {
	SearchliteFieldRegistry,
	SearchliteIndexRegistry,
	type SearchliteFieldMetadata,
	type SearchliteIndexMetadata,
	UnsupportedZodTypeError,
	InvalidZodSchemaError,
} from "./zod";
