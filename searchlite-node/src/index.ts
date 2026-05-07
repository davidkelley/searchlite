export type { S3IndexConfig, S3StaticCredentials } from "./embedded";
export { EmbeddedIndex } from "./embedded";
export type { RemoteIndexOptions } from "./remote";
export { RemoteIndex } from "./remote";
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
export type { SearchIndex } from "./search-index";

// --- Zod-native authoring ---
//
// Define your index once with Zod and the same schema will validate documents
// on insert, drive the native index definition, and type-check search results.

export {
	compileZodSchema,
	deriveResponseSchema,
	InvalidZodSchemaError,
	isZodIndexSchema,
	type SearchliteFieldMetadata,
	SearchliteFieldRegistry,
	type SearchliteIndexMetadata,
	SearchliteIndexRegistry,
	UnsupportedZodTypeError,
	type ZodCompiledJsonSchema,
	type ZodIndexSchema,
} from "./zod";
export type {
	IndexOpts,
	KeywordOpts,
	NumericOpts,
	TextOpts,
	VectorOpts,
} from "./zod/helpers";
export { sl } from "./zod/helpers";
