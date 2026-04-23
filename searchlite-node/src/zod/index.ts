// Public entry point for searchlite's Zod-native authoring API.
//
// Users import from `"searchlite-js"` which re-exports everything here via
// `src/index.ts`. This module is the single source of truth for what's public.

export { sl } from "./helpers";
export type {
	TextOpts,
	KeywordOpts,
	NumericOpts,
	VectorOpts,
	IndexOpts,
} from "./helpers";

export {
	compileZodSchema,
	deriveResponseSchema,
	isZodIndexSchema,
	type JsonSchemaOutput as ZodCompiledJsonSchema,
	type ZodIndexSchema,
} from "./compile";

export {
	SearchliteFieldRegistry,
	SearchliteIndexRegistry,
	type SearchliteFieldMetadata,
	type SearchliteIndexMetadata,
} from "./registries";

export { UnsupportedZodTypeError, InvalidZodSchemaError } from "./errors";
