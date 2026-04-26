// Public entry point for searchlite's Zod-native authoring API.
//
// Users import from `"searchlite-js"` which re-exports everything here via
// `src/index.ts`. This module is the single source of truth for what's public.

export {
	compileZodSchema,
	deriveResponseSchema,
	isZodIndexSchema,
	type JsonSchemaOutput as ZodCompiledJsonSchema,
	type ZodIndexSchema,
} from "./compile";
export { InvalidZodSchemaError, UnsupportedZodTypeError } from "./errors";
export type {
	IndexOpts,
	KeywordOpts,
	NumericOpts,
	TextOpts,
	VectorOpts,
} from "./helpers";
export { sl } from "./helpers";
export {
	type SearchliteFieldMetadata,
	SearchliteFieldRegistry,
	type SearchliteIndexMetadata,
	SearchliteIndexRegistry,
} from "./registries";
