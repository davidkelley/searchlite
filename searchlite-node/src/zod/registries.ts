import { z } from "zod";

// --- Metadata types ---

/**
 * Metadata attachable to a Zod schema used as a field in a searchlite index.
 *
 * The compiler reads this registry to determine the field kind and emits the
 * corresponding `searchlite:*` JSON Schema keywords. Explicit metadata always
 * wins over auto-promotion rules (e.g., `z.string().uuid()` → keyword).
 */
export interface SearchliteFieldMetadata {
	/** Field kind. Overrides any inferred kind (e.g., uuid → keyword). */
	kind?: "text" | "keyword" | "integer" | "float" | "vector";

	// Shared flags
	stored?: boolean;
	indexed?: boolean;
	fast?: boolean;

	// Text-only
	analyzer?: string;
	searchAnalyzer?: string;
	searchAsYouType?: { minGram: number; maxGram: number };

	// Vector-only
	dim?: number;
	metric?: "Cosine" | "L2";
	hnsw?: Record<string, unknown>;
}

/**
 * Root-level metadata attachable to the Zod object passed to `sl.index(...)`.
 */
export interface SearchliteIndexMetadata {
	docIdField?: string;
	analyzers?: unknown[];
}

// --- Registries ---

/**
 * Per-field metadata registry. Attach via `.register(SearchliteFieldRegistry, {...})`
 * or, more conveniently, via `sl.text()`, `sl.keyword()`, etc.
 */
export const SearchliteFieldRegistry = z.registry<SearchliteFieldMetadata>();

/**
 * Root index-level metadata registry. Populated by `sl.index(schema, opts)`.
 */
export const SearchliteIndexRegistry = z.registry<SearchliteIndexMetadata>();
