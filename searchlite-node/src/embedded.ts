import { existsSync } from "node:fs";
import { join } from "node:path";
import type { ZodType } from "zod";
import {
	DocumentSchema,
	DocumentsSchema,
	OpenOptionsSchema,
	SearchRequestSchema,
	SearchResultSchema,
	expandSchema,
} from "./schemas";
import type { SchemaDefinition, SearchRequest, SearchResult, TypedSearchResult } from "./schemas";
import type { SearchIndex } from "./search-index";
import {
	type RawSearchResult,
	requestToSnake,
	transformResult,
	validate,
	validateTypedResult,
} from "./transform";
import { type ZodIndexSchema, compileZodSchema, isZodIndexSchema } from "./zod/compile";

// --- Native binding loader ---

interface NativeIndex {
	add(doc: unknown): void;
	addMany(docs: unknown): number;
	commit(): void;
	search(query: unknown): unknown;
	compact(): void;
	close(): void;
}

interface NativeBinding {
	Index: new (path: string, options?: Record<string, unknown>) => NativeIndex;
}

const SUFFIX_MAP: Record<string, string> = {
	"win32-x64": "win32-x64-msvc",
	"win32-ia32": "win32-ia32-msvc",
	"win32-arm64": "win32-arm64-msvc",
	"darwin-x64": "darwin-x64",
	"darwin-arm64": "darwin-arm64",
	"linux-x64": "linux-x64-gnu",
	"linux-arm64": "linux-arm64-gnu",
};

function loadNativeBinding(): NativeBinding {
	const suffix = SUFFIX_MAP[`${process.platform}-${process.arch}`];
	if (!suffix) {
		throw new Error(`Unsupported platform: ${process.platform}-${process.arch}`);
	}

	const localPath = join(__dirname, "..", `searchlite.${suffix}.node`);
	if (existsSync(localPath)) {
		return require(localPath);
	}

	try {
		return require(`@searchlite-js/${suffix}`);
	} catch {
		throw new Error(
			`Failed to load native binding for ${suffix}. ` +
				`Tried: ${localPath} and @searchlite-js/${suffix}`,
		);
	}
}

let _native: NativeBinding | undefined;
function getNative(): NativeBinding {
	if (!_native) _native = loadNativeBinding();
	return _native;
}

// --- Schema discrimination ---

/**
 * Detect whether `value` is a Zod schema (has a `_def` descriptor). Used to
 * discriminate between the shorthand / raw-JSON-Schema path and the Zod path.
 */
function isZodLike(value: unknown): boolean {
	return (
		!!value && typeof value === "object" && typeof (value as { _def?: unknown })._def === "object"
	);
}

// Accepted shapes for the `schema` option.
type AnySchemaInput = SchemaDefinition | ZodIndexSchema | Record<string, unknown>;

// --- EmbeddedIndex ---

export class EmbeddedIndex<T = Record<string, unknown>> implements SearchIndex<T> {
	#native: NativeIndex;
	#closed = false;
	/** Set when the index was constructed with a Zod-authored schema. */
	#zodSchema: ZodIndexSchema | undefined;

	constructor(path: string, options?: { writeKey?: string; schema?: AnySchemaInput }) {
		if (typeof path !== "string" || path.length === 0) {
			throw new Error("path must be a non-empty string");
		}
		const parsed = validate(OpenOptionsSchema, options, "options");
		const nativeOpts: Record<string, unknown> = {};

		if (parsed) {
			nativeOpts.writeKey = parsed.writeKey;
			if (parsed.schema !== undefined) {
				const schemaInput = parsed.schema;

				if (isZodLike(schemaInput)) {
					// Zod-authored path: the schema must have been wrapped with `sl.index(...)`.
					if (!isZodIndexSchema(schemaInput)) {
						throw new Error(
							"Zod schemas passed to EmbeddedIndex must be wrapped with `sl.index(...)` " +
								"so the constructor can read index-level metadata (docIdField, analyzers).",
						);
					}
					this.#zodSchema = schemaInput;
					nativeOpts.schema = compileZodSchema(schemaInput);
				} else {
					// Shorthand / raw JSON Schema: delegate to the existing expander.
					nativeOpts.schema = expandSchema(schemaInput as Record<string, unknown>);
				}
			}
		}

		this.#native = new (getNative().Index)(
			path,
			nativeOpts.schema || nativeOpts.writeKey ? nativeOpts : undefined,
		);
	}

	async add(doc: T): Promise<void> {
		if (this.#zodSchema) {
			validate(this.#zodSchema as unknown as ZodType<T>, doc, "document");
		} else {
			validate(DocumentSchema, doc as Record<string, unknown>, "document");
		}
		this.#native.add(doc);
	}

	async addMany(docs: T[] | T): Promise<number> {
		if (this.#zodSchema) {
			// Validate each doc against the Zod schema.
			if (Array.isArray(docs)) {
				const zod = this.#zodSchema as unknown as ZodType<T>;
				for (let i = 0; i < docs.length; i++) {
					validate(zod, docs[i], `documents[${i}]`);
				}
			} else {
				validate(this.#zodSchema as unknown as ZodType<T>, docs, "document");
			}
		} else {
			validate(
				DocumentsSchema,
				docs as Record<string, unknown>[] | Record<string, unknown>,
				"documents",
			);
		}
		return this.#native.addMany(docs);
	}

	async commit(): Promise<void> {
		this.#native.commit();
	}

	async search<U>(schema: ZodType<U>, query: string): Promise<TypedSearchResult<U>>;
	async search<U>(schema: ZodType<U>, query: SearchRequest): Promise<TypedSearchResult<U>>;
	async search(query: string): Promise<SearchResult<T>>;
	async search(query: SearchRequest): Promise<SearchResult<T>>;
	async search<U = T>(
		queryOrSchema: string | SearchRequest | ZodType<U>,
		maybeQuery?: string | SearchRequest,
	): Promise<SearchResult<T> | TypedSearchResult<U>> {
		let fieldsSchema: ZodType<U> | undefined;
		let query: string | SearchRequest;

		if (maybeQuery !== undefined) {
			fieldsSchema = queryOrSchema as ZodType<U>;
			query = maybeQuery;
		} else {
			query = queryOrSchema as string | SearchRequest;
		}

		// Explicit per-call schema wins. Otherwise, fall back to the
		// construction-time Zod schema (if any) so hit fields are validated and
		// typed without the caller having to pass it again.
		const effectiveSchema: ZodType<unknown> | undefined =
			fieldsSchema ?? (this.#zodSchema as ZodType<unknown> | undefined);

		if (effectiveSchema) {
			if (typeof query === "string") {
				query = { query, returnStored: true };
			} else {
				query = { ...query, returnStored: true };
			}
		}

		let raw: RawSearchResult;
		if (typeof query === "string") {
			raw = this.#native.search(query) as RawSearchResult;
		} else {
			const validated = validate(SearchRequestSchema, query, "search request");
			const snaked = requestToSnake(validated);
			raw = this.#native.search(snaked) as RawSearchResult;
		}

		const result = validate(
			SearchResultSchema,
			transformResult(raw),
			"search result",
		) as SearchResult;

		if (effectiveSchema) {
			return validateTypedResult(result, effectiveSchema) as TypedSearchResult<U>;
		}

		return result as SearchResult<T>;
	}

	async compact(): Promise<void> {
		this.#native.compact();
	}

	async close(): Promise<void> {
		if (this.#closed) return;
		this.#closed = true;
		this.#native.close();
	}

	[Symbol.dispose](): void {
		if (this.#closed) return;
		this.#closed = true;
		this.#native.close();
	}
}
