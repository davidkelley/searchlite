import { existsSync } from "node:fs";
import { join } from "node:path";
import type { ZodType } from "zod";
import type { SchemaDefinition, SearchRequest, SearchResult, TypedSearchResult } from "./schemas";
import {
	DocumentSchema,
	DocumentsSchema,
	expandSchema,
	OpenOptionsSchema,
	SearchRequestSchema,
	SearchResultSchema,
} from "./schemas";
import type { SearchIndex } from "./search-index";
import {
	type RawSearchResult,
	requestToSnake,
	transformResult,
	validate,
	validateTypedResult,
} from "./transform";
import {
	compileZodSchema,
	deriveResponseSchema,
	isZodIndexSchema,
	type ZodIndexSchema,
} from "./zod/compile";
import { SearchliteIndexRegistry } from "./zod/registries";

// --- Native binding loader ---

interface NativeIndex {
	add(doc: unknown): void;
	addMany(docs: unknown): number;
	commit(): void;
	search(query: unknown): unknown;
	compact(): void;
	close(): void;
}

interface NativeIndexConstructor {
	new (path: string, options?: Record<string, unknown>): NativeIndex;
	fromS3(config: Record<string, unknown>): NativeIndex;
}

interface NativeBinding {
	Index: NativeIndexConstructor;
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

/**
 * Static credentials for an S3-compatible endpoint. Omit the
 * containing `credentials` field on `S3IndexConfig` to load
 * credentials from the standard AWS chain (env vars, shared
 * credentials file, IMDS, EC2 instance role).
 */
export interface S3StaticCredentials {
	accessKeyId: string;
	secretAccessKey: string;
	/** Optional session token for temporary credentials (STS, R2). */
	sessionToken?: string;
}

/**
 * Configuration for opening a read-only `EmbeddedIndex` against an
 * S3-compatible backend (AWS S3, Cloudflare R2, MinIO).
 *
 * The schema is read from the manifest in the bucket — there is no
 * constructor-time schema. Mutators (`add`, `addMany`, `commit`,
 * `compact`) on the resulting index will error.
 */
export interface S3IndexConfig {
	/** Bucket name. */
	bucket: string;
	/**
	 * AWS region. Defaults to `us-east-1` when unset (required by
	 * SigV4 even for R2 — pass `auto` for R2).
	 */
	region?: string;
	/** Optional namespace within the bucket. */
	prefix?: string;
	/**
	 * Endpoint URL. Set for R2
	 * (`https://<account>.r2.cloudflarestorage.com`) or MinIO /
	 * LocalStack. Leave unset for AWS S3.
	 */
	endpointUrl?: string;
	/**
	 * Path-style addressing (`https://endpoint/bucket/key`). Required
	 * for MinIO / LocalStack. Defaults to `false`.
	 */
	forcePathStyle?: boolean;
	/**
	 * Conditional PUT support (`If-Match` / `If-None-Match`). Defaults
	 * to `true` on AWS S3 and MinIO, and `false` on R2 (auto-detected
	 * from the endpoint hostname pattern `*.r2.cloudflarestorage.com`).
	 */
	conditionalPut?: boolean;
	/**
	 * Static credentials. Omit to use the standard AWS credential
	 * chain.
	 */
	credentials?: S3StaticCredentials;
	/**
	 * Checksum policy. Defaults to `"strict"` (per-segment SHA-256
	 * verification on every `Index::reader()`). Use `"trust-manifest"`
	 * to skip verification (cheaper opens for object storage), or
	 * `"audit"` to verify in the background.
	 */
	checksumPolicy?: "strict" | "trust-manifest" | "audit";
}

/** Internal init bag for `EmbeddedIndex.fromS3`. Not exported. */
interface NativeInit {
	__searchliteNativeInit: true;
	native: NativeIndex;
	zodSchema?: ZodIndexSchema;
}

function isNativeInit(value: unknown): value is NativeInit {
	return (
		!!value &&
		typeof value === "object" &&
		(value as { __searchliteNativeInit?: unknown }).__searchliteNativeInit === true
	);
}

// --- EmbeddedIndex ---

export class EmbeddedIndex<T = Record<string, unknown>> implements SearchIndex<T> {
	#native: NativeIndex;
	#closed = false;
	/** Set when the index was constructed with a Zod-authored schema. */
	#zodSchema: ZodIndexSchema | undefined;
	/**
	 * Derived partial schema used for auto-validating search hits. See
	 * `deriveResponseSchema()` — non-stored / missing fields don't fail
	 * validation, which matches the runtime reality of search results.
	 */
	#responseSchema: ZodType | undefined;
	/** docIdField resolved from the Zod index metadata, or undefined. */
	#docIdField: string | undefined;

	constructor(path: string, options?: { writeKey?: string; schema?: AnySchemaInput }) {
		// Internal `fromS3` path: bypass path validation and native
		// construction; install the supplied native index directly.
		if (isNativeInit(path)) {
			const init = path as unknown as NativeInit;
			this.#native = init.native;
			if (init.zodSchema) {
				this.#zodSchema = init.zodSchema;
				this.#responseSchema = deriveResponseSchema(init.zodSchema);
				this.#docIdField =
					SearchliteIndexRegistry.get(init.zodSchema as never)?.docIdField ?? "_id";
			}
			return;
		}

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
					this.#responseSchema = deriveResponseSchema(schemaInput);
					this.#docIdField = SearchliteIndexRegistry.get(schemaInput as never)?.docIdField ?? "_id";
					nativeOpts.schema = compileZodSchema(schemaInput);
				} else if (
					schemaInput !== null &&
					typeof schemaInput === "object" &&
					!Array.isArray(schemaInput)
				) {
					// Shorthand / raw JSON Schema: delegate to the existing expander.
					nativeOpts.schema = expandSchema(schemaInput as Record<string, unknown>);
				} else {
					throw new Error(
						`schema must be a plain object, a Zod index schema (from sl.index()), or a SchemaDefinition; received ${
							schemaInput === null ? "null" : typeof schemaInput
						}`,
					);
				}
			}
		}

		this.#native = new (getNative().Index)(
			path,
			nativeOpts.schema || nativeOpts.writeKey ? nativeOpts : undefined,
		);
	}

	/**
	 * Open a read-only `EmbeddedIndex` against an S3-compatible backend
	 * (AWS S3, Cloudflare R2, MinIO). The schema is read from the
	 * manifest stored in the bucket. Mutators (`add`, `addMany`,
	 * `commit`, `compact`) will error on the returned index.
	 *
	 * Pass an optional Zod-authored index schema for client-side typed
	 * search results — this does not validate the index's stored schema.
	 *
	 * @example AWS S3 (credentials from env)
	 * ```ts
	 * const idx = EmbeddedIndex.fromS3({
	 *   bucket: "my-search-indexes",
	 *   region: "us-east-1",
	 *   prefix: "products/v1",
	 * });
	 * ```
	 *
	 * @example Cloudflare R2
	 * ```ts
	 * const idx = EmbeddedIndex.fromS3({
	 *   bucket: "my-bucket",
	 *   region: "auto",
	 *   endpointUrl: "https://<account>.r2.cloudflarestorage.com",
	 *   credentials: { accessKeyId, secretAccessKey },
	 * });
	 * ```
	 *
	 * @example MinIO / LocalStack
	 * ```ts
	 * const idx = EmbeddedIndex.fromS3({
	 *   bucket: "my-bucket",
	 *   region: "us-east-1",
	 *   endpointUrl: "http://localhost:9000",
	 *   forcePathStyle: true,
	 *   credentials: { accessKeyId, secretAccessKey },
	 * });
	 * ```
	 */
	static fromS3<U = Record<string, unknown>>(
		s3Config: S3IndexConfig,
		options?: { schema?: ZodIndexSchema },
	): EmbeddedIndex<U> {
		if (!s3Config || typeof s3Config !== "object") {
			throw new Error("s3Config must be an object");
		}
		if (typeof s3Config.bucket !== "string" || s3Config.bucket.length === 0) {
			throw new Error("s3Config.bucket must be a non-empty string");
		}
		if (options?.schema !== undefined && !isZodIndexSchema(options.schema)) {
			throw new Error(
				"EmbeddedIndex.fromS3 `options.schema` must be a Zod index schema wrapped with `sl.index(...)`.",
			);
		}

		const nativeConfig: Record<string, unknown> = { bucket: s3Config.bucket };
		if (s3Config.region !== undefined) nativeConfig.region = s3Config.region;
		if (s3Config.prefix !== undefined) nativeConfig.prefix = s3Config.prefix;
		if (s3Config.endpointUrl !== undefined) nativeConfig.endpointUrl = s3Config.endpointUrl;
		if (s3Config.forcePathStyle !== undefined)
			nativeConfig.forcePathStyle = s3Config.forcePathStyle;
		if (s3Config.conditionalPut !== undefined)
			nativeConfig.conditionalPut = s3Config.conditionalPut;
		if (s3Config.checksumPolicy !== undefined)
			nativeConfig.checksumPolicy = s3Config.checksumPolicy;
		if (s3Config.credentials !== undefined) {
			nativeConfig.credentials = {
				accessKeyId: s3Config.credentials.accessKeyId,
				secretAccessKey: s3Config.credentials.secretAccessKey,
				sessionToken: s3Config.credentials.sessionToken,
			};
		}

		const native = getNative().Index.fromS3(nativeConfig);
		const init: NativeInit = {
			__searchliteNativeInit: true,
			native,
			zodSchema: options?.schema,
		};
		return new EmbeddedIndex<U>(init as unknown as string);
	}

	/**
	 * Validate `doc` against the stored Zod schema and return the parsed
	 * output. If the parsed output doesn't contain the configured docIdField
	 * (because the user didn't declare it in their Zod schema, e.g. the
	 * default `_id` is not part of `z.object({...})`), the original value is
	 * preserved from the input so the native engine can identify the doc.
	 */
	#parseDoc(doc: T, label: string): Record<string, unknown> {
		const schema = this.#zodSchema as unknown as ZodType<T>;
		const parsed = validate(schema, doc, label) as Record<string, unknown>;
		const idField = this.#docIdField;
		if (idField && !(idField in parsed) && doc && typeof doc === "object") {
			const src = doc as Record<string, unknown>;
			if (idField in src) parsed[idField] = src[idField];
		}
		return parsed;
	}

	async add(doc: T): Promise<void> {
		// Use the parsed/coerced Zod output when available — schemas using
		// coercion (e.g. `sl.integer()` accepts bigint → coerces to number)
		// or `.default()` rely on the output value, not the input.
		let toStore: unknown;
		if (this.#zodSchema) {
			toStore = this.#parseDoc(doc, "document");
		} else {
			validate(DocumentSchema, doc as Record<string, unknown>, "document");
			toStore = doc;
		}
		this.#native.add(toStore);
	}

	async addMany(docs: T[] | T): Promise<number> {
		if (this.#zodSchema) {
			if (Array.isArray(docs)) {
				const parsed: unknown[] = new Array(docs.length);
				for (let i = 0; i < docs.length; i++) {
					parsed[i] = this.#parseDoc(docs[i], `documents[${i}]`);
				}
				return this.#native.addMany(parsed);
			}
			return this.#native.addMany(this.#parseDoc(docs, "document"));
		}
		validate(
			DocumentsSchema,
			docs as Record<string, unknown>[] | Record<string, unknown>,
			"documents",
		);
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

		// Explicit per-call schema wins. Otherwise, fall back to the derived
		// response schema (a partial of the construction-time schema) so
		// hit.fields is validated without failing on non-stored / missing
		// fields. See `deriveResponseSchema()` for rationale.
		const effectiveSchema: ZodType<unknown> | undefined =
			fieldsSchema ?? (this.#responseSchema as ZodType<unknown> | undefined);

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
