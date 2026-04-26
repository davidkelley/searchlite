import type { ZodType } from "zod";
import type { SearchRequest, SearchResult, TypedSearchResult } from "./schemas";
import {
	DocumentSchema,
	DocumentsSchema,
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
import { deriveResponseSchema, isZodIndexSchema, type ZodIndexSchema } from "./zod/compile";
import { SearchliteIndexRegistry } from "./zod/registries";

export interface RemoteIndexOptions<T = Record<string, unknown>> {
	/** Write key sent via X-Write-Key header for protected indexes. */
	writeKey?: string;
	/** Custom fetch implementation for testing or custom transports. Defaults to global fetch. */
	fetch?: typeof globalThis.fetch;
	/**
	 * Optional Zod-authored index schema (from `sl.index(...)`). When provided,
	 * `add` / `addMany` validate documents against it, and `search()` auto-
	 * validates & types hit fields without requiring the schema per-call.
	 *
	 * Unlike `EmbeddedIndex`, the schema is NOT sent to the server — the server
	 * already has its own schema. This option exists purely for client-side
	 * validation and type flow.
	 */
	schema?: ZodIndexSchema;
}

export class RemoteIndex<T = Record<string, unknown>> implements SearchIndex<T> {
	readonly #baseUrl: string;
	readonly #indexName: string;
	readonly #writeKey?: string;
	readonly #fetch: typeof globalThis.fetch;
	readonly #zodSchema: ZodIndexSchema | undefined;
	/**
	 * Derived partial schema used for auto-validating search hits. See
	 * `deriveResponseSchema()` — non-stored / missing fields don't fail
	 * validation, which matches the runtime reality of search results.
	 */
	readonly #responseSchema: ZodType | undefined;
	/** docIdField resolved from the Zod index metadata, or undefined. */
	readonly #docIdField: string | undefined;

	constructor(baseUrl: string, indexName: string, options?: RemoteIndexOptions<T>) {
		if (typeof baseUrl !== "string" || baseUrl.length === 0) {
			throw new Error("baseUrl must be a non-empty string");
		}
		if (typeof indexName !== "string" || indexName.length === 0) {
			throw new Error("indexName must be a non-empty string");
		}
		this.#baseUrl = baseUrl.replace(/\/+$/, "");
		this.#indexName = indexName;
		this.#writeKey = options?.writeKey;
		this.#fetch = options?.fetch ?? globalThis.fetch;

		if (options?.schema !== undefined) {
			if (!isZodIndexSchema(options.schema)) {
				throw new Error(
					"RemoteIndex `schema` option must be a Zod index schema wrapped with `sl.index(...)`.",
				);
			}
			this.#zodSchema = options.schema;
			this.#responseSchema = deriveResponseSchema(options.schema);
			this.#docIdField = SearchliteIndexRegistry.get(options.schema as never)?.docIdField ?? "_id";
		} else {
			this.#responseSchema = undefined;
			this.#docIdField = undefined;
		}
	}

	/**
	 * Validate and parse a single document against the stored Zod schema,
	 * restoring the docIdField from the raw input if Zod stripped it (users
	 * commonly don't declare `_id` in their z.object when the default
	 * docIdField is used).
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
		// Use the parsed/coerced output so coercion and defaults flow into the
		// payload (e.g. `sl.integer()` coerces bigint → number).
		let toSend: unknown;
		if (this.#zodSchema) {
			toSend = this.#parseDoc(doc, "document");
		} else {
			validate(DocumentSchema, doc as Record<string, unknown>, "document");
			toSend = doc;
		}
		await this.#post("bulk", { docs: [toSend] });
	}

	async addMany(docs: T[] | T): Promise<number> {
		let payload: unknown[];
		if (this.#zodSchema) {
			const asArray = Array.isArray(docs) ? docs : [docs];
			payload = new Array(asArray.length);
			for (let i = 0; i < asArray.length; i++) {
				payload[i] = this.#parseDoc(asArray[i], `documents[${i}]`);
			}
		} else {
			validate(
				DocumentsSchema,
				docs as Record<string, unknown>[] | Record<string, unknown>,
				"documents",
			);
			payload = Array.isArray(docs) ? docs : [docs];
		}
		const body = await this.#post<{ queued: number }>("bulk", { docs: payload });
		return body.queued;
	}

	async commit(): Promise<void> {
		await this.#post("commit");
	}

	async compact(): Promise<void> {
		await this.#post("compact");
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

		// Per-call schema wins; otherwise fall back to the derived response
		// schema (partial) so non-stored fields don't fail hit validation.
		const effectiveSchema: ZodType<unknown> | undefined =
			fieldsSchema ?? (this.#responseSchema as ZodType<unknown> | undefined);

		if (effectiveSchema) {
			if (typeof query === "string") {
				query = { query, returnStored: true };
			} else {
				query = { ...query, returnStored: true };
			}
		}

		let snaked: Record<string, unknown>;
		if (typeof query === "string") {
			snaked = { query, limit: 10 };
		} else {
			const validated = validate(SearchRequestSchema, query, "search request");
			snaked = requestToSnake(validated);
		}

		const raw = await this.#post<RawSearchResult>("search", snaked);
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

	async close(): Promise<void> {
		// No-op: HTTP connections are stateless
	}

	// --- Internal helpers ---

	#url(endpoint: string): string {
		return `${this.#baseUrl}/indexes/${encodeURIComponent(this.#indexName)}/${endpoint}`;
	}

	#headers(): Record<string, string> {
		const headers: Record<string, string> = { "Content-Type": "application/json" };
		if (this.#writeKey) {
			headers["X-Write-Key"] = this.#writeKey;
		}
		return headers;
	}

	async #post<T = unknown>(endpoint: string, body?: unknown): Promise<T> {
		const response = await this.#fetch(this.#url(endpoint), {
			method: "POST",
			headers: this.#headers(),
			body: body !== undefined ? JSON.stringify(body) : undefined,
		});

		if (!response.ok) {
			let detail: string;
			try {
				const body = (await response.json()) as {
					error?: { type?: string; reason?: string };
				};
				detail = body.error?.reason ?? body.error?.type ?? response.statusText;
			} catch {
				detail = response.statusText;
			}
			throw new Error(`searchlite ${endpoint} failed (${response.status}): ${detail}`);
		}

		return (await response.json()) as T;
	}
}
