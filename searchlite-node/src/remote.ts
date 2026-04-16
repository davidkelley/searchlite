import type { ZodType } from "zod";
import {
	DocumentSchema,
	DocumentsSchema,
	SearchRequestSchema,
	SearchResultSchema,
} from "./schemas";
import type { SearchRequest, SearchResult, TypedSearchResult } from "./schemas";
import type { SearchIndex } from "./search-index";
import {
	type RawSearchResult,
	requestToSnake,
	transformResult,
	validate,
	validateTypedResult,
} from "./transform";
import { type ZodIndexSchema, isZodIndexSchema } from "./zod/compile";

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
		}
	}

	async add(doc: T): Promise<void> {
		if (this.#zodSchema) {
			validate(this.#zodSchema as unknown as ZodType<T>, doc, "document");
		} else {
			validate(DocumentSchema, doc as Record<string, unknown>, "document");
		}
		await this.#post("bulk", { docs: [doc] });
	}

	async addMany(docs: T[] | T): Promise<number> {
		if (this.#zodSchema) {
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
		const docsArray = Array.isArray(docs) ? docs : [docs];
		const body = await this.#post<{ queued: number }>("bulk", { docs: docsArray });
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

		const effectiveSchema: ZodType<unknown> | undefined =
			fieldsSchema ?? (this.#zodSchema as ZodType<unknown> | undefined);

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
