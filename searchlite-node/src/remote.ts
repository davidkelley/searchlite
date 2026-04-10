import type { ZodType } from "zod";
import type { SearchIndex } from "./search-index";
import {
	DocumentSchema,
	DocumentsSchema,
	SearchRequestSchema,
	SearchResultSchema,
} from "./schemas";
import type { SearchRequest, SearchResult, TypedSearchResult } from "./schemas";
import {
	type RawSearchResult,
	requestToSnake,
	transformResult,
	validate,
	validateTypedResult,
} from "./transform";

export interface RemoteIndexOptions {
	/** Write key sent via X-Write-Key header for protected indexes. */
	writeKey?: string;
	/** Custom fetch implementation for testing or custom transports. Defaults to global fetch. */
	fetch?: typeof globalThis.fetch;
}

export class RemoteIndex implements SearchIndex {
	readonly #baseUrl: string;
	readonly #indexName: string;
	readonly #writeKey?: string;
	readonly #fetch: typeof globalThis.fetch;

	constructor(baseUrl: string, indexName: string, options?: RemoteIndexOptions) {
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
	}

	async add(doc: Record<string, unknown>): Promise<void> {
		validate(DocumentSchema, doc, "document");
		await this.#post("bulk", { docs: [doc] });
	}

	async addMany(docs: Record<string, unknown>[] | Record<string, unknown>): Promise<number> {
		validate(DocumentsSchema, docs, "documents");
		const docsArray = Array.isArray(docs) ? docs : [docs];
		const body = await this.#post<{ added: number }>("bulk", { docs: docsArray });
		return body.added;
	}

	async commit(): Promise<void> {
		await this.#post("commit");
	}

	async compact(): Promise<void> {
		await this.#post("compact");
	}

	async search<T>(schema: ZodType<T>, query: string): Promise<TypedSearchResult<T>>;
	async search<T>(schema: ZodType<T>, query: SearchRequest): Promise<TypedSearchResult<T>>;
	async search(query: string): Promise<SearchResult>;
	async search(query: SearchRequest): Promise<SearchResult>;
	async search<T = unknown>(
		queryOrSchema: string | SearchRequest | ZodType<T>,
		maybeQuery?: string | SearchRequest,
	): Promise<SearchResult | TypedSearchResult<T>> {
		let fieldsSchema: ZodType<T> | undefined;
		let query: string | SearchRequest;

		if (maybeQuery !== undefined) {
			fieldsSchema = queryOrSchema as ZodType<T>;
			query = maybeQuery;
		} else {
			query = queryOrSchema as string | SearchRequest;
		}

		if (fieldsSchema) {
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

		if (fieldsSchema) {
			return validateTypedResult(result, fieldsSchema);
		}

		return result;
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
				const err = (await response.json()) as { kind?: string; reason?: string };
				detail = err.reason ?? err.kind ?? response.statusText;
			} catch {
				detail = response.statusText;
			}
			throw new Error(`searchlite ${endpoint} failed (${response.status}): ${detail}`);
		}

		return (await response.json()) as T;
	}
}
