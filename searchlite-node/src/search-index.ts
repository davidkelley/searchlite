import type { ZodType } from "zod";
import type { SearchRequest, SearchResult, TypedSearchResult } from "./schemas";

/**
 * Abstract interface for a Searchlite index.
 *
 * Both `EmbeddedIndex` (local native engine) and `RemoteIndex` (HTTP client)
 * implement this interface. All methods are async to support both local
 * synchronous operations and remote HTTP calls transparently.
 */
export interface SearchIndex {
	/** Add a single document. Must call `commit()` to make it searchable. */
	add(doc: Record<string, unknown>): Promise<void>;

	/** Add one or more documents. Returns the number of documents queued. */
	addMany(docs: Record<string, unknown>[] | Record<string, unknown>): Promise<number>;

	/** Persist queued documents so they become searchable. */
	commit(): Promise<void>;

	/** Merge all segments into one for better read performance. */
	compact(): Promise<void>;

	/** Search with a simple query string. */
	search(query: string): Promise<SearchResult>;

	/** Search with a structured request object. */
	search(query: SearchRequest): Promise<SearchResult>;

	/** Typed search: validates hit fields against a Zod schema. */
	search<T>(schema: ZodType<T>, query: string): Promise<TypedSearchResult<T>>;

	/** Typed search with structured request. */
	search<T>(schema: ZodType<T>, query: SearchRequest): Promise<TypedSearchResult<T>>;

	/** Release resources held by this index. */
	close(): Promise<void>;
}
