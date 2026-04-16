import type { ZodType } from "zod";
import type { SearchRequest, SearchResult, TypedSearchResult } from "./schemas";

/**
 * Abstract interface for a Searchlite index.
 *
 * Both `EmbeddedIndex` (local native engine) and `RemoteIndex` (HTTP client)
 * implement this interface. All methods are async to support both local
 * synchronous operations and remote HTTP calls transparently.
 *
 * The generic `T` carries the document shape. It defaults to
 * `Record<string, unknown>` so existing code keeps its current types
 * unchanged. When the index is constructed with a Zod-authored schema, users
 * can narrow `T` (e.g., `new EmbeddedIndex<z.infer<typeof UserSchema>>(...)`)
 * to flow typed fields through `add` / `addMany` / `search` results.
 */
export interface SearchIndex<T = Record<string, unknown>> {
	/** Add a single document. Must call `commit()` to make it searchable. */
	add(doc: T): Promise<void>;

	/** Add one or more documents. Returns the number of documents queued. */
	addMany(docs: T[] | T): Promise<number>;

	/** Persist queued documents so they become searchable. */
	commit(): Promise<void>;

	/** Merge all segments into one for better read performance. */
	compact(): Promise<void>;

	/** Search with a simple query string. */
	search(query: string): Promise<SearchResult<T>>;

	/** Search with a structured request object. */
	search(query: SearchRequest): Promise<SearchResult<T>>;

	/** Typed search: validates hit fields against a Zod schema. */
	search<U>(schema: ZodType<U>, query: string): Promise<TypedSearchResult<U>>;

	/** Typed search with structured request. */
	search<U>(schema: ZodType<U>, query: SearchRequest): Promise<TypedSearchResult<U>>;

	/** Release resources held by this index. */
	close(): Promise<void>;
}
