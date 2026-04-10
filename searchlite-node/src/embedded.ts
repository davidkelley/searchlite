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

// --- EmbeddedIndex ---

export class EmbeddedIndex implements SearchIndex {
	#native: NativeIndex;
	#closed = false;

	constructor(path: string, options?: { writeKey?: string; schema?: SchemaDefinition }) {
		if (typeof path !== "string" || path.length === 0) {
			throw new Error("path must be a non-empty string");
		}
		const parsed = validate(OpenOptionsSchema, options, "options");
		const nativeOpts: Record<string, unknown> = {};

		if (parsed) {
			nativeOpts.writeKey = parsed.writeKey;
			if (parsed.schema) {
				nativeOpts.schema = expandSchema(parsed.schema);
			}
		}

		this.#native = new (getNative().Index)(
			path,
			nativeOpts.schema || nativeOpts.writeKey ? nativeOpts : undefined,
		);
	}

	async add(doc: Record<string, unknown>): Promise<void> {
		validate(DocumentSchema, doc, "document");
		this.#native.add(doc);
	}

	async addMany(docs: Record<string, unknown>[] | Record<string, unknown>): Promise<number> {
		validate(DocumentsSchema, docs, "documents");
		return this.#native.addMany(docs);
	}

	async commit(): Promise<void> {
		this.#native.commit();
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

		if (fieldsSchema) {
			return validateTypedResult(result, fieldsSchema);
		}

		return result;
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
