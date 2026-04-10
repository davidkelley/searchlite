import { existsSync } from "node:fs";
import { join } from "node:path";
import { ZodError, type ZodType, prettifyError } from "zod";
import {
	DocumentSchema,
	DocumentsSchema,
	OpenOptionsSchema,
	SearchRequestSchema,
	SearchResultSchema,
	expandSchema,
} from "./schemas";

import type {
	SchemaDefinition,
	SearchRequest,
	SearchResult,
	TypedSearchResult,
} from "./schemas";

export type {
	FieldDefinition,
	FieldShorthand,
	Hit,
	KeywordFieldDef,
	NumericFieldDef,
	OpenOptions,
	SchemaDefinition,
	SearchRequest,
	SearchResult,
	TextFieldDef,
	TypedHit,
	TypedSearchResult,
} from "./schemas";

// --- Zod validation helper ---

function validate<T>(schema: ZodType<T>, data: unknown, label: string): T {
	try {
		return schema.parse(data);
	} catch (err) {
		if (err instanceof ZodError) {
			throw new Error(`Invalid ${label}:\n${prettifyError(err)}`);
		}
		throw err;
	}
}

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

const native = loadNativeBinding();

// --- camelCase -> snake_case for SearchRequest envelope ---

const REQUEST_KEY_MAP: Record<string, string> = {
	returnHits: "return_hits",
	returnStored: "return_stored",
	trackTotalHits: "track_total_hits",
	highlightField: "highlight_field",
	searchAfter: "search_after",
	candidateSize: "candidate_size",
	bmwBlockSize: "bmw_block_size",
};

function requestToSnake(obj: Record<string, unknown>): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	for (const [k, v] of Object.entries(obj)) {
		out[REQUEST_KEY_MAP[k] ?? k] = v;
	}
	if (out.fuzzy && typeof out.fuzzy === "object") {
		const fuzzy = out.fuzzy as Record<string, unknown>;
		const f: Record<string, unknown> = {};
		if ("maxEdits" in fuzzy) f.max_edits = fuzzy.maxEdits;
		if ("prefixLength" in fuzzy) f.prefix_length = fuzzy.prefixLength;
		out.fuzzy = f;
	}
	return out;
}

// --- snake_case -> camelCase for search results ---

interface RawHit {
	doc_id: string;
	score: number;
	vector_score?: number;
	sort_key?: unknown[];
	fields?: unknown;
	snippet?: string | null;
	explanation?: unknown;
	highlights?: Record<string, string[]>;
	inner_hits?: RawHit[];
}

interface RawSearchResult {
	total_hits_estimate: number;
	total_groups?: number;
	hits: RawHit[];
	next_cursor?: string;
	next_search_after?: unknown[];
	aggregations?: Record<string, unknown>;
	suggest?: Record<string, unknown>;
	profile?: unknown;
}

function transformHit(hit: RawHit): Record<string, unknown> {
	const out: Record<string, unknown> = {
		docId: hit.doc_id,
		score: hit.score,
	};
	if (hit.vector_score != null) out.vectorScore = hit.vector_score;
	if (hit.sort_key != null) out.sortKey = hit.sort_key;
	if ("fields" in hit) out.fields = hit.fields;
	if ("snippet" in hit) out.snippet = hit.snippet;
	if (hit.explanation != null) out.explanation = hit.explanation;
	if (hit.highlights != null) out.highlights = hit.highlights;
	if (hit.inner_hits != null) out.innerHits = hit.inner_hits.map(transformHit);
	return out;
}

function transformResult(raw: RawSearchResult) {
	return {
		totalHits: raw.total_hits_estimate,
		totalGroups: raw.total_groups,
		hits: (raw.hits ?? []).map(transformHit),
		nextCursor: raw.next_cursor,
		nextSearchAfter: raw.next_search_after,
		aggregations: raw.aggregations ?? {},
		suggest: raw.suggest ?? {},
		profile: raw.profile,
	};
}

// --- Public API ---

export class Index {
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

		this.#native = new native.Index(
			path,
			nativeOpts.schema || nativeOpts.writeKey ? nativeOpts : undefined,
		);
	}

	add(doc: Record<string, unknown>): void {
		validate(DocumentSchema, doc, "document");
		this.#native.add(doc);
	}

	addMany(docs: Record<string, unknown>[] | Record<string, unknown>): number {
		validate(DocumentsSchema, docs, "documents");
		return this.#native.addMany(docs);
	}

	commit(): void {
		this.#native.commit();
	}

	search<T>(schema: ZodType<T>, query: string): TypedSearchResult<T>;
	search<T>(schema: ZodType<T>, query: SearchRequest): TypedSearchResult<T>;
	search(query: string): SearchResult;
	search(query: SearchRequest): SearchResult;
	search<T = unknown>(
		queryOrSchema: string | SearchRequest | ZodType<T>,
		maybeQuery?: string | SearchRequest,
	): SearchResult | TypedSearchResult<T> {
		let fieldsSchema: ZodType<T> | undefined;
		let query: string | SearchRequest;

		if (maybeQuery !== undefined) {
			fieldsSchema = queryOrSchema as ZodType<T>;
			query = maybeQuery;
		} else {
			query = queryOrSchema as string | SearchRequest;
		}

		// Auto-set returnStored when a fields schema is provided
		if (fieldsSchema) {
			if (typeof query === "string") {
				query = { query, returnStored: true };
			} else {
				query = { ...query, returnStored: true };
			}
		}

		// Execute the search
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

		// Validate each hit's fields against the user's schema
		if (fieldsSchema) {
			const validateHitFields = (
				hit: SearchResult["hits"][number],
				path: string,
			): void => {
				const parsed = fieldsSchema!.safeParse(hit.fields);
				if (!parsed.success) {
					throw new Error(
						`Invalid fields on ${path} (docId: "${hit.docId}"):\n${prettifyError(parsed.error)}`,
					);
				}
				(hit as { fields: T }).fields = parsed.data;

				if (hit.innerHits) {
					for (let j = 0; j < hit.innerHits.length; j++) {
						validateHitFields(hit.innerHits[j], `${path}.innerHits[${j}]`);
					}
				}
			};

			for (let i = 0; i < result.hits.length; i++) {
				validateHitFields(result.hits[i], `hit ${i}`);
			}
			return result as unknown as TypedSearchResult<T>;
		}

		return result;
	}

	compact(): void {
		this.#native.compact();
	}

	close(): void {
		if (this.#closed) return;
		this.#closed = true;
		this.#native.close();
	}

	[Symbol.dispose](): void {
		this.close();
	}
}
