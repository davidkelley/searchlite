import { ZodError, type ZodType, prettifyError } from "zod";
import type { Hit, SearchResult, TypedSearchResult } from "./schemas";

// --- Zod validation helper ---

export function validate<T>(schema: ZodType<T>, data: unknown, label: string): T {
	try {
		return schema.parse(data);
	} catch (err) {
		if (err instanceof ZodError) {
			throw new Error(`Invalid ${label}:\n${prettifyError(err)}`);
		}
		throw err;
	}
}

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

export function requestToSnake(obj: Record<string, unknown>): Record<string, unknown> {
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

export interface RawHit {
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

export interface RawSearchResult {
	total_hits_estimate: number;
	total_groups?: number;
	hits: RawHit[];
	next_cursor?: string;
	next_search_after?: unknown[];
	aggregations?: Record<string, unknown>;
	suggest?: Record<string, unknown>;
	profile?: unknown;
}

export function transformHit(hit: RawHit): Record<string, unknown> {
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

export function transformResult(raw: RawSearchResult) {
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

// --- Typed search result validation ---

export function validateTypedResult<T>(
	result: SearchResult,
	fieldsSchema: ZodType<T>,
): TypedSearchResult<T> {
	const validateHitFields = (hit: Hit, path: string): void => {
		const parsed = fieldsSchema.safeParse(hit.fields);
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
