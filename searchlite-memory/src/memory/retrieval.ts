// Hybrid retrieval math: Reciprocal Rank Fusion (RRF) over two ranked lists,
// then a Generative-Agents-style re-score blending relevance + recency +
// importance + access frequency. Pure functions — no IO — so they're trivially
// testable and deterministic.

export const DEFAULT_RRF_K = 60;

/**
 * Reciprocal Rank Fusion: `score(d) = Σ_lists 1/(k + rank(d))` with 1-based
 * rank. Rank-based, so it needs no score normalization across the lists (BM25
 * is unbounded while cosine is bounded — a raw blend would be scale-mismatched).
 * Returns a map of id → fused score.
 */
export function rrfFuse(lists: string[][], k: number = DEFAULT_RRF_K): Map<string, number> {
	const scores = new Map<string, number>();
	for (const list of lists) {
		for (let i = 0; i < list.length; i++) {
			const id = list[i];
			scores.set(id, (scores.get(id) ?? 0) + 1 / (k + i + 1));
		}
	}
	return scores;
}

export interface ScoringWeights {
	rel: number;
	rec: number;
	imp: number;
	acc: number;
}

export const DEFAULT_WEIGHTS: ScoringWeights = { rel: 0.6, rec: 0.2, imp: 0.15, acc: 0.05 };

export interface RescoreCandidate {
	id: string;
	/** Fused RRF score (min-max normalized across the candidate set). */
	rrf: number;
	/** Importance in [0,1]. */
	importance: number;
	/** Hours since last access (or creation). */
	ageHours: number;
	/** Access count (recall/get hits). */
	accessCount: number;
}

export interface RescoreOptions {
	weights?: ScoringWeights;
	/** Recency decay half-life in hours (default 168 = 1 week). */
	halfLifeHours?: number;
	/** Access count that maps to ~full access boost (default 20). */
	accessCap?: number;
}

export interface ScoredCandidate {
	id: string;
	score: number;
}

/**
 * Re-score fused candidates:
 *   final = w_rel·norm(rrf) + w_rec·exp(-λ·age) + w_imp·importance + w_acc·access
 * where norm(rrf) is min-max over the candidate set, λ = ln2/halfLife, and
 * access = log1p(count)/log1p(cap). Sorted descending; id breaks ties so the
 * order is deterministic.
 */
export function rescore(
	candidates: RescoreCandidate[],
	opts: RescoreOptions = {},
): ScoredCandidate[] {
	const weights = opts.weights ?? DEFAULT_WEIGHTS;
	const halfLife = opts.halfLifeHours ?? 168;
	const accessCap = opts.accessCap ?? 20;
	const lambda = Math.LN2 / halfLife;
	const logCap = Math.log1p(Math.max(0, accessCap));

	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	for (const c of candidates) {
		if (c.rrf < min) min = c.rrf;
		if (c.rrf > max) max = c.rrf;
	}
	const range = max - min;

	const scored = candidates.map((c) => {
		const normRrf = range > 0 ? (c.rrf - min) / range : 1;
		const recency = Math.exp(-lambda * Math.max(0, c.ageHours));
		const importance = Math.min(1, Math.max(0, c.importance));
		const access = logCap > 0 ? Math.min(1, Math.log1p(Math.max(0, c.accessCount)) / logCap) : 0;
		const score =
			weights.rel * normRrf +
			weights.rec * recency +
			weights.imp * importance +
			weights.acc * access;
		return { id: c.id, score };
	});

	scored.sort((a, b) => b.score - a.score || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
	return scored;
}
