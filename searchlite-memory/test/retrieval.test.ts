import { describe, expect, it } from "vitest";
import { DEFAULT_RRF_K, rescore, rrfFuse } from "../src/memory/retrieval.js";

describe("rrfFuse", () => {
	it("computes 1/(k+rank) with 1-based rank", () => {
		const fused = rrfFuse([["a", "b"]], 60);
		expect(fused.get("a")).toBeCloseTo(1 / 61, 10);
		expect(fused.get("b")).toBeCloseTo(1 / 62, 10);
	});

	it("sums contributions across lists; overlap ranks higher", () => {
		// 'a' appears top of both lists; 'b' only in list 1; 'c' only in list 2.
		const fused = rrfFuse([
			["a", "b"],
			["a", "c"],
		]);
		const k = DEFAULT_RRF_K;
		expect(fused.get("a")).toBeCloseTo(1 / (k + 1) + 1 / (k + 1), 10);
		expect(fused.get("b")).toBeCloseTo(1 / (k + 2), 10);
		expect(fused.get("c")).toBeCloseTo(1 / (k + 2), 10);
		expect(fused.get("a") ?? 0).toBeGreaterThan(fused.get("b") ?? 0);
	});
});

describe("rescore", () => {
	it("ranks by the blended score and is deterministic", () => {
		const out = rescore([
			{ id: "hi", rrf: 1, importance: 1, ageHours: 0, accessCount: 20 },
			{ id: "lo", rrf: 0, importance: 0, ageHours: 10000, accessCount: 0 },
		]);
		expect(out[0].id).toBe("hi");
		expect(out[1].id).toBe("lo");
		expect(out[0].score).toBeGreaterThan(out[1].score);
	});

	it("recency decays by the configured half-life", () => {
		// Two identical candidates except age; with only recency weight, the
		// older one scores ~half at one half-life.
		const weights = { rel: 0, rec: 1, imp: 0, acc: 0 };
		const fresh = rescore([{ id: "f", rrf: 0, importance: 0, ageHours: 0, accessCount: 0 }], {
			weights,
			halfLifeHours: 168,
		})[0];
		const aged = rescore([{ id: "a", rrf: 0, importance: 0, ageHours: 168, accessCount: 0 }], {
			weights,
			halfLifeHours: 168,
		})[0];
		expect(fresh.score).toBeCloseTo(1, 6);
		expect(aged.score).toBeCloseTo(0.5, 2);
	});

	it("handles a single candidate (normalized relevance = 1)", () => {
		const out = rescore(
			[{ id: "only", rrf: 0.123, importance: 0.5, ageHours: 0, accessCount: 0 }],
			{
				weights: { rel: 1, rec: 0, imp: 0, acc: 0 },
			},
		);
		expect(out[0].score).toBeCloseTo(1, 6);
	});
});
