import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { canonicalJson } from "../src/memory/io.js";
import { materialize, readLedger, writeLedger } from "../src/memory/ledger.js";
import {
	contentHashOf,
	type MemoryRecord,
	makeAddRecord,
	makeForgetRecord,
	SCHEMA_VERSION,
} from "../src/memory/model.js";
import { dequantizeInt8, quantizeInt8 } from "../src/memory/vectors.js";

let cleanup: string[] = [];
function tempFile(name: string): string {
	const dir = mkdtempSync(join(tmpdir(), "slm-mem-"));
	cleanup.push(dir);
	return join(dir, name);
}
afterEach(() => {
	for (const d of cleanup) rmSync(d, { recursive: true, force: true });
	cleanup = [];
});

describe("canonicalJson", () => {
	it("is key-order independent", () => {
		expect(canonicalJson({ b: 1, a: 2 })).toBe(canonicalJson({ a: 2, b: 1 }));
		expect(canonicalJson({ a: 2, b: 1 })).toBe('{"a":2,"b":1}');
	});
});

describe("contentHashOf", () => {
	it("is stable under tag/entity reordering and duplication", () => {
		const a = contentHashOf({
			text: "hi",
			type: "semantic",
			namespace: "default",
			tags: ["b", "a", "a"],
			entities: ["y", "x"],
		});
		const b = contentHashOf({
			text: "hi",
			type: "semantic",
			namespace: "default",
			tags: ["a", "b"],
			entities: ["x", "y"],
		});
		expect(a).toBe(b);
	});
	it("differs when text/type/namespace differ", () => {
		const base = { text: "hi", type: "semantic", namespace: "default" } as const;
		expect(contentHashOf(base)).not.toBe(contentHashOf({ ...base, text: "bye" }));
		expect(contentHashOf(base)).not.toBe(contentHashOf({ ...base, namespace: "other" }));
	});
});

describe("ledger determinism", () => {
	it("writes byte-identical files regardless of input order", async () => {
		const r1 = makeAddRecord({ text: "one" }, new Date("2026-01-01T00:00:00Z"));
		const r2 = makeAddRecord({ text: "two" }, new Date("2026-01-02T00:00:00Z"));
		const r3 = makeAddRecord({ text: "three" }, new Date("2026-01-03T00:00:00Z"));

		const fileA = tempFile("a.jsonl");
		const fileB = tempFile("b.jsonl");
		await writeLedger(fileA, [r1, r2, r3]);
		await writeLedger(fileB, [r3, r1, r2]);
		expect(readFileSync(fileA, "utf8")).toBe(readFileSync(fileB, "utf8"));
	});

	it("round-trips through readLedger", async () => {
		const rec = makeAddRecord({ text: "hello", tags: ["x"] }, new Date("2026-01-01T00:00:00Z"));
		const file = tempFile("l.jsonl");
		await writeLedger(file, [rec]);
		const { records, malformed } = await readLedger(file, SCHEMA_VERSION);
		expect(malformed).toEqual([]);
		expect(records).toHaveLength(1);
		expect(records[0].text).toBe("hello");
	});
});

describe("materialize", () => {
	it("excludes a tombstoned id", () => {
		const add = makeAddRecord({ text: "secret" }, new Date("2026-01-01T00:00:00Z"));
		const forget = makeForgetRecord(add.id, new Date("2026-01-02T00:00:00Z"));
		const { live, tombstoned } = materialize([add, forget]);
		expect(live.length).toBe(0);
		expect(tombstoned).toContain(add.id);
	});

	it("orders by (opTs,id) not file order: a forget reordered before its add still deletes", () => {
		const add = makeAddRecord({ text: "secret" }, new Date("2026-01-01T00:00:00Z"));
		const forget = makeForgetRecord(add.id, new Date("2026-01-02T00:00:00Z"));
		// Pass the forget FIRST (as a union-merge could reorder lines):
		const { live } = materialize([forget, add]);
		expect(live.length).toBe(0);
	});

	it("dedups live records by contentHash keeping the lowest ULID", () => {
		const early = makeAddRecord({ text: "dup" }, new Date("2026-01-01T00:00:00Z"));
		const late = makeAddRecord({ text: "dup" }, new Date("2026-01-02T00:00:00Z"));
		expect(early.contentHash).toBe(late.contentHash);
		const { live } = materialize([late, early]);
		expect(live.length).toBe(1);
		expect(live[0].id).toBe(early.id < late.id ? early.id : late.id);
	});

	it("excludes invalidated records", () => {
		const add = makeAddRecord({ text: "stale" }, new Date("2026-01-01T00:00:00Z"));
		add.invalidAt = "2026-01-05T00:00:00Z";
		const { live } = materialize([add], new Date("2026-02-01T00:00:00Z"));
		expect(live.length).toBe(0);
	});
});

describe("int8 vector codec", () => {
	it("round-trips sign and unit-norm for vectors with negative components", () => {
		const v = [0.4, -0.7, 0.1, -0.2, 0.55];
		const b64 = quantizeInt8(v);
		const out = dequantizeInt8(b64, v.length);
		// unit norm
		let norm = 0;
		for (const x of out) norm += x * x;
		expect(Math.sqrt(norm)).toBeCloseTo(1, 5);
		// sign preserved (negative components stay negative — guards the i8/u8 trap)
		expect(out[1]).toBeLessThan(0);
		expect(out[3]).toBeLessThan(0);
		expect(out[0]).toBeGreaterThan(0);
		// direction preserved within int8 tolerance
		const unit = (() => {
			let n = 0;
			for (const x of v) n += x * x;
			n = Math.sqrt(n);
			return v.map((x) => x / n);
		})();
		for (let i = 0; i < v.length; i++) expect(out[i]).toBeCloseTo(unit[i], 1);
	});

	it("rejects non-finite and zero vectors", () => {
		expect(() => quantizeInt8([0, 0, 0])).toThrow();
		expect(() => quantizeInt8([1, Number.NaN])).toThrow();
	});

	it("throws on dim mismatch", () => {
		const b64 = quantizeInt8([1, 0, 0, 0]);
		expect(() => dequantizeInt8(b64, 3)).toThrow();
	});
});

// Type-only sanity: a forget record has no content fields.
const _exampleForget: MemoryRecord = makeForgetRecord("01J", new Date("2026-01-01T00:00:00Z"));
void _exampleForget;
