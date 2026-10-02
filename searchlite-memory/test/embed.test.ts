import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { cacheKey, VectorCache } from "../src/embed/cache.js";
import { createEmbedder } from "../src/embed/embedder.js";
import { NullEmbedder } from "../src/embed/null.js";
import { stubEmbedder } from "./helpers.js";

let cleanup: string[] = [];
function tempDir(): string {
	const dir = mkdtempSync(join(tmpdir(), "slm-embed-"));
	cleanup.push(dir);
	return dir;
}
afterEach(() => {
	for (const d of cleanup) rmSync(d, { recursive: true, force: true });
	cleanup = [];
});

describe("NullEmbedder", () => {
	it("is unavailable and produces no vectors", async () => {
		const e = new NullEmbedder();
		expect(e.available).toBe(false);
		expect(await e.embed(["x"])).toEqual([]);
	});
});

describe("createEmbedder", () => {
	it("returns the null embedder for provider 'none'", async () => {
		const e = await createEmbedder({ provider: "none", model: "", dim: 0, quant: "" });
		expect(e.available).toBe(false);
	});
	it("throws for unimplemented external providers", async () => {
		await expect(
			createEmbedder({ provider: "openai", model: "x", dim: 1536, quant: "" }),
		).rejects.toThrow(/not implemented/);
	});
});

describe("stubEmbedder", () => {
	it("is deterministic and content-dependent", async () => {
		const e = stubEmbedder(8);
		const [a1] = await e.embed(["hello"]);
		const [a2] = await e.embed(["hello"]);
		const [b] = await e.embed(["world"]);
		expect(Array.from(a1)).toEqual(Array.from(a2));
		expect(Array.from(a1)).not.toEqual(Array.from(b));
		expect(a1.length).toBe(8);
	});
});

describe("cacheKey", () => {
	it("changes with the embedder fingerprint", () => {
		expect(cacheKey("modelA@main@q8", "text")).not.toBe(cacheKey("modelB@main@q8", "text"));
		expect(cacheKey("modelA@main@q8", "a")).not.toBe(cacheKey("modelA@main@q8", "b"));
		expect(cacheKey("m@main@q8", "x")).toBe(cacheKey("m@main@q8", "x"));
	});
});

describe("VectorCache", () => {
	it("get/set and persists across load", async () => {
		const path = join(tempDir(), "embeddings.cache");
		const c1 = new VectorCache(path);
		await c1.load();
		const key = cacheKey("m@main@q8", "hello");
		expect(c1.get(key)).toBeUndefined();
		c1.set(key, "AAA=");
		await c1.flush();

		const c2 = new VectorCache(path);
		await c2.load();
		expect(c2.get(key)).toBe("AAA=");
	});

	it("flush is a no-op when nothing changed", async () => {
		const path = join(tempDir(), "embeddings.cache");
		const c = new VectorCache(path);
		await c.load();
		await c.flush(); // should not create a file
		const c2 = new VectorCache(path);
		await c2.load();
		expect(c2.get("anything")).toBeUndefined();
	});
});
