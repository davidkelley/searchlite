import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { loadConfig } from "../../src/config.js";
import { createEmbedder } from "../../src/embed/embedder.js";
import { MemoryStore } from "../../src/memory/store.js";

// Gated: only runs with RUN_E2E=1 (downloads the ONNX model on first use and
// does real CPU inference). Kept tiny to constrain cost. The default `npm test`
// lane never executes or downloads anything here.
const RUN = process.env.RUN_E2E === "1";

const LOCAL = {
	provider: "local" as const,
	model: "Xenova/all-MiniLM-L6-v2",
	dim: 384,
	quant: "q8",
	revision: "main",
};

describe.runIf(RUN)("real-model E2E (transformers.js all-MiniLM-L6-v2)", () => {
	const dirs: string[] = [];
	function freshRoot(): string {
		const d = mkdtempSync(join(tmpdir(), "slm-e2e-model-"));
		dirs.push(d);
		return join(d, ".searchlite-memory");
	}
	afterAll(() => {
		for (const d of dirs) rmSync(d, { recursive: true, force: true });
	});

	it("produces 384-dim embeddings from the real model", async () => {
		const e = await createEmbedder(LOCAL);
		expect(e.available).toBe(true);
		expect(e.dim).toBe(384);
		const [v] = await e.embed(["a short sentence"]);
		expect(v.length).toBe(384);
	}, 180_000);

	it("semantic recall: a paraphrase with no shared keywords finds the right memory", async () => {
		const config = loadConfig({
			SEARCHLITE_MEMORY_DIR: freshRoot(),
			SEARCHLITE_MEMORY_EMBEDDER: "local",
		});
		const store = await MemoryStore.open(config);
		expect(store.vectorsEnabled).toBe(true);

		const dog = await store.remember({
			text: "My canine companion loves fetching tennis balls at the park.",
		});
		await store.remember({ text: "The quarterly financial report is due next Friday." });
		await store.remember({ text: "Preheat the oven to 200 degrees before baking the bread." });

		// Shares no content words with the dog memory — only the vector half of
		// the hybrid search can surface it.
		const res = await store.recall("a pet that enjoys playing outside");
		expect(res.memories.length).toBeGreaterThan(0);
		expect(res.memories[0].id).toBe(dog.id);
		await store.close();
	}, 180_000);

	it("rebuild reuses the committed int8 sidecar vectors (no re-embed / no rewrite)", async () => {
		const config = loadConfig({
			SEARCHLITE_MEMORY_DIR: freshRoot(),
			SEARCHLITE_MEMORY_EMBEDDER: "local",
		});
		const store = await MemoryStore.open(config);
		const m = await store.remember({
			text: "Photosynthesis converts sunlight into chemical energy in plants.",
		});
		await store.close();

		expect(existsSync(config.paths.sidecar)).toBe(true);
		const sidecarBefore = readFileSync(config.paths.sidecar, "utf8");
		const entry = JSON.parse(sidecarBefore.trim().split("\n")[0]);
		expect(entry.dim).toBe(384);
		expect(entry.quant).toBe("i8");
		expect(entry.model).toContain("all-MiniLM-L6-v2");

		// Wipe the gitignored index; reopen forces a rebuild from ledger+sidecar.
		rmSync(config.paths.indexDir, { recursive: true, force: true });
		const store2 = await MemoryStore.open(config);
		const res = await store2.recall("how do plants make food from light");
		expect(res.memories.some((x) => x.id === m.id)).toBe(true);
		await store2.close();

		// Rebuild reads the sidecar but must not recompute/rewrite it.
		expect(readFileSync(config.paths.sidecar, "utf8")).toBe(sidecarBefore);
	}, 180_000);
});
