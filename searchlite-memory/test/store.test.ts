import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadConfig, type MemoryConfig } from "../src/config.js";
import { MemoryStore } from "../src/memory/store.js";
import { stubEmbedder } from "./helpers.js";

let dirs: string[] = [];
function freshRoot(): string {
	const d = mkdtempSync(join(tmpdir(), "slm-store-"));
	dirs.push(d);
	return join(d, ".searchlite-memory");
}
afterEach(() => {
	for (const d of dirs) rmSync(d, { recursive: true, force: true });
	dirs = [];
});

function cfg(root: string, provider = "none"): MemoryConfig {
	return loadConfig({ SEARCHLITE_MEMORY_DIR: root, SEARCHLITE_MEMORY_EMBEDDER: provider });
}

describe("MemoryStore (hybrid, stub embedder)", () => {
	it("remember -> recall -> get -> forget -> recall(gone)", async () => {
		const root = freshRoot();
		const store = await MemoryStore.open(cfg(root), stubEmbedder(16));
		const { id } = await store.remember({
			text: "the deploy uses release-plz on tags",
			tags: ["ci", "release"],
		});
		expect(id).toBeTruthy();

		const recalled = await store.recall("release process");
		expect(recalled.memories.length).toBeGreaterThan(0);
		expect(recalled.memories.some((m) => m.id === id)).toBe(true);

		const got = await store.get(id);
		expect(got?.text).toContain("release-plz");

		await store.forget(id);
		const after = await store.recall("release process");
		expect(after.memories.some((m) => m.id === id)).toBe(false);
		await store.close();
	});

	it("dedups identical content idempotently", async () => {
		const store = await MemoryStore.open(cfg(freshRoot()), stubEmbedder(16));
		const a = await store.remember({ text: "same fact" });
		const b = await store.remember({ text: "same fact" });
		expect(b.deduped).toBe(true);
		expect(b.id).toBe(a.id);
		await store.close();
	});

	it("works in full-text-only mode (no embedder)", async () => {
		const store = await MemoryStore.open(cfg(freshRoot(), "none"));
		expect(store.vectorsEnabled).toBe(false);
		const { id } = await store.remember({ text: "rust borrow checker tips" });
		const res = await store.recall("borrow checker");
		expect(res.memories.some((m) => m.id === id)).toBe(true);
		await store.close();
	});

	it("persists across reopen without rebuilding", async () => {
		const root = freshRoot();
		const store1 = await MemoryStore.open(cfg(root), stubEmbedder(16));
		const { id } = await store1.remember({ text: "persisted memory" });
		await store1.close();

		const store2 = await MemoryStore.open(cfg(root), stubEmbedder(16));
		expect((await store2.get(id))?.text).toBe("persisted memory");
		await store2.close();
	});

	it("rebuilds the index from the ledger when index/ is missing", async () => {
		const root = freshRoot();
		const config = cfg(root);
		const store1 = await MemoryStore.open(config, stubEmbedder(16));
		const { id } = await store1.remember({ text: "survives an index wipe" });
		await store1.close();

		// Simulate a fresh clone / gitignored index removed.
		rmSync(config.paths.indexDir, { recursive: true, force: true });

		const store2 = await MemoryStore.open(cfg(root), stubEmbedder(16));
		const res = await store2.recall("index wipe");
		expect(res.memories.some((m) => m.id === id)).toBe(true);
		await store2.close();
	});

	it("filters recall by namespace", async () => {
		const store = await MemoryStore.open(cfg(freshRoot()), stubEmbedder(16));
		const a = await store.remember({ text: "alpha note", namespace: "auth" });
		await store.remember({ text: "beta note", namespace: "ci" });
		const res = await store.recall("note", { namespace: "auth" });
		expect(res.memories.every((m) => m.namespace === "auth")).toBe(true);
		expect(res.memories.some((m) => m.id === a.id)).toBe(true);
		await store.close();
	});

	it("doctor reports healthy on a populated store", async () => {
		const store = await MemoryStore.open(cfg(freshRoot()), stubEmbedder(16));
		await store.remember({ text: "x", tags: ["t"] });
		const report = await store.doctor();
		expect(report.ok).toBe(true);
		await store.close();
	});

	it("supersedes replaces a memory atomically", async () => {
		const store = await MemoryStore.open(cfg(freshRoot()), stubEmbedder(16));
		const a = await store.remember({ text: "old policy: deploy on fridays" });
		const b = await store.remember({ text: "new policy: deploy any day", supersedes: a.id });
		expect(b.id).not.toBe(a.id);
		expect(await store.get(a.id)).toBeNull();
		expect((await store.get(b.id))?.text).toContain("any day");
		await store.close();
	});

	it("forget is idempotent and writes no phantom tombstones for unknown ids", async () => {
		const root = freshRoot();
		const config = cfg(root);
		const store = await MemoryStore.open(config, stubEmbedder(16));
		expect((await store.forget("nope")).forgotten).toBe(true);
		expect((await store.forget("nope")).forgotten).toBe(true);
		// Nothing live was forgotten → the ledger file is never created.
		expect(existsSync(config.paths.ledger)).toBe(false);
		await store.close();
	});

	it("merges concurrent writers without losing records", async () => {
		const root = freshRoot();
		const s1 = await MemoryStore.open(cfg(root, "none"));
		const s2 = await MemoryStore.open(cfg(root, "none"));
		const a = await s1.remember({ text: "alpha fact one" });
		const b = await s2.remember({ text: "bravo fact two" });
		await s1.close();
		await s2.close();

		const s3 = await MemoryStore.open(cfg(root, "none"));
		expect((await s3.get(a.id))?.text).toContain("alpha");
		expect((await s3.get(b.id))?.text).toContain("bravo");
		await s3.close();
	});
});
