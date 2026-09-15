import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { runInit } from "../src/init.js";

let dirs: string[] = [];
function tempProject(): string {
	const d = mkdtempSync(join(tmpdir(), "slm-init-"));
	dirs.push(d);
	return d;
}
afterEach(() => {
	for (const d of dirs) rmSync(d, { recursive: true, force: true });
	dirs = [];
});

describe("init", () => {
	it("scaffolds .mcp.json and .searchlite-memory config", async () => {
		const root = tempProject();
		const res = await runInit({ CLAUDE_PROJECT_DIR: root } as NodeJS.ProcessEnv);
		expect(res.mcpWritten).toBe(true);
		expect(existsSync(join(root, ".mcp.json"))).toBe(true);
		expect(existsSync(join(root, ".searchlite-memory", ".gitignore"))).toBe(true);
		expect(existsSync(join(root, ".searchlite-memory", ".gitattributes"))).toBe(true);

		const gi = readFileSync(join(root, ".searchlite-memory", ".gitignore"), "utf8");
		const ignoreLines = gi
			.split("\n")
			.map((l) => l.trim())
			.filter((l) => l.length > 0 && !l.startsWith("#"));
		expect(ignoreLines).toContain("index/");
		// The committed source-of-truth files must NOT be ignore patterns.
		expect(ignoreLines).not.toContain("memory.jsonl");
		expect(ignoreLines).not.toContain("vectors.jsonl");

		const ga = readFileSync(join(root, ".searchlite-memory", ".gitattributes"), "utf8");
		expect(ga).toContain("memory.jsonl merge=union");

		const mcp = JSON.parse(readFileSync(join(root, ".mcp.json"), "utf8"));
		expect(mcp.mcpServers["searchlite-memory"]).toBeTruthy();
		expect(mcp.mcpServers["searchlite-memory"].args).toContain("serve");
	});

	it("does not overwrite an existing .mcp.json", async () => {
		const root = tempProject();
		writeFileSync(join(root, ".mcp.json"), '{"mcpServers":{"other":{}}}');
		const res = await runInit({ CLAUDE_PROJECT_DIR: root } as NodeJS.ProcessEnv);
		expect(res.mcpWritten).toBe(false);
		expect(readFileSync(join(root, ".mcp.json"), "utf8")).toContain('"other"');
	});
});
