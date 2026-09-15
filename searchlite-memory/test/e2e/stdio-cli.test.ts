import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { afterEach, describe, expect, it } from "vitest";

// Gated: spawns the BUILT `dist/cli.js` (run `npm run test:e2e`, which builds
// first). FTS-only (EMBEDDER=none) so it is offline + fast. This is the only
// test that exercises the real bin shebang, ESM output, real stdio JSON-RPC
// transport, and that the server keeps stdout clean for the protocol.
const RUN = process.env.RUN_E2E === "1";
const CLI = fileURLToPath(new URL("../../dist/cli.js", import.meta.url));

function sc<T>(result: unknown): T {
	return (result as { structuredContent?: unknown }).structuredContent as T;
}

describe.runIf(RUN)("stdio CLI E2E (spawned server)", () => {
	const dirs: string[] = [];
	function tmp(): string {
		const d = mkdtempSync(join(tmpdir(), "slm-e2e-cli-"));
		dirs.push(d);
		return d;
	}
	afterEach(() => {
		for (const d of dirs) rmSync(d, { recursive: true, force: true });
		dirs.length = 0;
	});

	async function connect(root: string): Promise<Client> {
		const transport = new StdioClientTransport({
			command: process.execPath,
			args: [CLI, "serve"],
			env: {
				...(process.env as Record<string, string>),
				SEARCHLITE_MEMORY_DIR: root,
				SEARCHLITE_MEMORY_EMBEDDER: "none",
			},
		});
		const client = new Client({ name: "e2e-client", version: "0.0.0" });
		await client.connect(transport);
		return client;
	}

	it("round-trips remember/recall/get/forget over real stdio JSON-RPC", async () => {
		const client = await connect(join(tmp(), ".searchlite-memory"));

		const tools = (await client.listTools()).tools.map((t) => t.name).sort();
		expect(tools).toEqual(["forget", "get", "recall", "remember"]);

		const r = await client.callTool({
			name: "remember",
			arguments: { text: "the deploy is driven by release-plz on tags", tags: ["ci"] },
		});
		const { id } = sc<{ id: string }>(r);
		expect(id).toBeTruthy();

		const rec = await client.callTool({
			name: "recall",
			arguments: { query: "release-plz deploy" },
		});
		expect(sc<{ memories: { id: string }[] }>(rec).memories.some((m) => m.id === id)).toBe(true);

		const got = await client.callTool({ name: "get", arguments: { id } });
		expect(sc<{ found: boolean }>(got).found).toBe(true);

		await client.callTool({ name: "forget", arguments: { id } });
		const rec2 = await client.callTool({
			name: "recall",
			arguments: { query: "release-plz deploy" },
		});
		expect(sc<{ memories: { id: string }[] }>(rec2).memories.some((m) => m.id === id)).toBe(false);

		await client.close();
	}, 60_000);

	it("init + doctor work via the built CLI as subprocesses", () => {
		const project = tmp();
		execFileSync(process.execPath, [CLI, "init"], {
			env: { ...process.env, CLAUDE_PROJECT_DIR: project },
			stdio: "pipe",
		});
		expect(existsSync(join(project, ".mcp.json"))).toBe(true);
		expect(existsSync(join(project, ".searchlite-memory", ".gitignore"))).toBe(true);

		const out = execFileSync(process.execPath, [CLI, "doctor"], {
			env: {
				...process.env,
				SEARCHLITE_MEMORY_DIR: join(project, ".searchlite-memory"),
				SEARCHLITE_MEMORY_EMBEDDER: "none",
			},
			encoding: "utf8",
		});
		expect(out).toContain("All checks passed");
	});
});
