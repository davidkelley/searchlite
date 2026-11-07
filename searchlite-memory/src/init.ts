import { access, copyFile, mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

function assetsDir(): string {
	// dist/init.js -> ../assets (assets ships alongside dist in the package).
	return fileURLToPath(new URL("../assets/", import.meta.url));
}

async function exists(path: string): Promise<boolean> {
	try {
		await access(path);
		return true;
	} catch {
		return false;
	}
}

export interface InitResult {
	memDir: string;
	mcpWritten: boolean;
	lines: string[];
}

/**
 * Scaffold a repo for searchlite-memory: a project-scoped `.mcp.json` and the
 * `.searchlite-memory/` directory with its committed `.gitignore` (cache files)
 * and `.gitattributes` (union-merge for the ledger). Never overwrites an
 * existing `.mcp.json`.
 */
export async function runInit(env: NodeJS.ProcessEnv = process.env): Promise<InitResult> {
	const projectRoot = env.CLAUDE_PROJECT_DIR ?? process.cwd();
	const memDir = join(projectRoot, ".searchlite-memory");
	const assets = assetsDir();
	const lines: string[] = [];

	await mkdir(memDir, { recursive: true });
	await copyFile(join(assets, "gitignore.template"), join(memDir, ".gitignore"));
	await copyFile(join(assets, "gitattributes.template"), join(memDir, ".gitattributes"));
	lines.push(`Initialized ${memDir} (.gitignore + .gitattributes written).`);

	const mcpPath = join(projectRoot, ".mcp.json");
	let mcpWritten = false;
	if (await exists(mcpPath)) {
		const template = await readFile(join(assets, "mcp.template.json"), "utf8");
		lines.push(`.mcp.json already exists at ${mcpPath} — left untouched. Merge in:`);
		lines.push(template.trimEnd());
	} else {
		await copyFile(join(assets, "mcp.template.json"), mcpPath);
		mcpWritten = true;
		lines.push(`Wrote ${mcpPath}.`);
	}

	lines.push("Commit memory.jsonl + vectors.jsonl; index/ and caches are gitignored.");
	lines.push(
		"If your MCP host does not set CLAUDE_PROJECT_DIR, set SEARCHLITE_MEMORY_DIR to an " +
			"absolute path in .mcp.json.",
	);
	lines.push(`A skill describing the tools is bundled at ${join(assets, "SKILL.md")}.`);

	return { memDir, mcpWritten, lines };
}
