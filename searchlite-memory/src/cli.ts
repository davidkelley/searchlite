#!/usr/bin/env node
/**
 * searchlite-memory CLI entry point.
 *
 * Subcommands:
 *   serve   (default) — run the MCP stdio server
 *   rebuild [--reembed] — rebuild the searchlite index from the committed ledger
 *   doctor            — read-only health report (non-zero exit on problems)
 *   init              — scaffold .mcp.json + .searchlite-memory config (Stage 8)
 */
import { loadConfig } from "./config.js";
import { runInit } from "./init.js";
import { MemoryStore } from "./memory/store.js";
import { serveStdio } from "./server.js";

const USAGE = `searchlite-memory — repository-local memory MCP server (full-text + vector)

Usage:
  searchlite-memory [serve]            Run the MCP server over stdio (default)
  searchlite-memory rebuild [--reembed]
                                       Rebuild the index from the committed ledger
  searchlite-memory doctor             Print a health report; non-zero exit on problems
  searchlite-memory init               Scaffold .mcp.json and .searchlite-memory/ config
  searchlite-memory --help             Show this help

Environment:
  SEARCHLITE_MEMORY_DIR                Memory dir (default:
                                       $CLAUDE_PROJECT_DIR/.searchlite-memory or ./.searchlite-memory)
  SEARCHLITE_MEMORY_EMBEDDER           local | none | openai | voyage | cohere (default: local)
`;

const COMMANDS = ["serve", "rebuild", "doctor", "init"] as const;
type Command = (typeof COMMANDS)[number];

function isCommand(value: string): value is Command {
	return (COMMANDS as readonly string[]).includes(value);
}

function parseArgs(argv: string[]): { command: Command; rest: string[] } | null {
	const args = argv.slice(2);
	if (args.includes("--help") || args.includes("-h")) return null;
	const first = args[0];
	if (first === undefined || first.startsWith("-")) return { command: "serve", rest: args };
	if (isCommand(first)) return { command: first, rest: args.slice(1) };
	return null;
}

async function cmdServe(): Promise<void> {
	const store = await MemoryStore.open(loadConfig());
	// Note: do NOT write to stdout — it is the JSON-RPC channel. Logs go to stderr.
	await serveStdio(store);
}

async function cmdRebuild(reembed: boolean): Promise<void> {
	const store = await MemoryStore.open(loadConfig());
	await store.rebuild(reembed);
	await store.close();
	process.stderr.write(`searchlite-memory: index rebuilt${reembed ? " (re-embedded)" : ""}.\n`);
}

async function cmdDoctor(): Promise<void> {
	const store = await MemoryStore.open(loadConfig());
	const report = await store.doctor();
	await store.close();
	for (const c of report.checks) {
		process.stdout.write(`${c.ok ? "ok  " : "FAIL"}  ${c.name}: ${c.detail}\n`);
	}
	process.stdout.write(report.ok ? "\nAll checks passed.\n" : "\nSome checks failed.\n");
	if (!report.ok) process.exitCode = 1;
}

async function main(): Promise<void> {
	const parsed = parseArgs(process.argv);
	if (!parsed) {
		process.stdout.write(USAGE);
		return;
	}
	switch (parsed.command) {
		case "serve":
			return cmdServe();
		case "rebuild":
			return cmdRebuild(parsed.rest.includes("--reembed"));
		case "doctor":
			return cmdDoctor();
		case "init": {
			const result = await runInit();
			for (const line of result.lines) process.stdout.write(`${line}\n`);
			return;
		}
	}
}

main().catch((err) => {
	process.stderr.write(`searchlite-memory: ${err instanceof Error ? err.message : String(err)}\n`);
	process.exitCode = 1;
});
