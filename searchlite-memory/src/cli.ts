#!/usr/bin/env node
/**
 * searchlite-memory CLI entry point.
 *
 * Subcommands:
 *   serve   (default) — run the MCP stdio server (Stage 7)
 *   rebuild           — rebuild the searchlite index from the committed ledger (Stage 5)
 *   doctor            — read-only health report (Stage 5)
 *   init              — scaffold .mcp.json + .searchlite-memory config (Stage 8)
 */

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

interface ParsedArgs {
	command: Command;
	rest: string[];
}

/** Returns null when help should be shown (explicit --help or unknown command). */
function parseArgs(argv: string[]): ParsedArgs | null {
	const args = argv.slice(2);
	if (args.includes("--help") || args.includes("-h")) return null;
	const first = args[0];
	// No subcommand, or only flags → default to `serve`.
	if (first === undefined || first.startsWith("-")) {
		return { command: "serve", rest: args };
	}
	if (isCommand(first)) {
		return { command: first, rest: args.slice(1) };
	}
	return null;
}

async function main(): Promise<void> {
	const parsed = parseArgs(process.argv);
	if (!parsed) {
		process.stdout.write(USAGE);
		return;
	}
	// Command implementations land in later stages (serve: 7, rebuild/doctor: 5,
	// init: 8). The scaffold recognizes and dispatches them today.
	process.stderr.write(
		`searchlite-memory: '${parsed.command}' is not implemented yet (scaffold)\n`,
	);
	process.exitCode = 70; // EX_SOFTWARE
}

main().catch((err) => {
	process.stderr.write(`searchlite-memory: ${err instanceof Error ? err.message : String(err)}\n`);
	process.exitCode = 1;
});
