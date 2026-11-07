import { randomUUID } from "node:crypto";
import { readFile, rename, unlink, writeFile } from "node:fs/promises";

/**
 * Deterministic JSON serialization: object keys are emitted in sorted order at
 * every level so the same logical record always produces byte-identical output
 * (diff-friendly, hashable). Arrays preserve order (callers sort first where
 * order should not matter, e.g. tags). Values are otherwise standard JSON.
 */
export function canonicalJson(value: unknown): string {
	return JSON.stringify(sortKeys(value));
}

function sortKeys(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(sortKeys);
	if (value && typeof value === "object") {
		const obj = value as Record<string, unknown>;
		const out: Record<string, unknown> = {};
		for (const key of Object.keys(obj).sort()) {
			out[key] = sortKeys(obj[key]);
		}
		return out;
	}
	return value;
}

/**
 * Write `contents` to `path` atomically: write a sibling temp file, then
 * rename over the target. A crash leaves either the old file or the new file
 * intact — never a partially-written one. (rename is atomic within a
 * filesystem; the temp file is a sibling so it shares one.)
 */
export async function atomicWriteFile(path: string, contents: string): Promise<void> {
	const tmp = `${path}.tmp.${process.pid}.${randomUUID()}`;
	try {
		await writeFile(tmp, contents, "utf8");
		await rename(tmp, path);
	} catch (err) {
		// Don't leave the temp file behind on a failed write/rename.
		await unlink(tmp).catch(() => {});
		throw err;
	}
}

/** Read a UTF-8 file, returning null if it does not exist. Rethrows other errors. */
export async function readFileOrNull(path: string): Promise<string | null> {
	try {
		return await readFile(path, "utf8");
	} catch (err) {
		if ((err as NodeJS.ErrnoException)?.code === "ENOENT") return null;
		throw err;
	}
}

export interface JsonlLine<T> {
	lineNumber: number;
	value: T;
}

export interface JsonlParseResult<T> {
	records: JsonlLine<T>[];
	malformed: { lineNumber: number; error: string }[];
}

/**
 * Parse newline-delimited JSON leniently: blank lines are skipped, malformed
 * lines are collected (with 1-based line numbers) rather than aborting the
 * whole read. The caller decides whether to tolerate or fail on `malformed`.
 */
export function parseJsonl<T = unknown>(text: string | null): JsonlParseResult<T> {
	const records: JsonlLine<T>[] = [];
	const malformed: { lineNumber: number; error: string }[] = [];
	if (!text) return { records, malformed };
	const lines = text.split("\n");
	for (let i = 0; i < lines.length; i++) {
		const raw = lines[i].trim();
		if (raw.length === 0) continue;
		try {
			records.push({ lineNumber: i + 1, value: JSON.parse(raw) as T });
		} catch (err) {
			malformed.push({
				lineNumber: i + 1,
				error: err instanceof Error ? err.message : String(err),
			});
		}
	}
	return { records, malformed };
}
