import { atomicWriteFile, canonicalJson, parseJsonl, readFileOrNull } from "./io.js";
import { type MemoryRecord, MemoryRecordSchema } from "./model.js";

export interface LedgerReadResult {
	/** All parsed op records (adds + tombstones), in file order. */
	records: MemoryRecord[];
	/** Malformed/invalid lines, with 1-based line numbers. */
	malformed: { lineNumber: number; error: string }[];
	/** True if any record's schemaVersion exceeds `maxSchemaVersion`. */
	hasForwardVersion: boolean;
}

/**
 * Read and validate the ledger. Malformed JSON lines and schema-invalid records
 * are collected (with line numbers) rather than aborting — the caller (doctor /
 * store.open) decides whether to tolerate or fail. Unknown fields on valid
 * records are preserved (forward-compat).
 */
export async function readLedger(
	path: string,
	maxSchemaVersion: number,
): Promise<LedgerReadResult> {
	const text = await readFileOrNull(path);
	const parsed = parseJsonl(text);
	const malformed = [...parsed.malformed];
	const records: MemoryRecord[] = [];
	let hasForwardVersion = false;
	for (const line of parsed.records) {
		// Detect a forward-version record from its RAW schemaVersion BEFORE full
		// validation: a newer writer may also have changed a required shape (e.g.
		// `op`), which would otherwise fail `safeParse` and be dropped as merely
		// "malformed" — never tripping open()'s forward-version refusal, so the
		// next rewrite would silently lose that newer record.
		const rawVersion = (line.value as { schemaVersion?: unknown })?.schemaVersion;
		if (typeof rawVersion === "number" && rawVersion > maxSchemaVersion) {
			hasForwardVersion = true;
			continue;
		}
		const result = MemoryRecordSchema.safeParse(line.value);
		if (!result.success) {
			malformed.push({ lineNumber: line.lineNumber, error: result.error.message });
			continue;
		}
		records.push(result.data as MemoryRecord);
	}
	return { records, malformed, hasForwardVersion };
}

/** Pick the winning op for an id: highest opTs; on a tie, `forget` wins (deletion is safe). */
function winningOp(a: MemoryRecord, b: MemoryRecord): MemoryRecord {
	if (a.opTs !== b.opTs) return a.opTs > b.opTs ? a : b;
	if (a.op !== b.op) return a.op === "forget" ? a : b;
	return a; // identical ordering key + op → stable
}

export interface MaterializeResult {
	/** Live `add` records (current state), deduped by contentHash. */
	live: MemoryRecord[];
	/** Ids whose winning op is a tombstone. */
	tombstoned: string[];
}

/**
 * Reduce the append-only op log to current state. Ordering is by (opTs, id) —
 * NEVER file line-order — so a union-merge or canonical resort can't reorder a
 * `forget` behind its `add` and resurrect a deleted memory. A record is live if
 * its winning op is an `add` that is not invalidated (invalidAt in the past) or
 * superseded. Live records are then deduped by contentHash, keeping the
 * lowest-ULID winner (deterministic, clock-independent).
 */
export function materialize(records: MemoryRecord[], now: Date = new Date()): MaterializeResult {
	const winners = new Map<string, MemoryRecord>();
	for (const rec of records) {
		const prev = winners.get(rec.id);
		winners.set(rec.id, prev ? winningOp(prev, rec) : rec);
	}

	const nowMs = now.getTime();
	const tombstoned: string[] = [];
	const liveById: MemoryRecord[] = [];
	for (const rec of winners.values()) {
		if (rec.op === "forget") {
			tombstoned.push(rec.id);
			continue;
		}
		if (rec.supersededBy) continue;
		// Parse timestamps for comparison — string compare is unreliable across
		// ISO format/timezone/precision differences (e.g. `Z` vs `+00:00`).
		if (rec.invalidAt != null) {
			const invalidMs = Date.parse(rec.invalidAt);
			if (Number.isFinite(invalidMs) && invalidMs <= nowMs) continue;
		}
		liveById.push(rec);
	}

	// Dedup by contentHash: keep the lowest ULID (ids are sortable strings).
	const byHash = new Map<string, MemoryRecord>();
	const noHash: MemoryRecord[] = [];
	for (const rec of liveById) {
		if (!rec.contentHash) {
			noHash.push(rec);
			continue;
		}
		const prev = byHash.get(rec.contentHash);
		if (!prev || rec.id < prev.id) byHash.set(rec.contentHash, rec);
	}
	const live = [...byHash.values(), ...noHash].sort((a, b) =>
		a.id < b.id ? -1 : a.id > b.id ? 1 : 0,
	);
	return { live, tombstoned };
}

/**
 * Write the full op log atomically, sorted by (id, opTs, op) for byte-identical
 * determinism regardless of in-memory insertion order. The file stays
 * append-only in spirit (records are only ever added — adds + tombstones —
 * never edited in place); rewriting sorted keeps diffs minimal and pairs with
 * the `merge=union` gitattribute for conflict-free concurrent appends.
 */
export async function writeLedger(path: string, records: MemoryRecord[]): Promise<void> {
	await atomicWriteFile(path, serializeLedger(records));
}

/** Deterministic ledger serialization (sorted canonical lines). Used for both
 * writing and hashing (the gate's ledgerHash). */
export function serializeLedger(records: MemoryRecord[]): string {
	const sorted = [...records].sort(compareRecords);
	if (sorted.length === 0) return "";
	return `${sorted.map((r) => canonicalJson(r)).join("\n")}\n`;
}

function compareRecords(a: MemoryRecord, b: MemoryRecord): number {
	if (a.id !== b.id) return a.id < b.id ? -1 : 1;
	if (a.opTs !== b.opTs) return a.opTs < b.opTs ? -1 : 1;
	if (a.op !== b.op) return a.op < b.op ? -1 : 1;
	return 0;
}
