import { createHash } from "node:crypto";
import { ulid } from "ulid";
import { z } from "zod";
import { canonicalJson } from "./io.js";

export const SCHEMA_VERSION = 1;

export const MEMORY_TYPES = ["semantic", "episodic", "procedural"] as const;
export type MemoryType = (typeof MEMORY_TYPES)[number];

/**
 * A committed ledger record. `op:"add"` carries the full memory; `op:"forget"`
 * is a tombstone (only id/opTs are meaningful). Unknown fields are preserved on
 * read for forward-compatibility with future schema versions.
 */
export interface MemoryRecord {
	id: string;
	schemaVersion: number;
	op: "add" | "forget";
	/** RFC3339 monotonic ordering key — materialization uses (opTs, id), never file order. */
	opTs: string;
	text?: string;
	type?: MemoryType;
	namespace?: string;
	tags?: string[];
	entities?: string[];
	importance?: number;
	createdAt?: string;
	validFrom?: string;
	invalidAt?: string | null;
	supersededBy?: string | null;
	contentHash?: string;
	source?: string;
	[k: string]: unknown;
}

const MemoryTypeSchema = z.enum(MEMORY_TYPES);

// `.passthrough()` keeps unknown fields so a newer writer's extra fields survive
// a round-trip through an older reader (forward-compat).
export const MemoryRecordSchema = z
	.object({
		id: z.string().min(1),
		schemaVersion: z.number().int().positive(),
		op: z.enum(["add", "forget"]),
		opTs: z.string().min(1),
		text: z.string().optional(),
		type: MemoryTypeSchema.optional(),
		namespace: z.string().optional(),
		tags: z.array(z.string()).optional(),
		entities: z.array(z.string()).optional(),
		importance: z.number().optional(),
		createdAt: z.string().optional(),
		validFrom: z.string().optional(),
		invalidAt: z.string().nullable().optional(),
		supersededBy: z.string().nullable().optional(),
		contentHash: z.string().optional(),
		source: z.string().optional(),
	})
	.passthrough();

/** Clamp importance into [0,1]; non-finite or absent → default 0.5. */
export function clampImportance(value: number | undefined): number {
	if (value === undefined || !Number.isFinite(value)) return 0.5;
	return Math.min(1, Math.max(0, value));
}

/** Sort + de-duplicate a string list (so order/dupes don't affect identity). */
function normalizeList(list: string[] | undefined): string[] {
	if (!list || list.length === 0) return [];
	return Array.from(new Set(list)).sort();
}

/**
 * Content-identity hash used for dedup and as the vector-cache/sidecar key.
 * Covers the *semantic identity* of a memory — text, type, namespace, and the
 * (sorted, de-duplicated) tags/entities — and deliberately excludes volatile or
 * mutable metadata (id, op, timestamps, importance, supersededBy). Two
 * `remember` calls with identical content therefore dedupe even if importance
 * or tags-order differ.
 */
export function contentHashOf(input: {
	text: string;
	type: MemoryType;
	namespace: string;
	tags?: string[];
	entities?: string[];
}): string {
	const canonical = canonicalJson({
		text: input.text,
		type: input.type,
		namespace: input.namespace,
		tags: normalizeList(input.tags),
		entities: normalizeList(input.entities),
	});
	return `sha256:${createHash("sha256").update(canonical).digest("hex")}`;
}

export interface RememberInput {
	text: string;
	type?: MemoryType;
	namespace?: string;
	tags?: string[];
	entities?: string[];
	importance?: number;
	validFrom?: string;
}

/** Build a fresh `add` record from user input. `now` is injected for testability. */
export function makeAddRecord(input: RememberInput, now: Date = new Date()): MemoryRecord {
	const ts = now.toISOString();
	const type: MemoryType = input.type ?? "semantic";
	const namespace = input.namespace ?? "default";
	const tags = normalizeList(input.tags);
	const entities = normalizeList(input.entities);
	return {
		id: ulid(now.getTime()),
		schemaVersion: SCHEMA_VERSION,
		op: "add",
		opTs: ts,
		text: input.text,
		type,
		namespace,
		tags,
		entities,
		importance: clampImportance(input.importance),
		createdAt: ts,
		validFrom: input.validFrom ?? ts,
		invalidAt: null,
		supersededBy: null,
		contentHash: contentHashOf({ text: input.text, type, namespace, tags, entities }),
		source: "agent",
	};
}

/** Build a tombstone record for `id`. */
export function makeForgetRecord(id: string, now: Date = new Date()): MemoryRecord {
	return {
		id,
		schemaVersion: SCHEMA_VERSION,
		op: "forget",
		opTs: now.toISOString(),
	};
}

/**
 * The enriched text fed to the embedder: the memory text plus its tags and
 * entities, so semantically-relevant keywords influence the embedding (A-MEM
 * style). NOT an indexed searchlite field — BM25 targets `text` only.
 */
export function embedTextOf(record: Pick<MemoryRecord, "text" | "tags" | "entities">): string {
	const parts = [record.text ?? ""];
	if (record.tags && record.tags.length > 0) parts.push(`tags: ${record.tags.join(", ")}`);
	if (record.entities && record.entities.length > 0) {
		parts.push(`entities: ${record.entities.join(", ")}`);
	}
	return parts.join("\n");
}
