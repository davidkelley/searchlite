import { createHash } from "node:crypto";
import type { MemoryRecord } from "./model.js";
import { SCHEMA_VERSION } from "./model.js";

export interface IndexSchemaOptions {
	/** Vector dim, or null for a full-text-only schema (no vector field). */
	vectorDim: number | null;
	metric?: "Cosine" | "L2";
}

/**
 * Build the searchlite index schema (raw JSON Schema — the "already JSON
 * Schema" passthrough branch, NOT the Zod path, so no construction-time
 * response schema forces `returnStored:true`). `_id` is the implicit doc-id
 * field and must not be declared. `text` is explicitly stored so highlight
 * snippets are non-null. Timestamps are epoch-second fast fields for range
 * filters/sorts. The vector field is omitted entirely in FTS-only mode.
 */
export function buildIndexSchema(opts: IndexSchemaOptions): Record<string, unknown> {
	const properties: Record<string, unknown> = {
		text: { type: "string", "searchlite:stored": true },
		type: { type: "string", "searchlite:kind": "keyword" },
		namespace: { type: "string", "searchlite:kind": "keyword" },
		// Multi-valued keyword fields are declared scalar; documents may supply an
		// array of values (searchlite indexes each, and KeywordEq/In match any).
		tags: { type: "string", "searchlite:kind": "keyword" },
		entities: { type: "string", "searchlite:kind": "keyword" },
		importance: { type: "number", "searchlite:fast": true, "searchlite:stored": true },
		createdAtTs: { type: "integer", "searchlite:fast": true, "searchlite:stored": true },
		validFromTs: { type: "integer", "searchlite:fast": true, "searchlite:stored": true },
		invalidAtTs: {
			type: ["integer", "null"],
			"searchlite:fast": true,
			"searchlite:stored": true,
		},
	};
	if (opts.vectorDim != null) {
		properties.embedding = {
			type: "array",
			items: { type: "number" },
			"searchlite:vector": { dim: opts.vectorDim, metric: opts.metric ?? "Cosine" },
		};
	}
	return { type: "object", properties };
}

/**
 * A stable fingerprint of the index shape. A change (new schema version, vector
 * dim, or metric) forces a rebuild because the existing binary index is
 * incompatible.
 */
export function schemaFingerprint(opts: IndexSchemaOptions): string {
	const canonical = JSON.stringify({
		schemaVersion: SCHEMA_VERSION,
		vectorDim: opts.vectorDim ?? null,
		metric: opts.vectorDim != null ? (opts.metric ?? "Cosine") : null,
	});
	return createHash("sha256").update(canonical).digest("hex");
}

/** Convert an RFC3339 timestamp to epoch seconds, or null. */
export function epochSeconds(iso: string | null | undefined): number | null {
	if (!iso) return null;
	const ms = Date.parse(iso);
	return Number.isFinite(ms) ? Math.floor(ms / 1000) : null;
}

/**
 * Build the searchlite document for a live `add` record. `embedding` is included
 * only when a (dequantized, unit) vector is available; FTS-only records omit it.
 */
export function recordToDoc(
	record: MemoryRecord,
	embedding: Float32Array | number[] | null,
): Record<string, unknown> {
	const doc: Record<string, unknown> = {
		_id: record.id,
		text: record.text ?? "",
		type: record.type ?? "semantic",
		namespace: record.namespace ?? "default",
		tags: record.tags ?? [],
		entities: record.entities ?? [],
		importance: typeof record.importance === "number" ? record.importance : 0.5,
		createdAtTs: epochSeconds(record.createdAt) ?? 0,
		validFromTs: epochSeconds(record.validFrom) ?? epochSeconds(record.createdAt) ?? 0,
		invalidAtTs: epochSeconds(record.invalidAt),
	};
	if (embedding) {
		doc.embedding = Array.from(embedding);
	}
	return doc;
}
