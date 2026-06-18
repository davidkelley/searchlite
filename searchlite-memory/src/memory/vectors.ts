import { atomicWriteFile, canonicalJson, parseJsonl, readFileOrNull } from "./io.js";

/** A committed sidecar entry: one int8-quantized embedding per memory. */
export interface VectorSidecarEntry {
	id: string;
	contentHash: string;
	/** Embedder fingerprint `name@revision@quant` — detects drift. */
	model: string;
	dim: number;
	quant: "i8";
	/** base64 of the int8 (two's-complement) byte array. */
	vecB64: string;
}

function l2norm(vec: ArrayLike<number>): number {
	let sum = 0;
	for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
	return Math.sqrt(sum);
}

/**
 * Quantize a float vector to a base64 int8 string. The vector is L2-normalized
 * to unit length first (cosine similarity assumes unit vectors), then each
 * component is `round(v*127)` clamped to [-128,127], stored as int8 and
 * reinterpreted as unsigned bytes for base64. Throws on a non-finite or
 * zero-norm vector (which cannot be meaningfully normalized).
 *
 * The encoding is a self-contained round-trip with `dequantizeInt8`; it is NOT
 * coupled to the engine's on-disk format (the index stores f32).
 */
export function quantizeInt8(vec: ArrayLike<number>): string {
	const n = vec.length;
	for (let i = 0; i < n; i++) {
		if (!Number.isFinite(vec[i])) throw new Error("cannot quantize a non-finite vector");
	}
	const norm = l2norm(vec);
	if (!Number.isFinite(norm) || norm === 0) {
		throw new Error("cannot quantize a zero-norm vector");
	}
	const i8 = new Int8Array(n);
	for (let i = 0; i < n; i++) {
		const q = Math.round((vec[i] / norm) * 127);
		i8[i] = q < -128 ? -128 : q > 127 ? 127 : q;
	}
	return Buffer.from(new Uint8Array(i8.buffer, i8.byteOffset, i8.byteLength)).toString("base64");
}

/**
 * Decode a base64 int8 string back to a unit-normalized Float32Array. The bytes
 * are reinterpreted as signed int8 (using Uint8Array directly would corrupt
 * negative components), divided by 127, then re-normalized to remove
 * quantization-induced norm drift before being handed to the f32 index. Throws
 * if the decoded length does not equal `dim`.
 */
export function dequantizeInt8(vecB64: string, dim: number): Float32Array {
	// `new Uint8Array(buffer)` copies into a fresh, zero-offset ArrayBuffer so
	// the Int8Array view below is safe regardless of Buffer pooling.
	const u8 = new Uint8Array(Buffer.from(vecB64, "base64"));
	if (u8.length !== dim) {
		throw new Error(`vector length ${u8.length} does not match dim ${dim}`);
	}
	const i8 = new Int8Array(u8.buffer, 0, dim);
	const out = new Float32Array(dim);
	for (let i = 0; i < dim; i++) out[i] = i8[i] / 127;
	const norm = l2norm(out);
	if (norm > 0) {
		for (let i = 0; i < dim; i++) out[i] /= norm;
	}
	return out;
}

export interface SidecarReadResult {
	entries: VectorSidecarEntry[];
	malformed: { lineNumber: number; error: string }[];
}

export async function readSidecar(path: string): Promise<SidecarReadResult> {
	const text = await readFileOrNull(path);
	const parsed = parseJsonl<VectorSidecarEntry>(text);
	const entries: VectorSidecarEntry[] = [];
	const malformed = [...parsed.malformed];
	for (const line of parsed.records) {
		const v = line.value;
		if (
			v &&
			typeof v.id === "string" &&
			typeof v.contentHash === "string" &&
			typeof v.model === "string" &&
			typeof v.dim === "number" &&
			typeof v.vecB64 === "string"
		) {
			entries.push(v);
		} else {
			malformed.push({ lineNumber: line.lineNumber, error: "invalid sidecar entry shape" });
		}
	}
	return { entries, malformed };
}

/** Index sidecar entries by contentHash (last write wins for a given hash). */
export function indexByContentHash(entries: VectorSidecarEntry[]): Map<string, VectorSidecarEntry> {
	const map = new Map<string, VectorSidecarEntry>();
	for (const e of entries) map.set(e.contentHash, e);
	return map;
}

/** Write the sidecar atomically, sorted by (id, contentHash) for determinism. */
export async function writeSidecar(path: string, entries: VectorSidecarEntry[]): Promise<void> {
	await atomicWriteFile(path, serializeSidecar(entries));
}

/** Deterministic sidecar serialization — used for both writing and hashing. */
export function serializeSidecar(entries: VectorSidecarEntry[]): string {
	const sorted = [...entries].sort((a, b) => {
		if (a.id !== b.id) return a.id < b.id ? -1 : 1;
		if (a.contentHash !== b.contentHash) return a.contentHash < b.contentHash ? -1 : 1;
		return 0;
	});
	if (sorted.length === 0) return "";
	return `${sorted.map((e) => canonicalJson(e)).join("\n")}\n`;
}
