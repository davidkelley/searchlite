import { createHash } from "node:crypto";
import { mkdir, readdir, rm, stat, writeFile } from "node:fs/promises";
import { join, relative } from "node:path";
import type { MemoryConfig } from "../config.js";
import { cacheKey, VectorCache } from "../embed/cache.js";
import { createEmbedder, type Embedder } from "../embed/embedder.js";
import { withLock } from "../lock.js";
import { EmbeddedIndex, type SearchRequest } from "../searchlite.js";
import { atomicWriteFile, readFileOrNull } from "./io.js";
import { materialize, readLedger, serializeLedger, writeLedger } from "./ledger.js";
import {
	embedTextOf,
	type MemoryRecord,
	type MemoryType,
	makeAddRecord,
	makeForgetRecord,
	type RememberInput,
	SCHEMA_VERSION,
} from "./model.js";
import { rescore, rrfFuse } from "./retrieval.js";
import {
	buildIndexSchema,
	type IndexSchemaOptions,
	recordToDoc,
	schemaFingerprint,
} from "./schema.js";
import {
	dequantizeInt8,
	indexByContentHash,
	quantizeInt8,
	readSidecar,
	serializeSidecar,
	type VectorSidecarEntry,
	writeSidecar,
} from "./vectors.js";

const SNIPPET_MAX = 200;

interface Gate {
	ledgerHash: string;
	sidecarHash: string;
	schemaFingerprint: string;
	vectorFingerprint: string;
	indexGen: number;
}

interface AccessStat {
	lastAccessed: string;
	accessCount: number;
}

export interface RememberResult {
	id: string;
	deduped: boolean;
}

export interface RecallOptions {
	limit?: number;
	namespace?: string;
	type?: MemoryType | MemoryType[];
	tags?: string[];
	minImportance?: number;
}

export interface RecallHit {
	id: string;
	snippet: string;
	type: MemoryType;
	namespace: string;
	tags: string[];
	score: number;
	createdAt: string | null;
}

export interface DoctorReport {
	ok: boolean;
	checks: { name: string; ok: boolean; detail: string }[];
}

function sha256(s: string): string {
	return createHash("sha256").update(s).digest("hex");
}

function truncate(text: string, max = SNIPPET_MAX): string {
	const t = text.replace(/\s+/g, " ").trim();
	return t.length <= max ? t : `${t.slice(0, max - 1)}…`;
}

function normalizeVector(v: Float32Array): number[] {
	let norm = 0;
	for (const x of v) norm += x * x;
	norm = Math.sqrt(norm);
	const out = new Array<number>(v.length);
	for (let i = 0; i < v.length; i++) out[i] = norm > 0 ? v[i] / norm : 0;
	return out;
}

/**
 * Orchestrates the committed ledger + vector sidecar, the rebuildable searchlite
 * index (generation dirs + an atomic CURRENT pointer), the embedder, the
 * vector cache, and access stats. Every mutation and rebuild runs under a
 * cross-process lock; the slow embed runs BEFORE the lock.
 */
export class MemoryStore {
	#config: MemoryConfig;
	#embedder: Embedder;
	#cache: VectorCache;
	#index: InstanceType<typeof EmbeddedIndex> | null = null;
	#indexGen = 0;
	#openedGate: Gate | null = null;

	#allRecords: MemoryRecord[] = [];
	#live = new Map<string, MemoryRecord>();
	#sidecar = new Map<string, VectorSidecarEntry>(); // contentHash -> entry
	#access = new Map<string, AccessStat>();
	#malformed: { lineNumber: number; error: string }[] = [];
	#missingVectorIds: string[] = [];
	/** Per-file mtime stamp of the committed ledger+sidecar at last load — detects
	 * external (git pull / branch switch) changes a long-running server would
	 * otherwise miss. Per-file (not a single max) so a change to either file —
	 * including the older one, or a deletion — is always seen. */
	#sourceStamp = "";

	private constructor(config: MemoryConfig, embedder: Embedder) {
		this.#config = config;
		this.#embedder = embedder;
		this.#cache = new VectorCache(config.paths.cache);
	}

	/** Open the store. `embedder` overrides config-based creation (tests / custom providers). */
	static async open(config: MemoryConfig, embedder?: Embedder): Promise<MemoryStore> {
		const resolved = embedder ?? (await createEmbedder(config.embedder));
		const store = new MemoryStore(config, resolved);
		await store.#load();
		return store;
	}

	get vectorsEnabled(): boolean {
		return this.#embedder.available;
	}

	#schemaOpts(): IndexSchemaOptions {
		return { vectorDim: this.vectorsEnabled ? this.#embedder.dim : null };
	}

	#computeGate(indexGen: number): Gate {
		return {
			ledgerHash: sha256(serializeLedger(this.#allRecords)),
			sidecarHash: sha256(serializeSidecar([...this.#sidecar.values()])),
			schemaFingerprint: schemaFingerprint(this.#schemaOpts()),
			vectorFingerprint: this.vectorsEnabled ? this.#embedder.id : "none",
			indexGen,
		};
	}

	async #readGate(): Promise<Gate | null> {
		const raw = await readFileOrNull(this.#config.paths.gate);
		if (!raw) return null;
		try {
			return JSON.parse(raw) as Gate;
		} catch {
			return null;
		}
	}

	async #writeGate(gate: Gate): Promise<void> {
		await atomicWriteFile(this.#config.paths.gate, JSON.stringify(gate));
	}

	#genDir(gen: number): string {
		return join(this.#config.paths.indexDir, `gen-${gen}`);
	}

	async #readCurrentGen(): Promise<number | null> {
		const raw = await readFileOrNull(this.#config.paths.currentPointer);
		if (!raw) return null;
		const n = Number.parseInt(raw.trim(), 10);
		return Number.isFinite(n) ? n : null;
	}

	async #load(): Promise<void> {
		await mkdir(this.#config.paths.root, { recursive: true });
		await mkdir(this.#config.paths.indexDir, { recursive: true });
		// Ensure the lock target exists (proper-lockfile with realpath:false).
		if ((await readFileOrNull(this.#config.paths.lock)) === null) {
			await writeFile(this.#config.paths.lock, "");
		}

		await this.#cache.load();
		await this.#reloadState();
		await this.#openOrRebuild();
	}

	/**
	 * Re-read ledger + sidecar + access from disk and re-materialize live state.
	 * Called at open and at the start of every locked mutation/rebuild so a
	 * concurrent writer's committed records are MERGED, never lost (the ledger on
	 * disk is authoritative; the in-memory snapshot may be stale).
	 */
	async #reloadState(): Promise<void> {
		const ledger = await readLedger(this.#config.paths.ledger, SCHEMA_VERSION);
		if (ledger.hasForwardVersion) {
			throw new Error(
				"the memory ledger contains records from a newer schemaVersion than this " +
					"searchlite-memory understands; upgrade the package (refusing to rebuild and " +
					"risk dropping unrecognized records)",
			);
		}
		this.#malformed = ledger.malformed;
		this.#allRecords = ledger.records;
		this.#live = new Map(materialize(ledger.records).live.map((r) => [r.id, r]));
		const sidecar = await readSidecar(this.#config.paths.sidecar);
		this.#sidecar = indexByContentHash(sidecar.entries);
		this.#access = await this.#readAccess();
		this.#sourceStamp = await this.#currentSourceStamp();
	}

	/** Per-file mtime stamp of the committed ledger + sidecar ("-" if absent). */
	async #currentSourceStamp(): Promise<string> {
		const parts: string[] = [];
		for (const p of [this.#config.paths.ledger, this.#config.paths.sidecar]) {
			try {
				parts.push(String((await stat(p)).mtimeMs));
			} catch {
				parts.push("-");
			}
		}
		return parts.join("|");
	}

	#isStale(onDisk: Gate | null, currentGen: number | null, genDirExists: boolean): boolean {
		if (currentGen == null || !genDirExists || onDisk == null) return true;
		if (onDisk.indexGen !== currentGen) return true;
		const desired = this.#computeGate(currentGen);
		return (
			onDisk.ledgerHash !== desired.ledgerHash ||
			onDisk.sidecarHash !== desired.sidecarHash ||
			onDisk.schemaFingerprint !== desired.schemaFingerprint ||
			onDisk.vectorFingerprint !== desired.vectorFingerprint
		);
	}

	async #openOrRebuild(): Promise<void> {
		const onDisk = await this.#readGate();
		const currentGen = await this.#readCurrentGen();
		const genDirExists = currentGen != null && (await this.#dirExists(this.#genDir(currentGen)));
		if (this.#isStale(onDisk, currentGen, genDirExists)) {
			await withLock(this.#config.paths.lock, this.#lockOpts(), () => this.#rebuildLocked());
		} else if (currentGen != null && onDisk != null) {
			await this.#openIndex(currentGen, onDisk);
		}
	}

	/** (Re)open the index handle on a generation dir. Swaps the new handle in
	 * BEFORE closing the old one so a concurrent reader never sees a closed
	 * handle. */
	async #openIndex(gen: number, gate: Gate): Promise<void> {
		const old = this.#index;
		this.#index = new EmbeddedIndex(this.#genDir(gen), {
			schema: buildIndexSchema(this.#schemaOpts()),
		});
		this.#indexGen = gen;
		this.#openedGate = gate;
		if (old) await old.close();
	}

	async #dirExists(path: string): Promise<boolean> {
		try {
			await readdir(path);
			return true;
		} catch {
			return false;
		}
	}

	/**
	 * Rebuild into a fresh generation dir and flip CURRENT. MUST hold the lock.
	 * Double-checked: re-reads on-disk state under the lock and, if another
	 * process already produced a matching index, opens it instead of rebuilding
	 * (and allocates the next generation from the fresh CURRENT, never a stale one).
	 */
	async #rebuildLocked(): Promise<void> {
		await this.#reloadState();
		const onDisk = await this.#readGate();
		const currentGen = await this.#readCurrentGen();
		const genDirExists = currentGen != null && (await this.#dirExists(this.#genDir(currentGen)));
		if (!this.#isStale(onDisk, currentGen, genDirExists) && currentGen != null && onDisk != null) {
			await this.#openIndex(currentGen, onDisk);
			return;
		}

		const newGen = (currentGen ?? 0) + 1;
		const dir = this.#genDir(newGen);
		await rm(dir, { recursive: true, force: true });
		await mkdir(dir, { recursive: true });

		const idx = new EmbeddedIndex(dir, { schema: buildIndexSchema(this.#schemaOpts()) });
		this.#missingVectorIds = [];
		const live = [...this.#live.values()];
		const BATCH = 256;
		for (let i = 0; i < live.length; i += BATCH) {
			const docs = live.slice(i, i + BATCH).map((rec) => this.#docFor(rec));
			if (docs.length > 0) await idx.addMany(docs);
			await new Promise((r) => setImmediate(r)); // yield the event loop
		}
		await idx.commit();
		const old = this.#index;
		this.#index = idx;
		this.#indexGen = newGen;
		if (old) await old.close();

		// Atomic pointer flip, then best-effort GC of stale generations.
		await atomicWriteFile(this.#config.paths.currentPointer, String(newGen));
		await this.#gcGenerations(newGen);

		const gate = this.#computeGate(newGen);
		await this.#writeGate(gate);
		this.#openedGate = gate;
	}

	#docFor(rec: MemoryRecord): Record<string, unknown> {
		let embedding: Float32Array | null = null;
		if (this.vectorsEnabled && rec.contentHash) {
			const entry = this.#sidecar.get(rec.contentHash);
			// Only reuse a committed vector when it was produced by the CURRENT
			// embedder + dim. A model/revision/quant change (same dim) would
			// otherwise index semantically-stale vectors that are silently wrong
			// against fresh query embeddings. Drifted/mismatched → index FTS-only
			// for this record; `doctor`/`rebuild --reembed` repairs it.
			if (entry && entry.model === this.#embedder.id && entry.dim === this.#embedder.dim) {
				embedding = dequantizeInt8(entry.vecB64, entry.dim);
			} else {
				this.#missingVectorIds.push(rec.id);
			}
		}
		return recordToDoc(rec, embedding);
	}

	async #gcGenerations(keep: number): Promise<void> {
		let names: string[];
		try {
			names = await readdir(this.#config.paths.indexDir);
		} catch {
			return;
		}
		for (const name of names) {
			if (name.startsWith("gen-") && name !== `gen-${keep}`) {
				await rm(join(this.#config.paths.indexDir, name), { recursive: true, force: true }).catch(
					() => {},
				);
			}
		}
	}

	#lockOpts() {
		return {
			staleMs: this.#config.lockStaleMs,
			retries: this.#config.lockRetries,
			disabled: this.#config.lockDisabled,
		};
	}

	async #readAccess(): Promise<Map<string, AccessStat>> {
		const raw = await readFileOrNull(this.#config.paths.access);
		if (!raw) return new Map();
		try {
			const obj = JSON.parse(raw) as Record<string, AccessStat>;
			return new Map(Object.entries(obj));
		} catch {
			return new Map();
		}
	}

	async #writeAccess(): Promise<void> {
		// access.json is updated lock-free on read paths (recall/get). Merge the
		// latest on-disk stats before writing so a concurrent reader's increments
		// aren't clobbered (best-effort, not a strict lock): newest lastAccessed
		// wins, and accessCount takes the max of the two.
		const merged = await this.#readAccess();
		for (const [id, stat] of this.#access) {
			const disk = merged.get(id);
			merged.set(id, {
				lastAccessed:
					disk && disk.lastAccessed > stat.lastAccessed ? disk.lastAccessed : stat.lastAccessed,
				accessCount: Math.max(stat.accessCount, disk?.accessCount ?? 0),
			});
		}
		this.#access = merged;
		const obj: Record<string, AccessStat> = {};
		for (const [id, stat] of merged) obj[id] = stat;
		await atomicWriteFile(this.#config.paths.access, JSON.stringify(obj));
	}

	// --- Public operations ---

	async remember(input: RememberInput): Promise<RememberResult> {
		if (!input.text || input.text.trim().length === 0) {
			throw new Error("remember: text is required");
		}
		const record = makeAddRecord(input);
		const hash = record.contentHash as string;

		// Embed OUTSIDE the lock (slow, pure). Cache by fingerprint+text. We embed
		// unconditionally (no pre-lock dedup): the only authoritative dedup happens
		// under the lock against freshly-reloaded state — a stale in-memory match
		// could otherwise return an id another process already forgot. A duplicate
		// discovered under the lock just discards the (cheap, cached) vector.
		let vecB64: string | null = null;
		if (this.vectorsEnabled) {
			const key = cacheKey(this.#embedder.id, embedTextOf(record));
			const cached = this.#cache.get(key);
			if (cached) {
				vecB64 = cached;
			} else {
				const [vec] = await this.#embedder.embed([embedTextOf(record)]);
				vecB64 = quantizeInt8(vec);
				this.#cache.set(key, vecB64);
			}
		}

		return withLock(this.#config.paths.lock, this.#lockOpts(), async () => {
			await this.#reloadState(); // merge any concurrent writes before mutating
			await this.#ensureIndexFresh();

			const supersedeId =
				input.supersedes && this.#live.has(input.supersedes) ? input.supersedes : null;
			const existing = this.#findLiveByHash(hash);

			// Dedup: identical content already present. Still honor `supersedes`.
			if (existing) {
				if (supersedeId && supersedeId !== existing) {
					const sidecarMutated = this.#tombstone(supersedeId);
					if (sidecarMutated) {
						await writeSidecar(this.#config.paths.sidecar, [...this.#sidecar.values()]);
					}
					await writeLedger(this.#config.paths.ledger, this.#allRecords);
					await this.#index?.deleteMany([supersedeId]);
					await this.#bumpGate();
				}
				return { id: existing, deduped: true };
			}

			this.#allRecords.push(record);
			this.#live.set(record.id, record);
			let sidecarMutated = false;
			if (vecB64) {
				this.#sidecar.set(hash, {
					id: record.id,
					contentHash: hash,
					model: this.#embedder.id,
					dim: this.#embedder.dim,
					quant: "i8",
					vecB64,
				});
				sidecarMutated = true;
			}
			if (supersedeId && supersedeId !== record.id) {
				if (this.#tombstone(supersedeId)) sidecarMutated = true;
			}

			if (sidecarMutated) {
				await writeSidecar(this.#config.paths.sidecar, [...this.#sidecar.values()]);
			}
			await writeLedger(this.#config.paths.ledger, this.#allRecords);

			const embedding = vecB64 ? dequantizeInt8(vecB64, this.#embedder.dim) : null;
			await this.#index?.add(recordToDoc(record, embedding));
			if (supersedeId && supersedeId !== record.id) {
				await this.#index?.deleteMany([supersedeId]); // deletes + commits
			} else {
				await this.#index?.commit();
			}

			await this.#bumpGate();
			await this.#cache.flush();
			return { id: record.id, deduped: false };
		});
	}

	async forget(id: string): Promise<{ id: string; forgotten: boolean }> {
		return withLock(this.#config.paths.lock, this.#lockOpts(), async () => {
			await this.#reloadState(); // merge concurrent writes before mutating
			await this.#ensureIndexFresh();
			// Idempotent: nothing live to forget → no ledger churn / gate bump.
			if (!this.#live.has(id)) return { id, forgotten: true };

			const sidecarMutated = this.#tombstone(id);
			if (sidecarMutated) {
				await writeSidecar(this.#config.paths.sidecar, [...this.#sidecar.values()]);
			}
			await writeLedger(this.#config.paths.ledger, this.#allRecords);
			await this.#index?.deleteMany([id]);
			await this.#bumpGate();
			return { id, forgotten: true };
		});
	}

	/**
	 * Append a tombstone for `id` and drop it from live state, removing its
	 * sidecar entry when no other live record shares the contentHash. Returns
	 * whether the sidecar was mutated (so the caller persists it — regardless of
	 * vectorsEnabled, since a committed sidecar may exist even in FTS-only mode).
	 */
	#tombstone(id: string): boolean {
		const old = this.#live.get(id);
		this.#allRecords.push(makeForgetRecord(id));
		this.#live.delete(id);
		if (old?.contentHash && !this.#hashStillLive(old.contentHash)) {
			return this.#sidecar.delete(old.contentHash);
		}
		return false;
	}

	#hashStillLive(hash: string): boolean {
		for (const r of this.#live.values()) if (r.contentHash === hash) return true;
		return false;
	}

	/**
	 * Caller holds the mutation lock and has just `#reloadState()`d. Make the
	 * index reflect that fresh state: REBUILD if stale (e.g. an external git-pull
	 * changed the committed ledger but not the gitignored gate, so the desired
	 * gate now mismatches), otherwise reopen if another process advanced the gate.
	 * Calls `#rebuildLocked()` directly (not via `withLock`) since the lock is held.
	 */
	async #ensureIndexFresh(): Promise<void> {
		const onDisk = await this.#readGate();
		const currentGen = await this.#readCurrentGen();
		const genDirExists = currentGen != null && (await this.#dirExists(this.#genDir(currentGen)));
		if (this.#isStale(onDisk, currentGen, genDirExists)) {
			await this.#rebuildLocked();
		} else if (
			currentGen != null &&
			onDisk != null &&
			(!this.#openedGate || !this.#sameGate(this.#openedGate, onDisk))
		) {
			await this.#openIndex(currentGen, onDisk);
		}
	}

	#sameGate(a: Gate, b: Gate): boolean {
		return (
			a.indexGen === b.indexGen &&
			a.ledgerHash === b.ledgerHash &&
			a.sidecarHash === b.sidecarHash &&
			a.schemaFingerprint === b.schemaFingerprint &&
			a.vectorFingerprint === b.vectorFingerprint
		);
	}

	async get(id: string): Promise<MemoryRecord | null> {
		await this.#refreshIfStale();
		const rec = this.#live.get(id);
		if (!rec) return null;
		this.#bumpAccess([id]);
		await this.#writeAccess();
		return rec;
	}

	async recall(query: string, opts: RecallOptions = {}): Promise<{ memories: RecallHit[] }> {
		await this.#refreshIfStale();
		if (!query || query.trim().length === 0) return { memories: [] };
		const limit = opts.limit ?? this.#config.recallLimit;
		const pool = this.#config.poolSize;
		const filter = this.#buildFilter(opts);

		// BM25 call (snippet via highlightField; no per-call schema → no forced
		// returnStored). Metadata comes from the in-memory live records.
		const bmReq: SearchRequest = {
			query,
			fields: ["text"],
			limit: pool,
			highlightField: "text",
			...(filter ? { filter } : {}),
		};
		const bm = await this.#index?.search(bmReq);
		const bmHits = bm?.hits ?? [];
		const bmIds = bmHits.map((h) => h.docId);
		const snippets = new Map<string, string>();
		for (const h of bmHits) {
			if (typeof h.snippet === "string" && h.snippet.length > 0) {
				snippets.set(h.docId, truncate(h.snippet));
			}
		}

		// Vector call (pure vector via query node; snake_case inner keys).
		let vecIds: string[] = [];
		if (this.vectorsEnabled) {
			const [qvec] = await this.#embedder.embed([query]);
			const vector = normalizeVector(qvec);
			const vecReq: SearchRequest = {
				query: { type: "vector", field: "embedding", vector, k: pool, alpha: 0.0 },
				limit: pool,
				...(filter ? { filter } : {}),
			};
			const vr = await this.#index?.search(vecReq);
			vecIds = (vr?.hits ?? []).map((h) => h.docId);
		}

		const fused = rrfFuse([bmIds, vecIds], this.#config.rrfK);
		const now = Date.now();
		const candidates = [...fused.entries()]
			.map(([id, rrf]) => {
				const rec = this.#live.get(id);
				if (!rec) return null;
				const ref = this.#access.get(id)?.lastAccessed ?? rec.createdAt;
				const refMs = ref ? Date.parse(ref) : now;
				const ageHours = Math.max(0, (now - (Number.isFinite(refMs) ? refMs : now)) / 3_600_000);
				return {
					id,
					rrf,
					importance: typeof rec.importance === "number" ? rec.importance : 0.5,
					ageHours,
					accessCount: this.#access.get(id)?.accessCount ?? 0,
				};
			})
			.filter((c): c is NonNullable<typeof c> => c !== null);

		const ranked = rescore(candidates, {
			weights: this.#config.weights,
			halfLifeHours: this.#config.halfLifeHours,
			accessCap: this.#config.accessCap,
		}).slice(0, limit);

		const memories: RecallHit[] = ranked.map(({ id, score }) => {
			const rec = this.#live.get(id) as MemoryRecord;
			return {
				id,
				snippet: snippets.get(id) ?? truncate(rec.text ?? ""),
				type: (rec.type ?? "semantic") as MemoryType,
				namespace: rec.namespace ?? "default",
				tags: rec.tags ?? [],
				score,
				createdAt: rec.createdAt ?? null,
			};
		});

		if (memories.length > 0) {
			this.#bumpAccess(memories.map((m) => m.id));
			await this.#writeAccess();
		}
		return { memories };
	}

	/** Force a rebuild. `reembed` re-embeds live records missing a sidecar vector (outside the lock). */
	async rebuild(reembed = false): Promise<void> {
		if (reembed && this.vectorsEnabled) await this.#reembedMissing();
		await withLock(this.#config.paths.lock, this.#lockOpts(), () => this.#rebuildLocked());
	}

	#needsEmbed(rec: MemoryRecord): boolean {
		if (!rec.contentHash) return false;
		const entry = this.#sidecar.get(rec.contentHash);
		// Missing OR produced by a different model/dim (drift) → needs (re)embed.
		return !entry || entry.model !== this.#embedder.id || entry.dim !== this.#embedder.dim;
	}

	async #reembedMissing(): Promise<void> {
		const missing = [...this.#live.values()].filter((r) => this.#needsEmbed(r));
		if (missing.length === 0) return;
		// Embed OUTSIDE the lock, then take the lock only to persist the sidecar.
		const fresh: VectorSidecarEntry[] = [];
		const BATCH = 32;
		for (let i = 0; i < missing.length; i += BATCH) {
			const batch = missing.slice(i, i + BATCH);
			const vecs = await this.#embedder.embed(batch.map((r) => embedTextOf(r)));
			for (let j = 0; j < batch.length; j++) {
				fresh.push({
					id: batch[j].id,
					contentHash: batch[j].contentHash as string,
					model: this.#embedder.id,
					dim: this.#embedder.dim,
					quant: "i8",
					vecB64: quantizeInt8(vecs[j]),
				});
			}
		}
		await withLock(this.#config.paths.lock, this.#lockOpts(), async () => {
			await this.#reloadState(); // merge concurrent sidecar writes before persisting
			const liveHashes = new Set(
				[...this.#live.values()].map((r) => r.contentHash).filter((h): h is string => !!h),
			);
			let changed = false;
			for (const e of fresh) {
				// Skip vectors for records forgotten during embedding, or already
				// (re)embedded for the current model by a concurrent writer.
				const cur = this.#sidecar.get(e.contentHash);
				const stillNeeded = !cur || cur.model !== e.model || cur.dim !== e.dim;
				if (liveHashes.has(e.contentHash) && stillNeeded) {
					this.#sidecar.set(e.contentHash, e);
					changed = true;
				}
			}
			if (changed) await writeSidecar(this.#config.paths.sidecar, [...this.#sidecar.values()]);
		});
	}

	async doctor(): Promise<DoctorReport> {
		const checks: { name: string; ok: boolean; detail: string }[] = [];
		const add = (name: string, ok: boolean, detail: string) => checks.push({ name, ok, detail });

		const tombstones = this.#allRecords.filter((r) => r.op === "forget").length;
		add(
			"ledger",
			true,
			`${this.#allRecords.length} ops, ${this.#live.size} live, ${tombstones} tombstones`,
		);
		add("malformed lines", this.#malformed.length === 0, `${this.#malformed.length} malformed`);

		// A configured local/api embedder that silently fell back to FTS-only is a problem.
		const embedderOk = this.vectorsEnabled || this.#config.embedder.provider === "none";
		add(
			"embedder",
			embedderOk,
			this.vectorsEnabled
				? `available (${this.#embedder.id}, dim ${this.#embedder.dim})`
				: this.#config.embedder.provider === "none"
					? "full-text-only (embedder=none)"
					: `configured '${this.#config.embedder.provider}' but unavailable — running full-text-only`,
		);

		// Mixed schemaVersion across ledger ops (forward-compat / migration signal).
		const versions = new Set(this.#allRecords.map((r) => r.schemaVersion));
		add(
			"schemaVersion",
			versions.size <= 1,
			versions.size <= 1 ? `uniform (v${SCHEMA_VERSION})` : `mixed: ${[...versions].join(", ")}`,
		);

		// Dangling supersededBy references (point at ids that are not live).
		const dangling = [...this.#live.values()].filter(
			(r) => r.supersededBy && !this.#live.has(r.supersededBy),
		).length;
		add("supersededBy refs", dangling === 0, `${dangling} dangling`);

		// The committed source-of-truth files must not be accidentally gitignored.
		const ignored = await this.#gitIgnored([this.#config.paths.ledger, this.#config.paths.sidecar]);
		add(
			"git tracking",
			ignored.length === 0,
			ignored.length === 0
				? "memory.jsonl + vectors.jsonl are not gitignored"
				: `gitignored (will be lost on clone): ${ignored.join(", ")}`,
		);

		// Duplicate contentHash among live records (should be 0 after dedup).
		const seen = new Set<string>();
		let dups = 0;
		for (const r of this.#live.values()) {
			if (!r.contentHash) continue;
			if (seen.has(r.contentHash)) dups++;
			else seen.add(r.contentHash);
		}
		add("duplicate contentHash", dups === 0, `${dups} duplicates`);

		if (this.vectorsEnabled) {
			// Records with no usable vector for the CURRENT embedder (absent OR
			// produced by a different model/dim) — these index FTS-only until
			// `rebuild --reembed`.
			const missing = [...this.#live.values()].filter((r) => this.#needsEmbed(r)).length;
			add(
				"vectors present",
				missing === 0,
				missing === 0
					? "all live records have current-model vectors"
					: `${missing} live records need (re)embedding (run: searchlite-memory rebuild --reembed)`,
			);
			const drift = [...this.#sidecar.values()].filter((e) => e.model !== this.#embedder.id).length;
			add(
				"embedder fingerprint",
				drift === 0,
				drift === 0
					? "sidecar matches embedder"
					: `${drift} sidecar entries from a different model`,
			);
		}

		add(
			"CLAUDE_PROJECT_DIR",
			!this.#config.projectDirResolvedFromCwd,
			this.#config.projectDirResolvedFromCwd
				? "not set — memory dir resolved from cwd; set it for non-Claude-Code hosts"
				: "set",
		);

		return { ok: checks.every((c) => c.ok), checks };
	}

	/** Which of `paths` are gitignored. Empty (no false alarms) outside a git repo. */
	async #gitIgnored(paths: string[]): Promise<string[]> {
		const { execFile } = await import("node:child_process");
		const root = this.#config.paths.root;
		const results = await Promise.all(
			paths.map(
				(p) =>
					new Promise<string | null>((resolve) => {
						// Run from the repo and pass a path RELATIVE to cwd (git
						// check-ignore is most reliable with worktree-relative paths;
						// `--` guards a leading dash). Exit 0 (no err) = ignored, 1 =
						// not, 128 = not a repo. Only exit 0 means "ignored".
						execFile(
							"git",
							["check-ignore", "-q", "--", relative(root, p)],
							{ cwd: root },
							(err) => {
								resolve(err == null ? p : null);
							},
						);
					}),
			),
		);
		return results.filter((p): p is string => p !== null);
	}

	async close(): Promise<void> {
		await this.#cache.flush();
		if (this.#index) {
			await this.#index.close();
			this.#index = null;
		}
	}

	// --- internals ---

	#findLiveByHash(hash: string): string | null {
		for (const rec of this.#live.values()) {
			if (rec.contentHash === hash) return rec.id;
		}
		return null;
	}

	#buildFilter(opts: RecallOptions): Record<string, unknown> | undefined {
		const clauses: Record<string, unknown>[] = [];
		if (opts.namespace) {
			clauses.push({ KeywordEq: { field: "namespace", value: opts.namespace } });
		}
		if (opts.type) {
			const types = Array.isArray(opts.type) ? opts.type : [opts.type];
			clauses.push({ KeywordIn: { field: "type", values: types } });
		}
		if (opts.tags && opts.tags.length > 0) {
			clauses.push({ KeywordIn: { field: "tags", values: opts.tags } });
		}
		if (typeof opts.minImportance === "number") {
			clauses.push({ F64Range: { field: "importance", min: opts.minImportance, max: 1 } });
		}
		if (clauses.length === 0) return undefined;
		if (clauses.length === 1) return clauses[0];
		return { And: clauses };
	}

	#bumpAccess(ids: string[]): void {
		const now = new Date().toISOString();
		for (const id of ids) {
			const prev = this.#access.get(id);
			this.#access.set(id, { lastAccessed: now, accessCount: (prev?.accessCount ?? 0) + 1 });
		}
	}

	async #bumpGate(): Promise<void> {
		const gate = this.#computeGate(this.#indexGen);
		await this.#writeGate(gate);
		this.#openedGate = gate;
		// Record post-write source stamp so our own mutation isn't seen as an
		// external change on the next recall.
		this.#sourceStamp = await this.#currentSourceStamp();
	}

	/**
	 * Reopen + reload if the committed memory changed underneath a long-running
	 * server — either by another searchlite-memory process (detected via the gate)
	 * or by an external edit / `git pull` / branch switch (detected via the
	 * ledger+sidecar mtime, which the gitignored gate file would not reflect).
	 */
	async #refreshIfStale(): Promise<void> {
		if ((await this.#currentSourceStamp()) !== this.#sourceStamp) {
			await this.#load();
			return;
		}
		const onDisk = await this.#readGate();
		if (onDisk && this.#openedGate && !this.#sameGate(onDisk, this.#openedGate)) {
			await this.#load();
		}
	}
}
