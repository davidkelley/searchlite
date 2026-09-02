import { join, resolve } from "node:path";
import {
	DEFAULT_DIM,
	DEFAULT_LOCAL_MODEL,
	DEFAULT_QUANT,
	DEFAULT_REVISION,
	type EmbedderConfig,
	type EmbedderProvider,
} from "./embed/embedder.js";
import { DEFAULT_RRF_K, DEFAULT_WEIGHTS, type ScoringWeights } from "./memory/retrieval.js";

export interface MemoryPaths {
	root: string;
	ledger: string;
	sidecar: string;
	indexDir: string;
	currentPointer: string;
	access: string;
	gate: string;
	cache: string;
	lock: string;
	gitignore: string;
	gitattributes: string;
}

export interface MemoryConfig {
	paths: MemoryPaths;
	embedder: EmbedderConfig;
	/** Pool size per searchlite call before fusion. */
	poolSize: number;
	rrfK: number;
	/** Default number of memories returned by recall. */
	recallLimit: number;
	weights: ScoringWeights;
	halfLifeHours: number;
	accessCap: number;
	lockStaleMs: number;
	lockRetries: number;
	lockDisabled: boolean;
	/** True when CLAUDE_PROJECT_DIR was not set and we fell back to cwd. */
	projectDirResolvedFromCwd: boolean;
}

const PROVIDERS: EmbedderProvider[] = ["local", "none", "openai", "voyage", "cohere"];

function intEnv(value: string | undefined, fallback: number): number {
	if (value === undefined) return fallback;
	const n = Number.parseInt(value, 10);
	return Number.isFinite(n) ? n : fallback;
}

function floatEnv(value: string | undefined, fallback: number): number {
	if (value === undefined) return fallback;
	const n = Number.parseFloat(value);
	return Number.isFinite(n) ? n : fallback;
}

function resolveRoot(env: NodeJS.ProcessEnv): { root: string; fromCwd: boolean } {
	if (env.SEARCHLITE_MEMORY_DIR) {
		return { root: resolve(env.SEARCHLITE_MEMORY_DIR), fromCwd: false };
	}
	const base = env.CLAUDE_PROJECT_DIR ?? process.cwd();
	return {
		root: resolve(join(base, ".searchlite-memory")),
		fromCwd: env.CLAUDE_PROJECT_DIR == null,
	};
}

function resolveWeights(env: NodeJS.ProcessEnv): ScoringWeights {
	const raw = env.SEARCHLITE_MEMORY_WEIGHTS;
	if (!raw) return DEFAULT_WEIGHTS;
	try {
		const parsed = JSON.parse(raw) as Partial<ScoringWeights>;
		return {
			rel: typeof parsed.rel === "number" ? parsed.rel : DEFAULT_WEIGHTS.rel,
			rec: typeof parsed.rec === "number" ? parsed.rec : DEFAULT_WEIGHTS.rec,
			imp: typeof parsed.imp === "number" ? parsed.imp : DEFAULT_WEIGHTS.imp,
			acc: typeof parsed.acc === "number" ? parsed.acc : DEFAULT_WEIGHTS.acc,
		};
	} catch {
		return DEFAULT_WEIGHTS;
	}
}

/** Resolve configuration from environment variables (CLI flags layer on top elsewhere). */
export function loadConfig(env: NodeJS.ProcessEnv = process.env): MemoryConfig {
	const { root, fromCwd } = resolveRoot(env);
	const paths: MemoryPaths = {
		root,
		ledger: join(root, "memory.jsonl"),
		sidecar: join(root, "vectors.jsonl"),
		indexDir: join(root, "index"),
		currentPointer: join(root, "index", "CURRENT"),
		access: join(root, "access.json"),
		gate: join(root, ".ledger-hash"),
		cache: join(root, "embeddings.cache"),
		lock: join(root, ".lock"),
		gitignore: join(root, ".gitignore"),
		gitattributes: join(root, ".gitattributes"),
	};

	const providerRaw = (env.SEARCHLITE_MEMORY_EMBEDDER ?? "local") as EmbedderProvider;
	const provider = PROVIDERS.includes(providerRaw) ? providerRaw : "local";
	const embedder: EmbedderConfig = {
		provider,
		model: env.SEARCHLITE_MEMORY_MODEL ?? DEFAULT_LOCAL_MODEL,
		dim: intEnv(env.SEARCHLITE_MEMORY_DIM, DEFAULT_DIM),
		quant: env.SEARCHLITE_MEMORY_QUANT ?? DEFAULT_QUANT,
		revision: env.SEARCHLITE_MEMORY_REVISION ?? DEFAULT_REVISION,
	};

	return {
		paths,
		embedder,
		poolSize: intEnv(env.SEARCHLITE_MEMORY_POOL_SIZE, 50),
		rrfK: intEnv(env.SEARCHLITE_MEMORY_RRF_K, DEFAULT_RRF_K),
		recallLimit: intEnv(env.SEARCHLITE_MEMORY_RECALL_LIMIT, 8),
		weights: resolveWeights(env),
		halfLifeHours: floatEnv(env.SEARCHLITE_MEMORY_HALF_LIFE_HOURS, 168),
		accessCap: intEnv(env.SEARCHLITE_MEMORY_ACCESS_CAP, 20),
		lockStaleMs: intEnv(env.SEARCHLITE_MEMORY_LOCK_STALE, 30_000),
		lockRetries: intEnv(env.SEARCHLITE_MEMORY_LOCK_RETRIES, 10),
		lockDisabled: env.SEARCHLITE_MEMORY_NO_LOCK === "1" || env.SEARCHLITE_MEMORY_NO_LOCK === "true",
		projectDirResolvedFromCwd: fromCwd,
	};
}
