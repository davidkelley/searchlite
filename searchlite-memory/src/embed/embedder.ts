import { NullEmbedder } from "./null.js";

/**
 * Produces embeddings for memory text and queries. Implementations: a local
 * ONNX model (`local.ts`), external APIs (later), and a no-op FTS-only fallback
 * (`null.ts`, `available=false`).
 */
export interface Embedder {
	/** Fingerprint `name@revision@quant` — stored with each vector to detect drift. */
	readonly id: string;
	/** Embedding dimension (must equal the index's configured vector dim). */
	readonly dim: number;
	/** When false, the store runs full-text-only (no vectors written or queried). */
	readonly available: boolean;
	/** Embed a batch of texts. Implementations throw on transient failure so the
	 * caller can surface a tool error rather than silently degrade. */
	embed(texts: string[]): Promise<Float32Array[]>;
}

export type EmbedderProvider = "local" | "none" | "openai" | "voyage" | "cohere";

export interface EmbedderConfig {
	provider: EmbedderProvider;
	model: string;
	dim: number;
	quant: string;
	revision?: string;
}

export const DEFAULT_LOCAL_MODEL = "Xenova/all-MiniLM-L6-v2";
export const DEFAULT_DIM = 384;
export const DEFAULT_QUANT = "q8";
export const DEFAULT_REVISION = "main";

/**
 * Build an embedder from config. `none` → FTS-only. `local` → ONNX via
 * transformers.js, falling back to FTS-only (with a warning) if the optional
 * dependency or model is unavailable at startup. External providers are not
 * implemented yet.
 */
export async function createEmbedder(config: EmbedderConfig): Promise<Embedder> {
	switch (config.provider) {
		case "none":
			return new NullEmbedder();
		case "local": {
			// Lazy import keeps `local.ts` (and its heavy dep) off the FTS-only path.
			const { createLocalEmbedder } = await import("./local.js");
			const local = await createLocalEmbedder({
				model: config.model,
				dim: config.dim,
				quant: config.quant,
				revision: config.revision ?? DEFAULT_REVISION,
			});
			if (!local) {
				process.stderr.write(
					"searchlite-memory: local embedder unavailable (@huggingface/transformers " +
						"or the model could not be loaded); running full-text-only. Install the " +
						"optional dependency or set SEARCHLITE_MEMORY_EMBEDDER=none to silence this.\n",
				);
				return new NullEmbedder();
			}
			return local;
		}
		default:
			throw new Error(
				`embedder provider '${config.provider}' is not implemented yet; use 'local' or 'none'`,
			);
	}
}
