import type { Embedder } from "./embedder.js";

/**
 * Full-text-only embedder: produces no vectors. Selected when
 * `EMBEDDER=none`, or as the fallback when a configured local model can't be
 * loaded at startup. The store checks `available` and skips the vector field
 * and the vector retrieval call entirely.
 */
export class NullEmbedder implements Embedder {
	readonly id = "none";
	readonly dim = 0;
	readonly available = false;

	async embed(_texts: string[]): Promise<Float32Array[]> {
		return [];
	}
}
