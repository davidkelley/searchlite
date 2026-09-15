import { createHash } from "node:crypto";
import type { Embedder } from "../src/embed/embedder.js";

/**
 * Deterministic, content-dependent stub embedder for fast offline tests — same
 * text always yields the same vector, and similar text yields somewhat similar
 * vectors (shared hash bytes). Avoids ONNX/network in unit + integration tests.
 */
export function stubEmbedder(dim = 16): Embedder {
	return {
		id: `stub@v1@dim${dim}`,
		dim,
		available: true,
		async embed(texts: string[]): Promise<Float32Array[]> {
			return texts.map((t) => {
				const h = createHash("sha256").update(t).digest();
				const v = new Float32Array(dim);
				for (let i = 0; i < dim; i++) v[i] = (h[i % h.length] / 255) * 2 - 1;
				return v;
			});
		},
	};
}
