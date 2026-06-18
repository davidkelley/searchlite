import type { Embedder } from "./embedder.js";

export interface LocalEmbedderOptions {
	model: string;
	dim: number;
	quant: string;
	revision: string;
}

/**
 * Create a local ONNX embedder via `@huggingface/transformers`. Returns `null`
 * if the optional dependency or the model cannot be loaded at startup (the
 * factory then falls back to full-text-only). Once created, `embed` THROWS on a
 * transient failure so the store can surface a tool error rather than silently
 * writing a vectorless memory.
 *
 * NOTE: exercised by a gated integration test (`RUN_MODEL_TESTS`) because the
 * model download is slow and network-dependent; unit tests use a stub embedder.
 */
// Minimal structural view of the bits of `@huggingface/transformers` we use.
// Cast through `unknown` so we neither couple to its exact (version-specific)
// option unions nor require its types to be present for consumers who install
// without the optional dependency.
type FeaturePipeline = (
	texts: string[],
	opts: { pooling: string; normalize: boolean },
) => Promise<{ tolist(): number[][] }>;
interface TransformersModule {
	pipeline(
		task: "feature-extraction",
		model: string,
		opts: { dtype?: string; revision?: string },
	): Promise<FeaturePipeline>;
}

export async function createLocalEmbedder(opts: LocalEmbedderOptions): Promise<Embedder | null> {
	let pipe: FeaturePipeline;
	try {
		const transformers = (await import(
			"@huggingface/transformers"
		)) as unknown as TransformersModule;
		pipe = await transformers.pipeline("feature-extraction", opts.model, {
			dtype: opts.quant,
			revision: opts.revision,
		});
	} catch {
		return null;
	}

	const id = `${opts.model}@${opts.revision}@${opts.quant}`;
	const dim = opts.dim;

	return {
		id,
		dim,
		available: true,
		async embed(texts: string[]): Promise<Float32Array[]> {
			if (texts.length === 0) return [];
			// Mean pooling + L2 normalization → one unit vector per input.
			const output = await pipe(texts, { pooling: "mean", normalize: true });
			const rows = output.tolist();
			return rows.map((row) => {
				if (row.length !== dim) {
					throw new Error(`embedding dim ${row.length} does not match configured dim ${dim}`);
				}
				return Float32Array.from(row);
			});
		},
	};
}
