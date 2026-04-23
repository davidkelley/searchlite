/**
 * Zod-native example: product catalog with nested variants.
 *
 * Demonstrates:
 *   - Nested `z.object({...})` for metadata
 *   - `z.array(z.object({...}))` for multi-valued variants
 *   - Explicit stored / fast options on numerics
 *   - Modeling a boolean-like field with `z.enum(["yes","no"])`
 *
 * (Dense vectors via `sl.vector(...)` require the `vectors` feature flag at
 * compile time — see `docs/vectors.md`. This example keeps the default build.)
 *
 * Run:
 *   npm install
 *   npm run example
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { EmbeddedIndex, sl } from "searchlite-js";

const ProductSchema = sl.index(
	z.object({
		id: z.string().uuid(),
		name: z.string(),
		description: z.string(),
		brand: sl.keyword({ fast: true }),
		category: z.enum(["electronics", "home", "apparel", "toys"]),
		price: sl.float({ stored: true }),
		inStock: z.enum(["yes", "no"]), // modeled as keyword because core has no boolean
		meta: z.object({
			sku: sl.keyword(),
			weightKg: sl.float({ stored: true }),
		}),
		variants: z.array(
			z.object({
				color: sl.keyword(),
				sizeUs: sl.integer({ stored: true }),
				priceDelta: sl.float({ stored: true }),
			}),
		),
	}),
	{ docIdField: "id" },
);

type Product = z.infer<typeof ProductSchema>;

async function main() {
	const dir = mkdtempSync(join(tmpdir(), "searchlite-zod-products-"));
	try {
		const index = new EmbeddedIndex<Product>(join(dir, "idx"), {
			schema: ProductSchema,
		});

		const products: Product[] = [
			{
				id: "550e8400-e29b-41d4-a716-446655440011",
				name: "Wireless Noise-Cancelling Headphones",
				description: "Over-ear bluetooth headphones with active noise cancellation.",
				brand: "AudioCo",
				category: "electronics",
				price: 149.99,
				inStock: "yes",
				meta: { sku: "AC-HP-01", weightKg: 0.28 },
				variants: [
					{ color: "black", sizeUs: 0, priceDelta: 0 },
					{ color: "silver", sizeUs: 0, priceDelta: 10 },
				],
			},
			{
				id: "550e8400-e29b-41d4-a716-446655440012",
				name: "USB Condenser Microphone",
				description: "Studio-quality USB mic for podcasting and streaming.",
				brand: "SoundPro",
				category: "electronics",
				price: 89.0,
				inStock: "yes",
				meta: { sku: "SP-MIC-12", weightKg: 0.95 },
				variants: [{ color: "black", sizeUs: 0, priceDelta: 0 }],
			},
			{
				id: "550e8400-e29b-41d4-a716-446655440013",
				name: "Running Shoes",
				description: "Lightweight cushioned running shoes for daily training.",
				brand: "StrideCo",
				category: "apparel",
				price: 119.0,
				inStock: "yes",
				meta: { sku: "SC-RS-88", weightKg: 0.24 },
				variants: [
					{ color: "red", sizeUs: 9, priceDelta: 0 },
					{ color: "red", sizeUs: 10, priceDelta: 0 },
					{ color: "blue", sizeUs: 9, priceDelta: 0 },
				],
			},
		];

		await index.addMany(products);
		await index.commit();

		// Full-text search for "headphones"
		const headphones = await index.search("headphones");
		console.log(`Products matching "headphones": ${headphones.totalHits}`);
		for (const hit of headphones.hits) {
			console.log(
				`  • ${hit.fields?.name} — $${hit.fields?.price.toFixed(2)} (weight ${hit.fields?.meta.weightKg}kg)`,
			);
		}

		// Filter by brand
		const audioco = await index.search({
			query: { type: "match_all" },
			filter: { KeywordEq: { field: "brand", value: "AudioCo" } },
		});
		console.log(`\nAudioCo products: ${audioco.totalHits}`);
		for (const hit of audioco.hits) {
			console.log(`  • ${hit.fields?.name}`);
		}

		// Structured query with price range filter (float field)
		const under100 = await index.search({
			query: { type: "match_all" },
			filter: { F64Range: { field: "price", min: 0, max: 100 } },
		});
		console.log(`\nProducts under $100: ${under100.totalHits}`);
		for (const hit of under100.hits) {
			console.log(`  • ${hit.fields?.name} ($${hit.fields?.price.toFixed(2)})`);
		}

		// Validation catches malformed docs — e.g., wrong type for a numeric:
		try {
			await index.add({
				...products[0],
				id: "550e8400-e29b-41d4-a716-446655440044",
				// @ts-expect-error — demonstrates Zod catching a type error at runtime
				price: "ninety-nine",
			});
		} catch (err) {
			const first = (err as Error).message.split("\n").slice(0, 3).join("\n");
			console.log(`\nRejected by Zod (type error):\n${first}`);
		}

		await index.close();
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}

main().catch((err) => {
	console.error(err);
	process.exit(1);
});
