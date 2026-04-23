/**
 * Zod-native example: blog posts.
 *
 * A minimal, end-to-end demonstration of using a single Zod schema to drive
 * indexing, validation, and type-safe search results.
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

// One schema. Validates documents, drives the native index, types search
// results.
const BlogSchema = sl.index(
	z.object({
		id: z.string().uuid(),
		title: z.string(),
		slug: sl.keyword(),
		body: z.string(),
		tags: z.array(z.object({ label: sl.keyword() })),
		status: z.enum(["draft", "published", "archived"]),
		views: sl.integer({ stored: true }),
	}),
	{ docIdField: "id" },
);

type BlogPost = z.infer<typeof BlogSchema>;

async function main() {
	const dir = mkdtempSync(join(tmpdir(), "searchlite-zod-blog-"));
	try {
		const index = new EmbeddedIndex<BlogPost>(join(dir, "idx"), {
			schema: BlogSchema,
		});

		const posts: BlogPost[] = [
			{
				id: "550e8400-e29b-41d4-a716-446655440001",
				title: "Getting started with Searchlite",
				slug: "getting-started",
				body: "Searchlite is a fast, embeddable full-text search engine.",
				tags: [{ label: "intro" }, { label: "tutorial" }],
				status: "published",
				views: 42,
			},
			{
				id: "550e8400-e29b-41d4-a716-446655440002",
				title: "Zod-native schemas",
				slug: "zod-native",
				body: "Define your index once with Zod and get validation, types, and indexing for free.",
				tags: [{ label: "typescript" }, { label: "zod" }],
				status: "published",
				views: 7,
			},
			{
				id: "550e8400-e29b-41d4-a716-446655440003",
				title: "Draft post",
				slug: "draft",
				body: "A thought in progress.",
				tags: [],
				status: "draft",
				views: 0,
			},
		];

		await index.addMany(posts);
		await index.commit();

		// Simple query — hit fields are auto-typed as BlogPost.
		const hello = await index.search("searchlite");
		console.log(`Search "searchlite" → ${hello.totalHits} hit(s):`);
		for (const hit of hello.hits) {
			console.log(`  • ${hit.fields?.title} (slug=${hit.fields?.slug}, views=${hit.fields?.views})`);
		}

		// Structured request — filter by status.
		const published = await index.search({
			query: "zod",
			filter: { KeywordEq: { field: "status", value: "published" } },
		});
		console.log(`\nPublished posts mentioning "zod" → ${published.totalHits}:`);
		for (const hit of published.hits) {
			console.log(`  • ${hit.fields?.title}`);
		}

		// Validation demo: trying to add an invalid doc throws.
		try {
			await index.add({
				id: "not-a-uuid",
				title: "x",
				slug: "x",
				body: "x",
				tags: [],
				status: "published",
				views: 0,
			});
		} catch (err) {
			console.log(`\nInvalid doc rejected at add-time: ${(err as Error).message.split("\n")[0]}`);
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
