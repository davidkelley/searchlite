// Type-level tests for the Zod authoring path.
//
// This file is NOT executed at runtime. It's compiled by `tsc` (via the
// `typecheck:tests` script) to verify that the public API provides correct
// compile-time types. A failure here shows up as a build error, not a test
// failure.

import { expectTypeOf } from "vitest";
import { z } from "zod";

import type {
	Hit,
	SearchResult,
	TypedSearchResult,
	ZodIndexSchema,
} from "../src";
import {
	EmbeddedIndex,
	RemoteIndex,
	compileZodSchema,
	isZodIndexSchema,
	sl,
} from "../src";

// ── sl.* helper return types ─────────────────────────────────────────────────

{
	expectTypeOf(sl.text()).toEqualTypeOf<z.ZodString>();
	expectTypeOf(sl.keyword()).toEqualTypeOf<z.ZodString>();
	// sl.integer() coerces bigint → number so NAPI-returned large i64 values
	// don't break validation. `z.infer<typeof sl.integer()>` is still `number`.
	expectTypeOf<z.infer<ReturnType<typeof sl.integer>>>().toEqualTypeOf<number>();
	expectTypeOf(sl.float()).toEqualTypeOf<z.ZodNumber>();
	expectTypeOf(sl.vector({ dim: 4, metric: "Cosine" })).toEqualTypeOf<
		z.ZodArray<z.ZodNumber>
	>();
}

// ── sl.index() preserves the inner schema's shape for z.infer<> ──────────────

{
	const UserSchema = sl.index(
		z.object({
			id: z.string().uuid(),
			name: z.string(),
			age: sl.integer(),
		}),
		{ docIdField: "id" },
	);

	type User = z.infer<typeof UserSchema>;

	expectTypeOf<User>().toEqualTypeOf<{
		id: string;
		name: string;
		age: number;
	}>();
}

// ── sl.index() return type is branded (ZodIndexSchema<...>) ──────────────────

{
	const S = sl.index(z.object({ a: z.string() }));
	// Branded type is assignable to the plain ZodObject via structural typing:
	expectTypeOf(S).toMatchTypeOf<z.ZodObject<{ a: z.ZodString }>>();
	// The ZodIndexSchema brand is a phantom marker, so it's a subtype.
	type Branded = ZodIndexSchema<z.ZodObject<{ a: z.ZodString }>>;
	expectTypeOf<typeof S>().toMatchTypeOf<Branded>();
}

// ── compileZodSchema takes a branded schema ──────────────────────────────────

{
	const branded = sl.index(z.object({ a: z.string() }));
	const _result = compileZodSchema(branded);
	// Return is `JsonSchemaOutput` (i.e., ZodCompiledJsonSchema), which has
	// `type: "object"` and `properties`.
	expectTypeOf(_result.type).toEqualTypeOf<"object">();
	expectTypeOf(_result.properties).toEqualTypeOf<
		Record<string, Record<string, unknown>>
	>();
}

// ── isZodIndexSchema type predicate narrows correctly ────────────────────────

{
	const maybe: unknown = sl.index(z.object({}));
	if (isZodIndexSchema(maybe)) {
		// After narrowing, maybe is ZodIndexSchema.
		expectTypeOf(maybe).toMatchTypeOf<ZodIndexSchema>();
	}
}

// ── EmbeddedIndex default generic preserves today's untyped behavior ─────────

{
	const idx = new EmbeddedIndex("/tmp/idx");
	expectTypeOf(idx).toEqualTypeOf<EmbeddedIndex<Record<string, unknown>>>();

	// `.search("q")` returns SearchResult<Record<string, unknown>>.
	const searchReturn = idx.search("q");
	expectTypeOf(searchReturn).toEqualTypeOf<
		Promise<SearchResult<Record<string, unknown>>>
	>();

	// `.add(doc)` accepts an arbitrary record (today's behavior).
	idx.add({ anything: "goes" });
}

// ── EmbeddedIndex<T> with explicit generic narrows add/search types ──────────

{
	interface User {
		id: string;
		name: string;
		role: "admin" | "user";
	}
	const idx = new EmbeddedIndex<User>("/tmp/idx");

	// add() accepts User, rejects non-User:
	idx.add({ id: "1", name: "a", role: "admin" });
	// @ts-expect-error — missing required fields
	idx.add({ id: "1" });
	// @ts-expect-error — wrong type
	idx.add({ id: 1, name: "a", role: "admin" });
	// @ts-expect-error — invalid enum
	idx.add({ id: "1", name: "a", role: "hacker" });

	// addMany accepts User[] or single User:
	idx.addMany([{ id: "1", name: "a", role: "user" }]);
	idx.addMany({ id: "1", name: "a", role: "user" });

	// search() auto-types to SearchResult<User>:
	const r = idx.search("query");
	expectTypeOf(r).toEqualTypeOf<Promise<SearchResult<User>>>();

	// hit.fields has the right shape (nullable for default behavior):
	r.then((res) => {
		expectTypeOf<typeof res.hits[number]>().toMatchTypeOf<Hit<User>>();
	});
}

// ── Per-call search<U>() takes precedence with its own type ──────────────────

{
	interface User {
		id: string;
		name: string;
	}
	const idx = new EmbeddedIndex<User>("/tmp/idx");
	const Subset = z.object({ name: z.string() });
	const r = idx.search(Subset, "q");
	expectTypeOf(r).toEqualTypeOf<Promise<TypedSearchResult<{ name: string }>>>();
}

// ── RemoteIndex mirrors the same generic contract ────────────────────────────

{
	interface User {
		id: string;
		name: string;
	}
	const idx = new RemoteIndex<User>("http://srv", "users");

	idx.add({ id: "1", name: "a" });
	// @ts-expect-error
	idx.add({ id: "1" });

	const r = idx.search("q");
	expectTypeOf(r).toEqualTypeOf<Promise<SearchResult<User>>>();
}

// ── Constructor accepts schema in multiple shapes ────────────────────────────

{
	// 1. Shorthand
	new EmbeddedIndex("/tmp/a", { schema: { title: "text" } });
	// 2. Raw JSON Schema
	new EmbeddedIndex("/tmp/b", {
		schema: { type: "object", properties: { title: { type: "string" } } },
	});
	// 3. Zod branded
	new EmbeddedIndex("/tmp/c", {
		schema: sl.index(z.object({ title: z.string() })),
	});
}

// ── sl.text overload (existing ZodString passthrough) ────────────────────────

{
	const existing: z.ZodString = z.string().min(3);
	const tagged = sl.text(existing, { analyzer: "simple" });
	expectTypeOf(tagged).toEqualTypeOf<z.ZodString>();
}

// ── sl.integer applies .int() and coerces bigint → number ───────────────────

{
	const n = sl.integer();
	// Output type is `number`; input accepts `number | bigint` (NAPI i64
	// values above i32 range come back as BigInt).
	expectTypeOf<z.infer<typeof n>>().toEqualTypeOf<number>();
	// `.int()` was applied internally — parsing non-integers will fail at runtime.
}

// ── sl.vector runtime validates dimensionality ───────────────────────────────

{
	const v = sl.vector({ dim: 8, metric: "L2" });
	// The underlying Zod type is `ZodArray<ZodNumber>`, `z.infer<>` is number[]
	type V = z.infer<typeof v>;
	expectTypeOf<V>().toEqualTypeOf<number[]>();
}

// ── Explicit annotation: this file is consumed only by tsc (type tests) ──────
export {};
