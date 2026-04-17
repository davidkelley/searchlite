import { z } from "zod";
import { type ZodIndexSchema } from "./compile";
import {
	type SearchliteFieldMetadata,
	type SearchliteIndexMetadata,
	SearchliteFieldRegistry,
	SearchliteIndexRegistry,
} from "./registries";

// ── Option types ─────────────────────────────────────────────────────────────

export interface TextOpts {
	analyzer?: string;
	searchAnalyzer?: string;
	stored?: boolean;
	indexed?: boolean;
	searchAsYouType?: { minGram: number; maxGram: number };
}

export interface KeywordOpts {
	stored?: boolean;
	indexed?: boolean;
	fast?: boolean;
}

export interface NumericOpts {
	stored?: boolean;
	fast?: boolean;
}

export interface VectorOpts {
	dim: number;
	metric: "Cosine" | "L2";
	hnsw?: { m?: number; efConstruction?: number };
}

export interface IndexOpts {
	docIdField?: string;
	analyzers?: unknown[];
}

// ── Helper implementations ───────────────────────────────────────────────────

/**
 * Register field metadata onto a Zod schema and return the same schema with
 * the metadata attached. The Zod schema reference is unchanged; only the
 * registry is mutated.
 */
function attach<T extends z.ZodType>(schema: T, meta: SearchliteFieldMetadata): T {
	SearchliteFieldRegistry.add(schema as never, meta);
	return schema;
}

/**
 * Declare a text-searchable string field. Equivalent to `z.string()` plus
 * text-kind metadata and any analyzer / stored / indexed overrides.
 *
 * Overloads:
 *   - `sl.text(opts?)` — creates a new `z.string()`
 *   - `sl.text(inner, opts?)` — attaches metadata to an existing string schema
 */
export function text(opts?: TextOpts): z.ZodString;
export function text<S extends z.ZodString>(inner: S, opts?: TextOpts): S;
export function text(
	innerOrOpts?: z.ZodString | TextOpts,
	maybeOpts?: TextOpts,
): z.ZodString {
	let inner: z.ZodString;
	let opts: TextOpts | undefined;
	if (isZodString(innerOrOpts)) {
		inner = innerOrOpts;
		opts = maybeOpts;
	} else {
		inner = z.string();
		opts = innerOrOpts;
	}
	return attach(inner, { kind: "text", ...opts });
}

/**
 * Declare an exact-match keyword field. Keyword fields are stored, indexed,
 * and fast-enabled by default.
 */
export function keyword(opts?: KeywordOpts): z.ZodString;
export function keyword<S extends z.ZodString>(inner: S, opts?: KeywordOpts): S;
export function keyword(
	innerOrOpts?: z.ZodString | KeywordOpts,
	maybeOpts?: KeywordOpts,
): z.ZodString {
	let inner: z.ZodString;
	let opts: KeywordOpts | undefined;
	if (isZodString(innerOrOpts)) {
		inner = innerOrOpts;
		opts = maybeOpts;
	} else {
		inner = z.string();
		opts = innerOrOpts;
	}
	return attach(inner, { kind: "keyword", ...opts });
}

/**
 * A Zod schema that accepts numbers or BigInts representing integers. This
 * is the return type of `sl.integer()` — it covers the case where the NAPI
 * binding surfaces large i64 values as JS `BigInt` (values above the i32
 * range). Both inputs are coerced to `number` after validation.
 */
export type ZodIntegerLike = z.ZodType<number, number | bigint>;

/**
 * Declare an integer field.
 *
 * The returned schema uses a `z.union([z.number(), z.bigint()])` input guard
 * piped through `z.coerce.number().int()`. This accepts JS `number` and
 * `BigInt` — but NOT strings, booleans, or other types — so NAPI-returned
 * large i64 values (which surface as BigInt for values above the i32 range)
 * are handled seamlessly without silently coercing arbitrary inputs.
 */
export function integer(opts?: NumericOpts): ZodIntegerLike;
export function integer<S extends z.ZodNumber>(inner: S, opts?: NumericOpts): S;
export function integer(
	innerOrOpts?: z.ZodNumber | NumericOpts,
	maybeOpts?: NumericOpts,
): z.ZodType<number> {
	let inner: z.ZodType<number>;
	let opts: NumericOpts | undefined;
	if (isZodNumber(innerOrOpts)) {
		inner = innerOrOpts.int();
		opts = maybeOpts;
	} else {
		// Accept number OR bigint, then coerce to number + validate .int().
		// This is tighter than a blanket z.coerce.number() which would also
		// accept strings, booleans, null, and Date objects.
		// Accept number | bigint, then coerce to number and validate .int().
		// The double cast is needed because Zod's pipe types are strict about
		// intermediate _zod.input shapes that don't align with our ZodType<number>.
		// At runtime, the pipeline is: accept number|bigint → coerce → int check.
		const guard = z.union([z.number(), z.bigint()]) as z.ZodType<number | bigint>;
		inner = guard.pipe(z.coerce.number().int() as z.ZodType<number, number | bigint>) as unknown as z.ZodType<number>;
		opts = innerOrOpts as NumericOpts | undefined;
	}
	return attach(inner, { kind: "integer", ...opts });
}

/**
 * Declare a floating-point numeric field.
 */
export function float(opts?: NumericOpts): z.ZodNumber;
export function float<S extends z.ZodNumber>(inner: S, opts?: NumericOpts): S;
export function float(
	innerOrOpts?: z.ZodNumber | NumericOpts,
	maybeOpts?: NumericOpts,
): z.ZodNumber {
	let inner: z.ZodNumber;
	let opts: NumericOpts | undefined;
	if (isZodNumber(innerOrOpts)) {
		inner = innerOrOpts;
		opts = maybeOpts;
	} else {
		inner = z.number();
		opts = innerOrOpts;
	}
	return attach(inner, { kind: "float", ...opts });
}

/**
 * Declare a vector field. The runtime Zod type is a length-bounded
 * `z.array(z.number())` so values are validated for correct dimensionality.
 */
export function vector(opts: VectorOpts): z.ZodArray<z.ZodNumber> {
	if (!Number.isInteger(opts.dim) || opts.dim <= 0) {
		throw new Error("sl.vector: `dim` must be a positive integer");
	}
	if (opts.metric !== "Cosine" && opts.metric !== "L2") {
		throw new Error('sl.vector: `metric` must be "Cosine" or "L2"');
	}
	const inner = z.array(z.number()).length(opts.dim);
	return attach<z.ZodArray<z.ZodNumber>>(inner, {
		kind: "vector",
		dim: opts.dim,
		metric: opts.metric,
		hnsw: opts.hnsw as Record<string, unknown> | undefined,
	});
}

/**
 * Mark a `z.object({...})` as the root schema of a searchlite index. Attaches
 * index-level metadata (`docIdField`, `analyzers`) and brands the return type
 * so that constructors can detect a Zod index schema at both the type and
 * runtime level.
 */
export function index<TSchema extends z.ZodObject<z.ZodRawShape>>(
	schema: TSchema,
	opts?: IndexOpts,
): ZodIndexSchema<TSchema> {
	const meta: SearchliteIndexMetadata = {};
	if (opts?.docIdField !== undefined) {
		if (typeof opts.docIdField !== "string" || opts.docIdField.length === 0) {
			throw new Error("sl.index: `docIdField` must be a non-empty string");
		}
		meta.docIdField = opts.docIdField;
	}
	if (opts?.analyzers !== undefined) {
		if (!Array.isArray(opts.analyzers)) {
			throw new Error("sl.index: `analyzers` must be an array");
		}
		meta.analyzers = opts.analyzers;
	}
	SearchliteIndexRegistry.add(schema as never, meta);
	return schema as ZodIndexSchema<TSchema>;
}

// ── Typed factory functions ───────────────────────────────────────────────────
//
// These eliminate the `new EmbeddedIndex<z.infer<typeof Schema>>(...)` ceremony
// by inferring `T` from the schema argument. They're the preferred entry point
// for Zod-first users.

// Lazy imports to avoid pulling in the entire embedded/remote module when only
// the helpers are used (e.g., in a shared schema package that doesn't index).
function lazyEmbedded() {
	// eslint-disable-next-line @typescript-eslint/no-var-requires
	return require("../embedded") as typeof import("../embedded");
}
function lazyRemote() {
	// eslint-disable-next-line @typescript-eslint/no-var-requires
	return require("../remote") as typeof import("../remote");
}

/**
 * Create a typed `EmbeddedIndex` from a Zod-branded schema. The document type
 * is inferred automatically — no explicit generic needed.
 *
 * @example
 * ```ts
 * const idx = sl.embedded("./data", UserSchema);
 * await idx.add({ id: "...", name: "Alice" }); // typed as User
 * ```
 */
export function embedded<TSchema extends z.ZodObject<z.ZodRawShape>>(
	path: string,
	schema: ZodIndexSchema<TSchema>,
	options?: { writeKey?: string },
): import("../embedded").EmbeddedIndex<z.infer<TSchema>> {
	const { EmbeddedIndex } = lazyEmbedded();
	return new EmbeddedIndex(path, {
		...options,
		schema: schema as unknown as ZodIndexSchema,
	}) as import("../embedded").EmbeddedIndex<z.infer<TSchema>>;
}

/**
 * Create a typed `RemoteIndex` from a Zod-branded schema. The document type
 * is inferred automatically.
 *
 * @example
 * ```ts
 * const idx = sl.remote("http://srv", "users", UserSchema);
 * const r = await idx.search("alice"); // r.hits[0].fields typed as User
 * ```
 */
export function remote<TSchema extends z.ZodObject<z.ZodRawShape>>(
	baseUrl: string,
	indexName: string,
	schema: ZodIndexSchema<TSchema>,
	options?: { writeKey?: string; fetch?: typeof globalThis.fetch },
): import("../remote").RemoteIndex<z.infer<TSchema>> {
	const { RemoteIndex } = lazyRemote();
	return new RemoteIndex(baseUrl, indexName, {
		...options,
		schema: schema as unknown as ZodIndexSchema,
	}) as import("../remote").RemoteIndex<z.infer<TSchema>>;
}

// ── Internal helpers ─────────────────────────────────────────────────────────

function isZodString(x: unknown): x is z.ZodString {
	if (!x || typeof x !== "object") return false;
	const def = (x as { _def?: { type?: unknown } })._def;
	return def?.type === "string";
}

function isZodNumber(x: unknown): x is z.ZodNumber {
	if (!x || typeof x !== "object") return false;
	const def = (x as { _def?: { type?: unknown } })._def;
	return def?.type === "number";
}

// ── Public namespace ─────────────────────────────────────────────────────────

/**
 * `sl.*` — fluent helpers for authoring searchlite index fields with Zod.
 *
 * @example
 * ```ts
 * const UserSchema = sl.index(
 *   z.object({
 *     id: z.string().uuid(),         // auto-promoted to keyword
 *     name: z.string(),               // text
 *     tags: z.array(sl.keyword()),    // keyword (explicit)
 *     age: sl.integer(),
 *     score: sl.float(),
 *   }),
 *   { docIdField: "id" },
 * );
 * ```
 */
export const sl = {
	text,
	keyword,
	integer,
	float,
	vector,
	index,
	embedded,
	remote,
} as const;
