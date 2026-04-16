import type { SearchliteFieldMetadata } from "./registries";
import type { SearchliteFieldRegistry } from "./registries";

// ── Auto-promotion rules ─────────────────────────────────────────────────────

/**
 * String-format refinements that auto-promote `z.string()` to the `keyword`
 * kind because they represent identifiers or canonical values that are never
 * useful as full-text-searchable content.
 *
 * Users can override with `sl.text()` or `.meta({ kind: "text" })`.
 */
export const KEYWORD_AUTO_PROMOTE_FORMATS: ReadonlySet<string> = new Set([
	"uuid",
	"guid",
	"cuid",
	"cuid2",
	"ulid",
	"nanoid",
	"email",
	"url",
]);

// ── Internal helpers on Zod `_def` ───────────────────────────────────────────

/** Internal shape of a v4 check entry on `_def.checks`. */
interface ZodCheck {
	_zod?: { def?: { check?: string; format?: string } };
	def?: { check?: string; format?: string };
}

/** Read a check's discriminator ("string_format", "safeint", "regex", ...). */
function checkDiscriminator(check: ZodCheck): string | undefined {
	return check._zod?.def?.check ?? check.def?.check;
}

/** Read a check's `format` (uuid, email, url, ...), if any. */
function checkFormat(check: ZodCheck): string | undefined {
	return check._zod?.def?.format ?? check.def?.format;
}

/**
 * Return the string-format name for a `ZodString`, or `undefined` for plain
 * strings. Only the FIRST format check is considered — conflicting formats
 * would be an unusual user error.
 */
export function getStringFormat(schema: unknown): string | undefined {
	const checks = getChecks(schema);
	if (!checks) return undefined;
	for (const c of checks) {
		if (checkDiscriminator(c) === "string_format") {
			const format = checkFormat(c);
			if (format) return format;
		}
	}
	return undefined;
}

/**
 * Return true if a `ZodNumber` was refined with `.int()` or `.safe()`.
 * In Zod v4.3 this appears as a `number_format` check with `format: "safeint"`.
 */
export function isIntegerNumber(schema: unknown): boolean {
	const checks = getChecks(schema);
	if (!checks) return false;
	for (const c of checks) {
		const disc = checkDiscriminator(c);
		const format = checkFormat(c);
		if (disc === "number_format" && (format === "safeint" || format === "int")) {
			return true;
		}
		// Older Zod variants (pre-4.3) used `.def.check === "int"`; kept for robustness.
		if (disc === "int" || disc === "safeint") return true;
	}
	return false;
}

function getChecks(schema: unknown): ZodCheck[] | undefined {
	if (!schema || typeof schema !== "object") return undefined;
	const def = (schema as { _def?: { checks?: ZodCheck[] } })._def;
	return def?.checks;
}

// ── Kind inference ───────────────────────────────────────────────────────────

export type InferredKind = "text" | "keyword" | "integer" | "float";

/**
 * Infer the default field kind for a Zod primitive, applying auto-promotion
 * rules. Returns `undefined` for types that don't map to a leaf kind.
 *
 * Explicit metadata (`sl.*` helper or `.meta({kind})`) always wins over this
 * inference — this helper is only consulted when no explicit kind is present.
 */
export function inferKind(schema: unknown): InferredKind | undefined {
	const type = getDefType(schema);
	switch (type) {
		case "string": {
			const format = getStringFormat(schema);
			if (format && KEYWORD_AUTO_PROMOTE_FORMATS.has(format)) {
				return "keyword";
			}
			return "text";
		}
		case "number":
			return isIntegerNumber(schema) ? "integer" : "float";
		case "literal":
			return inferLiteralKind(schema);
		case "enum":
			return "keyword";
		default:
			return undefined;
	}
}

function inferLiteralKind(schema: unknown): InferredKind | undefined {
	const def = (schema as { _def?: { values?: unknown[] } })._def;
	const values = def?.values;
	if (!values || values.length === 0) return undefined;

	// All literal values must agree on the JSON primitive kind. We check the
	// first and trust the rest (Zod already enforces homogeneous literal sets).
	const first = values[0];
	if (typeof first === "string") return "keyword";
	if (typeof first === "number") {
		return Number.isInteger(first) ? "integer" : "float";
	}
	return undefined;
}

// ── Type discriminator ───────────────────────────────────────────────────────

/** Return `_def.type` as a string, or `undefined`. */
export function getDefType(schema: unknown): string | undefined {
	if (!schema || typeof schema !== "object") return undefined;
	const def = (schema as { _def?: { type?: unknown } })._def;
	return typeof def?.type === "string" ? def.type : undefined;
}

// ── Metadata resolution (precedence: helper > .meta > auto-promote) ──────────

/**
 * Compose the effective field metadata for a Zod schema:
 *   1. Field-registry metadata (set by `sl.*` helpers) — highest precedence
 *   2. Global `.meta({kind: ...})` metadata
 *   3. Auto-promotion inference (uuid → keyword, etc.)
 *
 * Returned metadata may still be partial (e.g., only `kind` set); the caller
 * layers in format defaults (stored, indexed, fast, analyzer).
 */
export function resolveFieldMetadata(
	schema: unknown,
	fieldRegistry: typeof SearchliteFieldRegistry,
	globalRegistry: { get(schema: never): Record<string, unknown> | undefined },
): SearchliteFieldMetadata {
	// Unwrap wrappers (optional/nullable/default) so metadata attached to the
	// inner schema is also picked up when the helper was called on the primitive
	// before wrapping — e.g., `sl.keyword().optional()`.
	const unwrapped = unwrapSchema(schema);

	const fieldMeta =
		fieldRegistry.get(schema as never) ??
		fieldRegistry.get(unwrapped as never) ??
		{};

	const globalMeta =
		(globalRegistry.get(schema as never) as Record<string, unknown> | undefined) ??
		(globalRegistry.get(unwrapped as never) as Record<string, unknown> | undefined) ??
		{};

	// Extract kind-like fields from global meta if present (users using `.meta({kind:...})`).
	const out: SearchliteFieldMetadata = { ...fieldMeta };
	if (globalMeta.kind && !out.kind) {
		out.kind = globalMeta.kind as SearchliteFieldMetadata["kind"];
	}
	// Copy over any other known searchlite-relevant fields from global meta.
	for (const key of [
		"stored",
		"indexed",
		"fast",
		"analyzer",
		"searchAnalyzer",
		"searchAsYouType",
		"dim",
		"metric",
		"hnsw",
	] as const) {
		if (globalMeta[key] !== undefined && out[key] === undefined) {
			(out as Record<string, unknown>)[key] = globalMeta[key];
		}
	}

	return out;
}

/**
 * Unwrap a Zod schema through wrapper types (optional, nullable, default),
 * returning the leaf schema. Does NOT unwrap branded types because the brand
 * is transparent (brand._def.type is already the underlying type).
 */
export function unwrapSchema(schema: unknown): unknown {
	let cur = schema;
	for (let depth = 0; depth < 32; depth++) {
		const t = getDefType(cur);
		if (t === "optional" || t === "nullable" || t === "default") {
			const def = (cur as { _def?: { innerType?: unknown } })._def;
			if (def?.innerType) {
				cur = def.innerType;
				continue;
			}
		}
		break;
	}
	return cur;
}

/** Compute wrapper flags by walking optional/nullable/default layers. */
export interface WrapperState {
	optional: boolean;
	nullable: boolean;
	hasDefault: boolean;
	inner: unknown;
}

export function wrapperState(schema: unknown): WrapperState {
	const state: WrapperState = {
		optional: false,
		nullable: false,
		hasDefault: false,
		inner: schema,
	};
	let cur: unknown = schema;
	for (let depth = 0; depth < 32; depth++) {
		const t = getDefType(cur);
		const def = (cur as { _def?: { innerType?: unknown } })._def;
		if (t === "optional" && def?.innerType) {
			state.optional = true;
			cur = def.innerType;
			continue;
		}
		if (t === "nullable" && def?.innerType) {
			state.nullable = true;
			cur = def.innerType;
			continue;
		}
		if (t === "default" && def?.innerType) {
			state.hasDefault = true;
			cur = def.innerType;
			continue;
		}
		break;
	}
	state.inner = cur;
	return state;
}
