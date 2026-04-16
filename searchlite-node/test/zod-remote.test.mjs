import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { RemoteIndex, sl } from "../dist/index.js";

// --- Mock fetch helper (mirrors remote.test.mjs) ---

function mockFetch(responses) {
	const calls = [];
	let callIndex = 0;
	const fn = vi.fn(async (url, init) => {
		calls.push({
			url,
			method: init?.method,
			headers: init?.headers,
			body: init?.body ? JSON.parse(init.body) : undefined,
		});
		const response =
			typeof responses === "function"
				? responses(callIndex)
				: Array.isArray(responses)
					? responses[callIndex]
					: responses;
		callIndex++;
		return {
			ok: response.ok ?? true,
			status: response.status ?? 200,
			statusText: response.statusText ?? "OK",
			json: async () => response.body,
		};
	});
	fn._calls = calls;
	return fn;
}

const BULK_RESPONSE = { body: { queued: 1 } };
const COMMIT_RESPONSE = { body: {} };

function userResultResponse(doc) {
	return {
		body: {
			total_hits_estimate: 1,
			hits: [{ doc_id: doc.id, score: 1.0, fields: doc }],
			aggregations: {},
		},
	};
}

const UserSchema = sl.index(
	z.object({
		id: z.string().uuid(),
		name: z.string(),
		email: z.string().email(),
		role: z.enum(["admin", "user", "guest"]),
	}),
	{ docIdField: "id" },
);

const VALID_USER = {
	id: "550e8400-e29b-41d4-a716-446655440000",
	name: "Alice",
	email: "alice@example.com",
	role: "admin",
};

// ── Construction ─────────────────────────────────────────────────────────────

describe("RemoteIndex Zod path: construction", () => {
	it("accepts a Zod index schema", () => {
		const idx = new RemoteIndex("http://srv", "users", {
			schema: UserSchema,
			fetch: mockFetch(BULK_RESPONSE),
		});
		expect(idx).toBeDefined();
	});

	it("rejects a bare ZodObject (must be wrapped with sl.index())", () => {
		const raw = z.object({ id: z.string() });
		expect(
			() =>
				new RemoteIndex("http://srv", "users", {
					schema: raw,
					fetch: mockFetch(BULK_RESPONSE),
				}),
		).toThrowError(/must be.*wrapped.*sl\.index/s);
	});

	it("still works without any schema (backwards compat)", async () => {
		const fetch = mockFetch(BULK_RESPONSE);
		const idx = new RemoteIndex("http://srv", "users", { fetch });
		await idx.add({ id: "1", foo: "bar" });
		expect(fetch._calls[0].url).toContain("/indexes/users/bulk");
	});
});

// ── Client-side validation on add() ──────────────────────────────────────────

describe("RemoteIndex Zod path: add() validation", () => {
	it("accepts a valid document", async () => {
		const fetch = mockFetch(BULK_RESPONSE);
		const idx = new RemoteIndex("http://srv", "users", { schema: UserSchema, fetch });
		await idx.add(VALID_USER);
		expect(fetch._calls).toHaveLength(1);
		expect(fetch._calls[0].body).toEqual({ docs: [VALID_USER] });
	});

	it("rejects invalid UUID before hitting the server", async () => {
		const fetch = mockFetch(BULK_RESPONSE);
		const idx = new RemoteIndex("http://srv", "users", { schema: UserSchema, fetch });
		await expect(
			idx.add({ ...VALID_USER, id: "not-a-uuid" }),
		).rejects.toThrow();
		expect(fetch._calls).toHaveLength(0);
	});

	it("rejects invalid enum before hitting the server", async () => {
		const fetch = mockFetch(BULK_RESPONSE);
		const idx = new RemoteIndex("http://srv", "users", { schema: UserSchema, fetch });
		await expect(
			idx.add({ ...VALID_USER, role: "unknown" }),
		).rejects.toThrow();
		expect(fetch._calls).toHaveLength(0);
	});

	it("rejects invalid email before hitting the server", async () => {
		const fetch = mockFetch(BULK_RESPONSE);
		const idx = new RemoteIndex("http://srv", "users", { schema: UserSchema, fetch });
		await expect(
			idx.add({ ...VALID_USER, email: "not-an-email" }),
		).rejects.toThrow();
		expect(fetch._calls).toHaveLength(0);
	});

	it("does NOT send the Zod schema over the wire", async () => {
		const fetch = mockFetch(BULK_RESPONSE);
		const idx = new RemoteIndex("http://srv", "users", { schema: UserSchema, fetch });
		await idx.add(VALID_USER);
		expect(fetch._calls[0].body.schema).toBeUndefined();
	});
});

// ── addMany validation ───────────────────────────────────────────────────────

describe("RemoteIndex Zod path: addMany", () => {
	it("validates each document", async () => {
		const fetch = mockFetch({ body: { queued: 2 } });
		const idx = new RemoteIndex("http://srv", "users", { schema: UserSchema, fetch });
		const a = { ...VALID_USER };
		const b = { ...VALID_USER, id: "550e8400-e29b-41d4-a716-446655440001", name: "Bob" };
		const queued = await idx.addMany([a, b]);
		expect(queued).toBe(2);
	});

	it("reports the index of the failing document", async () => {
		const fetch = mockFetch({ body: { queued: 0 } });
		const idx = new RemoteIndex("http://srv", "users", { schema: UserSchema, fetch });
		await expect(
			idx.addMany([VALID_USER, { ...VALID_USER, id: "bad" }]),
		).rejects.toThrow(/documents\[1\]/);
		expect(fetch._calls).toHaveLength(0);
	});
});

// ── search() auto-validation ─────────────────────────────────────────────────

describe("RemoteIndex Zod path: search auto-validation", () => {
	it("auto-validates hit fields against the stored Zod schema", async () => {
		const fetch = mockFetch(userResultResponse(VALID_USER));
		const idx = new RemoteIndex("http://srv", "users", { schema: UserSchema, fetch });
		const result = await idx.search("alice");
		expect(result.totalHits).toBe(1);
		expect(result.hits[0].fields).toEqual(VALID_USER);
	});

	it("raises a validation error when the server returns a shape mismatch", async () => {
		const bad = { ...VALID_USER, email: "not-an-email" };
		const fetch = mockFetch(userResultResponse(bad));
		const idx = new RemoteIndex("http://srv", "users", { schema: UserSchema, fetch });
		await expect(idx.search("anything")).rejects.toThrowError(/hit 0.*docId/s);
	});

	it("per-call schema overrides the stored schema", async () => {
		const fetch = mockFetch(userResultResponse(VALID_USER));
		const idx = new RemoteIndex("http://srv", "users", { schema: UserSchema, fetch });
		const Subset = z.object({ name: z.string() });
		const result = await idx.search(Subset, "alice");
		// Fields are filtered/validated against the subset schema:
		expect(result.hits[0].fields).toEqual({ name: "Alice" });
	});

	it("returnStored is set when a Zod schema is present", async () => {
		const fetch = mockFetch(userResultResponse(VALID_USER));
		const idx = new RemoteIndex("http://srv", "users", { schema: UserSchema, fetch });
		await idx.search({ query: "alice" });
		const body = fetch._calls[0].body;
		expect(body.return_stored).toBe(true);
	});
});

// ── No-schema behavior preserved ─────────────────────────────────────────────

describe("RemoteIndex Zod path: no-schema backward compat", () => {
	it("search() without a stored schema returns SearchResult (fields: unknown)", async () => {
		const fetch = mockFetch({
			body: {
				total_hits_estimate: 1,
				hits: [{ doc_id: "1", score: 1, fields: { foo: "bar" } }],
				aggregations: {},
			},
		});
		const idx = new RemoteIndex("http://srv", "users", { fetch });
		const result = await idx.search("anything");
		expect(result.hits[0].fields).toEqual({ foo: "bar" });
	});

	it("add() without a stored schema still runs basic validation", async () => {
		const fetch = mockFetch(BULK_RESPONSE);
		const idx = new RemoteIndex("http://srv", "users", { fetch });
		await expect(idx.add("not an object")).rejects.toThrow();
		expect(fetch._calls).toHaveLength(0);
	});
});
