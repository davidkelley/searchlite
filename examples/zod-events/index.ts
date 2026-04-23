/**
 * Zod-native example: event log with epoch-ms timestamps.
 *
 * Demonstrates the "no date kind" pattern: searchlite-core has no native
 * date type, so timestamps are modeled as `z.number().int()` storing epoch
 * milliseconds. Convert to/from JavaScript `Date` at your application
 * boundary — the schema stays a simple integer field that works with range
 * filters and sort-by-time queries out of the box.
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

// Helper: convert Date → epoch-ms integer with Zod's brand.
// You could also do this with a plain preprocessor; brand stays transparent
// to the compiler so the index field is still an integer.
const EpochMs = sl.integer({ stored: true }).brand<"EpochMs">();

const EventSchema = sl.index(
	z.object({
		id: z.string().uuid(),
		type: z.enum(["login", "logout", "click", "error", "purchase"]),
		actorId: z.string().uuid(),
		message: z.string(),
		// Epoch-ms timestamp. The Rust side sees an integer field.
		occurredAt: EpochMs,
		receivedAt: EpochMs,
		// Severity as keyword so we can filter cheaply.
		severity: z.enum(["info", "warn", "error"]),
	}),
	{ docIdField: "id" },
);

type Event = z.infer<typeof EventSchema>;

// Convenience helpers to bridge Date <-> the index.
const toEpoch = (d: Date) => d.getTime() as z.infer<typeof EpochMs>;
const fromEpoch = (ms: number) => new Date(ms);

async function main() {
	const dir = mkdtempSync(join(tmpdir(), "searchlite-zod-events-"));
	try {
		const index = new EmbeddedIndex<Event>(join(dir, "idx"), {
			schema: EventSchema,
		});

		const now = Date.now();
		const events: Event[] = [
			{
				id: "550e8400-e29b-41d4-a716-446655440101",
				type: "login",
				actorId: "550e8400-e29b-41d4-a716-446655440201",
				message: "User alice logged in",
				occurredAt: toEpoch(new Date(now - 60_000)),
				receivedAt: toEpoch(new Date(now - 59_500)),
				severity: "info",
			},
			{
				id: "550e8400-e29b-41d4-a716-446655440102",
				type: "error",
				actorId: "550e8400-e29b-41d4-a716-446655440201",
				message: "Payment gateway timeout",
				occurredAt: toEpoch(new Date(now - 30_000)),
				receivedAt: toEpoch(new Date(now - 29_800)),
				severity: "error",
			},
			{
				id: "550e8400-e29b-41d4-a716-446655440103",
				type: "purchase",
				actorId: "550e8400-e29b-41d4-a716-446655440202",
				message: "User bob completed checkout for $42.50",
				occurredAt: toEpoch(new Date(now - 10_000)),
				receivedAt: toEpoch(new Date(now - 9_900)),
				severity: "info",
			},
		];

		await index.addMany(events);
		await index.commit();

		// Full-text search
		const r = await index.search("timeout");
		console.log(`Events matching "timeout": ${r.totalHits}`);
		for (const hit of r.hits) {
			const occurred = fromEpoch(hit.fields?.occurredAt as number).toISOString();
			console.log(
				`  • [${hit.fields?.severity}] ${hit.fields?.type} at ${occurred}: ${hit.fields?.message}`,
			);
		}

		// Range filter: events in the last 45 seconds — integer range filters
		// work natively on our epoch-ms fields.
		const cutoff = toEpoch(new Date(now - 45_000));
		const recent = await index.search({
			query: { type: "match_all" },
			filter: {
				I64Range: { field: "occurredAt", min: cutoff, max: toEpoch(new Date(now + 1000)) },
			},
		});
		console.log(`\nEvents in the last 45s: ${recent.totalHits}`);
		for (const hit of recent.hits) {
			const ms = Number(hit.fields?.occurredAt);
			const occurred = fromEpoch(ms).toISOString();
			console.log(`  • ${occurred} — ${hit.fields?.type} (${hit.fields?.severity})`);
		}

		// Validation: z.date() would be rejected at compile time — but we can
		// catch a typo at insert time too.
		try {
			await index.add({
				...events[0],
				id: "550e8400-e29b-41d4-a716-446655440144",
				occurredAt: "not a number" as unknown as z.infer<typeof EpochMs>,
			});
		} catch (err) {
			console.log(`\nZod caught bad timestamp: ${(err as Error).message.split("\n")[0]}`);
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
