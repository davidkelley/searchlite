import { describe, expect, it } from "vitest";
import { EmbeddedIndex } from "../dist/index.js";

// These tests verify the JS-side input validation and the native binding's
// presence. They do not exercise a real S3 round-trip — that's covered by
// the Rust-side wiremock integration tests in `searchlite-s3/tests/`.
//
// `EmbeddedIndex.fromS3` is async (network I/O must not block the event
// loop), so input-validation failures surface as promise rejections.

describe("EmbeddedIndex.fromS3 input validation", () => {
	it("exposes fromS3 as a static method", () => {
		expect(typeof EmbeddedIndex.fromS3).toBe("function");
	});

	it("rejects on missing config", async () => {
		await expect(EmbeddedIndex.fromS3(undefined)).rejects.toThrowError(
			/s3Config must be an object/,
		);
	});

	it("rejects on non-object config", async () => {
		await expect(EmbeddedIndex.fromS3("oops")).rejects.toThrowError(/s3Config must be an object/);
	});

	it("rejects on missing bucket", async () => {
		await expect(EmbeddedIndex.fromS3({})).rejects.toThrowError(
			/bucket must be a non-empty string/,
		);
	});

	it("rejects on empty bucket", async () => {
		await expect(EmbeddedIndex.fromS3({ bucket: "" })).rejects.toThrowError(
			/bucket must be a non-empty string/,
		);
	});

	it("rejects on whitespace-only bucket", async () => {
		// Without trim() the JS-side check would let this through, and
		// the AWS SDK would surface an opaque error deep in request
		// signing rather than a clean validation failure at the seam.
		await expect(EmbeddedIndex.fromS3({ bucket: "   " })).rejects.toThrowError(
			/bucket must be a non-empty string/,
		);
	});

	it("treats null credentials as 'use AWS chain' (no TypeError)", async () => {
		// `JSON.parse` output commonly turns missing fields into `null`.
		// Before the `!= null` fix this dereferenced `null.accessKeyId`
		// for a confusing TypeError. Now it should fall through to the
		// LoadFromEnv credential path and only fail on the network
		// reach to the unreachable endpoint.
		await expect(
			EmbeddedIndex.fromS3({
				bucket: "smoke-test-bucket",
				region: "us-east-1",
				endpointUrl: "http://127.0.0.1:1",
				forcePathStyle: true,
				// @ts-expect-error — exercising the null path explicitly
				credentials: null,
			}),
		).rejects.toThrow(/^(?!.*Cannot read properties of null).*/);
	});

	it("rejects on non-object credentials", async () => {
		await expect(
			EmbeddedIndex.fromS3({
				bucket: "valid-bucket",
				// @ts-expect-error — intentional bad value
				credentials: "not-an-object",
			}),
		).rejects.toThrowError(/credentials must be an object/);
	});

	it("rejects on invalid checksumPolicy", async () => {
		// Reaches the native layer where the policy string is parsed.
		await expect(
			EmbeddedIndex.fromS3({
				bucket: "valid-bucket",
				region: "us-east-1",
				// @ts-expect-error — intentional bad value
				checksumPolicy: "garbage",
				credentials: { accessKeyId: "x", secretAccessKey: "y" },
			}),
		).rejects.toThrowError(/invalid checksumPolicy/);
	});

	it("rejects when options.schema is not a Zod index schema", async () => {
		await expect(
			EmbeddedIndex.fromS3(
				{ bucket: "valid-bucket" },
				// @ts-expect-error — plain object isn't a Zod schema
				{ schema: { not: "zod" } },
			),
		).rejects.toThrowError(/Zod index schema/);
	});

	it("rejects with a network/auth error against an unreachable endpoint", async () => {
		// Pointing at localhost on a port that nothing is listening on
		// surfaces a connection error from the AWS SDK during the first
		// HEAD on MANIFEST.json. This proves the wiring reaches the
		// network layer; what matters is that we get a thrown error.
		await expect(
			EmbeddedIndex.fromS3({
				bucket: "smoke-test-bucket",
				region: "us-east-1",
				endpointUrl: "http://127.0.0.1:1",
				forcePathStyle: true,
				credentials: { accessKeyId: "x", secretAccessKey: "y" },
			}),
		).rejects.toThrow();
	});
});
