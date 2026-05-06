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
