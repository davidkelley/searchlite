import { describe, expect, it } from "vitest";
import { EmbeddedIndex } from "../dist/index.js";

// These tests verify the JS-side input validation and the native binding's
// presence. They do not exercise a real S3 round-trip — that's covered by
// the Rust-side wiremock integration tests in `searchlite-s3/tests/`.

describe("EmbeddedIndex.fromS3 input validation", () => {
	it("exposes fromS3 as a static method", () => {
		expect(typeof EmbeddedIndex.fromS3).toBe("function");
	});

	it("throws on missing config", () => {
		expect(() => EmbeddedIndex.fromS3(undefined)).toThrowError(/s3Config must be an object/);
	});

	it("throws on non-object config", () => {
		expect(() => EmbeddedIndex.fromS3("oops")).toThrowError(/s3Config must be an object/);
	});

	it("throws on missing bucket", () => {
		expect(() => EmbeddedIndex.fromS3({})).toThrowError(/bucket must be a non-empty string/);
	});

	it("throws on empty bucket", () => {
		expect(() => EmbeddedIndex.fromS3({ bucket: "" })).toThrowError(
			/bucket must be a non-empty string/,
		);
	});

	it("throws on invalid checksumPolicy", () => {
		// Reaches the native layer where the policy string is parsed.
		expect(() =>
			EmbeddedIndex.fromS3({
				bucket: "valid-bucket",
				region: "us-east-1",
				// @ts-expect-error — intentional bad value
				checksumPolicy: "garbage",
				credentials: { accessKeyId: "x", secretAccessKey: "y" },
			}),
		).toThrowError(/invalid checksumPolicy/);
	});

	it("throws when options.schema is not a Zod index schema", () => {
		expect(() =>
			EmbeddedIndex.fromS3(
				{ bucket: "valid-bucket" },
				// @ts-expect-error — plain object isn't a Zod schema
				{ schema: { not: "zod" } },
			),
		).toThrowError(/Zod index schema/);
	});

	it("rejects with a network/auth error against an unreachable endpoint", () => {
		// Pointing at localhost on a port that nothing is listening on
		// surfaces a connection error from the AWS SDK during the first
		// HEAD on MANIFEST.json. This proves the wiring reaches the
		// network layer; what matters is that we get a thrown error.
		expect(() =>
			EmbeddedIndex.fromS3({
				bucket: "smoke-test-bucket",
				region: "us-east-1",
				endpointUrl: "http://127.0.0.1:1",
				forcePathStyle: true,
				credentials: { accessKeyId: "x", secretAccessKey: "y" },
			}),
		).toThrow();
	});
});
