import { describe, expect, it } from "vitest";
// Guards the CJS->ESM interop: the `./searchlite.js` helper loads the
// (CommonJS) searchlite-js binding via createRequire, which must work
// identically under vitest and real Node. If the binding fails to load, this
// test fails loudly here rather than deep inside the store.
import { EmbeddedIndex } from "../src/searchlite.js";

describe("searchlite-js interop (createRequire)", () => {
	it("exposes the EmbeddedIndex constructor", () => {
		expect(typeof EmbeddedIndex).toBe("function");
	});
});
