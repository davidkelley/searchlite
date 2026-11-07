import { createRequire } from "node:module";

// `searchlite-js` ships CommonJS. Loading it from this ESM package is
// surprisingly environment-dependent: empirically, real Node ESM only exposes
// it via a *default* import (default = module.exports) while bundlers
// (Vite/vitest) only expose it via *named* imports (they honor the `__esModule`
// marker and find no `default`). `createRequire` sidesteps the whole interop
// disagreement by using Node's actual CommonJS resolver — it returns
// `module.exports` (with the named properties) identically under Node and
// under vitest. The type is recovered via `typeof import(...)`, which is a
// type-only construct and emits no runtime import.
const nodeRequire = createRequire(import.meta.url);
const sl = nodeRequire("searchlite-js") as typeof import("searchlite-js");

export const EmbeddedIndex = sl.EmbeddedIndex;

// Type-only re-exports (erased at runtime — safe regardless of interop).
export type { Hit, SearchRequest, SearchResult } from "searchlite-js";
