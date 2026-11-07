import { defineConfig } from "vitest/config";

export default defineConfig({
	test: {
		// Confine discovery to this package's tests (the repo root may contain
		// git worktrees / other packages with their own test files).
		root: import.meta.dirname,
		include: ["test/**/*.test.ts"],
		exclude: ["node_modules/**", "dist/**", "**/.claude/**", "**/.worktrees/**"],
	},
});
