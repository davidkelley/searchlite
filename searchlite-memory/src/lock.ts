import { createRequire } from "node:module";

// proper-lockfile is CommonJS; load it via createRequire for the same
// interop-robustness reasons as searchlite-js (see src/searchlite.ts).
const nodeRequire = createRequire(import.meta.url);
const lockfile = nodeRequire("proper-lockfile") as typeof import("proper-lockfile");

export interface LockOptions {
	/** Age (ms) after which an orphaned lock is reclaimed. Default ~30s. */
	staleMs: number;
	/** Retry attempts under contention (window must stay below staleMs). */
	retries: number;
	/** Bypass locking entirely (e.g. NFS where O_EXCL/mkdir is unreliable). */
	disabled?: boolean;
}

/**
 * Run `fn` while holding a cross-process advisory lock on `lockPath`. The lock
 * spans the whole critical section (mutation or rebuild); the slow embed is
 * computed by the caller BEFORE entering. `realpath:false` lets us lock a path
 * whose parent exists without requiring the target file to pre-exist.
 */
export async function withLock<T>(
	lockPath: string,
	opts: LockOptions,
	fn: () => Promise<T>,
): Promise<T> {
	if (opts.disabled) return fn();
	const release = await lockfile.lock(lockPath, {
		stale: opts.staleMs,
		realpath: false,
		retries: { retries: opts.retries, minTimeout: 100, maxTimeout: 1000 },
	});
	try {
		return await fn();
	} finally {
		await release();
	}
}
