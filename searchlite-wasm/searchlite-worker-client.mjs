function structuredError(type, reason) {
  return { type, reason };
}

function normalizeError(err) {
  if (err && typeof err === "object") {
    if (typeof err.type === "string" && typeof err.reason === "string") {
      return err;
    }
    if (typeof err.message === "string") {
      return structuredError("worker_error", err.message);
    }
  }
  if (typeof err === "string") {
    return structuredError("worker_error", err);
  }
  return structuredError("worker_error", String(err));
}

export function supportsModuleWorkers() {
  try {
    if (typeof Worker === "undefined") {
      return false;
    }
    const blob = new Blob([""], { type: "text/javascript" });
    const url = URL.createObjectURL(blob);
    const worker = new Worker(url, { type: "module" });
    worker.terminate();
    URL.revokeObjectURL(url);
    return true;
  } catch (_err) {
    return false;
  }
}

export class SearchliteWorkerClient {
  constructor(workerUrl = new URL("./searchlite-demo-worker.mjs", import.meta.url)) {
    this.workerUrl = workerUrl;
    this.worker = null;
    this.seq = 0;
    this.pending = new Map();
    this.config = null;
    this.ready = false;
  }

  async initIndex(dbName, schemaJson, storage = "indexeddb") {
    this.config = { dbName, schemaJson, storage };
    this.ready = false;
    await this.#call("init_index", this.config);
    this.ready = true;
  }

  async addDocuments(docs) {
    await this.#ensureReady();
    return this.#call("add_documents", { docs });
  }

  async searchRequest(request, options = {}) {
    await this.#ensureReady();
    const timeoutMs =
      typeof options.timeoutMs === "number" ? options.timeoutMs : null;
    if (
      timeoutMs !== null &&
      (!Number.isFinite(timeoutMs) || timeoutMs < 0)
    ) {
      throw structuredError(
        "invalid_timeout",
        "timeoutMs must be a non-negative finite number"
      );
    }
    const delayMs = typeof options.delayMs === "number" ? options.delayMs : null;
    if (delayMs !== null && (!Number.isFinite(delayMs) || delayMs < 0)) {
      throw structuredError(
        "invalid_argument",
        "delayMs must be a non-negative finite number"
      );
    }
    return this.#call(
      "search_request",
      { request, timeoutMs, delayMs },
      { signal: options.signal, timeoutMs }
    );
  }

  async resetIndex() {
    await this.#ensureReady();
    return this.#call("reset_index", {});
  }

  async flushStorage() {
    await this.#ensureReady();
    return this.#call("flush_storage", {});
  }

  async storageUsage() {
    return this.#call("storage_usage", {});
  }

  async dispose() {
    this.ready = false;
    this.config = null;
    this.#terminateWorker("worker_disposed", true, "worker disposed");
  }

  async #ensureReady() {
    if (this.ready) {
      return;
    }
    if (!this.config) {
      throw structuredError(
        "index_not_initialized",
        "worker index is not initialized"
      );
    }
    await this.initIndex(
      this.config.dbName,
      this.config.schemaJson,
      this.config.storage
    );
  }

  #ensureWorker() {
    if (this.worker) {
      return this.worker;
    }
    const worker = new Worker(this.workerUrl, { type: "module" });
    worker.onmessage = (event) => {
      const msg = event.data || {};
      const pending = this.pending.get(msg.id);
      if (!pending) {
        return;
      }
      this.pending.delete(msg.id);
      pending.cleanup();
      if (msg.ok) {
        pending.resolve(msg.payload);
      } else {
        pending.reject(normalizeError(msg.error));
      }
    };
    worker.onerror = (event) => {
      const reason = event?.message || "worker runtime error";
      this.#terminateWorker("worker_crashed", true, reason);
    };
    this.worker = worker;
    return worker;
  }

  #terminateWorker(type, rejectPending, reason = "worker terminated") {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.ready = false;
    const pendingEntries = Array.from(this.pending.values());
    this.pending.clear();
    const error = structuredError(type, reason);
    for (const pending of pendingEntries) {
      pending.cleanup();
      if (rejectPending) {
        pending.reject(error);
      }
    }
  }

  #call(action, payload, options = {}) {
    const worker = this.#ensureWorker();
    const id = ++this.seq;
    const signal = options.signal || null;
    const timeoutMs =
      typeof options.timeoutMs === "number" ? options.timeoutMs : null;
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(structuredError("aborted", "operation aborted by AbortSignal"));
        return;
      }
      let timer = null;
      let abortHandler = null;
      const cleanup = () => {
        if (timer !== null) {
          clearTimeout(timer);
          timer = null;
        }
        if (abortHandler) {
          signal?.removeEventListener("abort", abortHandler);
          abortHandler = null;
        }
      };
      if (timeoutMs !== null) {
        timer = setTimeout(() => {
          this.pending.delete(id);
          cleanup();
          this.#terminateWorker(
            "worker_restarted",
            true,
            "worker restarted after timeout"
          );
          reject(
            structuredError(
              "timeout",
              `operation exceeded timeout_ms=${timeoutMs}`
            )
          );
        }, timeoutMs);
      }
      if (signal) {
        abortHandler = () => {
          this.pending.delete(id);
          cleanup();
          this.#terminateWorker(
            "worker_restarted",
            true,
            "worker restarted after abort"
          );
          reject(structuredError("aborted", "operation aborted by AbortSignal"));
        };
        signal.addEventListener("abort", abortHandler, { once: true });
      }
      this.pending.set(id, { resolve, reject, cleanup });
      try {
        worker.postMessage({ id, action, payload });
      } catch (err) {
        this.pending.delete(id);
        cleanup();
        reject(normalizeError(err));
      }
    });
  }
}
