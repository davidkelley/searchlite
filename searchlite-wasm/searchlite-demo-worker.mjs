import initWasm, { Searchlite } from "./pkg/searchlite_wasm.js";

let wasmReady = false;
let index = null;
let currentConfig = null;

function normalizeError(err) {
  if (err && typeof err === "object") {
    if (typeof err.type === "string" && typeof err.reason === "string") {
      return err;
    }
    if (typeof err.message === "string") {
      return { type: "worker_error", reason: err.message };
    }
  }
  if (typeof err === "string") {
    return { type: "worker_error", reason: err };
  }
  return { type: "worker_error", reason: String(err) };
}

function respondOk(id, payload) {
  self.postMessage({ id, ok: true, payload });
}

function respondErr(id, err) {
  self.postMessage({ id, ok: false, error: normalizeError(err) });
}

async function ensureWasm() {
  if (wasmReady) {
    return;
  }
  await initWasm();
  wasmReady = true;
}

async function ensureIndex() {
  if (index) {
    return;
  }
  if (!currentConfig) {
    throw { type: "index_not_initialized", reason: "worker index is not initialized" };
  }
  index = await Searchlite.init(
    currentConfig.dbName,
    currentConfig.schemaJson,
    currentConfig.storage
  );
}

async function handleInit(payload) {
  const dbName = String(payload?.dbName || "");
  const schemaJson = String(payload?.schemaJson || "");
  const storage = payload?.storage || "indexeddb";
  if (!dbName) {
    throw { type: "invalid_argument", reason: "dbName is required" };
  }
  if (!schemaJson) {
    throw { type: "invalid_argument", reason: "schemaJson is required" };
  }
  currentConfig = { dbName, schemaJson, storage };
  index = await Searchlite.init(dbName, schemaJson, storage);
  return { db_name: dbName };
}

async function handleSearch(payload) {
  await ensureIndex();
  const request = payload?.request ?? {};
  const timeoutMs =
    typeof payload?.timeoutMs === "number" ? payload.timeoutMs : null;
  const delayMs =
    typeof payload?.delayMs === "number" && payload.delayMs > 0
      ? payload.delayMs
      : 0;
  if (delayMs > 0) {
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
  return index.search_request_value_controlled(request, null, timeoutMs);
}

async function handleMessage(message) {
  const id = message?.id;
  const action = message?.action;
  const payload = message?.payload || {};
  await ensureWasm();
  switch (action) {
    case "init_index":
      return handleInit(payload);
    case "add_documents":
      await ensureIndex();
      index.add_documents(payload.docs || []);
      await index.commit();
      return { added: Array.isArray(payload.docs) ? payload.docs.length : 0 };
    case "search_request":
      return handleSearch(payload);
    case "flush_storage":
      await ensureIndex();
      await index.flush_storage();
      return { flushed: true };
    case "storage_usage":
      return Searchlite.storage_usage();
    case "reset_index":
      if (!currentConfig) {
        throw { type: "index_not_initialized", reason: "worker index is not initialized" };
      }
      await Searchlite.clear_index(currentConfig.dbName);
      index = await Searchlite.init(
        currentConfig.dbName,
        currentConfig.schemaJson,
        currentConfig.storage
      );
      return { reset: true };
    default:
      throw { type: "invalid_action", reason: `unknown worker action: ${action}` };
  }
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  try {
    const payload = await handleMessage(msg);
    respondOk(msg.id, payload);
  } catch (err) {
    respondErr(msg.id, err);
  }
};
