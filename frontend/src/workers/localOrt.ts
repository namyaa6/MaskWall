import { env } from "@huggingface/transformers";

/**
 * Load the ONNX Runtime WebAssembly files from this site (/ort/, copied by scripts/copy_ort.mjs) instead of Transformers.js's
 * default, cdn.jsdelivr.net. Must run before the first model is loaded. Called by both model workers.
 */
export function serveOrtLocally(): void {
  const wasm = (env.backends.onnx as { wasm?: { wasmPaths?: unknown } }).wasm;
  if (wasm) wasm.wasmPaths = new URL(`${import.meta.env.BASE_URL}ort/`, self.location.origin).href;
}
