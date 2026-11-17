// Copies the ONNX Runtime WebAssembly files that Transformers.js needs into public/ort/ so the site serves them itself.
// Without this, Transformers.js fetches them from cdn.jsdelivr.net (a third party that would see every visitor's IP).
// Runs automatically before `npm run dev` and `npm run build`. public/ort/ is gitignored (it comes from node_modules).
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "@huggingface", "transformers", "dist");
const out = join(root, "public", "ort");
mkdirSync(out, { recursive: true });
for (const f of ["ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"]) copyFileSync(join(src, f), join(out, f));
console.log(`copied ONNX Runtime WASM files -> ${out}`);
