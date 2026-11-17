// Copies the SERVED injection model (backend/model/minilm, made by backend/scripts/prepare_model.py) into
// frontend/public/models/maskwall-minilm/ in the layout Transformers.js expects. The copy is gitignored (binaries are
// never committed); Day 4 hosts the same files on the Hugging Face Hub.
//
//   cd frontend; node scripts/prepare_browser_model.mjs
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");
const served = join(repo, "backend", "model", "minilm");
const metaPath = join(served, "model_meta.json");
if (!existsSync(metaPath)) {
  console.error(`${metaPath} not found: run backend/scripts/prepare_model.py minilm first`);
  process.exit(1);
}
const meta = JSON.parse(readFileSync(metaPath, "utf8"));
const ckpt = join(repo, "models", "ckpt", meta.source);
const out = join(repo, "frontend", "public", "models", "maskwall-minilm");
mkdirSync(join(out, "onnx"), { recursive: true });

copyFileSync(join(served, meta.onnx_file), join(out, "onnx", "model_quantized.onnx"));
copyFileSync(join(served, meta.tokenizer_file), join(out, "tokenizer.json"));
copyFileSync(metaPath, join(out, "model_meta.json"));

// config.json and tokenizer_config.json come from the training checkpoint; if it is absent, write the minimum needed
const cfgSrc = join(ckpt, "config.json");
if (existsSync(cfgSrc)) copyFileSync(cfgSrc, join(out, "config.json"));
else writeFileSync(join(out, "config.json"), JSON.stringify({ model_type: "bert", id2label: { 0: "benign", 1: "attack" } }, null, 2));
const tcSrc = join(ckpt, "tokenizer_config.json");
if (existsSync(tcSrc)) copyFileSync(tcSrc, join(out, "tokenizer_config.json"));
else writeFileSync(join(out, "tokenizer_config.json"), JSON.stringify({ tokenizer_class: "BertTokenizer", do_lower_case: true }, null, 2));

const block = Math.max(meta.thr_f1, meta.thr_fpr1pct);
const warn = Math.min(meta.thr_f1, meta.thr_fpr1pct);
console.log(`copied ${meta.name} (${meta.source}, block ${block.toFixed(3)}, warn ${warn.toFixed(3)}) -> ${out}`);
