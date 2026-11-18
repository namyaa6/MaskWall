/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // the PII tests read the backend's shared fixture (backend/tests/fixtures/pii_cases.json)
    fs: { allow: [".."] },
  },
  worker: { format: "es" },
  // Transformers.js ships its own ESM build and loads onnxruntime-web dynamically; pre-bundling it breaks the WASM loader
  optimizeDeps: { exclude: ["@huggingface/transformers"] },
  test: {
    include: ["src/**/*.test.ts", "scripts/**/*.test.ts", "eslint-rules/**/*.test.ts"],
    exclude: ["e2e/**", "node_modules/**"],
    environment: "node",
  },
});
