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
  test: {
    include: ["src/**/*.test.ts", "scripts/**/*.test.ts"],
    environment: "node",
  },
});
