import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end test against the LOCAL backend (start it first, see backend/README.md):
 *   .\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --app-dir backend --port 8000
 * then: cd frontend; npm run e2e
 * Two modes, both on port 5173 (the origin the backend's default CORS setting allows):
 *   npm run e2e       production build via `vite preview`
 *   npm run e2e:dev   the Vite DEV server (React StrictMode, effect-return warnings, the dev-only /__crash route). The production
 *                     build hid a crash that only dev mode shows, so both must pass (npm run e2e:all).
 * No API keys are needed: the chat test sends an attack, which the firewall blocks before any AI is called.
 * Needs network once: the name model (about 66 MB) is downloaded from the Hugging Face Hub on first use.
 */
export default defineConfig({
  testDir: "e2e",
  timeout: 5 * 60_000,
  expect: { timeout: 30_000 },
  retries: 0,
  reporter: [["list"]],
  use: { baseURL: "http://localhost:5173", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: process.env.E2E_MODE === "dev" ? "npx vite --port 5173 --strictPort" : "npm run build && npx vite preview --port 5173 --strictPort",
    url: "http://localhost:5173",
    reuseExistingServer: true,
    timeout: 180_000,
  },
});
