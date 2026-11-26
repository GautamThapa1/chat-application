import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the Vite server (port 5173) forwards API and WebSocket
// traffic to FastAPI on port 8000, so cookies stay same-origin.
const backend = process.env.BACKEND_URL || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": backend,
      "/ws": { target: backend, ws: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
  test: {
    environment: "jsdom",
    restoreMocks: true,
  },
});
