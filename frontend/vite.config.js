import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to FastAPI, so the browser never makes a
// cross-origin request. In production FastAPI serves dist/, so the app is
// single-origin there too. No CORS configuration is required either way.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://127.0.0.1:8000",
        changeOrigin: false,
      },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
