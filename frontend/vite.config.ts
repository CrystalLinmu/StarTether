import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

export default defineConfig(() => ({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    hmr: process.env.DISABLE_HMR !== "true",
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/sessions": "http://127.0.0.1:8000",
      "/documents": "http://127.0.0.1:8000",
      "/ingest": "http://127.0.0.1:8000",
      "/chat": "http://127.0.0.1:8000",
      "/folders": "http://127.0.0.1:8000",
      "/api": "http://127.0.0.1:8000",
    },
  },
}));
