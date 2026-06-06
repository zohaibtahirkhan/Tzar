import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Tauri dev server — prevent vite from obscuring rust errors
  clearScreen: false,
  server: {
    port: 3000,
    strictPort: true,
    proxy: {
      // Proxy API calls to FastAPI during dev (no CORS issues)
      "/chat": "http://127.0.0.1:8000",
      "/memory": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/tool": "http://127.0.0.1:8000",
      "/settings": "http://127.0.0.1:8000",
    },
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    // Tauri uses Chromium on Windows and WebKit on macOS/Linux
    target: ["es2021", "chrome105", "safari15"],
    minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
    sourcemap: !!process.env.TAURI_DEBUG,
  },
});
