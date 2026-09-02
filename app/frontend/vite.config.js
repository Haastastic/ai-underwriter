import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the app talks to the API through this proxy, so the browser
// makes same-origin requests and no CORS round-trip is needed. Point it at a
// non-default backend with VITE_API_TARGET. A production build instead reads
// VITE_API_BASE_URL at build time (see src/api.js) and the API must then allow
// the app's origin via AIU_CORS_ORIGINS.
export default defineConfig(({ mode }) => {
  const target = process.env.VITE_API_TARGET || "http://localhost:8000";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
    build: { outDir: "dist" },
    // keep `mode` referenced so custom modes (e.g. --mode staging) are valid
    define: { __APP_MODE__: JSON.stringify(mode) },
  };
});
