import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/compare": "http://localhost:8080",
      "/query": "http://localhost:8080",
      "/results": "http://localhost:8080",
      "/health": "http://localhost:8080",
      "/ready": "http://localhost:8080",
      "/debug": "http://localhost:8080",
    },
  },
});
