import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// No dev proxy: the dashboard is static and reads the generated report.json out
// of public/, so what it shows is always what is committed in the repository.
export default defineConfig({ plugins: [react()] });
