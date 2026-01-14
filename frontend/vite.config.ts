import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [tailwindcss(), react()],
  preview: {
    port: 3800,
    strictPort: true,
  },
  server: {
    port: 3800,
    strictPort: true,
    host: true,
    origin: "http://0.0.0.0:3800",
  },
});
