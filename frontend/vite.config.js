import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Optional: proxy API through Vite dev server (avoids CORS). Not used by default —
    // api.js calls VITE_API_URL directly. To enable, set VITE_API_URL=/api in .env and
    // uncomment proxy below (target = your backend host).
    // proxy: {
    //   "/api": {
    //     target: "http://192.168.1.46:12212",
    //     changeOrigin: true,
    //     rewrite: (path) => path.replace(/^\/api/, ""),
    //   },
    // },
  },
});
