import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

function buildRemoteProxy(env) {
  const target = (env.VITE_API_URL || "").trim().replace(/\/$/, "");
  if (!target || !/^https?:\/\//.test(target)) return undefined;

  const user = (env.VITE_API_USER || "").trim();
  const password = (env.VITE_API_PASSWORD || "").trim();

  return {
    "/api": {
      target,
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ""),
      configure: (proxy) => {
        proxy.on("proxyReq", (proxyReq) => {
          proxyReq.setHeader("X-Pinggy-No-Screen", "true");
          if (user && password) {
            const token = Buffer.from(`${user}:${password}`).toString("base64");
            proxyReq.setHeader("Authorization", `Basic ${token}`);
          }
        });
      },
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const proxy = buildRemoteProxy(env);

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy,
    },
    preview: {
      proxy,
    },
  };
});
