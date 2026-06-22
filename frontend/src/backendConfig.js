/**
 * Backend URL and optional HTTP Basic Auth (e.g. Pinggy Pro bauth on the tunnel).
 *
 * In dev, remote URLs are reached via the Vite proxy at /api so Pinggy basic auth
 * does not break browser CORS preflight (OPTIONS).
 */
const DEFAULT_BACKEND = "http://127.0.0.1:12212";

function envTrim(key) {
  const value = import.meta.env[key];
  return value && String(value).trim() ? String(value).trim() : "";
}

function normalizeBackendUrl(url) {
  return String(url).trim().replace(/\/$/, "");
}

const REMOTE_URL = envTrim("VITE_API_URL")
  ? normalizeBackendUrl(envTrim("VITE_API_URL"))
  : null;

const useDevProxy =
  import.meta.env.DEV && REMOTE_URL && /^https?:\/\//.test(REMOTE_URL);

export const BACKEND = useDevProxy ? "/api" : REMOTE_URL || DEFAULT_BACKEND;

const API_USER = envTrim("VITE_API_USER");
const API_PASSWORD = envTrim("VITE_API_PASSWORD");

/** Authorization header for direct (non-proxied) requests. */
export function getBasicAuthHeader() {
  if (useDevProxy) return null;
  if (!API_USER || !API_PASSWORD) return null;
  return `Basic ${btoa(`${API_USER}:${API_PASSWORD}`)}`;
}

/** URL for <video src> / <a href> — proxied in dev, embedded creds otherwise. */
export function withBackendAuth(urlString) {
  if (useDevProxy) {
    if (urlString.startsWith("/api")) return urlString;
    try {
      const url = new URL(urlString);
      return `/api${url.pathname}${url.search}`;
    } catch {
      return urlString;
    }
  }
  if (!API_USER || !API_PASSWORD) return urlString;
  try {
    const url = new URL(urlString);
    url.username = API_USER;
    url.password = API_PASSWORD;
    return url.toString();
  } catch {
    return urlString;
  }
}

if (import.meta.env.DEV) {
  const via = useDevProxy ? ` via proxy → ${REMOTE_URL}` : "";
  const authNote = API_USER && API_PASSWORD ? " (basic auth)" : "";
  console.info(`[API] Backend: ${BACKEND}${via}${authNote}`);
}
