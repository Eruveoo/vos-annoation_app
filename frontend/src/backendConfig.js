/**
 * Backend always at localhost:12212 on the machine running the browser.
 *
 * Use an SSH tunnel or local proxy in a terminal (no frontend config):
 *
 *   Puhti:  ssh -N -L 12212:r02g03.bullx:12212 gregormi@puhti.csc.fi
 *   WSL + localhost.run on Mac:
 *           scripts/mac-localhost-run-forward.sh https://xxxx.lhr.life
 *
 * Optional override: VITE_API_URL in frontend/.env (usually not needed).
 */
const DEFAULT_BACKEND = "http://127.0.0.1:12212";

function normalizeBackendUrl(url) {
  return String(url).trim().replace(/\/$/, "");
}

const fromEnv = import.meta.env.VITE_API_URL;
export const BACKEND =
  fromEnv && String(fromEnv).trim() ? normalizeBackendUrl(fromEnv) : DEFAULT_BACKEND;

if (import.meta.env.DEV) {
  console.info(`[API] Backend: ${BACKEND}`);
}
