"""VOS annotation backend: FastAPI app entry point (run: uvicorn server:app)."""
import os
import shutil
import subprocess
import time
import shlex
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from vos.config import PATHS_CONFIG, log
from vos.routes import upload, init_ids, behavior, tracking, frames, correction, mask_editing, results


app = FastAPI()

# Add CORS middleware to allow frontend requests
# Allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Progress endpoints polled frequently — omit from request/access logs.
_QUIET_LOG_PATH_PREFIXES = (
    "/prepare_upload_progress/",
    "/track_progress/",
)


class _QuietAccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(prefix in msg for prefix in _QUIET_LOG_PATH_PREFIXES)


# -------------------------
# Request logging middleware
# -------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    path = request.url.path
    quiet = any(path.startswith(prefix) for prefix in _QUIET_LOG_PATH_PREFIXES)
    t0 = time.perf_counter()
    response = await call_next(request)
    if not quiet:
        dt = (time.perf_counter() - t0) * 1000
        log.info(f"{request.method} {path} -> {response.status_code} ({dt:.1f} ms)")
    return response


# -------------------------
# Startup: Auto-load ffmpeg module if needed
# -------------------------
@app.on_event("startup")
async def startup_load_ffmpeg():
    """Startup: quiet access logs for poll endpoints; load ffmpeg if missing."""
    logging.getLogger("uvicorn.access").addFilter(_QuietAccessLogFilter())

    if shutil.which("ffmpeg"):
        return
    
    module = PATHS_CONFIG.get("ffmpeg_module")
    if not module:
        log.warning("ffmpeg not found in PATH (set ffmpeg_module in config/paths.local.json to auto-load an environment module)")
        return

    log.info(f"ffmpeg not found, attempting 'module load {module}'...")
    try:
        result = subprocess.run(
            ["bash", "-c", f"module load {shlex.quote(module)} && echo $PATH"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            os.environ["PATH"] = result.stdout.strip()
            if shutil.which("ffmpeg"):
                log.info(f"Successfully loaded module {module}")
            else:
                log.warning(f"'module load {module}' ran but ffmpeg still not found in PATH")
        else:
            log.warning(f"Could not load module {module}")
    except Exception as e:
        log.debug(f"Could not load module {module}: {e}")


app.include_router(upload.router)
app.include_router(init_ids.router)
app.include_router(behavior.router)
app.include_router(tracking.router)
app.include_router(frames.router)
app.include_router(correction.router)
app.include_router(mask_editing.router)
app.include_router(results.router)


@app.get("/health")
def health():
    """Health check endpoint for testing connectivity"""
    return {"status": "ok", "message": "Backend is running"}
