"""Paths and runtime constants (config/paths.json + config/paths.local.json)."""
import json
import os
from pathlib import Path
from typing import Dict, Any
import logging


# IMPORTANT: use uvicorn logger (so logs show up in uvicorn output)
log = logging.getLogger("uvicorn.error")


# -------------------------
# Config / paths (config/paths.json, overridden by config/paths.local.json)
# -------------------------
BACKEND_ROOT = Path(__file__).resolve().parent.parent
PATHS_CONFIG_FILE = BACKEND_ROOT / "config" / "paths.json"
PATHS_LOCAL_CONFIG_FILE = Path(
    os.environ.get("VOS_PATHS_CONFIG", BACKEND_ROOT / "config" / "paths.local.json")
)


def _load_paths_config() -> Dict[str, Any]:
    with PATHS_CONFIG_FILE.open(encoding="utf-8") as f:
        config = json.load(f)
    if PATHS_LOCAL_CONFIG_FILE.exists():
        with PATHS_LOCAL_CONFIG_FILE.open(encoding="utf-8") as f:
            local = json.load(f)
        unknown = set(local) - set(config)
        if unknown:
            raise ValueError(f"{PATHS_LOCAL_CONFIG_FILE}: unknown keys {sorted(unknown)}")
        config.update(local)
        log.info(f"Loaded path overrides from {PATHS_LOCAL_CONFIG_FILE}")
    return config


def _config_path(key: str) -> Path:
    """Path from config; relative paths are resolved against the backend folder."""
    path = Path(os.path.expandvars(os.path.expanduser(str(PATHS_CONFIG[key]))))
    return path if path.is_absolute() else BACKEND_ROOT / path


PATHS_CONFIG = _load_paths_config()

XMEM_REPO = _config_path("xmem_repo")
XMEM_MODEL = _config_path("xmem_model")
VIDEO_NAME = "video1"
JPEG_QUALITY = 90
MASK_THRESHOLD = 0.5

RUNS_ROOT = _config_path("runs_root")
try:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
except OSError as e:
    raise RuntimeError(
        f"runs_root {RUNS_ROOT} is not writable ({e}); set runs_root in {PATHS_LOCAL_CONFIG_FILE}"
    ) from e
log.info(f"Using runs directory: {RUNS_ROOT}")


def config_dir(key: str) -> Path:
    """Writable directory from config (torch_cache_dir, tmp_dir), created on first use."""
    path = _config_path(key)
    path.mkdir(parents=True, exist_ok=True)
    return path

LOG_EVERY_FRAMES_EXTRACT = 200
LOG_EVERY_FRAMES_RENDER = 200
