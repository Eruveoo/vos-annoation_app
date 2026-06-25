import json
import os
import sys
import shutil
import subprocess
import uuid
import time
import base64
import tempfile
import threading
import zipfile
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException, Request, Query, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
import sam3.model_builder
import pkg_resources

# Fix sam3.__file__ if it's None (namespace package issue with editable installs)
import sam3
if not hasattr(sam3, '__file__') or sam3.__file__ is None:
    if hasattr(sam3.model_builder, '__file__') and sam3.model_builder.__file__:
        sam3.__file__ = str(Path(sam3.model_builder.__file__).parent.parent / "__init__.py")

# IMPORTANT: use uvicorn logger (so logs show up in uvicorn output)
log = logging.getLogger("uvicorn.error")

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
    
    log.info("ffmpeg not found, attempting to load module...")
    try:
        # Run module load and capture the updated PATH
        result = subprocess.run(
            ["bash", "-c", "module load ffmpeg && echo $PATH"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # Update the current process's PATH with the module-loaded PATH
            new_path = result.stdout.strip()
            os.environ["PATH"] = new_path
            if shutil.which("ffmpeg"):
                log.info("Successfully loaded ffmpeg module")
            else:
                log.warning("Module load command ran but ffmpeg still not found in PATH")
        else:
            log.warning("Could not load ffmpeg module")
    except Exception as e:
        log.debug(f"Could not load ffmpeg module: {e}")


# -------------------------
# Config / paths
# -------------------------
XMEM_REPO = "./XMem"
XMEM_MODEL = os.path.join(XMEM_REPO, "saves", "XMem.pth")
VIDEO_NAME = "video1"
JPEG_QUALITY = 90
MASK_THRESHOLD = 0.5

# Dynamically detect project number from the current file's path
# This handles different users with different project codes (e.g., 2015338, 2016918, etc.)
def _get_project_number() -> Optional[str]:
    """Extract project number from the server.py file path (e.g., 'project_2015338' -> '2015338')"""
    try:
        # Get the directory containing this script
        server_dir = Path(__file__).resolve().parent
        # Search up the path for a directory matching project_XXXXX pattern
        for parent in [server_dir] + list(server_dir.parents):
            match = re.search(r'project_(\d+)', str(parent))
            if match:
                return match.group(1)
    except Exception as e:
        log.debug(f"Could not extract project number: {e}")
    return None

# Use scratch space (has most room for files):
# - Home: 10GB space, 100K files (9.1G used, 6K files - space tight)
# - Scratch: 4TB space, 1M files (54G used, 210K files - PLENTY OF ROOM!)
# - Projappl: 50GB space, 100K files (21G used, 101K files - FILE LIMIT EXCEEDED!)
# Dynamically construct the scratch path based on the current user's project code
_project_num = _get_project_number()
if _project_num:
    _scratch_path = Path(f"/scratch/project_{_project_num}/vos_annotation_runs")
    log.info(f"Detected project number: {_project_num}")
else:
    _scratch_path = Path("/scratch/vos_annotation_runs")
    log.warning("Could not detect project number from path, using generic scratch path")

try:
    _scratch_path.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT = _scratch_path
    log.info(f"Using scratch directory: {RUNS_ROOT}")
except PermissionError:
    # Fall back to local "runs" directory if scratch is not writable
    RUNS_ROOT = Path("./runs")
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    log.warning(f"Scratch path {_scratch_path} not writable, falling back to local runs directory: {RUNS_ROOT}")


def scratch_subdir(name: str) -> Path:
    """Writable scratch subdirectory for the current project (torch cache, tmp, etc.)."""
    candidates: List[Path] = []
    if _project_num:
        candidates.append(Path(f"/scratch/project_{_project_num}") / name)
    if str(RUNS_ROOT).startswith("/scratch"):
        candidates.append(RUNS_ROOT.parent / name)
    candidates.append(RUNS_ROOT / name)
    last_err: Optional[OSError] = None
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError as e:
            last_err = e
            log.debug(f"scratch_subdir: cannot use {path}: {e}")
    raise PermissionError(last_err or PermissionError(f"No writable scratch path for '{name}'"))


# Progress tracking for upload/prepare operations
# Key: run_id, Value: {"stage": "upload"|"extract", "progress": 0-100, "message": str}
prepare_progress: Dict[str, Dict] = {}

# Progress tracking for tracking operations
# Key: run_id, Value: {"stage": "tracking"|"rendering", "progress": 0-100, "message": str, "current_frame": int, "total_frames": int}
track_progress: Dict[str, Dict] = {}

LOG_EVERY_FRAMES_EXTRACT = 200
LOG_EVERY_FRAMES_RENDER = 200


def ensure_clean_dir(path: Path):
    if path.exists():
        log.info(f"Removing directory: {path}")
        shutil.rmtree(path)
    log.info(f"Creating directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_files_parallel(src_dst_pairs: list, max_workers: int = 8):
    """
    Copy multiple files in parallel using ThreadPoolExecutor.
    
    Args:
        src_dst_pairs: List of (src_path, dst_path) tuples
        max_workers: Maximum number of parallel copy operations
    
    Returns:
        Number of successfully copied files
    """
    def copy_one(src_dst):
        src, dst = src_dst
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            log.warning(f"Failed to copy {src} to {dst}: {e}")
            return False
    
    if not src_dst_pairs:
        return 0
    
    copied = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(copy_one, pair): pair for pair in src_dst_pairs}
        for future in as_completed(futures):
            if future.result():
                copied += 1
    
    return copied


def parse_meta_file(meta_path: Path) -> dict:
    """Parse a meta.txt file, skipping empty lines and lines without '='."""
    if not meta_path.exists():
        return {}
    meta_lines = [
        line.strip().split("=", 1) 
        for line in meta_path.read_text().splitlines() 
        if line.strip() and "=" in line
    ]
    return dict(meta_lines)


def get_annotation_mode(run_dir: Path) -> str:
    meta = parse_meta_file(run_dir / "meta.txt")
    mode = (meta.get("annotation_mode") or "standard").strip().lower()
    return mode if mode in ("standard", "behavior") else "standard"


def update_meta_key(meta_path: Path, key: str, value: str) -> None:
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    meta_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")


# -------------------------
# Behaviour annotation: labels + segment storage (3 dimensions)
# -------------------------
BEHAVIOR_LABEL_NONE = "none"
NOT_VISIBLE_LABEL_ID = "not_visible"
NOT_SEEN_LABEL_ID = "not_seen"

BEHAVIOR_LABELS_ACTIVITY: List[Dict[str, Any]] = [
    {
        "id": "walk",
        "name_fi": "Kävelee",
        "description_fi": (
            "Eläin ottaa useita askeleita siirtyäkseen paikasta toiseen. Eläin liikkuu kohtalaisen "
            "hitaasti siirtäen yhtä jalkaa kerrallaan eteenpäin. Myös peruuttaminen ja kääntyminen."
        ),
    },
    {
        "id": "trot_gallop",
        "name_fi": "Ravaa tai laukkaa",
        "description_fi": (
            "Eläin ottaa useita askeleita siirtyäkseen paikasta toiseen. Eläin liikkuu kohtalaisen "
            "nopeasti tai nopeasti joko symmetrisesti (ravi) tai laukaten."
        ),
    },
    {
        "id": "stand",
        "name_fi": "Seisoo",
        "description_fi": "Eläin seisoo paikoillaan vähintään kolmen jalan ollessa kosketuksissa maahan.",
    },
    {
        "id": "lie_down",
        "name_fi": "Makuulle laskeutuminen",
        "description_fi": (
            "Makuulle laskeutuminen alkaa, kun eläimen kyynärnivel taipuu ja laskeutuu (ennen maahan "
            "kosketusta). Liike päättyy, kun takapuoli on maassa ja etujalka on vedetty alta."
        ),
    },
    {
        "id": "get_up",
        "name_fi": "Ylös nouseminen",
        "description_fi": (
            "Ylös nouseminen alkaa eläimen kohottaessa päätään ja jännittäessä etuosan lihaksia, "
            "päätä heilauttaen eteen ja nostaen takapäätä; lopuksi eläin nousee seisomaan."
        ),
    },
    {
        "id": "abnormal_motion",
        "name_fi": "Epänormaalit liikesarjat",
        "description_fi": (
            "Epänormaali laskeutuminen (takapää ensin, istuva asento) tai nouseminen "
            "(etujalat ensin, istuvasta ponnistus)."
        ),
    },
    {
        "id": "lying",
        "name_fi": "Makaa",
        "description_fi": (
            "Eläimen vartalo lepää maassa alemmanpuoleisen takajalan ja reiden, vatsan ja "
            "etujalkojen tai toisen kyljen varassa."
        ),
    },
    {
        "id": "other_posture",
        "name_fi": "Muu asento",
        "description_fi": "Esim. selkään hyppääminen, kaatuminen tai kompurointi.",
    },
    {
        "id": NOT_VISIBLE_LABEL_ID,
        "name_fi": "Ei näkyvissä",
        "description_fi": "Eläin ei ole kuvassa (poistui ruudusta).",
    },
]

# Label 2: syöminen, hoito, toimijan sosiaalinen/kiima-käyttäytyminen, ei näy
BEHAVIOR_LABELS_LABEL2: List[Dict[str, Any]] = [
    {
        "id": BEHAVIOR_LABEL_NONE,
        "name_fi": "Ei valittu",
        "description_fi": "Ei käyttäytymistä tässä kategoriassa (oletus).",
        "group_fi": "",
    },
    {
        "id": "inactive_ruminate",
        "name_fi": "Toimeton tai märehtii (ei labelia)",
        "description_fi": "Eläin on toimeton tai märehtii; ei muuta Label 2 -käyttäytymistä.",
        "group_fi": "",
    },
    {
        "id": "feed_head_down",
        "name_fi": "Ruokintapöydällä pää alhaalla",
        "description_fi": (
            "Eläin seisoo pää ruokintapöydän vieressä pää ruokintaesteen etupuolella, pää alhaalla "
            "(syö, tutkii rehua ym.). Nopea rehun heittely sisältyy."
        ),
        "group_fi": "Syömiskäyttäytyminen",
    },
    {
        "id": "feed_head_up",
        "name_fi": "Ruokintapöydällä pää ylhäällä",
        "description_fi": (
            "Eläin seisoo pää ruokintapöydän vieressä pää ruokintaesteen etupuolella, pää ylhäällä "
            "(pureskelee, on toimeton ym.)."
        ),
        "group_fi": "Syömiskäyttäytyminen",
    },
    {
        "id": "drink",
        "name_fi": "Juo",
        "description_fi": (
            "Juo, laskee päänsä kuppiin/altaaseen – nostaa sen pois kupista/altaasta. "
            "Sisältää muutamien sekuntien tauot."
        ),
        "group_fi": "Syömiskäyttäytyminen",
    },
    {
        "id": "groom_self",
        "name_fi": "Kehon hoito itse",
        "description_fi": "Nuolee, raapii tai hankaa itseään.",
        "group_fi": "Kehon hoito",
    },
    {
        "id": "scratch_neck_rail",
        "name_fi": "Rapsuttelu karjaharjalla",
        "description_fi": "Kehon rapsuttelu karjaharjaan.",
        "group_fi": "Kehon hoito",
    },
    {
        "id": "scratch_other",
        "name_fi": "Rapsuttelu muuhun",
        "description_fi": "Kehon rapsuttelu muuhun kuin karjaharjaan (parret ym.).",
        "group_fi": "Kehon hoito",
    },
    {
        "id": "social_lick_actor",
        "name_fi": "Sosiaalinen nuoleminen (toimija)",
        "description_fi": "Eläin nuolee toista eläintä yleensä päästä, kaulasta tai hartioista.",
        "group_fi": "Sosiaalinen käyttäytyminen",
    },
    {
        "id": "pushing",
        "name_fi": "Puskeminen",
        "description_fi": (
            "Eläin sysää otsan tai pään ylöspäin suuntautuvalla liikkeellä vasten toisen eläimen "
            "niskaa, hartioita, kylkeä tai takaosaa."
        ),
        "group_fi": "Sosiaalinen käyttäytyminen",
    },
    {
        "id": "displacement_actor",
        "name_fi": "Syrjäyttäminen (toimija)",
        "description_fi": (
            "Eläin puskee tai esim. vartalollaan tönimällä syrjäyttää toisen eläimen pois "
            "ruokintapaikalta, juomakupilta tai makuupaikalta."
        ),
        "group_fi": "Sosiaalinen käyttäytyminen",
    },
    {
        "id": "chin_rest_actor",
        "name_fi": "Leuan lepuuttaminen (toimija)",
        "description_fi": (
            "Eläin testaa toisen lehmän seisomisrefleksiä ennen selkään hyppäämistä painamalla "
            "leukaansa ja kurkkuaan lehmän takapuolen tai selän päälle."
        ),
        "group_fi": "Kiimakäyttäytyminen",
    },
    {
        "id": "mount_actor",
        "name_fi": "Selkään hyppääminen (toimija)",
        "description_fi": (
            "Eläin ponnistaa etujalkansa irti maasta ja nostaa ryntäänsä toisen lehmän selän päälle, "
            "sijoittaen etujalkansa juuri lehmän lonkkakyhmyjen etupuolelle pitäen kiinni lehmästä."
        ),
        "group_fi": "Kiimakäyttäytyminen",
    },
    {
        "id": "other_label2",
        "name_fi": "Muu",
        "description_fi": "Ei ole mitään yllä mainittua (ei toimeton tai märehdi).",
        "group_fi": "Kiimakäyttäytyminen",
    },
    {
        "id": NOT_SEEN_LABEL_ID,
        "name_fi": "Ei näy",
        "description_fi": "Käyttäytymistä ei näe.",
        "group_fi": "",
    },
]

# Label 3: vastaanottajan sosiaalinen ja kiima-käyttäytyminen
BEHAVIOR_LABELS_LABEL3: List[Dict[str, Any]] = [
    {
        "id": BEHAVIOR_LABEL_NONE,
        "name_fi": "Ei valittu",
        "description_fi": "Ei käyttäytymistä tässä kategoriassa (oletus).",
        "group_fi": "",
    },
    {
        "id": "social_lick_receiver",
        "name_fi": "Sosiaalinen nuoleminen (vastaanottaja)",
        "description_fi": (
            "Toinen eläin nuolee eläintä yleensä päästä, kaulasta tai hartioista. "
            "Nuoltavana oleva eläin ojentaa usein kaulaansa ja päätään eteen."
        ),
        "group_fi": "Sosiaalinen käyttäytyminen",
    },
    {
        "id": "displacement_receiver",
        "name_fi": "Syrjäyttäminen (vastaanottaja)",
        "description_fi": (
            "Toinen eläin puskee tai esim. vartalollaan tönimällä syrjäyttää eläimen pois "
            "ruokintapaikalta, juomakupilta tai makuupaikalta. Syrjäytettävä siirtyy noin lehmän mitan pois."
        ),
        "group_fi": "Sosiaalinen käyttäytyminen",
    },
    {
        "id": "chin_rest_receiver",
        "name_fi": "Leuan lepuuttaminen (vastaanottaja)",
        "description_fi": "Toinen eläin lepuuttaa leukaa tämän lehmän selän tai takapuolen päällä.",
        "group_fi": "Kiimakäyttäytyminen",
    },
    {
        "id": "mount_receiver",
        "name_fi": "Selkään hyppääminen (vastaanottaja)",
        "description_fi": "Toinen eläin hyppää tämän lehmän selkään.",
        "group_fi": "Kiimakäyttäytyminen",
    },
]

BEHAVIOR_DIMENSIONS = ("activity", "label2", "label3")
BEHAVIOR_DIMENSION_META: Dict[str, Dict[str, Any]] = {
    "activity": {
        "file": "behavior_labels_activity.json",
        "labels": BEHAVIOR_LABELS_ACTIVITY,
        "default_label": "stand",
        "title_fi": "Label 1: Aktivisuus",
        "required": True,
        "affects_preview": True,
        "hidden_ids": set(),
    },
    "label2": {
        "file": "behavior_labels_label2.json",
        "labels": BEHAVIOR_LABELS_LABEL2,
        "default_label": BEHAVIOR_LABEL_NONE,
        "title_fi": "Label 2: Toimija / syöminen / hoito",
        "required": False,
        "affects_preview": True,
        "hidden_ids": set(),
    },
    "label3": {
        "file": "behavior_labels_label3.json",
        "labels": BEHAVIOR_LABELS_LABEL3,
        "default_label": BEHAVIOR_LABEL_NONE,
        "title_fi": "Label 3: Vastaanottaja",
        "required": False,
        "affects_preview": True,
        "hidden_ids": set(),
    },
}

DEFAULT_BEHAVIOR_LABEL_ID = "stand"
BEHAVIOR_FILE_NAME = "behavior_labels_activity.json"


def _valid_label_ids_for_dimension(dimension: str) -> set:
    return {label["id"] for label in BEHAVIOR_DIMENSION_META[dimension]["labels"]}


def _default_label_for_dimension(dimension: str) -> str:
    return BEHAVIOR_DIMENSION_META[dimension]["default_label"]


def _segment_visible(label_id: str, dimension: str) -> bool:
    if dimension == "activity":
        return label_id != NOT_VISIBLE_LABEL_ID
    if dimension == "label2":
        return label_id not in (BEHAVIOR_LABEL_NONE, NOT_SEEN_LABEL_ID)
    # label3: only receiver behaviours count as active segments
    return label_id != BEHAVIOR_LABEL_NONE


def behavior_label_by_id(label_id: str, dimension: str = "activity") -> Dict[str, Any]:
    for label in BEHAVIOR_DIMENSION_META[dimension]["labels"]:
        if label["id"] == label_id:
            return label
    raise KeyError(label_id)


def behavior_file_path(run_dir: Path, dimension: str = "activity") -> Path:
    return run_dir / BEHAVIOR_DIMENSION_META[dimension]["file"]


def empty_behavior_data(cow_ids: Optional[List[int]] = None, dimension: str = "activity") -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "version": 1,
        "dimension": dimension,
        "segments": [],
        "cow_ids": sorted(cow_ids or []),
    }
    if dimension == "activity":
        data["preview_in_sync"] = True
    return data


def load_behavior_dimension(run_dir: Path, dimension: str) -> Optional[Dict[str, Any]]:
    path = behavior_file_path(run_dir, dimension)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _sort_behavior_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(segments, key=lambda s: (int(s["cow_id"]), int(s["start_frame"])))


def _prune_invalid_behavior_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pruned: List[Dict[str, Any]] = []
    for seg in segments:
        start = int(seg["start_frame"])
        end = seg.get("end_frame")
        if end is not None and int(end) < start:
            continue
        pruned.append(seg)
    return pruned


def _normalize_behavior_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _sort_behavior_segments(_prune_invalid_behavior_segments(segments))


def save_behavior_dimension(run_dir: Path, dimension: str, data: Dict[str, Any]) -> None:
    path = behavior_file_path(run_dir, dimension)
    data = dict(data)
    data["segments"] = _normalize_behavior_segments(data.get("segments", []))
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_behavior_data(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Activity dimension (Label 1); used for golden preview overlay."""
    return load_behavior_dimension(run_dir, "activity")


def save_behavior_data(run_dir: Path, data: Dict[str, Any]) -> None:
    save_behavior_dimension(run_dir, "activity", data)


def mark_behavior_preview_out_of_sync(run_dir: Path) -> None:
    data = load_behavior_dimension(run_dir, "activity")
    if data is None:
        return
    data["preview_in_sync"] = False
    save_behavior_dimension(run_dir, "activity", data)


def mark_behavior_preview_in_sync(run_dir: Path) -> None:
    data = load_behavior_dimension(run_dir, "activity")
    if data is None:
        return
    data["preview_in_sync"] = True
    save_behavior_dimension(run_dir, "activity", data)


def _validate_behavior_label_id(label_id: str, dimension: str) -> None:
    if label_id not in _valid_label_ids_for_dimension(dimension):
        raise ValueError(f"Unknown label_id for {dimension}: {label_id}")
    meta = BEHAVIOR_DIMENSION_META[dimension]
    if meta["required"] and label_id == BEHAVIOR_LABEL_NONE:
        raise ValueError(f"Label 1 (activity) cannot be '{BEHAVIOR_LABEL_NONE}'")


def create_initial_segments(
    cow_ids: List[int],
    start_frame: int,
    labels_by_cow: Dict[int, str],
    dimension: str = "activity",
) -> Dict[str, Any]:
    """One open-ended segment per cow starting at start_frame."""
    segments: List[Dict[str, Any]] = []
    default_label = _default_label_for_dimension(dimension)
    for cow_id in sorted(cow_ids):
        label_id = labels_by_cow.get(cow_id, default_label)
        _validate_behavior_label_id(label_id, dimension)
        segments.append(
            {
                "cow_id": int(cow_id),
                "start_frame": int(start_frame),
                "end_frame": None,
                "label_id": label_id,
                "visible": _segment_visible(label_id, dimension),
            }
        )
    return empty_behavior_data(cow_ids, dimension) | {"segments": segments}


def _pre_detection_label_for_dimension(dimension: str) -> str:
    """Label for frames before a cow first appears in masks (late add_mask / correction)."""
    if dimension == "activity":
        return NOT_VISIBLE_LABEL_ID
    if dimension == "label2":
        return NOT_SEEN_LABEL_ID
    return BEHAVIOR_LABEL_NONE


def register_late_behavior_cows(
    run_dir: Path, cow_ids: List[int], first_visible_frame: int
) -> List[int]:
    """
    Register cows that first appear after frame 0 (e.g. add_mask + apply_correction).
    Adds segments from frame 0 through first_visible_frame - 1 with a pre-detection label,
    then an open-ended segment from first_visible_frame with the dimension default.
    """
    if get_annotation_mode(run_dir) != "behavior":
        return []

    new_cow_ids = sorted({int(c) for c in cow_ids if int(c) > 0})
    if not new_cow_ids:
        return []

    first_visible_frame = int(first_visible_frame)
    registered: List[int] = []

    for dimension in BEHAVIOR_DIMENSIONS:
        data = load_behavior_dimension(run_dir, dimension)
        if data is None:
            data = empty_behavior_data([], dimension)

        existing = {int(c) for c in data.get("cow_ids", [])}
        to_add = [cid for cid in new_cow_ids if cid not in existing]
        if not to_add:
            continue

        segments: List[Dict[str, Any]] = list(data.get("segments", []))
        pre_label = _pre_detection_label_for_dimension(dimension)
        default_label = _default_label_for_dimension(dimension)

        for cow_id in to_add:
            if first_visible_frame > 0:
                segments.append(
                    {
                        "cow_id": cow_id,
                        "start_frame": 0,
                        "end_frame": first_visible_frame - 1,
                        "label_id": pre_label,
                        "visible": _segment_visible(pre_label, dimension),
                    }
                )
            segments.append(
                {
                    "cow_id": cow_id,
                    "start_frame": first_visible_frame,
                    "end_frame": None,
                    "label_id": default_label,
                    "visible": _segment_visible(default_label, dimension),
                }
            )
            registered.append(cow_id)

        data["cow_ids"] = sorted(existing | set(to_add))
        data["segments"] = _normalize_behavior_segments(segments)
        save_behavior_dimension(run_dir, dimension, data)

    registered_unique = sorted(set(registered))
    if registered_unique:
        mark_behavior_preview_out_of_sync(run_dir)
        log.info(
            f"[BEHAVIOR] Registered late cows {registered_unique} "
            f"first_visible_frame={first_visible_frame} run_dir={run_dir.name}"
        )
    return registered_unique


def _find_covering_behavior_segment(
    segments: List[Dict[str, Any]], cow_id: int, frame: int
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for seg in segments:
        if int(seg["cow_id"]) != cow_id:
            continue
        start = int(seg["start_frame"])
        end = seg.get("end_frame")
        if frame < start:
            continue
        if end is not None and frame > int(end):
            continue
        if best is None or start >= int(best["start_frame"]):
            best = seg
    return best


def _next_behavior_segment_start(
    segments: List[Dict[str, Any]], cow_id: int, frame: int
) -> Optional[int]:
    starts = [
        int(s["start_frame"])
        for s in segments
        if int(s["cow_id"]) == cow_id and int(s["start_frame"]) > frame
    ]
    return min(starts) if starts else None


def _append_behavior_segment(
    segments: List[Dict[str, Any]],
    cow_id: int,
    start_frame: int,
    end_frame: Optional[int],
    label_id: str,
    dimension: str,
) -> None:
    segments.append(
        {
            "cow_id": int(cow_id),
            "start_frame": int(start_frame),
            "end_frame": int(end_frame) if end_frame is not None else None,
            "label_id": label_id,
            "visible": _segment_visible(label_id, dimension),
        }
    )


def set_label_from_frame(
    data: Dict[str, Any],
    cow_id: int,
    frame: int,
    label_id: str,
    dimension: str = "activity",
) -> Dict[str, Any]:
    """
    Set label for cow_id from frame onward. Splits an existing segment when frame falls
    in the middle (e.g. A on 0–30, B on 30+, then C at 15 → 0–14 A, 15–29 C, 30+ B).
    """
    _validate_behavior_label_id(label_id, dimension)
    frame = int(frame)
    cow_id = int(cow_id)
    segments: List[Dict[str, Any]] = list(data.get("segments", []))

    for seg in segments:
        if int(seg["cow_id"]) == cow_id and int(seg["start_frame"]) == frame:
            seg["label_id"] = label_id
            seg["visible"] = _segment_visible(label_id, dimension)
            data["segments"] = _normalize_behavior_segments(segments)
            if cow_id not in data.get("cow_ids", []):
                data.setdefault("cow_ids", []).append(cow_id)
                data["cow_ids"] = sorted(data["cow_ids"])
            return data

    covering = _find_covering_behavior_segment(segments, cow_id, frame)
    next_start = _next_behavior_segment_start(segments, cow_id, frame)

    if covering is not None and int(covering["start_frame"]) < frame:
        cover_end = covering.get("end_frame")
        covering["end_frame"] = frame - 1
        if next_start is not None:
            new_end = next_start - 1
        elif cover_end is not None:
            new_end = int(cover_end)
        else:
            new_end = None
        _append_behavior_segment(segments, cow_id, frame, new_end, label_id, dimension)
    else:
        new_end = (next_start - 1) if next_start is not None else None
        _append_behavior_segment(segments, cow_id, frame, new_end, label_id, dimension)

    data["segments"] = _normalize_behavior_segments(segments)
    cow_ids_set = set(data.get("cow_ids", []))
    cow_ids_set.add(cow_id)
    data["cow_ids"] = sorted(cow_ids_set)
    return data


def delete_label_from_frame(
    data: Dict[str, Any],
    cow_id: int,
    frame: int,
    dimension: str = "activity",
) -> Dict[str, Any]:
    """
    Remove a behaviour change that starts at frame (undo split). Extends the previous
    segment to cover the deleted segment's range.
    """
    frame = int(frame)
    cow_id = int(cow_id)
    if frame <= 0:
        raise ValueError("Cannot delete the initial behaviour at frame 0")

    segments: List[Dict[str, Any]] = list(data.get("segments", []))
    target_idx: Optional[int] = None
    for i, seg in enumerate(segments):
        if int(seg["cow_id"]) == cow_id and int(seg["start_frame"]) == frame:
            target_idx = i
            break
    if target_idx is None:
        raise ValueError(f"No behaviour change at frame {frame} for cow {cow_id}")

    deleted = segments.pop(target_idx)
    deleted_end = deleted.get("end_frame")

    prev: Optional[Dict[str, Any]] = None
    for seg in segments:
        if int(seg["cow_id"]) != cow_id:
            continue
        if int(seg["start_frame"]) < frame:
            if prev is None or int(seg["start_frame"]) > int(prev["start_frame"]):
                prev = seg
    if prev is None:
        raise ValueError("Cannot delete: no preceding segment")

    prev["end_frame"] = int(deleted_end) if deleted_end is not None else None
    data["segments"] = _normalize_behavior_segments(segments)
    return data


def get_behavior_label_at_frame(data: Dict[str, Any], cow_id: int, frame: int) -> Optional[str]:
    frame = int(frame)
    cow_id = int(cow_id)
    best: Optional[Dict[str, Any]] = None
    for seg in data.get("segments", []):
        if seg["cow_id"] != cow_id:
            continue
        start = int(seg["start_frame"])
        end = seg.get("end_frame")
        if frame < start:
            continue
        if end is not None and frame > int(end):
            continue
        if best is None or start >= int(best["start_frame"]):
            best = seg
    return best["label_id"] if best else None


def labels_at_frame(data: Dict[str, Any], frame: int) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for cow_id in data.get("cow_ids", []):
        label = get_behavior_label_at_frame(data, int(cow_id), frame)
        if label is not None:
            out[int(cow_id)] = label
    return out


def behavior_overlay_lines_at_frame(run_dir: Path, cow_id: int, frame_idx: int) -> List[str]:
    """Finnish label lines for golden preview: activity, then label2, then label3."""
    if get_annotation_mode(run_dir) != "behavior":
        return []
    lines: List[str] = []
    for dim in BEHAVIOR_DIMENSIONS:
        data = load_behavior_dimension(run_dir, dim)
        if not data:
            continue
        label_id = get_behavior_label_at_frame(data, cow_id, frame_idx)
        if not label_id:
            continue
        if dim == "activity" and label_id == NOT_VISIBLE_LABEL_ID:
            continue
        if dim == "label2" and label_id in (BEHAVIOR_LABEL_NONE, NOT_SEEN_LABEL_ID):
            continue
        if dim == "label3" and label_id == BEHAVIOR_LABEL_NONE:
            continue
        try:
            lines.append(behavior_label_by_id(label_id, dim)["name_fi"])
        except KeyError:
            lines.append(label_id)
    return lines


_UNICODE_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def _load_unicode_font(size_px: int) -> ImageFont.ImageFont:
    for path in _UNICODE_FONT_PATHS:
        p = Path(path)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size_px)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_unicode_text_on_bgr(
    frame_bgr: np.ndarray,
    text: str,
    center_x: int,
    top_y: int,
    font_px: int = 15,
    fill_rgb: Tuple[int, int, int] = (0, 0, 0),
    outline_rgb: Optional[Tuple[int, int, int]] = (255, 255, 255),
    outline_width: int = 1,
) -> np.ndarray:
    if not text:
        return frame_bgr
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_unicode_font(font_px)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    x = int(center_x - text_w / 2)
    y = int(top_y)
    if outline_rgb and outline_width > 0:
        for ox in range(-outline_width, outline_width + 1):
            for oy in range(-outline_width, outline_width + 1):
                if ox == 0 and oy == 0:
                    continue
                draw.text((x + ox, y + oy), text, font=font, fill=outline_rgb)
    draw.text((x, y), text, font=font, fill=fill_rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_cow_overlay_with_behavior(
    frame: np.ndarray,
    mask: np.ndarray,
    cow_id: int,
    color: Tuple[int, int, int],
    behavior_lines: Optional[List[str]] = None,
    overlay_alpha: float = 0.4,
    id_font_scale: float = 0.8,
) -> np.ndarray:
    """Draw mask tint, cow ID, and behaviour labels (stacked) below the ID."""
    if not mask.any():
        return frame

    overlay = frame.copy()
    overlay[mask] = color
    frame = cv2.addWeighted(frame, 1.0 - overlay_alpha, overlay, overlay_alpha, 0)

    ys, xs = np.where(mask)
    cx, cy = int(xs.mean()), int(ys.mean())

    id_text = str(cow_id)
    thickness = 2
    (id_w, id_h), baseline = cv2.getTextSize(
        id_text, cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, thickness
    )
    label_font_px = 14
    label_line_gap = 3
    lines = [ln for ln in (behavior_lines or []) if ln]
    labels_block_h = len(lines) * (label_font_px + label_line_gap) if lines else 0
    stack_gap = 4
    stack_h = id_h + (stack_gap + labels_block_h if lines else 0)
    id_y = cy + id_h // 2 - stack_h // 2 + id_h
    id_x = cx - id_w // 2

    cv2.putText(
        frame,
        id_text,
        (id_x, id_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        id_font_scale,
        (0, 0, 0),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        id_text,
        (id_x, id_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        id_font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    y = id_y + stack_gap
    for line in lines:
        frame = _draw_unicode_text_on_bgr(
            frame,
            line,
            cx,
            y,
            font_px=label_font_px,
            fill_rgb=(0, 0, 0),
            outline_rgb=(255, 255, 255),
            outline_width=1,
        )
        y += label_font_px + label_line_gap

    return frame


GOLDEN_PREVIEW_WITH_LABELS = "golden_preview.mp4"
GOLDEN_PREVIEW_MASKS_ONLY = "golden_preview_masks.mp4"


def golden_preview_video_path(run_dir: Path, masks_only: bool = False) -> Path:
    name = GOLDEN_PREVIEW_MASKS_ONLY if masks_only else GOLDEN_PREVIEW_WITH_LABELS
    return run_dir / "golden" / name


def _finalize_golden_preview_segment(seg_path: Path, fps: float) -> None:
    if not seg_path.exists():
        return
    seg_reencoded = seg_path.parent / f"{seg_path.stem}_reencoded.mp4"
    if _ffmpeg_reencode_video(seg_path, seg_reencoded, fps):
        seg_reencoded.replace(seg_path)


def _append_to_golden_preview_file(preview_path: Path, segment_path: Path, fps: float) -> None:
    if not segment_path.exists():
        return
    ensure_dir(preview_path.parent)
    if preview_path.exists():
        _ffmpeg_concat(preview_path, segment_path, preview_path, fps)
    else:
        preview_path.write_bytes(segment_path.read_bytes())


def append_masks_only_golden_segment(
    run_dir: Path, fps: float, n_ids: int, start_idx: int, end_idx: int
) -> None:
    """Render mask + ID overlay (no behaviour labels) and append to golden_preview_masks.mp4."""
    if get_annotation_mode(run_dir) != "behavior":
        return
    start_idx, end_idx = int(start_idx), int(end_idx)
    if end_idx < start_idx:
        return
    seg_path = run_dir / "golden_segments" / f"{start_idx:05d}_{end_idx:05d}_masks.mp4"
    ensure_dir(seg_path.parent)
    _render_segment_from_golden(
        run_dir, fps, n_ids, start_idx, end_idx, seg_path, include_behavior=False
    )
    _finalize_golden_preview_segment(seg_path, fps)
    _append_to_golden_preview_file(golden_preview_video_path(run_dir, masks_only=True), seg_path, fps)


def rebuild_golden_preview_video(run_dir: Path) -> bool:
    """Re-render golden preview video(s) from committed golden frames."""
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        return False
    meta = parse_meta_file(meta_path)
    fps = float(meta["fps"])
    n_ids = int(meta.get("ids", 0) or 0)
    n_total = int(meta["frames"])
    if n_ids < 1:
        return False

    _, _, max_idx = golden_progress(run_dir, n_total)
    if max_idx is None:
        return False

    behavior_mode = get_annotation_mode(run_dir) == "behavior"
    targets = [(True, GOLDEN_PREVIEW_WITH_LABELS)]
    if behavior_mode:
        targets.append((False, GOLDEN_PREVIEW_MASKS_ONLY))

    ok_any = False
    for include_behavior, filename in targets:
        golden_preview = run_dir / "golden" / filename
        tmp_out = run_dir / "golden" / f"{filename}_rebuild_tmp.mp4"
        ensure_dir(golden_preview.parent)

        log.info(
            f"[GOLDEN_PREVIEW] Rebuilding {filename} 0..{max_idx} "
            f"(behavior_labels={include_behavior}) for run {run_dir.name}"
        )
        _render_segment_from_golden(
            run_dir, fps, n_ids, 0, int(max_idx), tmp_out, include_behavior=include_behavior
        )
        if not tmp_out.exists() or tmp_out.stat().st_size == 0:
            log.error(f"[GOLDEN_PREVIEW] Rebuild produced empty output for {filename}")
            continue

        golden_preview_tmp = run_dir / "golden" / f"{filename}_tmp.mp4"
        if _ffmpeg_reencode_video(tmp_out, golden_preview_tmp, fps):
            golden_preview_tmp.replace(golden_preview)
        else:
            tmp_out.replace(golden_preview)
        if tmp_out.exists():
            tmp_out.unlink()
        log.info(f"[GOLDEN_PREVIEW] Rebuild complete: {golden_preview}")
        ok_any = True
    return ok_any


def masks_to_label_map(masks_bool):
    H, W = masks_bool[0].shape
    label = np.zeros((H, W), dtype=np.uint8)
    for i, m in enumerate(masks_bool, start=1):
        label[m] = i
    return label


def random_color(seed: int):
    rng = np.random.RandomState(seed)
    return tuple(int(x) for x in rng.randint(50, 255, size=3))


# -------------------------
# Helper functions for common operations
# -------------------------

def get_color_for_id(id: int, min_val: int = 50) -> Tuple[int, int, int]:
    """
    Get a consistent color for a given ID.
    Uses RandomState to ensure same ID always gets same color.
    
    Args:
        id: The ID to get a color for
        min_val: Minimum RGB value (default 50 for better visibility)
    
    Returns:
        Tuple of (R, G, B) values
    """
    rng = np.random.RandomState(id)
    return tuple(int(x) for x in rng.randint(min_val, 255, size=3))


def encode_frame_to_base64(frame: np.ndarray, quality: int = 90) -> str:
    """
    Encode a frame (numpy array) to base64 JPEG string.
    
    Args:
        frame: Frame as numpy array (BGR format from cv2)
        quality: JPEG quality (0-100)
    
    Returns:
        Base64-encoded JPEG string
    
    Raises:
        HTTPException: If encoding fails
    """
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode frame")
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def load_frame_safely(frame_path: Path, frame_idx: Optional[int] = None) -> np.ndarray:
    """
    Load a frame from disk safely.
    
    Args:
        frame_path: Path to frame image
        frame_idx: Optional frame index for error messages
    
    Returns:
        Frame as numpy array (BGR format)
    
    Raises:
        HTTPException: If frame cannot be read
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        idx_msg = f" {frame_idx}" if frame_idx is not None else ""
        raise HTTPException(500, f"Could not read frame{idx_msg} from {frame_path}")
    return frame


def load_masks_safely(masks_file: Path) -> List[np.ndarray]:
    """
    Load masks from .npy file and ensure they are boolean numpy arrays.
    
    Args:
        masks_file: Path to .npy file containing masks
    
    Returns:
        List of boolean numpy arrays
    """
    masks_raw = np.load(masks_file, allow_pickle=True)
    masks = []
    for i, m in enumerate(masks_raw):
        if not isinstance(m, np.ndarray):
            log.warning(f"Loaded mask {i} is not a numpy array (type: {type(m)}), converting.")
            m = np.asarray(m)
        if m.dtype != bool:
            log.warning(f"Loaded mask {i} is not boolean (dtype: {m.dtype}), converting.")
            m = (m > 0.5).astype(bool)
        masks.append(m)
    return masks


def load_assignments_or_default(assignments_file: Path, n_masks: int) -> Dict[int, int]:
    """
    Load ID assignments from file or create default mapping.
    
    Args:
        assignments_file: Path to .npy file containing assignments dict
        n_masks: Number of masks (for default mapping)
    
    Returns:
        Dictionary mapping mask_index -> final_id
    """
    if assignments_file.exists():
        assignments = np.load(assignments_file, allow_pickle=True).item()
        return assignments
    else:
        # Default: mask_idx -> mask_idx + 1
        return {i: i + 1 for i in range(n_masks)}


def render_mask_overlay(frame: np.ndarray, mask: np.ndarray, mask_id: int, color: Tuple[int, int, int], 
                        alpha: float = 0.4, font_scale: float = 0.8) -> np.ndarray:
    """
    Render a mask overlay on a frame with ID label.
    
    Args:
        frame: Frame as numpy array (BGR format)
        mask: Boolean mask array
        mask_id: ID to display on mask
        color: RGB color tuple for overlay
        alpha: Overlay transparency (0.0-1.0)
        font_scale: Font scale for ID text
    
    Returns:
        Frame with mask overlay and ID label
    """
    if not mask.any():
        return frame
    
    overlay = frame.copy()
    overlay[mask] = color
    frame = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)
    
    # Add ID label at centroid
    ys, xs = np.where(mask)
    if len(ys) > 0:
        cx, cy = int(xs.mean()), int(ys.mean())
        text = str(mask_id)
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = cx - text_width // 2
        text_y = cy + text_height // 2
        
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    
    return frame


# Path helper functions
def get_golden_ann_dir(run_dir: Path) -> Path:
    """Get golden annotations directory path."""
    return run_dir / "golden" / "Annotations" / VIDEO_NAME


def get_golden_jpeg_dir(run_dir: Path) -> Path:
    """Get golden JPEG images directory path."""
    return run_dir / "golden" / "JPEGImages" / VIDEO_NAME


def get_jpeg_dir(run_dir: Path) -> Path:
    """Get source JPEG images directory path."""
    return run_dir / "xmem_generic" / "JPEGImages" / VIDEO_NAME


def get_init_masks_file(run_dir: Path) -> Path:
    """Get init masks file path."""
    return run_dir / "init" / "init_masks.npy"


def get_correction_masks_file(run_dir: Path, frame_idx: int) -> Path:
    """Get correction masks file path for a specific frame."""
    return run_dir / "correction_masks" / f"correction_masks_{frame_idx}.npy"


def get_correction_assignments_file(run_dir: Path, frame_idx: int) -> Path:
    """Get correction assignments file path for a specific frame."""
    return run_dir / "correction_masks" / f"correction_assignments_{frame_idx}.npy"


# -------------------------
# Model (lazy load)
# -------------------------
MODEL = None
PROCESSOR = None
VIDEO_PREDICTOR = None


_SAM3_READY = False


def _ensure_sam3_ready() -> None:
    """Tokenizer path fix + SAM 3.1 addmm_act dtype patch (github.com/facebookresearch/sam3/issues/507)."""
    global _SAM3_READY
    sam3_package_dir = Path(sam3.model_builder.__file__).parent
    bpe_file = sam3_package_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    pkg_resources_path = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    if not Path(pkg_resources_path).exists() and bpe_file.exists():
        original_fn = pkg_resources.resource_filename

        def patched_fn(package, resource):
            if package == "sam3" and "bpe_simple_vocab_16e6.txt.gz" in resource:
                return str(bpe_file)
            return original_fn(package, resource)

        pkg_resources.resource_filename = patched_fn

    if _SAM3_READY:
        return
    try:
        from sam3.perflib import fused

        addmm_act_op = torch.ops.aten._addmm_activation

        def addmm_act_fixed(activation, linear, mat1):
            if torch.is_grad_enabled():
                raise ValueError("Expected grad to be disabled.")
            orig_dtype = mat1.dtype
            bias = linear.bias.detach().to(torch.bfloat16)
            mat1_bf = mat1.to(torch.bfloat16)
            weight = linear.weight.detach().to(torch.bfloat16)
            flat = mat1_bf.view(-1, mat1_bf.shape[-1])
            use_gelu = activation in (torch.nn.functional.gelu, torch.nn.GELU)
            if activation not in (
                torch.nn.functional.relu,
                torch.nn.ReLU,
                torch.nn.functional.gelu,
                torch.nn.GELU,
            ):
                raise ValueError(f"Unexpected activation {activation}")
            y = addmm_act_op(bias, flat, weight.t(), beta=1, alpha=1, use_gelu=use_gelu)
            return y.view(mat1_bf.shape[:-1] + (y.shape[-1],)).to(orig_dtype)

        fused.addmm_act = addmm_act_fixed
        _SAM3_READY = True
        log.info("[SAM3] Ready (addmm_act dtype patch applied)")
    except Exception as e:
        log.warning(f"[SAM3] addmm_act patch failed: {e}")


def infer_sam3_text_prompt(processor: Sam3Processor, img: Image.Image, prompt: str) -> Dict[str, Any]:
    """Text-prompt segmentation on one image."""
    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(img)
                return processor.set_text_prompt(state=state, prompt=prompt)
        state = processor.set_image(img)
        return processor.set_text_prompt(state=state, prompt=prompt)


def get_model():
    global MODEL, PROCESSOR
    if MODEL is None:
        log.info("Loading SAM-3 model...")
        t0 = time.perf_counter()
        _ensure_sam3_ready()
        MODEL = build_sam3_image_model()
        PROCESSOR = Sam3Processor(MODEL)
        log.info(f"SAM-3 loaded in {time.perf_counter() - t0:.2f}s")
    return PROCESSOR


def get_video_predictor(force_reinit=False):
    """SAM-3 video predictor for point-based refinement (1-frame sessions)."""
    global VIDEO_PREDICTOR
    if VIDEO_PREDICTOR is None or force_reinit:
        if force_reinit and VIDEO_PREDICTOR is not None:
            log.warning("[VIDEO_PREDICTOR] Reinitializing")
            VIDEO_PREDICTOR = None
        log.info("Loading SAM-3 video predictor...")
        t0 = time.perf_counter()
        _ensure_sam3_ready()
        gpu_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        VIDEO_PREDICTOR = build_sam3_video_predictor(gpus_to_use=gpu_ids)
        log.info(f"SAM-3 video predictor loaded in {time.perf_counter() - t0:.2f}s")
    return VIDEO_PREDICTOR


def extract_instances_from_formatted(formatted0, img_w=None, img_h=None):
    """
    Extract instances from formatted SAM3 output (exactly like testing_backend.py).
    Returns list of dicts: [{"obj_id": int, "mask": HxW float(0/1), "score": float}, ...]
    """
    import torch
    
    instances: List[Dict[str, Any]] = []
    
    # Common pattern A: dict with "masks" + "obj_ids" (+ optional scores)
    if isinstance(formatted0, dict):
        if "masks" in formatted0 and formatted0["masks"] is not None:
            masks = formatted0["masks"]
            obj_ids = formatted0.get("obj_ids", formatted0.get("object_ids"))
            scores = formatted0.get("scores", formatted0.get("ious", formatted0.get("iou_predictions")))
            masks_np = np.asarray(masks)
            if masks_np.ndim == 2:
                masks_np = masks_np[None, :, :]
            if obj_ids is None:
                obj_ids_list = list(range(masks_np.shape[0]))
            else:
                obj_ids_list = [int(x) for x in np.asarray(obj_ids).reshape(-1).tolist()]
            if scores is None:
                scores_list = [1.0] * len(obj_ids_list)
            else:
                scores_list = [float(x) for x in np.asarray(scores).reshape(-1).tolist()]
            for i, oid in enumerate(obj_ids_list):
                mask_val = masks_np[i]
                if isinstance(mask_val, torch.Tensor):
                    mask_val = mask_val.squeeze().cpu().numpy()
                instances.append(
                    dict(obj_id=int(oid), mask=mask_val.astype(np.float32), score=float(scores_list[i] if i < len(scores_list) else 1.0))
                )
            return instances
        
        # Common pattern B: dict keyed by obj_id -> dict with "mask"
        keys = list(formatted0.keys())
        looks_like_obj_map = len(keys) > 0 and all((isinstance(k, int) or (isinstance(k, str) and k.isdigit())) for k in keys)
        if looks_like_obj_map:
            for k in keys:
                oid = int(k) if not isinstance(k, int) else k
                v = formatted0[k]
                if isinstance(v, dict):
                    m = v.get("mask", v.get("masks"))
                    if m is None:
                        continue
                    if isinstance(m, torch.Tensor):
                        m = m.squeeze().cpu().numpy()
                    score = v.get("score", v.get("iou", 1.0))
                    instances.append(dict(obj_id=int(oid), mask=np.asarray(m).astype(np.float32), score=float(score)))
                else:
                    # value itself is mask
                    if isinstance(v, torch.Tensor):
                        v = v.squeeze().cpu().numpy()
                    instances.append(dict(obj_id=int(oid), mask=np.asarray(v).astype(np.float32), score=1.0))
            return instances
        
        # Common pattern C: dict with "objects" list
        if "objects" in formatted0 and isinstance(formatted0["objects"], list):
            for obj in formatted0["objects"]:
                if not isinstance(obj, dict):
                    continue
                oid = obj.get("obj_id", obj.get("id", obj.get("object_id")))
                m = obj.get("mask", obj.get("masks"))
                if oid is None or m is None:
                    continue
                if isinstance(m, torch.Tensor):
                    m = m.squeeze().cpu().numpy()
                score = obj.get("score", obj.get("iou", 1.0))
                instances.append(dict(obj_id=int(oid), mask=np.asarray(m).astype(np.float32), score=float(score)))
            return instances
    
    # Pattern D: list of objects
    if isinstance(formatted0, list):
        for i, obj in enumerate(formatted0):
            if isinstance(obj, dict):
                oid = obj.get("obj_id", obj.get("id", obj.get("object_id", i)))
                m = obj.get("mask", obj.get("masks"))
                if m is None:
                    continue
                if isinstance(m, torch.Tensor):
                    m = m.squeeze().cpu().numpy()
                score = obj.get("score", obj.get("iou", 1.0))
                instances.append(dict(obj_id=int(oid), mask=np.asarray(m).astype(np.float32), score=float(score)))
            else:
                # list of masks
                if isinstance(obj, torch.Tensor):
                    obj = obj.squeeze().cpu().numpy()
                instances.append(dict(obj_id=i, mask=np.asarray(obj).astype(np.float32), score=1.0))
        return instances
    
    return instances


def safe_mask_hw(mask, h: int, w: int):
    """
    Ensure mask is HxW float32 in [0,1] (exactly like testing_backend.py).
    """
    m = mask.astype(np.float32)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    # normalize if it came as 0/255
    if m.max() > 1.5:
        m = (m > 127).astype(np.float32)
    return (m > 0.5).astype(np.float32)

def _ffmpeg_reencode_video(in_mp4: Path, out_mp4: Path, fps: float) -> bool:
    """
    Re-encode video to ensure browser-compatible H.264 format.
    """
    log.info(f"_ffmpeg_reencode_video: in={in_mp4} (exists={in_mp4.exists()}) -> out={out_mp4}")
    if not in_mp4.exists():
        log.error(f"Input video does not exist: {in_mp4}")
        return False
    
    in_size = in_mp4.stat().st_size
    in_dur = _probe_duration(in_mp4)
    log.info(f"Input video: size={in_size} bytes, duration={in_dur}s, fps={fps}")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_mp4),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-r", f"{fps:.10f}",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    log.info(f"Re-encoding video: {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        if p.returncode != 0:
            log.error(f"ffmpeg re-encode failed with return code {p.returncode}")
            log.error(f"ffmpeg output:\n{p.stdout[-2000:] if p.stdout else '(no output)'}")
            return False
    except FileNotFoundError:
        log.warning(f"ffmpeg not found in PATH. Skipping re-encoding - video may not be browser-compatible.")
        log.warning(f"To fix: install ffmpeg or load the ffmpeg module (e.g., 'module load ffmpeg')")
        # Copy the original file as-is (may not be browser-compatible)
        try:
            shutil.copy2(in_mp4, out_mp4)
            log.warning(f"Copied original video without re-encoding: {out_mp4}")
            return True
        except Exception as e:
            log.error(f"Failed to copy video: {e}")
            return False
    
    if not out_mp4.exists():
        log.error(f"Output video was not created: {out_mp4}")
        return False
    
    out_size = out_mp4.stat().st_size
    out_dur = _probe_duration(out_mp4)
    if out_size == 0:
        log.error(f"Output video is empty: {out_mp4}")
        return False
    
    log.info(f"Output video created: size={out_size} bytes, duration={out_dur}s")
    return True


def _ffmpeg_drop_seed_frame(in_mp4: Path, out_mp4: Path, fps: float) -> bool:
    """
    Create out_mp4 from in_mp4 but skipping the first frame (seed),
    using frame-index select (robust, avoids timestamp/keyframe issues).
    """
    log.info(f"_ffmpeg_drop_seed_frame: in={in_mp4} (exists={in_mp4.exists()}) -> out={out_mp4}")
    if not in_mp4.exists():
        log.error(f"Input video does not exist: {in_mp4}")
        return False
    
    in_size = in_mp4.stat().st_size
    in_dur = _probe_duration(in_mp4)
    log.info(f"Input video: size={in_size} bytes, duration={in_dur}s, fps={fps}")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_mp4),
        "-vf", f"select='gte(n,1)',setpts=N/({fps:.10f}*TB)",
        "-r", f"{fps:.10f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    log.info(f"Running ffmpeg drop-seed: {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        if p.returncode != 0:
            log.error(f"ffmpeg drop-seed failed with return code {p.returncode}")
            log.error(f"ffmpeg output:\n{p.stdout[-2000:] if p.stdout else '(no output)'}")
            return False
    except FileNotFoundError:
        log.warning(f"ffmpeg not found in PATH. Cannot drop seed frame - video may not be browser-compatible.")
        log.warning(f"To fix: install ffmpeg or load the ffmpeg module (e.g., 'module load ffmpeg')")
        # Copy the original file as-is (may not be browser-compatible)
        try:
            shutil.copy2(in_mp4, out_mp4)
            log.warning(f"Copied original video without dropping seed frame: {out_mp4}")
            return True
        except Exception as e:
            log.error(f"Failed to copy video: {e}")
            return False
    
    if not out_mp4.exists():
        log.error(f"Output video was not created: {out_mp4}")
        return False
    
    out_size = out_mp4.stat().st_size
    out_dur = _probe_duration(out_mp4)
    if out_size == 0:
        log.error(f"Output video is empty: {out_mp4}")
        return False
    
    log.info(f"Output video created: size={out_size} bytes, duration={out_dur}s")
    return True

# -------------------------
# Core pipeline
# -------------------------
def extract_frames(video_path: str, jpeg_dir: Path, progress_callback=None):
    """
    Extract frames from video using OpenCV, with ffmpeg fallback if OpenCV fails.
    
    Args:
        video_path: Path to video file
        jpeg_dir: Directory to save extracted frames
        progress_callback: Optional callback(progress: float, message: str) for progress updates
    """
    log.info(f"Extracting frames from {video_path}")
    t0 = time.perf_counter()

    # Try OpenCV first
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        
        frames = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fname = f"{idx:05d}.jpg"
            cv2.imwrite(str(jpeg_dir / fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            frames.append(fname)
            idx += 1
            
            # Update progress
            if progress_callback and total_frames:
                progress = min(100, (idx / total_frames) * 100)
                if idx % 50 == 0 or idx == total_frames:
                    progress_callback(progress, f"Extracted {idx}/{total_frames} frames...")
            elif progress_callback and (idx % 100 == 0 or idx % LOG_EVERY_FRAMES_EXTRACT == 0):
                estimated_total = 3000
                estimated_progress = min(95, (idx / estimated_total) * 100)
                progress_callback(estimated_progress, f"Extracted {idx} frames...")

        cap.release()
        
        if progress_callback:
            progress_callback(100, f"Extracted {len(frames)} frames")
        
        log.info(f"Frame extraction done: {len(frames)} frames ({time.perf_counter()-t0:.2f}s)")
        return frames, fps
        
    except Exception as e:
        log.warning(f"OpenCV extraction failed: {e}, trying ffmpeg fallback")
        return _extract_frames_ffmpeg(video_path, jpeg_dir, progress_callback)


def _extract_frames_ffmpeg(video_path: str, jpeg_dir: Path, progress_callback=None):
    """
    Extract frames using ffmpeg (more robust for problematic videos).
    
    Args:
        video_path: Path to video file
        jpeg_dir: Directory to save extracted frames
        progress_callback: Optional callback(progress: float, message: str) for progress updates
    """
    log.info(f"Using ffmpeg to extract frames from {video_path}")
    t0 = time.perf_counter()
    
    # Get FPS and frame count using ffprobe
    fps_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        fps_result = subprocess.run(fps_cmd, capture_output=True, text=True)
        if fps_result.returncode != 0:
            raise RuntimeError(f"ffprobe failed to get FPS: {fps_result.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found in PATH. Please install ffmpeg (which includes ffprobe).")
    
    fps_str = fps_result.stdout.strip()
    if "/" in fps_str:
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den > 0 else 30.0
    else:
        fps = float(fps_str) if fps_str else 30.0
    
    log.info(f"Detected FPS: {fps}")
    
    # Get total frame count for progress
    count_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    total_frames = None
    try:
        count_result = subprocess.run(count_cmd, capture_output=True, text=True, timeout=10)
        if count_result.returncode == 0:
            total_frames = int(count_result.stdout.strip()) if count_result.stdout.strip() else None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass  # Can't get frame count, will estimate
    
    if progress_callback:
        progress_callback(0, "Starting frame extraction with ffmpeg...")
    
    # Extract frames using ffmpeg
    output_pattern = str(jpeg_dir / "%05d.jpg")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-q:v", "2",  # High quality JPEG
        "-vsync", "0",  # Extract all frames
        output_pattern
    ]
    
    log.info(f"Running ffmpeg: {' '.join(cmd)}")
    try:
        # For ffmpeg, we can't easily track progress during extraction
        # We'll update progress after completion
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            log.error(f"ffmpeg extraction failed: {result.stderr}")
            raise RuntimeError(f"ffmpeg failed to extract frames: {result.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg or load the ffmpeg module (e.g., 'module load ffmpeg'). Frame extraction requires ffmpeg when OpenCV fails.")
    
    # List extracted frames
    frames = sorted([f.name for f in jpeg_dir.glob("*.jpg")])
    
    if len(frames) == 0:
        raise RuntimeError("ffmpeg extracted 0 frames")
    
    if progress_callback:
        progress_callback(100, f"Extracted {len(frames)} frames")
    
    log.info(f"Frame extraction done: {len(frames)} frames ({time.perf_counter()-t0:.2f}s)")
    return frames, fps


def compute_iou(mask1, mask2) -> float:
    """Compute Intersection over Union (IoU) between two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_centroid_distance(mask1, mask2) -> float:
    """Compute distance between centroids of two masks."""
    ys1, xs1 = np.where(mask1)
    ys2, xs2 = np.where(mask2)
    if len(xs1) == 0 or len(xs2) == 0:
        return float('inf')
    cx1, cy1 = xs1.mean(), ys1.mean()
    cx2, cy2 = xs2.mean(), ys2.mean()
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


def auto_assign_ids(new_masks: list, prev_label_map, iou_threshold: float = 0.2, allow_new_ids: bool = True) -> dict:
    """
    Auto-assign IDs to new masks based on previous frame's masks.
    Uses greedy matching similar to user's script: matches existing IDs first,
    then optionally assigns new IDs to unmatched detections.
    Returns dict mapping new_mask_index -> assigned_id
    
    Args:
        new_masks: List of boolean numpy arrays (new masks from SAM)
        prev_label_map: Previous frame's label map (uint8, 0=background, 1..N=object IDs)
        iou_threshold: Minimum IoU to consider a match (default 0.2)
        allow_new_ids: If False, prevents creating new IDs (unmatched masks are dropped)
    """
    
    if prev_label_map.max() == 0:
        # No previous masks, assign new IDs only if allowed
        if allow_new_ids:
            return {i: i + 1 for i in range(len(new_masks))}
        else:
            # No previous masks and new IDs not allowed - return empty assignments
            log.warning("No previous masks and allow_new_ids=False, returning empty assignments")
            return {}
    
    # Extract previous masks by ID
    prev_masks_by_id = {}
    for obj_id in range(1, int(prev_label_map.max()) + 1):
        mask = (prev_label_map == obj_id)
        if mask.any():  # Only include non-empty masks
            prev_masks_by_id[obj_id] = mask
    
    if not prev_masks_by_id:
        # No valid previous masks, assign new IDs only if allowed
        if allow_new_ids:
            return {i: i + 1 for i in range(len(new_masks))}
        else:
            log.warning("No valid previous masks and allow_new_ids=False, returning empty assignments")
            return {}
    
    prev_ids = sorted(prev_masks_by_id.keys())
    assignments = {}
    used_new_indices = set()
    used_prev_ids = set()
    
    # STEP 1: Greedy matching - for each previous ID, find the best matching new mask
    # This matches existing IDs first, ensuring stable IDs across reinitializations
    for prev_id in prev_ids:
        best_new_idx = None
        best_iou = 0.0
        
        for new_idx, new_mask in enumerate(new_masks):
            if new_idx in used_new_indices:
                continue
            
            iou = compute_iou(new_mask, prev_masks_by_id[prev_id])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_new_idx = new_idx
        
        if best_new_idx is not None:
            assignments[best_new_idx] = prev_id
            used_new_indices.add(best_new_idx)
            used_prev_ids.add(prev_id)
            log.info(f"Matched new mask {best_new_idx} -> prev ID {prev_id} (IoU={best_iou:.3f})")
    
    # STEP 2: If we have unmatched previous IDs and unmatched new masks, try to match them
    # with a lower threshold to prevent new IDs (especially when mask counts are similar)
    remaining_new_indices = [i for i in range(len(new_masks)) if i not in used_new_indices]
    remaining_prev_ids = [pid for pid in prev_ids if pid not in used_prev_ids]
    
    if remaining_new_indices and remaining_prev_ids:
        # Try to match remaining masks with a lower threshold to conserve IDs
        # This prevents one mask from splitting into multiple IDs
        low_threshold = 0.05  # Very low threshold for remaining matches
        
        # Build IoU matrix for remaining masks
        remaining_iou_matrix = np.zeros((len(remaining_new_indices), len(remaining_prev_ids)), dtype=np.float32)
        for r_new_idx, new_idx in enumerate(remaining_new_indices):
            for r_prev_idx, prev_id in enumerate(remaining_prev_ids):
                remaining_iou_matrix[r_new_idx, r_prev_idx] = compute_iou(new_masks[new_idx], prev_masks_by_id[prev_id])
        
        # Greedy matching: sort all pairs by IoU and match best pairs first
        matches = []
        for r_new_idx, new_idx in enumerate(remaining_new_indices):
            for r_prev_idx, prev_id in enumerate(remaining_prev_ids):
                iou = remaining_iou_matrix[r_new_idx, r_prev_idx]
                matches.append((iou, new_idx, prev_id))
        matches.sort(key=lambda x: x[0], reverse=True)
        
        # Match best pairs, ensuring 1-to-1
        for iou, new_idx, prev_id in matches:
            if new_idx in used_new_indices or prev_id in used_prev_ids:
                continue
            if iou >= low_threshold:
                assignments[new_idx] = prev_id
                used_new_indices.add(new_idx)
                used_prev_ids.add(prev_id)
                log.info(f"Conserved: matched new mask {new_idx} -> prev ID {prev_id} (IoU={iou:.3f})")
    
    # STEP 3: Assign new IDs to unmatched new masks (only if allow_new_ids=True)
    if allow_new_ids:
        next_new_id = max(prev_ids) + 1
        for new_idx in range(len(new_masks)):
            if new_idx not in used_new_indices:
                assignments[new_idx] = next_new_id
                next_new_id += 1
                log.info(f"Assigned new ID {assignments[new_idx]} to unmatched mask {new_idx}")
    else:
        # Drop unmatched new masks (don't assign them any ID)
        dropped_count = len(new_masks) - len(used_new_indices)
        if dropped_count > 0:
            log.info(f"Dropped {dropped_count} unmatched new masks (allow_new_ids=False)")
    
    log.info(f"Auto-assigned IDs: {assignments} (matched {len(used_prev_ids)}/{len(prev_ids)} previous IDs, allow_new_ids={allow_new_ids})")
    return assignments


def _masks_from_sam3_output(out: Dict[str, Any], width: int, height: int) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    for m in out["masks"]:
        mask = m.squeeze().cpu().numpy()
        mask = np.array(Image.fromarray(mask).resize((width, height), Image.NEAREST)) > MASK_THRESHOLD
        if mask.sum() > 500:
            masks.append(mask)
    return masks


def run_sam3_on_frame(prompt: str, frame_path: Path) -> list:
    """Run SAM-3 on a frame; return boolean masks."""
    log.info(f"SAM-3 on frame {frame_path}, prompt={prompt}")
    img = Image.open(frame_path).convert("RGB")
    W, H = img.size
    out = infer_sam3_text_prompt(get_model(), img, prompt)
    log.info(f"SAM-3 raw masks: {len(out['masks'])}")
    masks = _masks_from_sam3_output(out, W, H)
    if not masks:
        raise RuntimeError("No valid masks from SAM-3")
    log.info(f"SAM-3 kept {len(masks)} masks")
    return masks


def run_sam3_on_first_frame(prompt, jpeg_dir, ann_dir, frames):
    first_path = jpeg_dir / frames[0]
    masks = run_sam3_on_frame(prompt, first_path)
    label_map = masks_to_label_map(masks)
    ann0 = ann_dir / frames[0].replace(".jpg", ".png")
    Image.fromarray(label_map).save(ann0)
    log.info(f"SAM-3 kept {label_map.max()} masks")
    return int(label_map.max()), str(first_path)


def golden_progress(run_dir: Path, n_total: int):
    golden_ann_dir = get_golden_ann_dir(run_dir)
    if not golden_ann_dir.exists():
        log.info(f"[GOLDEN_PROGRESS] Golden annotations dir does not exist: {golden_ann_dir}")
        return 0, 0.0, None

    pngs = sorted(golden_ann_dir.glob("*.png"))
    def idx_from_name(p: Path) -> int:
        return int(p.stem)

    frame_indices = [idx_from_name(p) for p in pngs]
    if not frame_indices:
        return 0, 0.0, None
    max_idx = max(frame_indices)
    processed = max_idx + 1
    pct = (processed / max(n_total, 1)) * 100.0
    
    log.info(f"[GOLDEN_PROGRESS] Found {len(pngs)} frames in golden: min={min(frame_indices)}, max={max_idx}")
    log.info(f"[GOLDEN_PROGRESS] Frame indices: {sorted(frame_indices)[:20]}{'...' if len(frame_indices) > 20 else ''}")
    
    return processed, pct, max_idx


def make_chunk_dataset(run_dir: Path, seed_idx: int, end_idx: int, seed_ann_path: Path = None) -> Path:
    """
    Create an XMem generic dataset for frames [seed_idx .. end_idx] (inclusive),
    renumbered to 00000.jpg.., with annotation 00000.png taken from golden seed frame
    or provided seed_ann_path (for auto-reset).
    This lets XMem "continue" from the last golden frame or a SAM-reinitialized frame.
    """
    # Validate range
    if end_idx < seed_idx:
        raise RuntimeError(f"Invalid chunk range: seed_idx={seed_idx}, end_idx={end_idx} (end < start)")
    
    src_root = run_dir / "xmem_generic"
    src_jpeg = src_root / "JPEGImages" / VIDEO_NAME

    # Use provided seed annotation (from auto-reset) or fall back to golden
    if seed_ann_path:
        seed_ann = seed_ann_path  # Will fail on copy if missing
        log.info(f"Using auto-reset seed annotation: {seed_ann}")
    else:
        golden_ann_dir = get_golden_ann_dir(run_dir)
        seed_ann = golden_ann_dir / f"{seed_idx:05d}.png"  # Will fail on copy if missing

    dst_root = run_dir / "work_chunk" / f"{seed_idx:05d}_{end_idx:05d}"
    dst_jpeg = dst_root / "JPEGImages" / VIDEO_NAME
    dst_ann = dst_root / "Annotations" / VIDEO_NAME

    # fresh
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_jpeg.mkdir(parents=True, exist_ok=True)
    dst_ann.mkdir(parents=True, exist_ok=True)

    # Copy frames seed..end, renumber to 00000.. in the chunk dataset
    n = 0
    for orig_idx in range(seed_idx, end_idx + 1):
        src = src_jpeg / f"{orig_idx:05d}.jpg"
        dst = dst_jpeg / f"{n:05d}.jpg"
        shutil.copy2(src, dst)  # Will raise FileNotFoundError if src missing
        n += 1

    # Copy seed annotation to 00000.png (required by XMem)
    shutil.copy2(seed_ann, dst_ann / "00000.png")

    log.info(f"Chunk dataset prepared: {dst_root} (orig {seed_idx}..{end_idx}, frames={n})")
    return dst_root


def run_xmem(dataset_root: Path, xmem_output: Path):
    log.info(f"Starting XMem on dataset_root={dataset_root}")
    ensure_clean_dir(xmem_output)

    cmd = [
        "python", "eval.py",
        "--model", os.path.abspath(XMEM_MODEL),
        "--output", os.path.abspath(xmem_output),
        "--dataset", "G",
        "--generic_path", os.path.abspath(dataset_root),
    ]

    # Set TORCH_HOME and TMPDIR to use scratch space (avoid /tmp being full)
    # PyTorch will cache pretrained models (like ResNet50) here
    # TMPDIR is used for temporary extraction during download
    torch_cache_dir = scratch_subdir("torch_cache")
    tmp_dir = scratch_subdir("tmp")
    env = os.environ.copy()
    env["TORCH_HOME"] = str(torch_cache_dir)
    env["TMPDIR"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)  # Some tools use TMP instead of TMPDIR
    log.info(f"[XMem] Using TORCH_HOME={torch_cache_dir} for model cache")
    log.info(f"[XMem] Using TMPDIR={tmp_dir} for temporary files")

    proc = subprocess.Popen(
        cmd,
        cwd=XMEM_REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    logs = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        logs.append(line)
        log.info(f"[XMem] {line}")

    if proc.wait() != 0:
        raise RuntimeError("XMem failed")

    log.info("XMem finished")
    return logs


def find_xmem_pngs(xmem_output: Path):
    found = []
    for root, _, files in os.walk(xmem_output):
        if os.path.basename(root) == VIDEO_NAME:
            for f in files:
                if f.endswith(".png"):
                    found.append(os.path.join(root, f))
    found.sort()
    log.info(f"Found {len(found)} XMem masks")
    return found


def render_video(
    jpeg_dir: Path,
    frames,
    found_pngs,
    out_video: Path,
    fps: float,
    n_ids: int,
    run_dir: Optional[Path] = None,
    behavior_frame_offset: int = 0,
    include_behavior: bool = True,
):
    """
    Render all provided frames list (same length as found_pngs ideally).
    Uses direct ffmpeg encoding from processed frame images (faster, more reliable).
    """
    if not frames:
        raise RuntimeError("render_video got empty frames list")

    log.info(f"Rendering preview video: {out_video}")
    first = cv2.imread(str(jpeg_dir / frames[0]))
    if first is None:
        raise RuntimeError("Could not read first frame for rendering.")
    H, W = first.shape[:2]

    out_video.parent.mkdir(parents=True, exist_ok=True)
    colors = {i: random_color(i) for i in range(1, n_ids + 1)}
    draw_behavior = (
        include_behavior
        and run_dir is not None
        and get_annotation_mode(run_dir) == "behavior"
    )

    T = min(len(frames), len(found_pngs))
    
    # Process frames and write to temporary directory, then encode with ffmpeg
    with tempfile.TemporaryDirectory(prefix="render_video_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        log.debug(f"Processing {T} frames to temporary directory: {tmpdir_path}")
        for t in range(T):
            frame = cv2.imread(str(jpeg_dir / frames[t]))
            if frame is None:
                raise RuntimeError(f"Could not read frame {frames[t]}")
            mask = np.array(Image.open(found_pngs[t]))
            try:
                abs_frame_idx = int(Path(frames[t]).stem) + int(behavior_frame_offset)
            except ValueError:
                abs_frame_idx = t + int(behavior_frame_offset)

            for cid, col in colors.items():
                m = (mask == cid)
                if not m.any():
                    continue
                behavior_lines = None
                if draw_behavior:
                    behavior_lines = behavior_overlay_lines_at_frame(
                        run_dir, cid, abs_frame_idx
                    )
                frame = draw_cow_overlay_with_behavior(
                    frame, m, cid, col, behavior_lines=behavior_lines
                )

            # Write processed frame to temp directory (use quality 85 for faster I/O)
            frame_path = tmpdir_path / f"frame_{t:05d}.jpg"
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            if (t + 1) % LOG_EVERY_FRAMES_RENDER == 0:
                log.debug(f"  processed {t+1}/{T} frames")

        # Encode video directly with ffmpeg (single pass, H.264)
        log.debug(f"Encoding video with ffmpeg from {T} frames...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", f"{fps:.10f}",
            "-i", str(tmpdir_path / "frame_%05d.jpg"),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",  # Good quality for preview videos
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_video),
        ]
        
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
            if p.returncode != 0:
                log.error(f"ffmpeg encoding failed with return code {p.returncode}")
                log.error(f"ffmpeg output:\n{p.stdout[-2000:] if p.stdout else '(no output)'}")
                raise RuntimeError(f"Failed to encode video with ffmpeg: {out_video}")
        except FileNotFoundError:
            log.error("ffmpeg not found in PATH. Cannot render video without ffmpeg.")
            raise RuntimeError("ffmpeg is required for video rendering but was not found in PATH")

    log.info(f"Rendering done: {out_video}")
    return T


def _render_segment_from_golden(
    run_dir: Path,
    fps: float,
    n_ids: int,
    start_idx: int,
    end_idx: int,
    out_path: Path,
    include_behavior: bool = True,
):
    """
    Render golden overlay segment for frames start_idx..end_idx inclusive into out_path.
    Uses original JPEGs + golden label PNGs.
    """
    
    log.info(f"[RENDER_GOLDEN] Rendering segment: frames {start_idx}..{end_idx} -> {out_path}")
    
    src_root = run_dir / "xmem_generic"
    src_jpeg = src_root / "JPEGImages" / VIDEO_NAME
    golden_ann = get_golden_ann_dir(run_dir)

    log.info(f"[RENDER_GOLDEN] Source JPEG dir: {src_jpeg}")
    log.info(f"[RENDER_GOLDEN] Golden annotations dir: {golden_ann}")

    # Build frame list + mask list aligned
    frames = []
    masks = []
    for i in range(start_idx, end_idx + 1):
        jpg = src_jpeg / f"{i:05d}.jpg"
        png = golden_ann / f"{i:05d}.png"
        
        log.info(f"[RENDER_GOLDEN] Frame {i}: JPEG={jpg} (exists: {jpg.exists()}), Mask={png} (exists: {png.exists()})")
        
        if not jpg.exists() or not png.exists():
            raise RuntimeError(f"Missing for golden segment: {jpg} or {png}")
        
        # Load and log mask info
        mask = np.array(Image.open(png))
        max_id = int(mask.max())
        unique_ids = sorted(list(set(mask.flatten())))
        unique_ids = [id for id in unique_ids if id > 0]  # Remove background
        log.info(f"[RENDER_GOLDEN] Frame {i} mask: max_id={max_id}, IDs={unique_ids}, path={png}")
        
        frames.append(jpg.name)
        masks.append(str(png))

    log.info(f"[RENDER_GOLDEN] Rendering {len(frames)} frames with {len(masks)} masks to {out_path}")
    render_video(
        jpeg_dir=src_jpeg,
        frames=frames,
        found_pngs=masks,
        out_video=out_path,
        fps=fps,
        n_ids=n_ids,
        run_dir=run_dir,
        include_behavior=include_behavior,
    )
    log.info(f"[RENDER_GOLDEN] Segment rendering complete: {out_path}")


def _ffmpeg_concat(a: Path, b: Path, out: Path, fps: float) -> bool:
    """
    Concat a+b into out via a temp output, then atomic replace.
    Always re-encodes to ensure browser-compatible codec/container.
    """
    log.info(f"_ffmpeg_concat: a={a} (exists={a.exists()}, size={a.stat().st_size if a.exists() else 0})")
    log.info(f"_ffmpeg_concat: b={b} (exists={b.exists()}, size={b.stat().st_size if b.exists() else 0})")
    log.info(f"_ffmpeg_concat: out={out}, fps={fps}")
    
    if not a.exists():
        log.error(f"First video does not exist: {a}")
        return False
    if not b.exists():
        log.error(f"Second video does not exist: {b}")
        return False
    
    a_dur = _probe_duration(a)
    b_dur = _probe_duration(b)
    log.info(f"Input videos: a duration={a_dur}s, b duration={b_dur}s")
    
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_list = out.parent / f"concat_{uuid.uuid4().hex[:8]}.txt"
    tmp_out  = out.parent / f"concat_{uuid.uuid4().hex[:8]}.mp4"

    list_content = f"file '{a.resolve()}'\nfile '{b.resolve()}'\n"
    tmp_list.write_text(list_content, encoding="utf-8")
    log.info(f"Created concat list file: {tmp_list}\nContent:\n{list_content}")

    # Always re-encode to ensure browser-compatible codec/container
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(tmp_list),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-r", f"{fps:.10f}",  # Set frame rate explicitly
        "-pix_fmt", "yuv420p",  # Ensure browser-compatible pixel format
        "-movflags", "+faststart",  # Enable fast start for web playback
        str(tmp_out),
    ]
    log.info(f"Re-encoding concat (browser-compatible): {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        log.error(f"ffmpeg not found in PATH. Cannot concatenate videos.")
        log.error(f"To fix: install ffmpeg or load the ffmpeg module (e.g., 'module load ffmpeg')")
        # Clean up temp files
        if tmp_list.exists():
            tmp_list.unlink()
        return False
    
    if not (p.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size > 0):
        # If concat failed, the first video (a) might be corrupted - try re-encoding it first
        log.warning("Re-encode concat failed, trying to re-encode first video and retry...")
        log.warning(f"ffmpeg output:\n{p.stdout[-2000:] if p.stdout else '(no output)'}")
        
        # Re-encode first video to temp file
        a_reencoded = out.parent / f"concat_a_reencoded_{uuid.uuid4().hex[:8]}.mp4"
        if _ffmpeg_reencode_video(a, a_reencoded, fps):
            log.info("Successfully re-encoded first video, retrying concat...")
            # Update concat list with re-encoded video
            list_content = f"file '{a_reencoded.resolve()}'\nfile '{b.resolve()}'\n"
            tmp_list.write_text(list_content, encoding="utf-8")
            log.info(f"Updated concat list:\n{list_content}")
            
            # Retry concat
            try:
                p2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            except FileNotFoundError:
                log.error(f"ffmpeg not found during retry. Cannot concatenate videos.")
                # Clean up temp files
                if tmp_list.exists():
                    tmp_list.unlink()
                if a_reencoded.exists():
                    a_reencoded.unlink()
                return False
            if p2.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size > 0:
                log.info("Concat succeeded after re-encoding first video")
                a_reencoded.unlink(missing_ok=True)  # Clean up temp file
            else:
                log.error("Concat still failed after re-encoding first video")
                log.error(f"ffmpeg output:\n{p2.stdout[-2000:] if p2.stdout else '(no output)'}")
                a_reencoded.unlink(missing_ok=True)
                tmp_list.unlink(missing_ok=True)
                tmp_out.unlink(missing_ok=True)
                return False
        else:
            log.error("Failed to re-encode first video, giving up")
            tmp_list.unlink(missing_ok=True)
            tmp_out.unlink(missing_ok=True)
            return False
    
    log.info("Re-encode concat succeeded")
    tmp_size = tmp_out.stat().st_size
    tmp_dur = _probe_duration(tmp_out)
    log.info(f"Temporary output: size={tmp_size} bytes, duration={tmp_dur}s")

    # atomic replace
    old_size = out.stat().st_size if out.exists() else 0
    tmp_out.replace(out)
    tmp_list.unlink(missing_ok=True)
    
    final_size = out.stat().st_size
    final_dur = _probe_duration(out)
    log.info(f"Final output: size={final_size} bytes (was {old_size}), duration={final_dur}s")
    return True



# -------------------------
# API
# -------------------------
@app.post("/prepare")
def prepare(video_path: str):
    """
    Prepare a new annotation session WITHOUT running SAM:
    1. Create run directory
    2. Extract frames (this is the expensive part)
    3. Write meta.txt (prompt left empty for now)
    4. Return run_id + basic video metadata + source preview URL
    """

    log.info(f"/prepare video={video_path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = RUNS_ROOT / run_id

    custom_root = run_dir / "xmem_generic"
    jpeg_dir = custom_root / "JPEGImages" / VIDEO_NAME
    ann_dir = custom_root / "Annotations" / VIDEO_NAME

    jpeg_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    frames, fps = extract_frames(video_path, jpeg_dir)

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else None
    cap.release()

    # Save metadata (prompt empty for now - will be set in /init_sam)
    (run_dir / "meta.txt").write_text(
        f"video_path={video_path}\n"
        f"prompt=\n"
        f"fps={fps}\n"
        f"frames={len(frames)}\n"
        f"ids=0\n"
        f"annotation_mode=standard\n"
    )

    source_url = f"/source/{run_id}"

    return {
        "run_id": run_id,
        "fps": fps,
        "n_frames_total": len(frames),
        "width": width,
        "height": height,
        "source_url": source_url,
    }


def _do_frame_extraction(run_id: str, video_path: Path, jpeg_dir: Path, run_dir: Path, safe_name: str):
    """
    Background task to extract frames and save metadata.
    This allows the frontend to poll for progress during extraction.
    """
    
    try:
        # Update progress for frame extraction
        prepare_progress[run_id] = {"stage": "extract", "progress": 0, "message": "Starting frame extraction..."}
        log.info(f"[PREPARE_UPLOAD] Starting frame extraction for {run_id}")

        # Progress callback for frame extraction
        last_logged_progress = [-1]  # Use list to allow modification in closure
        def update_extract_progress(progress, message):
            if progress is not None:
                prepare_progress[run_id] = {
                    "stage": "extract",
                    "progress": progress,
                    "message": message
                }
                # Only log every 10% to reduce verbosity
                if int(progress) // 10 != int(last_logged_progress[0]) // 10:
                    log.info(f"[PREPARE_UPLOAD] Extract progress: {progress:.1f}% - {message}")
                    last_logged_progress[0] = progress
            else:
                # Keep current progress, just update message
                current = prepare_progress.get(run_id, {})
                prepare_progress[run_id] = {
                    "stage": "extract",
                    "progress": current.get("progress", 0),
                    "message": message
                }

        # Extract frames
        frames, fps = extract_frames(str(video_path), jpeg_dir, progress_callback=update_extract_progress)
        log.info(f"[PREPARE_UPLOAD] Frame extraction complete: {len(frames)} frames")

        # Best-effort metadata
        width = None
        height = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            cap.release()
        except Exception:
            pass

        # Save metadata (prompt empty for now - will be set in /init_sam)
        (run_dir / "meta.txt").write_text(
            f"video_path={video_path}\n"
            f"prompt=\n"
            f"fps={fps}\n"
            f"frames={len(frames)}\n"
            f"ids=0\n"
            f"annotation_mode=standard\n"
        )

        # Mark as completed but keep it for a bit so frontend can see it
        prepare_progress[run_id] = {
            "stage": "extract",
            "progress": 100,
            "message": "Completed"
        }
        
        # Clear progress after 10 seconds (give frontend time to poll)
        def clear_progress_later():
            time.sleep(10)
            prepare_progress.pop(run_id, None)
            log.info(f"[PREPARE_UPLOAD] Cleared progress for {run_id}")
        
        threading.Thread(target=clear_progress_later, daemon=True).start()
        
    except Exception as e:
        log.error(f"[PREPARE_UPLOAD] Frame extraction failed for {run_id}: {e}")
        prepare_progress[run_id] = {
            "stage": "extract",
            "progress": 0,
            "message": f"Error: {str(e)}"
        }
        raise


@app.post("/prepare_upload")
async def prepare_upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Prepare a new annotation session from a video uploaded via multipart/form-data.
    Saves the uploaded video into the run directory, then extracts frames (like /prepare).
    Returns run_id immediately after upload, extraction happens in background.
    """
    
    if not file or not file.filename:
        raise HTTPException(400, "No file uploaded")

    log.info(f"/prepare_upload filename={file.filename} content_type={file.content_type}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = RUNS_ROOT / run_id

    custom_root = run_dir / "xmem_generic"
    jpeg_dir = custom_root / "JPEGImages" / VIDEO_NAME
    ann_dir = custom_root / "Annotations" / VIDEO_NAME

    jpeg_dir.mkdir(parents=True, exist_ok=True)  # Creates all parent dirs including run_dir
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Initialize progress tracking
    prepare_progress[run_id] = {"stage": "upload", "progress": 0, "message": "Starting upload..."}
    log.info(f"[PREPARE_UPLOAD] Started, run_id={run_id}, filename={file.filename}")

    # Save uploaded file into the run directory
    uploads_dir = run_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name  # strip any path components
    video_path = uploads_dir / safe_name

    try:
        # Get file size for progress tracking
        # Note: FastAPI's UploadFile might not support seek, so we'll track as we read
        uploaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        log.info(f"[PREPARE_UPLOAD] Starting file upload to {video_path}")
        
        with open(video_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                uploaded += len(chunk)
                # Update progress (we don't know total size, so estimate based on chunks)
                # For now, we'll just show "uploading" until done
                prepare_progress[run_id] = {
                    "stage": "upload",
                    "progress": min(95, uploaded / (10 * 1024 * 1024) * 100),  # Estimate: assume ~10MB for 95%
                    "message": f"Uploading... {uploaded / (1024*1024):.1f} MB"
                }
                log.debug(f"[PREPARE_UPLOAD] Uploaded {uploaded} bytes")
    except Exception as e:
        log.error(f"[PREPARE_UPLOAD] Upload error: {e}")
        prepare_progress.pop(run_id, None)
        raise
    finally:
        try:
            await file.close()
        except Exception:
            pass

    if not video_path.exists() or video_path.stat().st_size == 0:
        log.error(f"[PREPARE_UPLOAD] Upload failed: file is empty or missing")
        prepare_progress.pop(run_id, None)
        raise HTTPException(500, "Uploaded file save failed (empty file)")

    log.info(f"[PREPARE_UPLOAD] Upload complete, file size: {video_path.stat().st_size} bytes")
    prepare_progress[run_id] = {"stage": "upload", "progress": 100, "message": "Upload complete, starting extraction..."}

    # Schedule frame extraction in background
    # This allows us to return run_id immediately so frontend can start polling
    background_tasks.add_task(_do_frame_extraction, run_id, video_path, jpeg_dir, run_dir, safe_name)

    # Return immediately with run_id (extraction will happen in background)
    # Frontend will poll for progress and get final results when done
    return JSONResponse({
        "run_id": run_id,
        "fps": None,  # Will be available after extraction
        "n_frames_total": None,  # Will be available after extraction
        "width": None,  # Will be available after extraction
        "height": None,  # Will be available after extraction
        "source_url": f"/source/{run_id}",
        "uploaded_filename": safe_name,
        "status": "uploaded",  # Indicates extraction is in progress
    })


@app.get("/prepare_upload_progress/{run_id}")
def get_prepare_upload_progress(run_id: str):
    """
    Get progress for prepare_upload operation.
    Returns progress info if operation is in progress, or final metadata if completed.
    """
    progress = prepare_progress.get(run_id)
    if progress is None:
        # Check if extraction is actually done by checking if meta.txt exists
        run_dir = RUNS_ROOT / run_id
        meta_path = run_dir / "meta.txt"
        if meta_path.exists():
            try:
                meta = parse_meta_file(meta_path)
                # Get video dimensions
                video_path = meta.get("video_path", "")
                width = None
                height = None
                if video_path and os.path.exists(video_path):
                    try:
                        cap = cv2.VideoCapture(str(video_path))
                        if cap.isOpened():
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
                        cap.release()
                    except Exception:
                        pass
                
                log.info(f"[PROGRESS] {run_id}: completed (found meta.txt)")
                return {
                    "status": "completed",
                    "progress": 100,
                    "message": "Completed",
                    "fps": float(meta.get("fps", 0)) if meta.get("fps") else None,
                    "n_frames_total": int(meta.get("frames", 0)) if meta.get("frames") else None,
                    "width": width,
                    "height": height,
                }
            except Exception as e:
                log.warning(f"[PROGRESS] {run_id}: meta.txt exists but couldn't parse: {e}")
        
        log.info(f"[PROGRESS] {run_id}: not found (completed or never started)")
        return {"status": "completed", "progress": 100, "message": "Completed"}
    
    return {
        "status": "in_progress",
        "stage": progress["stage"],
        "progress": progress["progress"],
        "message": progress["message"]
    }


@app.post("/init_sam/{run_id}")
def init_sam(run_id: str, prompt: str):
    """
    Run SAM initialization on frame 0 for an EXISTING prepared run.
    This avoids re-extracting frames and makes Page 1 \"Load Video\" meaningful.
    """
    
    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found (missing meta.txt)")

    meta = parse_meta_file(meta_path)
    video_path = meta.get("video_path", "")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(400, f"Video not found for run_id: {video_path}")

    custom_root = run_dir / "xmem_generic"
    jpeg_dir = custom_root / "JPEGImages" / VIDEO_NAME  # Note: custom_root may differ from run_dir
    ann_dir = custom_root / "Annotations" / VIDEO_NAME
    if not jpeg_dir.exists():
        raise HTTPException(400, "Frames not prepared. Run /prepare first.")
    ann_dir.mkdir(parents=True, exist_ok=True)

    fps = float(meta.get("fps", 30.0))

    frames = sorted([p.name for p in jpeg_dir.glob("*.jpg")])
    if len(frames) == 0:
        raise HTTPException(400, "No extracted frames found. Run /prepare first.")

    log.info(f"/init_sam run_id={run_id} prompt={prompt} frames={len(frames)} fps={fps}")

    log.info(f"[INIT_SAM] Running SAM-3 on frame 0, prompt={prompt}")
    first_path = jpeg_dir / frames[0]
    masks = run_sam3_on_frame(prompt, first_path)
    img = Image.open(first_path).convert("RGB")
    n_masks = len(masks)
    log.info(f"[INIT_SAM] SAM-3 kept {n_masks} masks")

    # Save init masks
    masks_file = get_init_masks_file(run_dir)
    masks_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(masks_file, masks)

    # Render preview image with auto-assigned IDs (1, 2, 3, ...)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    for mask_idx, mask in enumerate(masks, start=1):
        assigned_id = mask_idx
        col = get_color_for_id(assigned_id, min_val=0)
        overlay = frame.copy()
        overlay[mask] = col
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(
            frame,
            str(assigned_id),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    image_b64 = encode_frame_to_base64(frame, quality=90)

    mask_assignments = [{"mask_index": i, "auto_assigned_id": i + 1} for i in range(n_masks)]

    # Update meta with prompt (keep fps/frames/ids)
    meta["prompt"] = prompt
    meta_path.write_text("\n".join(f"{k}={v}" for k, v in meta.items()))

    log.info(f"[INIT_SAM] Returning {n_masks} masks for ID assignment")
    return JSONResponse(
        content={
            "run_id": run_id,
            "fps": fps,
            "n_frames_total": len(frames),
            "image": f"data:image/jpeg;base64,{image_b64}",
            "mask_assignments": mask_assignments,
        }
    )


class IDMapping(BaseModel):
    mapping: Dict[str, int]


class ApplyInitPayload(BaseModel):
    mapping: Dict[str, int]
    behavior_by_cow_id: Optional[Dict[str, str]] = None
    behavior_label2_by_cow_id: Optional[Dict[str, str]] = None
    behavior_label3_by_cow_id: Optional[Dict[str, str]] = None


class AnnotationModePayload(BaseModel):
    mode: str


class BehaviorSetLabelPayload(BaseModel):
    cow_id: int
    frame: int
    label_id: str
    dimension: str = "activity"


class BehaviorDeleteLabelPayload(BaseModel):
    cow_id: int
    frame: int
    dimension: str = "activity"


class PreviewUpdate(BaseModel):
    mapping: Dict[str, int]  # mask_index -> final_id (0 means delete)


@app.post("/match_init_ids/{run_id}")
def match_init_ids(run_id: str, file: UploadFile = File(...)):
    """
    Match new SAM masks from init to IDs from a previous golden mask file.
    Uses the same IoU-based matching as auto_assign_ids.
    Accepts an uploaded PNG mask file.
    """
    
    log.info(f"/match_init_ids run_id={run_id} uploaded_file={file.filename}")
    
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"Run not found: {run_id}")
    
    # Load saved masks from init
    masks_file = get_init_masks_file(run_dir)
    new_masks = np.load(masks_file, allow_pickle=True)  # Will raise FileNotFoundError if missing
    log.info(f"[MATCH_INIT_IDS] Loaded {len(new_masks)} new masks from init")
    
    # Save uploaded file temporarily and load it
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_path = Path(tmp_file.name)
        # Read uploaded file content
        content = file.file.read()
        tmp_path.write_bytes(content)
        log.info(f"[MATCH_INIT_IDS] Saved uploaded mask to temporary file: {tmp_path}")
    
    try:
        prev_label_map = np.array(Image.open(tmp_path))
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()
            log.info(f"[MATCH_INIT_IDS] Cleaned up temporary file: {tmp_path}")
    max_prev_id = int(prev_label_map.max())
    unique_prev_ids = sorted([id for id in np.unique(prev_label_map) if id > 0])
    log.info(f"[MATCH_INIT_IDS] Loaded previous mask with {len(unique_prev_ids)} object IDs: {unique_prev_ids}, max_id={max_prev_id}")
    
    # Match new masks to previous IDs using auto_assign_ids
    assignments = auto_assign_ids(new_masks, prev_label_map, iou_threshold=0.2, allow_new_ids=True)
    log.info(f"[MATCH_INIT_IDS] ID matching completed: {assignments}")
    
    # Count how many were matched vs new
    matched_count = len([aid for aid in assignments.values() if aid <= max_prev_id])
    total_count = len(assignments)
    
    # Render preview image with matched IDs
    meta = parse_meta_file(run_dir / "meta.txt")
    custom_root = run_dir / "xmem_generic"
    jpeg_dir = custom_root / "JPEGImages" / VIDEO_NAME  # Note: custom_root may differ from run_dir
    first_path = jpeg_dir / "00000.jpg"
    
    if not first_path.exists():
        raise HTTPException(404, "Frame 0 not found")
    
    frame = cv2.imread(str(first_path))
    if frame is None:
        raise HTTPException(500, "Could not read frame 0")
    
    # Render masks with matched IDs
    for mask_idx, mask in enumerate(new_masks):
        matched_id = assignments.get(mask_idx, mask_idx + 1)
        col = get_color_for_id(matched_id, min_val=0)
        overlay = frame.copy()
        overlay[mask] = col
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        
        text = str(matched_id)
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = cx - text_width // 2
        text_y = cy + text_height // 2
        
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    
    # Encode preview image
    image_b64 = encode_frame_to_base64(frame, quality=90)
    
    # Prepare response
    mask_assignments = [
        {
            "mask_index": mask_idx,
            "auto_assigned_id": mask_idx + 1,  # Original auto-assigned (sequential)
            "matched_id": matched_id,  # Matched ID from previous mask
        }
        for mask_idx, matched_id in sorted(assignments.items())
    ]
    
    log.info(f"[MATCH_INIT_IDS] Returning {len(mask_assignments)} matched assignments ({matched_count}/{total_count} matched to previous IDs)")
    return JSONResponse(content={
        "mask_assignments": mask_assignments,
        "matched_count": matched_count,
        "total_count": total_count,
        "image": f"data:image/jpeg;base64,{image_b64}",
    })


@app.post("/preview_init_update/{run_id}")
def preview_init_update(run_id: str, preview_update: PreviewUpdate):
    """
    Regenerate frame 0 preview image with current ID mappings and deletions.
    Used for real-time preview updates as user edits the table.
    """
    
    log.info(f"/preview_init_update run_id={run_id}")
    log.info(f"Preview update mapping: {preview_update.mapping}")
    
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"Run not found: {run_id}")
    
    # Load saved masks from init
    masks_file = get_init_masks_file(run_dir)
    masks = np.load(masks_file, allow_pickle=True)  # Will raise FileNotFoundError if missing
    log.info(f"[PREVIEW_INIT_UPDATE] Loaded {len(masks)} masks from init")
    
    # Load frame 0 image
    src_root = run_dir / "xmem_generic"
    jpeg_dir = src_root / "JPEGImages" / VIDEO_NAME
    frame_path = jpeg_dir / "00000.jpg"
    frame = load_frame_safely(frame_path, frame_idx=0)  # Will raise HTTPException if missing
    frame = frame.copy()  # Make a copy to avoid modifying original
    
    # Render masks with user's current ID mappings (skip deleted ones)
    rendered_count = 0
    for mask_idx_str, final_id in preview_update.mapping.items():
        mask_idx = int(mask_idx_str)
        if mask_idx >= len(masks):
            continue
        
        # Skip deleted masks (ID 0 or negative)
        if final_id <= 0:
            continue
        
        mask = masks[mask_idx]  # Already validated as boolean by load_masks_safely()
        col = get_color_for_id(final_id, min_val=0)
        overlay = frame.copy()
        overlay[mask] = col
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        
        text = str(final_id)
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = cx - text_width // 2
        text_y = cy + text_height // 2
        
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        rendered_count += 1
    
    log.info(f"Rendered {rendered_count} masks in preview")
    
    image_b64 = encode_frame_to_base64(frame, quality=90)
    
    return JSONResponse(content={
        "image": f"data:image/jpeg;base64,{image_b64}",
    })


@app.get("/behavior/labels")
def list_behavior_labels():
    return {
        "labels": BEHAVIOR_LABELS_ACTIVITY,
        "labels_activity": BEHAVIOR_LABELS_ACTIVITY,
        "labels_label2": BEHAVIOR_LABELS_LABEL2,
        "labels_label3": BEHAVIOR_LABELS_LABEL3,
        "dimensions": {
            dim: {
                "title_fi": BEHAVIOR_DIMENSION_META[dim]["title_fi"],
                "required": BEHAVIOR_DIMENSION_META[dim]["required"],
                "default_label": BEHAVIOR_DIMENSION_META[dim]["default_label"],
            }
            for dim in BEHAVIOR_DIMENSIONS
        },
    }


def _behavior_dimension_api_payload(data: Optional[Dict[str, Any]], dimension: str, frame: Optional[int]) -> Dict[str, Any]:
    meta = BEHAVIOR_DIMENSION_META[dimension]
    payload: Dict[str, Any] = {
        "segments": [],
        "cow_ids": [],
        "labels_at_frame": {},
        "labels": meta["labels"],
        "title_fi": meta["title_fi"],
        "required": meta["required"],
        "default_label": meta["default_label"],
    }
    if dimension == "activity":
        payload["preview_in_sync"] = True
    if data:
        payload["segments"] = data.get("segments", [])
        payload["cow_ids"] = data.get("cow_ids", [])
        if dimension == "activity":
            payload["preview_in_sync"] = bool(data.get("preview_in_sync", True))
        if frame is not None:
            payload["labels_at_frame"] = {
                str(k): v for k, v in labels_at_frame(data, int(frame)).items()
            }
    return payload


@app.post("/run/{run_id}/annotation_mode")
def set_annotation_mode(run_id: str, payload: AnnotationModePayload):
    mode = (payload.mode or "").strip().lower()
    if mode not in ("standard", "behavior"):
        raise HTTPException(400, "mode must be 'standard' or 'behavior'")
    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found (missing meta.txt)")
    update_meta_key(meta_path, "annotation_mode", mode)
    log.info(f"[ANNOTATION_MODE] run_id={run_id} mode={mode}")
    return {"run_id": run_id, "annotation_mode": mode}


@app.get("/behavior/{run_id}")
def get_behavior(run_id: str, frame: Optional[int] = Query(None)):
    run_dir = RUNS_ROOT / run_id
    if not (run_dir / "meta.txt").exists():
        raise HTTPException(404, "run_id not found")
    mode = get_annotation_mode(run_dir)
    activity_data = load_behavior_dimension(run_dir, "activity")
    result: Dict[str, Any] = {
        "run_id": run_id,
        "annotation_mode": mode,
        "labels": BEHAVIOR_LABELS_ACTIVITY,
        "labels_activity": BEHAVIOR_LABELS_ACTIVITY,
        "labels_label2": BEHAVIOR_LABELS_LABEL2,
        "labels_label3": BEHAVIOR_LABELS_LABEL3,
        "preview_in_sync": bool(activity_data.get("preview_in_sync", True)) if activity_data else True,
        "dimensions": {},
    }
    for dim in BEHAVIOR_DIMENSIONS:
        dim_data = load_behavior_dimension(run_dir, dim)
        result["dimensions"][dim] = _behavior_dimension_api_payload(dim_data, dim, frame)
    return result


@app.post("/behavior/{run_id}/set_label")
def behavior_set_label(run_id: str, payload: BehaviorSetLabelPayload):
    run_dir = RUNS_ROOT / run_id
    if not (run_dir / "meta.txt").exists():
        raise HTTPException(404, "run_id not found")
    if get_annotation_mode(run_dir) != "behavior":
        raise HTTPException(400, "Run is not in behavior annotation mode")
    dimension = (payload.dimension or "activity").strip().lower()
    if dimension not in BEHAVIOR_DIMENSIONS:
        raise HTTPException(400, f"dimension must be one of {BEHAVIOR_DIMENSIONS}")
    data = load_behavior_dimension(run_dir, dimension)
    if not data:
        raise HTTPException(400, "No behavior data for this run; complete ID assignment first")
    if payload.frame < 0:
        raise HTTPException(400, "frame must be >= 0")
    try:
        set_label_from_frame(data, payload.cow_id, payload.frame, payload.label_id, dimension)
    except ValueError as e:
        raise HTTPException(400, str(e))
    save_behavior_dimension(run_dir, dimension, data)
    preview_in_sync = None
    if BEHAVIOR_DIMENSION_META[dimension]["affects_preview"]:
        mark_behavior_preview_out_of_sync(run_dir)
        activity_data = load_behavior_dimension(run_dir, "activity")
        preview_in_sync = bool(activity_data.get("preview_in_sync", False)) if activity_data else False
    log.info(
        f"[BEHAVIOR] set_label run_id={run_id} dim={dimension} cow_id={payload.cow_id} "
        f"frame={payload.frame} label={payload.label_id}"
    )
    labels_at = {
        str(k): v for k, v in labels_at_frame(data, payload.frame).items()
    }
    return {
        "run_id": run_id,
        "dimension": dimension,
        "cow_id": payload.cow_id,
        "frame": payload.frame,
        "label_id": payload.label_id,
        "preview_in_sync": preview_in_sync,
        "label_at_frame": labels_at,
        "labels_at_frame": labels_at,
    }


@app.post("/behavior/{run_id}/delete_label")
def behavior_delete_label(run_id: str, payload: BehaviorDeleteLabelPayload):
    run_dir = RUNS_ROOT / run_id
    if not (run_dir / "meta.txt").exists():
        raise HTTPException(404, "run_id not found")
    if get_annotation_mode(run_dir) != "behavior":
        raise HTTPException(400, "Run is not in behavior annotation mode")
    dimension = (payload.dimension or "activity").strip().lower()
    if dimension not in BEHAVIOR_DIMENSIONS:
        raise HTTPException(400, f"dimension must be one of {BEHAVIOR_DIMENSIONS}")
    data = load_behavior_dimension(run_dir, dimension)
    if not data:
        raise HTTPException(400, "No behavior data for this run; complete ID assignment first")
    if payload.frame < 0:
        raise HTTPException(400, "frame must be >= 0")
    try:
        delete_label_from_frame(data, payload.cow_id, payload.frame, dimension)
    except ValueError as e:
        raise HTTPException(400, str(e))
    save_behavior_dimension(run_dir, dimension, data)
    preview_in_sync = None
    if BEHAVIOR_DIMENSION_META[dimension]["affects_preview"]:
        mark_behavior_preview_out_of_sync(run_dir)
        activity_data = load_behavior_dimension(run_dir, "activity")
        preview_in_sync = bool(activity_data.get("preview_in_sync", False)) if activity_data else False
    log.info(
        f"[BEHAVIOR] delete_label run_id={run_id} dim={dimension} cow_id={payload.cow_id} "
        f"frame={payload.frame}"
    )
    labels_at = {
        str(k): v for k, v in labels_at_frame(data, payload.frame).items()
    }
    return {
        "run_id": run_id,
        "dimension": dimension,
        "cow_id": payload.cow_id,
        "frame": payload.frame,
        "preview_in_sync": preview_in_sync,
        "labels_at_frame": labels_at,
    }


@app.post("/golden/{run_id}/rebuild_preview")
def golden_rebuild_preview(run_id: str):
    """Re-render golden preview video (e.g. after behaviour label changes or legacy runs)."""
    run_dir = RUNS_ROOT / run_id
    if not (run_dir / "meta.txt").exists():
        raise HTTPException(404, "run_id not found")
    if get_annotation_mode(run_dir) != "behavior":
        return {"run_id": run_id, "preview_rebuilt": False, "message": "Not in behavior mode"}
    ok = rebuild_golden_preview_video(run_dir)
    if not ok:
        raise HTTPException(500, "Failed to rebuild golden preview")
    mark_behavior_preview_in_sync(run_dir)
    return {"run_id": run_id, "preview_rebuilt": True, "preview_in_sync": True}


@app.post("/apply_init_ids/{run_id}")
def apply_init_ids(run_id: str, payload: ApplyInitPayload):
    """
    Apply user's ID mapping to frame 0 masks and complete initialization.
    This creates the annotation file, golden folder, and preview video.
  Optional behavior_by_cow_id (cow_id str -> label_id) for behavior annotation mode.
    """
    log.info(f"/apply_init_ids run_id={run_id} mapping={payload.mapping}")
    
    run_dir = RUNS_ROOT / run_id
    
    # Load saved masks
    masks_file = get_init_masks_file(run_dir)
    masks = np.load(masks_file, allow_pickle=True)
    n_masks = len(masks)
    
    # Read metadata
    meta_path = run_dir / "meta.txt"
    meta = parse_meta_file(meta_path)
    fps = float(meta["fps"])
    
    # Apply user's ID mapping
    custom_root = run_dir / "xmem_generic"
    jpeg_dir = custom_root / "JPEGImages" / VIDEO_NAME
    ann_dir = custom_root / "Annotations" / VIDEO_NAME
    
    H, W = masks[0].shape
    label_map = np.zeros((H, W), dtype=np.uint8)
    
    for mask_idx_str, final_id in payload.mapping.items():
        mask_idx = int(mask_idx_str)
        final_id = int(final_id)
        if final_id <= 0:  # Skip deleted masks
            continue
        label_map[masks[mask_idx]] = final_id
    
    # Save annotation
    ann0 = ann_dir / "00000.png"
    Image.fromarray(label_map).save(ann0)
    n_ids = int(label_map.max())
    log.info(f"[APPLY_INIT_IDS] Saved annotation with {n_ids} objects")
    
    # Create golden folder + seed frame0 annotation
    golden_ann_dir = get_golden_ann_dir(run_dir)
    ensure_dir(golden_ann_dir)
    shutil.copy2(ann0, golden_ann_dir / "00000.png")
    
    # Also copy frame0 JPEG to golden/JPEGImages/video1/
    golden_jpeg_dir = get_golden_jpeg_dir(run_dir)
    ensure_dir(golden_jpeg_dir)
    shutil.copy2(jpeg_dir / "00000.jpg", golden_jpeg_dir / "00000.jpg")
    
    # Update metadata with final n_ids (before preview render)
    meta_content = meta_path.read_text()
    meta_path.write_text(meta_content.replace("ids=0", f"ids={n_ids}"))

    # Behaviour labels: initial segments at frame 0 (3 JSON files; activity before preview)
    if get_annotation_mode(run_dir) == "behavior":
        cow_ids = sorted({int(v) for v in payload.mapping.values() if int(v) > 0})

        def _labels_from_payload(
            optional_map: Optional[Dict[str, str]],
            dimension: str,
        ) -> Dict[int, str]:
            out: Dict[int, str] = {}
            default = _default_label_for_dimension(dimension)
            if optional_map:
                for cow_key, label_id in optional_map.items():
                    out[int(cow_key)] = label_id
            for cow_id in cow_ids:
                if cow_id not in out:
                    out[cow_id] = default
            return out

        for dim in BEHAVIOR_DIMENSIONS:
            if dim == "activity":
                init_map = _labels_from_payload(payload.behavior_by_cow_id, dim)
            elif dim == "label2":
                init_map = _labels_from_payload(payload.behavior_label2_by_cow_id, dim)
            else:
                init_map = _labels_from_payload(payload.behavior_label3_by_cow_id, dim)
            dim_data = create_initial_segments(cow_ids, 0, init_map, dim)
            save_behavior_dimension(run_dir, dim, dim_data)
        log.info(f"[APPLY_INIT_IDS] Saved behavior segments (3 dims) for cows={cow_ids}")

    # Initialize golden preview video with frame0
    seg0 = run_dir / "golden_segments" / "00000_00000.mp4"
    ensure_dir(seg0.parent)
    _render_segment_from_golden(run_dir, fps, n_ids, 0, 0, seg0)
    
    golden_preview_init = run_dir / "golden" / "golden_preview.mp4"
    ensure_dir(golden_preview_init.parent)
    golden_preview_init.write_bytes(seg0.read_bytes())
    
    # Re-encode to ensure browser-compatible format
    golden_preview_tmp = run_dir / "golden" / "golden_preview_tmp.mp4"
    if _ffmpeg_reencode_video(golden_preview_init, golden_preview_tmp, fps):
        golden_preview_tmp.replace(golden_preview_init)

    if get_annotation_mode(run_dir) == "behavior":
        append_masks_only_golden_segment(run_dir, fps, n_ids, 0, 0)
    
    # Clean up temporary masks file
    masks_file.unlink()

    log.info(f"[APPLY_INIT_IDS] Initialization complete: run_id={run_id}, n_ids={n_ids}")
    return {"run_id": run_id, "n_ids": n_ids}

    
@app.post("/track")
def track(run_id: str, n_frames: int, auto_reset_interval: Optional[int] = Query(None)):
    """
    CONTINUATION tracking:
    - Finds last golden frame g (max idx)
    - Tracks chunk from g .. min(g+n_frames, end)
      (so you get n_frames NEW frames after the seed)
    - Stores chunk under chunks/<g>_<end>/
    - Renders preview to tracked.mp4
    
    If auto_reset_interval is set (e.g., 50), automatically reinitializes with SAM
    every K frames by running SAM on the seed frame and matching IDs with previous frame.
    This helps handle drift and reappearing objects.
    """
    log.info(f"/track run_id={run_id} n_frames={n_frames} auto_reset_interval={auto_reset_interval}")

    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found (missing meta.txt)")

    # Always start a tracking run with a clean chunks directory so we don't mix
    # old tracked chunks with a new tracking session. Older tracked chunks are
    # not needed once a new tracking run starts.
    chunks_dir = run_dir / "chunks"
    ensure_clean_dir(chunks_dir)

    meta = parse_meta_file(meta_path)
    fps = float(meta["fps"])
    n_ids = int(meta["ids"])
    n_total = int(meta["frames"])

    if n_frames < 1:
        raise HTTPException(400, "n_frames must be >= 1")

    processed, pct, max_idx = golden_progress(run_dir, n_total)
    log.info(f"[TRACK] Golden progress: processed={processed}, pct={pct}, max_idx={max_idx}")
    if max_idx is None:
        raise HTTPException(500, "Golden has no seed frame (unexpected). Re-run /init.")

    seed_idx = int(max_idx)  # last committed frame
    log.info(f"[TRACK] Using seed_idx={seed_idx} (from golden max_idx={max_idx})")
    end_idx = min(seed_idx + int(n_frames), n_total - 1)
    log.info(f"[TRACK] Will track frames {seed_idx+1}..{end_idx} (seed={seed_idx}, n_frames={n_frames})")

    if end_idx <= seed_idx:
        return {
            "run_id": run_id,
            "seed_idx": seed_idx,
            "end_idx": end_idx,
            "message": "Already at end of video.",
        }

    log.info(f"Tracking chunk seed={seed_idx} -> end={end_idx} (new frames: {seed_idx+1}..{end_idx})")
    
    # Initialize tracking progress IMMEDIATELY (before any processing)
    total_frames_to_track = end_idx - seed_idx
    track_progress[run_id] = {
        "stage": "tracking",
        "progress": 0,
        "message": f"Preparing to track {total_frames_to_track} frames...",
        "current_frame": seed_idx,
        "total_frames": end_idx,
    }
    log.info(f"[TRACK] Progress initialized: 0% - Preparing to track {total_frames_to_track} frames...")

    # If auto_reset_interval is set, split tracking into multiple chunks
    if auto_reset_interval is not None and auto_reset_interval > 0:
        log.info(f"🔄 Auto-reset enabled: splitting {seed_idx}..{end_idx} into chunks of {auto_reset_interval} frames")
        
        # Track in chunks: [seed_idx..seed_idx+K-1], [seed_idx+K..seed_idx+2K-1], etc.
        current_seed = seed_idx
        all_masks = []
        all_chunk_roots = []
        prev_chunk_seed_ann_path = None  # Seed annotation from previous chunk for next chunk
        
        while current_seed <= end_idx:
            # Calculate chunk end first
            chunk_end = min(current_seed + auto_reset_interval - 1, end_idx)
            
            log.info(f"[DEBUG] Chunk calculation: current_seed={current_seed}, auto_reset_interval={auto_reset_interval}, end_idx={end_idx}, chunk_end={chunk_end}")
            
            # Validate chunk range before processing
            if chunk_end < current_seed:
                log.error(f"Invalid chunk detected: current_seed={current_seed}, chunk_end={chunk_end}, end_idx={end_idx} - this should not happen!")
                log.error(f"Breaking loop to prevent invalid chunk creation")
                break
            
            # Ensure we have at least 2 frames for XMem (seed + at least one more)
            # If we only have the seed frame left, we can't create a valid chunk
            if chunk_end == current_seed:
                # Only one frame - this should only happen if current_seed == end_idx
                # In that case, we've already processed everything
                if current_seed == end_idx:
                    log.info(f"Reached end: only frame {current_seed} remains (already processed or will be handled)")
                else:
                    log.warning(f"Single frame chunk {current_seed} detected but end_idx={end_idx}, this shouldn't happen")
                break
            
            log.info(f"--- Processing chunk: frames {current_seed}..{chunk_end} ---")
            
            # Update progress
            frames_processed = current_seed - seed_idx
            progress_pct = min(90, int((frames_processed / total_frames_to_track) * 100))
            track_progress[run_id] = {
                "stage": "tracking",
                "progress": progress_pct,
                "message": f"Tracking chunk {current_seed}..{chunk_end} ({frames_processed}/{total_frames_to_track} frames)",
                "current_frame": current_seed,
                "total_frames": end_idx,
            }
            
            # Check if we need SAM reset for this chunk (before processing)
            # If we have a seed from previous chunk, use it; otherwise check for SAM reset
            seed_ann_path = prev_chunk_seed_ann_path
            # Reset every auto_reset_interval frames from the initial seed_idx
            # So if seed_idx=9 and interval=10, reset at 9, 19, 29, etc.
            should_reset = (current_seed > seed_idx) and ((current_seed - seed_idx) % auto_reset_interval == 0)
            
            if should_reset:
                
                log.info(f"🔄 Auto-reset: Running SAM on frame {current_seed} (reset interval: {auto_reset_interval})")
                
                # Run SAM on seed frame
                prompt = meta.get("prompt", "object")
                src_root = run_dir / "xmem_generic"
                jpeg_dir = src_root / "JPEGImages" / VIDEO_NAME
                frame_path = jpeg_dir / f"{current_seed:05d}.jpg"
                
                if not frame_path.exists():
                    raise HTTPException(404, f"Frame {current_seed} not found for SAM reset")
                
                new_masks = run_sam3_on_frame(prompt, frame_path)
                log.info(f"SAM detected {len(new_masks)} masks on frame {current_seed}")
                
                # Match IDs with previous frame if available
                # Try to get from the previous chunk first, then fall back to golden
                prev_frame_idx = current_seed - 1
                prev_label_map = None
                
                if prev_frame_idx >= 0:
                    # First, try to get from the previous chunk we just tracked
                    if all_chunk_roots:
                        # Find the chunk that contains prev_frame_idx
                        for prev_chunk_root in all_chunk_roots:
                            prev_chunk_name = prev_chunk_root.name  # e.g., "00030_00039"
                            prev_chunk_start, prev_chunk_end = map(int, prev_chunk_name.split("_"))  # Use different variable names to avoid shadowing
                            if prev_chunk_start <= prev_frame_idx <= prev_chunk_end:
                                # This chunk contains the previous frame
                                prev_chunk_ann_dir = prev_chunk_root / "Annotations" / VIDEO_NAME
                                local_idx = prev_frame_idx - prev_chunk_start
                                prev_mask_path = prev_chunk_ann_dir / f"{local_idx:05d}.png"
                                if prev_mask_path.exists():
                                    prev_label_map = np.array(Image.open(prev_mask_path))
                                    log.info(f"Using previous frame {prev_frame_idx} from chunk {prev_chunk_name} (local idx {local_idx})")
                                    break
                    
                    # Fall back to golden if not found in chunks
                    if prev_label_map is None:
                        golden_ann_dir = get_golden_ann_dir(run_dir)
                        prev_ann_path = golden_ann_dir / f"{prev_frame_idx:05d}.png"
                        if prev_ann_path.exists():
                            prev_label_map = np.array(Image.open(prev_ann_path))
                            log.info(f"Using previous frame {prev_frame_idx} from golden")
                
                if prev_label_map is not None:
                    # During tracking reinitialization, don't allow new IDs - only match existing ones
                    # This ensures stable IDs and prevents new masks from appearing
                    assignments = auto_assign_ids(new_masks, prev_label_map, iou_threshold=0.2, allow_new_ids=False)
                    
                    # Create label map from assignments
                    H, W = new_masks[0].shape
                    label_map = np.zeros((H, W), dtype=np.uint8)
                    for mask_idx, assigned_id in assignments.items():
                        label_map[new_masks[mask_idx]] = assigned_id
                    
                    # Update max ID if needed (shouldn't happen with allow_new_ids=False, but just in case)
                    new_max_id = int(label_map.max())
                    if new_max_id > n_ids:
                        n_ids = new_max_id
                        meta["ids"] = str(n_ids)
                        meta_path.write_text("\n".join(f"{k}={v}" for k, v in meta.items()))
                        log.info(f"Updated max ID to {n_ids}")
                else:
                    # No previous frame, assign sequential IDs
                    label_map = masks_to_label_map(new_masks)
                    n_ids = int(label_map.max())
                    log.info(f"No previous frame found, assigned sequential IDs (max: {n_ids})")
                
                # Save temporary seed annotation for this chunk
                temp_seed_dir = run_dir / "temp_seeds"
                ensure_dir(temp_seed_dir)
                seed_ann_path = temp_seed_dir / f"{current_seed:05d}.png"
                Image.fromarray(label_map).save(seed_ann_path)
                log.info(f"✅ Created reset seed annotation: {seed_ann_path} (max ID: {n_ids})")
            
            # Validate chunk range before creating dataset (double-check)
            if chunk_end < current_seed:
                log.error(f"CRITICAL: Invalid chunk range detected: current_seed={current_seed}, chunk_end={chunk_end}, end_idx={end_idx}")
                raise RuntimeError(f"Invalid chunk range before make_chunk_dataset: current_seed={current_seed}, chunk_end={chunk_end}, end_idx={end_idx}")
            
            if chunk_end == current_seed:
                log.warning(f"Single-frame chunk detected: {current_seed}, skipping XMem")
                # Create chunk directory with just the seed annotation
                chunk_root = run_dir / "chunks" / f"{current_seed:05d}_{chunk_end:05d}"
                chunk_ann_dir = chunk_root / "Annotations" / VIDEO_NAME
                ensure_clean_dir(chunk_root)
                ensure_dir(chunk_ann_dir)
                
                # Copy seed annotation
                if seed_ann_path:
                    shutil.copy2(seed_ann_path, chunk_ann_dir / "00000.png")  # Will fail if missing
                else:
                    golden_ann_dir = get_golden_ann_dir(run_dir)
                    golden_seed = golden_ann_dir / f"{current_seed:05d}.png"
                    shutil.copy2(golden_seed, chunk_ann_dir / "00000.png")  # Will fail if missing
                
                all_chunk_roots.append(chunk_root)
                log.info(f"✅ Chunk {current_seed} (single frame, no XMem)")
                
                # For single-frame chunk, the seed for next chunk is this frame's annotation
                single_frame_ann = chunk_ann_dir / "00000.png"
                prev_chunk_seed_ann_path = single_frame_ann
                log.info(f"Saved seed annotation for next chunk: {prev_chunk_seed_ann_path} (frame {current_seed})")
                
                # Move to next chunk
                next_seed = chunk_end + 1
                
                if next_seed > end_idx:
                    # We've reached the requested end, stop
                    log.info(f"Reached requested end_idx {end_idx}, stopping chunk processing")
                    break
                
                log.info(f"Continuing to next chunk: will use frame {next_seed} as seed (from single-frame chunk {current_seed})")
                current_seed = next_seed
                continue
            
            # Build chunk dataset for this sub-chunk
            log.info(f"Creating chunk dataset: seed={current_seed}, end={chunk_end}")
            chunk_ds = make_chunk_dataset(run_dir, current_seed, chunk_end, seed_ann_path=seed_ann_path)
            
            # Check if seed annotation is empty (all zeros) - if so, skip XMem and create empty masks
            seed_ann_file = chunk_ds / "Annotations" / VIDEO_NAME / "00000.png"
            seed_ann = np.array(Image.open(seed_ann_file))
            is_empty_seed = (seed_ann.max() == 0)
            
            xmem_output = run_dir / "xmem_outputs" / f"{current_seed:05d}_{chunk_end:05d}"
            ensure_dir(xmem_output.parent)
            
            if is_empty_seed:
                log.info(f"[TRACK] Chunk {current_seed}..{chunk_end}: Seed annotation is empty (all zeros), skipping XMem and creating empty masks")
                logs = []
                
                # Update progress (same as XMem would)
                track_progress[run_id] = {
                    "stage": "tracking",
                    "progress": progress_pct,
                    "message": f"Creating empty masks for chunk {current_seed}..{chunk_end} (no objects to track)...",
                    "current_frame": current_seed,
                    "total_frames": end_idx,
                }
                
                # Create empty masks for all frames (same size as seed annotation)
                ensure_clean_dir(xmem_output)
                xmem_ann_dir = xmem_output / VIDEO_NAME
                ensure_dir(xmem_ann_dir)
                
                # Create empty mask (all zeros) for each frame
                empty_mask = np.zeros_like(seed_ann, dtype=np.uint8)
                n_frames = chunk_end - current_seed + 1
                for i in range(n_frames):
                    mask_path = xmem_ann_dir / f"{i:05d}.png"
                    Image.fromarray(empty_mask).save(mask_path)
                
                masks = find_xmem_pngs(xmem_output)
                log.info(f"[TRACK] Created {len(masks)} empty masks for chunk {current_seed}..{chunk_end}")
            else:
                # Update progress before XMem
                track_progress[run_id] = {
                    "stage": "tracking",
                    "progress": progress_pct,
                    "message": f"Running XMem on chunk {current_seed}..{chunk_end}...",
                    "current_frame": current_seed,
                    "total_frames": end_idx,
                }
                
                logs = run_xmem(chunk_ds, xmem_output)
                masks = find_xmem_pngs(xmem_output)
            
            # Store masks into chunk folder
            chunk_root = run_dir / "chunks" / f"{current_seed:05d}_{chunk_end:05d}"
            chunk_ann_dir = chunk_root / "Annotations" / VIDEO_NAME
            ensure_clean_dir(chunk_root)
            ensure_dir(chunk_ann_dir)
            
            for p in masks:
                shutil.copy2(p, chunk_ann_dir / Path(p).name)
            
            all_chunk_roots.append(chunk_root)
            log.info(f"✅ Chunk {current_seed}..{chunk_end} tracked ({len(masks)} masks)")
            
            # Prepare seed annotation for next chunk from this chunk's last frame
            # The last frame of this chunk is at local index (chunk_end - current_seed)
            last_frame_local_idx = chunk_end - current_seed
            last_frame_ann = chunk_ann_dir / f"{last_frame_local_idx:05d}.png"
            
            if not last_frame_ann.exists():
                log.error(f"⚠️  Last frame {chunk_end} of chunk {current_seed}..{chunk_end} not found: {last_frame_ann}")
                log.error(f"⚠️  Cannot continue to next chunk - stopping")
                break
            
            # Save this as the seed for the next chunk
            prev_chunk_seed_ann_path = last_frame_ann
            log.info(f"Saved seed annotation for next chunk: {prev_chunk_seed_ann_path} (frame {chunk_end})")
            
            # Move to next chunk
            next_seed = chunk_end + 1
            
            if next_seed > end_idx:
                # We've reached the requested end, stop
                log.info(f"Reached requested end_idx {end_idx}, stopping chunk processing")
                break
            
            log.info(f"Continuing to next chunk: will use frame {next_seed} as seed (from tracked chunk {current_seed}..{chunk_end}, annotation: {prev_chunk_seed_ann_path})")
            current_seed = next_seed
        
        # Merge all chunks into one final chunk for rendering
        # Use the last chunk as the "main" one for metadata
        chunk_root = all_chunk_roots[-1]
        seed_idx_final = seed_idx
        end_idx_final = end_idx
        
        # Build combined dataset for rendering
        chunk_ds = make_chunk_dataset(run_dir, seed_idx, end_idx, seed_ann_path=None)
        
        # Collect all masks from all chunks for rendering
        all_masks = []
        for chunk_root_item in all_chunk_roots:
            chunk_ann_dir_item = chunk_root_item / "Annotations" / VIDEO_NAME
            masks_item = sorted([str(p) for p in chunk_ann_dir_item.glob("*.png")])
            all_masks.extend(masks_item)
        
        # Reorder masks by frame index
        def get_frame_idx(path_str):
            # Extract frame index from path like "chunks/00010_00019/Annotations/video1/00000.png"
            # We need to map local indices back to global
            path = Path(path_str)
            # Path structure: chunks/00010_00019/Annotations/video1/00000.png
            # So we need: path.parent.parent.parent.name to get "00010_00019"
            chunk_name = path.parent.parent.parent.name  # e.g., "00010_00019"
            local_idx = int(path.stem)  # e.g., 0 from "00000.png"
            chunk_seed = int(chunk_name.split("_")[0])
            return chunk_seed + local_idx
        
        all_masks.sort(key=get_frame_idx)
        
        # For rendering, we need masks in the chunk dataset order (00000.png, 00001.png, ...)
        # Map global frame indices to chunk dataset local indices
        jpeg_dir = chunk_ds / "JPEGImages" / VIDEO_NAME
        frames = sorted([p.name for p in jpeg_dir.glob("*.jpg")])
        
        # Create a mapping: global_idx -> mask_path
        global_to_mask = {}
        for mask_path_str in all_masks:
            global_idx = get_frame_idx(mask_path_str)
            global_to_mask[global_idx] = mask_path_str
        
        # Build mask list in chunk dataset order
        masks_for_render = []
        for i, frame_name in enumerate(frames):
            global_idx = seed_idx + i
            if global_idx in global_to_mask:
                masks_for_render.append(global_to_mask[global_idx])
            else:
                log.warning(f"Missing mask for global frame {global_idx} (chunk local {i})")
        
        # Use the combined masks for rendering
        masks = masks_for_render
        logs = []  # Combined logs from all chunks (we could collect them, but for now just empty)
        
    else:
        # Original behavior: single chunk, no auto-reset
        track_progress[run_id] = {
            "stage": "tracking",
            "progress": 10,
            "message": f"Preparing to track {total_frames_to_track} frames...",
            "current_frame": seed_idx,
            "total_frames": end_idx,
        }
        
        seed_ann_path = None
        chunk_ds = make_chunk_dataset(run_dir, seed_idx, end_idx, seed_ann_path=seed_ann_path)

        # Check if seed annotation is empty (all zeros) - if so, skip XMem and create empty masks
        seed_ann_file = chunk_ds / "Annotations" / VIDEO_NAME / "00000.png"
        seed_ann = np.array(Image.open(seed_ann_file))
        is_empty_seed = (seed_ann.max() == 0)
        
        if is_empty_seed:
            log.info(f"[TRACK] Seed annotation is empty (all zeros), skipping XMem and creating empty masks for all frames")
            logs = []
            
            # Update progress (same as XMem would)
            track_progress[run_id] = {
                "stage": "tracking",
                "progress": 30,
                "message": "Creating empty masks (no objects to track)...",
                "current_frame": seed_idx,
                "total_frames": end_idx,
            }
            
            # Create empty masks for all frames (same size as seed annotation)
            xmem_output = run_dir / "xmem_outputs" / f"{seed_idx:05d}_{end_idx:05d}"
            ensure_clean_dir(xmem_output)
            xmem_ann_dir = xmem_output / VIDEO_NAME
            ensure_dir(xmem_ann_dir)
            
            # Create empty mask (all zeros) for each frame
            empty_mask = np.zeros_like(seed_ann, dtype=np.uint8)
            n_frames = end_idx - seed_idx + 1
            for i in range(n_frames):
                mask_path = xmem_ann_dir / f"{i:05d}.png"
                Image.fromarray(empty_mask).save(mask_path)
            
            masks = find_xmem_pngs(xmem_output)
            log.info(f"[TRACK] Created {len(masks)} empty masks (no tracking performed)")
        else:
            # Update progress before XMem
            track_progress[run_id] = {
                "stage": "tracking",
                "progress": 30,
                "message": "Running XMem...",
                "current_frame": seed_idx,
                "total_frames": end_idx,
            }

            # Run XMem on this chunk dataset
            xmem_output = run_dir / "xmem_outputs" / f"{seed_idx:05d}_{end_idx:05d}"
            ensure_dir(xmem_output.parent)
            logs = run_xmem(chunk_ds, xmem_output)
            masks = find_xmem_pngs(xmem_output)
        
        # Update progress after XMem
        track_progress[run_id] = {
            "stage": "tracking",
            "progress": 80,
            "message": "Processing masks...",
            "current_frame": end_idx,
            "total_frames": end_idx,
        }

        # Store masks into stable chunk folder (still renumbered 00000..)
        chunk_root = run_dir / "chunks" / f"{seed_idx:05d}_{end_idx:05d}"
        chunk_ann_dir = chunk_root / "Annotations" / VIDEO_NAME
        ensure_clean_dir(chunk_root)
        ensure_dir(chunk_ann_dir)

        copied = 0
        for p in masks:
            shutil.copy2(p, chunk_ann_dir / Path(p).name)
            copied += 1
        log.info(f"Stored chunk masks: {chunk_ann_dir} ({copied} pngs)")

    # Update progress: rendering
    track_progress[run_id] = {
        "stage": "rendering",
        "progress": 95,
        "message": "Rendering preview video...",
        "current_frame": end_idx,
        "total_frames": end_idx,
    }
    
    # Render preview for the chunk dataset
    jpeg_dir = chunk_ds / "JPEGImages" / VIDEO_NAME
    frames = sorted([p.name for p in jpeg_dir.glob("*.jpg")])

    used = render_video(
        jpeg_dir=jpeg_dir,
        frames=frames,
        found_pngs=masks,
        out_video=run_dir / "tracked.mp4",
        fps=fps,
        n_ids=n_ids,
        run_dir=run_dir,
        behavior_frame_offset=seed_idx,
    )

    tracked_path = run_dir / "tracked.mp4"
    # Note: tracked.mp4 is already H.264 encoded by render_video() using direct ffmpeg, no re-encoding needed
    
    chunk_new = chunk_root / "chunk_new.mp4"   # stored with the chunk
    log.info(f"Preparing chunk_new.mp4: dropping seed frame from {tracked_path} -> {chunk_new}")
    ok = _ffmpeg_drop_seed_frame(tracked_path, chunk_new, fps)
    if ok:
        chunk_new_size = chunk_new.stat().st_size
        chunk_new_dur = _probe_duration(chunk_new)
        log.info(f"✅ Prepared chunk_new video: {chunk_new} (size={chunk_new_size} bytes, duration={chunk_new_dur}s)")
    else:
        log.warning("❌ Could not prepare chunk_new.mp4; commit will fallback to rendering.")


    # remember last chunk for commit
    (run_dir / "last_chunk.txt").write_text(str(chunk_root), encoding="utf-8")
    (run_dir / "last_chunk_meta.txt").write_text(
        f"seed_idx={seed_idx}\nend_idx={end_idx}\n",
        encoding="utf-8",
    )

    log.info(f"/track done rendered={used}")
    
    # Mark tracking as complete
    track_progress[run_id] = {
        "stage": "completed",
        "progress": 100,
        "message": "Tracking complete!",
        "current_frame": end_idx,
        "total_frames": end_idx,
    }
    
    # Clear progress after 5 seconds
    def clear_track_progress_later():
        time.sleep(5)
        track_progress.pop(run_id, None)
        log.debug(f"[TRACK] Cleared progress for {run_id}")
    
    threading.Thread(target=clear_track_progress_later, daemon=True).start()
    
    return {
        "run_id": run_id,
        "seed_idx": seed_idx,
        "end_idx": end_idx,
        "n_frames_rendered": used,
        "chunk_dir": str(chunk_root),
        "log_tail": logs[-30:],
    }

def _probe_duration(path: Path) -> float | None:
    """Get video duration using ffprobe. Returns None if ffprobe is not available or fails."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return None
        try:
            return float(p.stdout.strip())
        except:
            return None
    except FileNotFoundError:
        # ffprobe not found in PATH - this is non-critical, just return None
        log.warning(f"ffprobe not found in PATH, cannot probe video duration for {path}")
        return None
    except Exception as e:
        log.warning(f"Error probing video duration: {e}")
        return None



@app.post("/commit")
def commit(run_id: str):
    """
    Commit the last tracked chunk into golden/Annotations/video1/.
    IMPORTANT: commits only NEW frames (seed+1..end), not the seed frame itself.
    Also appends to golden preview video (best-effort).
    """
    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found")

    last_chunk_file = run_dir / "last_chunk.txt"
    last_chunk_meta = run_dir / "last_chunk_meta.txt"
    if not last_chunk_file.exists() or not last_chunk_meta.exists():
        raise HTTPException(400, "No chunk to commit yet. Run /track first.")

    meta = parse_meta_file(meta_path)
    fps = float(meta["fps"])
    n_ids = int(meta["ids"])
    n_total = int(meta["frames"])

    chunk_root = Path(last_chunk_file.read_text(encoding="utf-8").strip())
    chunk_ann_dir = chunk_root / "Annotations" / VIDEO_NAME

    # Prefer deriving seed/end from the chunk folder name (e.g. chunks/00050_00099),
    # because we've observed `last_chunk_meta.txt` can get out of sync with `last_chunk.txt`.
    # If they mismatch, commit will write masks to the wrong global indices (catastrophic).
    seed_idx = None
    end_idx = None
    try:
        # chunk_root is .../chunks/00050_00099
        name = chunk_root.name
        if "_" in name:
            a, b = name.split("_", 1)
            seed_idx = int(a)
            end_idx = int(b)
    except Exception:
        seed_idx = None
        end_idx = None

    chunk_kv = dict(line.split("=", 1) for line in last_chunk_meta.read_text(encoding="utf-8").splitlines())
    seed_idx_meta = int(chunk_kv["seed_idx"])
    end_idx_meta = int(chunk_kv["end_idx"])

    if seed_idx is None or end_idx is None:
        seed_idx = seed_idx_meta
        end_idx = end_idx_meta
        log.warning(f"[COMMIT] Could not parse seed/end from chunk folder name ({chunk_root.name}); falling back to last_chunk_meta.txt")
    else:
        if seed_idx != seed_idx_meta or end_idx != end_idx_meta:
            log.warning(
                f"[COMMIT] last_chunk mismatch: chunk_root name implies seed={seed_idx}, end={end_idx}, "
                f"but last_chunk_meta.txt says seed={seed_idx_meta}, end={end_idx_meta}. "
                f"Using chunk_root-derived values."
            )

    log.info(f"[COMMIT] Starting commit: seed_idx={seed_idx}, end_idx={end_idx}")
    
    if not chunk_ann_dir.exists():
        raise HTTPException(500, f"Chunk annotations missing: {chunk_ann_dir}")

    golden_ann_dir = get_golden_ann_dir(run_dir)
    ensure_dir(golden_ann_dir)
    
    # Find all chunks that need to be committed.
    # IMPORTANT: don't assume golden is contiguous (we've observed holes like missing 1..50 while having 51..99).
    # So we scan chunks and include any chunk that can "fill" at least one missing golden frame.
    processed, pct, max_golden_idx = golden_progress(run_dir, n_total)
    last_committed_frame = max_golden_idx if max_golden_idx is not None else -1
    
    chunks_dir = run_dir / "chunks"
    all_chunks_to_commit = []
    if chunks_dir.exists():
        all_chunk_folders = sorted([f for f in chunks_dir.iterdir() if f.is_dir()])
        
        # Determine an upper bound we are willing to commit up to for this call.
        # Use the end_idx derived from last_chunk, but also consider the maximum chunk_end we see on disk.
        max_chunk_end_on_disk = None
        for chunk_folder in all_chunk_folders:
            try:
                name = chunk_folder.name
                if "_" in name:
                    ce = int(name.split("_")[1])
                    max_chunk_end_on_disk = ce if max_chunk_end_on_disk is None else max(max_chunk_end_on_disk, ce)
            except Exception:
                continue
        commit_end_limit = end_idx
        if max_chunk_end_on_disk is not None and max_chunk_end_on_disk > commit_end_limit:
            commit_end_limit = max_chunk_end_on_disk

        for chunk_folder in all_chunk_folders:
            try:
                # Parse chunk name like "00050_00099"
                name = chunk_folder.name
                if "_" in name:
                    chunk_seed = int(name.split("_")[0])
                    chunk_end = int(name.split("_")[1])
                    # Include this chunk if:
                    # - it is within our commit limit
                    # - AND it can fill at least one missing golden annotation in its (seed+1..end) range
                    within_limit = chunk_end <= commit_end_limit
                    has_missing = False
                    if within_limit:
                        for gi in range(chunk_seed + 1, chunk_end + 1):
                            if not (golden_ann_dir / f"{gi:05d}.png").exists():
                                has_missing = True
                                break
                    should_include = within_limit and has_missing
                    if should_include:
                        all_chunks_to_commit.append((chunk_seed, chunk_end, chunk_folder))
            except (ValueError, IndexError) as e:
                log.warning(f"[COMMIT] Failed to parse chunk folder {chunk_folder.name}: {e}")
                continue
    
    # If we found chunks, commit them all; otherwise fall back to the last chunk
    if len(all_chunks_to_commit) > 0:
        log.debug(f"[COMMIT] Found {len(all_chunks_to_commit)} chunks to commit")
    else:
        # Fall back to single chunk commit (original behavior)
        all_chunks_to_commit = [(seed_idx, end_idx, chunk_root)]

    # Commit NEW frames only: seed+1..end
    # Also copy JPEG frames to golden/JPEGImages/video1/
    golden_jpeg_dir = get_golden_jpeg_dir(run_dir)
    ensure_dir(golden_jpeg_dir)
    
    src_root = run_dir / "xmem_generic"
    src_jpeg = src_root / "JPEGImages" / VIDEO_NAME
    
    committed = 0
    skipped_corrected = 0
    
    # Commit all chunks (important for auto-reset where multiple chunks are created)
    for chunk_seed, chunk_end, chunk_folder in all_chunks_to_commit:
        chunk_ann_dir_this = chunk_folder / "Annotations" / VIDEO_NAME
        if not chunk_ann_dir_this.exists():
            log.warning(f"[COMMIT] Skipping chunk {chunk_folder.name} - annotations missing")
            continue
        
        log.debug(f"[COMMIT] Processing chunk {chunk_folder.name}: seed={chunk_seed}, end={chunk_end}")
        
        # Check for corrected frames in this chunk's range
        chunk_last_corrected = None
        for orig_idx in range(chunk_seed + 1, chunk_end + 1):
            golden_mask = golden_ann_dir / f"{orig_idx:05d}.png"
            if golden_mask.exists():
                chunk_rel = orig_idx - chunk_seed
                chunk_mask_path = chunk_ann_dir_this / f"{chunk_rel:05d}.png"
                if chunk_mask_path.exists():
                    golden_mask_data = np.array(Image.open(golden_mask))
                    chunk_mask_data = np.array(Image.open(chunk_mask_path))
                    if not np.array_equal(golden_mask_data, chunk_mask_data):
                        chunk_last_corrected = orig_idx
                        log.debug(f"[COMMIT] Frame {orig_idx} is corrected (golden mask differs from chunk mask)")
        
        # Determine actual end for this chunk (respect corrected frames)
        chunk_commit_end = chunk_last_corrected if chunk_last_corrected is not None else chunk_end
        if chunk_last_corrected is not None:
            log.debug(f"[COMMIT] Chunk {chunk_folder.name}: found corrected frame at {chunk_last_corrected}, will only commit up to this frame")
        
        # First, check if the seed frame exists in golden - if not, copy it
        # (The seed frame is at relative index 0 in the chunk)
        seed_mask_src = chunk_ann_dir_this / "00000.png"
        seed_mask_dst = golden_ann_dir / f"{chunk_seed:05d}.png"
        if seed_mask_src.exists() and not seed_mask_dst.exists():
            log.debug(f"[COMMIT] Seed frame {chunk_seed} not in golden, copying it")
            shutil.copy2(seed_mask_src, seed_mask_dst)
            # Also copy JPEG frame
            seed_jpeg_src = src_jpeg / f"{chunk_seed:05d}.jpg"
            if seed_jpeg_src.exists():
                seed_jpeg_dst = golden_jpeg_dir / f"{chunk_seed:05d}.jpg"
                shutil.copy2(seed_jpeg_src, seed_jpeg_dst)
            committed += 1
        
        # Commit frames from this chunk: seed+1 to commit_end (inclusive)
        # Collect all files to copy for parallel processing
        files_to_copy = []
        jpeg_files_to_copy = []
        
        for orig_idx in range(chunk_seed + 1, chunk_commit_end + 1):
            rel = orig_idx - chunk_seed  # in chunk dataset, seed=0, next frame=1, ...
            src = chunk_ann_dir_this / f"{rel:05d}.png"
            
            if not src.exists():
                # If the file doesn't exist, it means we've reached the end of the chunk
                log.warning(f"[COMMIT] Missing chunk mask for frame {orig_idx} (relative {rel}) - reached end of chunk, stopping")
                break

            dst = golden_ann_dir / f"{orig_idx:05d}.png"
            
            # Validate frame index alignment
            log.debug(f"[COMMIT] Copying frame: chunk_seed={chunk_seed}, orig_idx={orig_idx}, rel={rel}, src={src.name}, dst={dst.name}")
            
            # Check if this frame already has a corrected mask in golden
            if dst.exists():
                # Load both masks to compare
                existing_mask = np.array(Image.open(dst))
                chunk_mask = np.array(Image.open(src))
                
                # Validate dimensions match (safety check for frame alignment)
                if existing_mask.shape != chunk_mask.shape:
                    log.error(f"[COMMIT] ⚠️  DIMENSION MISMATCH for frame {orig_idx}! Existing: {existing_mask.shape}, Chunk: {chunk_mask.shape}")
                    log.error(f"[COMMIT] This suggests a frame index mismatch! Skipping this frame.")
                    continue
                
                # Check if masks are different (not just same IDs)
                masks_different = not np.array_equal(existing_mask, chunk_mask)
                
                if masks_different:
                    log.info(f"[COMMIT] Frame {orig_idx} has corrected mask in golden (differs from chunk), skipping overwrite")
                    skipped_corrected += 1
                    # Don't overwrite - keep the corrected mask
                else:
                    # Masks are identical, safe to overwrite
                    files_to_copy.append((src, dst))
                    log.debug(f"[COMMIT] Will copy frame {orig_idx}: {src.name} -> {dst.name}")
            else:
                # Frame doesn't exist in golden, safe to copy
                files_to_copy.append((src, dst))
                log.debug(f"[COMMIT] Will copy NEW frame {orig_idx}: {src.name} -> {dst.name}")
            
            # Always collect JPEG frame for copying (even if mask was skipped)
            src_jpeg_frame = src_jpeg / f"{orig_idx:05d}.jpg"
            if src_jpeg_frame.exists():
                dst_jpeg_frame = golden_jpeg_dir / f"{orig_idx:05d}.jpg"
                jpeg_files_to_copy.append((src_jpeg_frame, dst_jpeg_frame))
        
        # Copy all files in parallel
        if files_to_copy:
            copied_count = copy_files_parallel(files_to_copy, max_workers=8)
            committed += copied_count
            log.debug(f"[COMMIT] Copied {copied_count} mask files in parallel")
        
        if jpeg_files_to_copy:
            copy_files_parallel(jpeg_files_to_copy, max_workers=8)
            log.debug(f"[COMMIT] Copied {len(jpeg_files_to_copy)} JPEG files in parallel")

    # Calculate actual committed range
    if all_chunks_to_commit:
        first_chunk_seed = all_chunks_to_commit[0][0]
        last_chunk_end = all_chunks_to_commit[-1][1]
        committed_range = f"{first_chunk_seed+1}..{last_chunk_end}"
    else:
        committed_range = f"{seed_idx+1}..{end_idx}"
    
    log.info(f"[COMMIT] Commit complete:")
    log.info(f"[COMMIT]   Committed {committed} NEW frames to golden: {golden_ann_dir} ({committed_range})")
    if skipped_corrected > 0:
        log.info(f"[COMMIT]   Skipped {skipped_corrected} frames that had corrected masks in golden")
    log.info(f"[COMMIT]   Copied {committed + skipped_corrected} JPEG frames to golden: {golden_jpeg_dir}")

    # Update golden preview video
    # Simple logic:
    # - If no corrected frames: just append chunk_new.mp4 (tracked preview) to golden preview
    # - If corrected frames exist: append tracked preview up to first corrected frame, then render corrected frames from golden
    log.debug("[COMMIT] Updating golden preview video")
    try:
        golden_preview = run_dir / "golden" / "golden_preview.mp4"
        chunk_new = chunk_root / "chunk_new.mp4"
        tracked_path = run_dir / "tracked.mp4"
        
        # Find the LAST corrected frame in the commit range (if any)
        # If found, we only commit up to that frame and discard everything after
        last_corrected_frame = None
        for orig_idx in range(seed_idx + 1, end_idx + 1):
            golden_mask = golden_ann_dir / f"{orig_idx:05d}.png"
            if golden_mask.exists():
                # Check if it's different from chunk mask (would indicate correction)
                chunk_rel = orig_idx - seed_idx
                chunk_mask_path = chunk_ann_dir / f"{chunk_rel:05d}.png"
                if chunk_mask_path.exists():
                    golden_mask_data = np.array(Image.open(golden_mask))
                    chunk_mask_data = np.array(Image.open(chunk_mask_path))
                    if not np.array_equal(golden_mask_data, chunk_mask_data):
                        last_corrected_frame = orig_idx  # Keep updating to find the LAST one
                        log.debug(f"[COMMIT] Found corrected frame: {last_corrected_frame}")
        
        if last_corrected_frame is not None:
            log.info(f"[COMMIT] Last corrected frame is {last_corrected_frame}, will only commit up to this frame")
            # Update end_idx to only commit up to the corrected frame
            end_idx = last_corrected_frame
        
        if last_corrected_frame is not None:
            # Partial commit: commit only up to the corrected frame, discard everything after
            log.info(f"[COMMIT] Partial commit: frames {seed_idx+1}..{end_idx} (corrected frame at {last_corrected_frame}, discarding frames {end_idx+1}..{chunk_kv.get('end_idx', '?')})")
            
            # Extract tracked segment from tracked.mp4: frames seed+1..(last_corrected_frame-1)
            # tracked.mp4 contains frames seed..original_end_idx (seed is frame 0 in the video)
            # We want frames 1..(last_corrected_frame-seed_idx-1) from tracked.mp4
            if last_corrected_frame > seed_idx + 1:
                # Extract tracked segment up to corrected frame
                tracked_seg_start = 1  # Skip seed frame (frame 0 in video)
                tracked_seg_end = last_corrected_frame - seed_idx - 1  # Last frame before correction
                tracked_seg_path = run_dir / "golden_segments" / f"tracked_{seed_idx+1}_{last_corrected_frame-1}.mp4"
                ensure_dir(tracked_seg_path.parent)
                
                log.info(f"[COMMIT] Extracting tracked segment: frames {tracked_seg_start}..{tracked_seg_end} from tracked.mp4 (absolute frames {seed_idx+1}..{last_corrected_frame-1})")
                # Extract frames using ffmpeg
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(tracked_path),
                    "-vf", f"select='gte(n,{tracked_seg_start})*lt(n,{tracked_seg_end+1})',setpts=N/({fps:.10f}*TB)",
                    "-r", f"{fps:.10f}",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "20",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(tracked_seg_path),
                ]
                log.info(f"[COMMIT] Running: {' '.join(cmd)}")
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if p.returncode == 0 and tracked_seg_path.exists():
                    log.info(f"[COMMIT] ✅ Extracted tracked segment: {tracked_seg_path}")
                    # Append tracked segment to golden preview
                    if golden_preview.exists():
                        _ffmpeg_concat(golden_preview, tracked_seg_path, golden_preview, fps)
                    else:
                        golden_preview.write_bytes(tracked_seg_path.read_bytes())
                    if get_annotation_mode(run_dir) == "behavior":
                        append_masks_only_golden_segment(
                            run_dir, fps, n_ids, seed_idx + 1, last_corrected_frame - 1
                        )
                else:
                    log.error(f"[COMMIT] Failed to extract tracked segment: {p.stdout[-500:] if p.stdout else 'no output'}")
            
            # Render corrected frame(s) from golden: last_corrected_frame only (or range if multiple corrected)
            log.info(f"[COMMIT] Rendering corrected frame(s) from golden: {last_corrected_frame}")
            corrected_seg_path = run_dir / "golden_segments" / f"{last_corrected_frame:05d}_{last_corrected_frame:05d}.mp4"
            ensure_dir(corrected_seg_path.parent)
            _render_segment_from_golden(run_dir, fps, n_ids, last_corrected_frame, last_corrected_frame, corrected_seg_path)
            
            if corrected_seg_path.exists():
                seg_reencoded = run_dir / "golden_segments" / f"{last_corrected_frame:05d}_{last_corrected_frame:05d}_reencoded.mp4"
                if _ffmpeg_reencode_video(corrected_seg_path, seg_reencoded, fps):
                    seg_reencoded.replace(corrected_seg_path)
                
                # Append corrected frame to golden preview
                if golden_preview.exists():
                    _ffmpeg_concat(golden_preview, corrected_seg_path, golden_preview, fps)
                else:
                    golden_preview.write_bytes(corrected_seg_path.read_bytes())
                if get_annotation_mode(run_dir) == "behavior":
                    append_masks_only_golden_segment(
                        run_dir, fps, n_ids, last_corrected_frame, last_corrected_frame
                    )
                log.debug(f"[COMMIT] Golden preview updated: tracked frames {seed_idx+1}..{last_corrected_frame-1} + corrected frame {last_corrected_frame}")
                log.debug(f"[COMMIT] Frames {last_corrected_frame+1}..{int(chunk_kv.get('end_idx', end_idx))} were discarded (will be re-tracked from frame {last_corrected_frame})")
        else:
            # Full commit: no corrections, just append chunk_new.mp4
            log.info(f"[COMMIT] Full commit: no corrected frames, appending chunk_new.mp4")
            if chunk_new.exists():
                chunk_new_size = chunk_new.stat().st_size
                chunk_new_dur = _probe_duration(chunk_new)
                log.info(f"[COMMIT] chunk_new.mp4: size={chunk_new_size} bytes, duration={chunk_new_dur}s")
                
                if golden_preview.exists():
                    golden_preview_size = golden_preview.stat().st_size
                    golden_preview_dur = _probe_duration(golden_preview)
                    log.info(f"[COMMIT] Existing golden_preview.mp4: size={golden_preview_size} bytes, duration={golden_preview_dur}s")
                    log.info(f"[COMMIT] Appending chunk_new to golden_preview...")
                    
                    ok = _ffmpeg_concat(golden_preview, chunk_new, golden_preview, fps)
                    
                    if golden_preview.exists():
                        final_size = golden_preview.stat().st_size
                        final_dur = _probe_duration(golden_preview)
                        log.info(f"[COMMIT] After concat - golden_preview.mp4: size={final_size} bytes, duration={final_dur}s")
                    
                    if ok:
                        log.debug(f"[COMMIT] Successfully appended chunk_new to golden_preview.mp4")
                    else:
                        log.error("[COMMIT] ❌ Concat failed (non-fatal).")
                    if get_annotation_mode(run_dir) == "behavior":
                        append_masks_only_golden_segment(run_dir, fps, n_ids, seed_idx + 1, end_idx)
                else:
                    log.info("[COMMIT] golden_preview.mp4 does not exist, initializing from chunk_new.mp4...")
                    golden_preview.parent.mkdir(parents=True, exist_ok=True)
                    golden_preview.write_bytes(chunk_new.read_bytes())
                    init_size = golden_preview.stat().st_size
                    init_dur = _probe_duration(golden_preview)
                    log.debug(f"[COMMIT] Initialized golden_preview.mp4 from chunk_new.mp4: size={init_size} bytes, duration={init_dur}s")
                    if get_annotation_mode(run_dir) == "behavior":
                        append_masks_only_golden_segment(run_dir, fps, n_ids, seed_idx + 1, end_idx)
            else:
                log.warning(f"[COMMIT] ❌ chunk_new.mp4 missing at {chunk_new}; falling back to rendering from golden")
                seg_path = run_dir / "golden_segments" / f"{seed_idx+1:05d}_{end_idx:05d}.mp4"
                ensure_dir(seg_path.parent)
                log.info(f"[COMMIT] Rendering segment from golden: {seed_idx+1}..{end_idx}")
                _render_segment_from_golden(run_dir, fps, n_ids, seed_idx + 1, end_idx, seg_path)
                
                if seg_path.exists():
                    seg_reencoded = run_dir / "golden_segments" / f"{seed_idx+1:05d}_{end_idx:05d}_reencoded.mp4"
                    if _ffmpeg_reencode_video(seg_path, seg_reencoded, fps):
                        seg_reencoded.replace(seg_path)
                    
                    if golden_preview.exists():
                        _ffmpeg_concat(golden_preview, seg_path, golden_preview, fps)
                    else:
                        golden_preview.write_bytes(seg_path.read_bytes())
                    if get_annotation_mode(run_dir) == "behavior":
                        append_masks_only_golden_segment(run_dir, fps, n_ids, seed_idx + 1, end_idx)
                    log.debug(f"[COMMIT] Golden preview updated from rendered segment")
    except Exception as e:
        log.error(f"[COMMIT] ❌ Golden preview update failed (non-fatal): {e}", exc_info=True)
    
    log.info("=" * 60)


    processed, pct, max_idx = golden_progress(run_dir, n_total)

    if get_annotation_mode(run_dir) == "behavior" and load_behavior_data(run_dir) is not None:
        mark_behavior_preview_out_of_sync(run_dir)

    return {
        "run_id": run_id,
        "committed_new_frames": committed,
        "golden_processed": processed,
        "golden_percent": pct,
        "golden_max_idx": max_idx,
        "seed_idx": seed_idx,
        "end_idx": end_idx,
        "preview_in_sync": False if get_annotation_mode(run_dir) == "behavior" else None,
    }


@app.get("/get_frame_from_time/{run_id}")
def get_frame_from_time(run_id: str, video_time: float):
    """
    Get frame number from video playback time for tracked video.
    Returns relative frame number (relative to chunk start).
    """
    log.info(f"/get_frame_from_time run_id={run_id} video_time={video_time}")
    
    run_dir = RUNS_ROOT / run_id
    meta = parse_meta_file(run_dir / "meta.txt")
    source_fps = float(meta.get("fps", 30.0))
    
    # Get actual video properties
    tracked_video_path = run_dir / "tracked.mp4"
    cap = cv2.VideoCapture(str(tracked_video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Calculate relative frame from video time
    relative_frame = int(video_time * video_fps)
    relative_frame = max(0, min(relative_frame, video_frame_count - 1))
    
    # Get last golden frame to calculate absolute frame
    processed, pct, max_idx = golden_progress(run_dir, int(meta["frames"]))
    absolute_frame = max_idx + relative_frame if max_idx is not None else relative_frame
    
    return {
        "relative_frame": relative_frame,
        "absolute_frame": absolute_frame,
        "video_time": video_time,
        "video_fps": float(video_fps),
        "video_frame_count": video_frame_count,
    }


@app.get("/track_progress/{run_id}")
def get_track_progress(run_id: str):
    """
    Get progress for tracking operation.
    Returns progress info if tracking is in progress, or None if completed/not found.
    """
    progress = track_progress.get(run_id)
    if progress is None:
        log.debug(f"[TRACK_PROGRESS] {run_id}: not found (not started yet or completed)")
        # Return "not_started" instead of "completed" - frontend will keep polling
        return {"status": "not_started", "progress": 0, "message": "Waiting to start..."}
    
    # Check if it's actually completed
    if progress.get("stage") == "completed":
        result = {
            "status": "completed",
            "stage": "completed",
            "progress": 100,
            "message": progress.get("message", "Completed"),
            "current_frame": progress.get("current_frame"),
            "total_frames": progress.get("total_frames"),
        }
        log.debug(f"[TRACK_PROGRESS] {run_id}: completed")
        return result
    
    result = {
        "status": "in_progress",
        "stage": progress["stage"],
        "progress": progress["progress"],
        "message": progress["message"],
        "current_frame": progress.get("current_frame"),
        "total_frames": progress.get("total_frames"),
    }
    log.debug(f"[TRACK_PROGRESS] {run_id}: {progress['stage']} {progress['progress']}% - {progress['message']}")
    return result


@app.get("/progress/{run_id}")
def progress(run_id: str):
    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found")

    meta = parse_meta_file(meta_path)
    n_total = int(meta["frames"])
    fps = float(meta.get("fps", 30.0))

    processed, pct, max_idx = golden_progress(run_dir, n_total)
    
    # Get last chunk info if available (for frame offset calculation)
    last_chunk_meta = run_dir / "last_chunk_meta.txt"
    seed_idx = None
    if last_chunk_meta.exists():
        chunk_kv = dict(line.split("=", 1) for line in last_chunk_meta.read_text(encoding="utf-8").splitlines())
        seed_idx = int(chunk_kv.get("seed_idx", max_idx if max_idx is not None else 0))
    
    # Get last chunk info for frame calculation
    last_chunk_meta = run_dir / "last_chunk_meta.txt"
    chunk_frames = None
    if last_chunk_meta.exists():
        chunk_kv = dict(line.split("=", 1) for line in last_chunk_meta.read_text(encoding="utf-8").splitlines())
        seed_idx = int(chunk_kv.get("seed_idx", max_idx if max_idx is not None else 0))
        end_idx = int(chunk_kv.get("end_idx", seed_idx))
        # Number of frames in the tracked video (includes seed frame)
        chunk_frames = end_idx - seed_idx + 1
    
    return {
        "run_id": run_id,
        "total_frames": n_total,
        "fps": fps,
        "golden_processed": processed,
        "golden_percent": pct,
        "golden_max_idx": max_idx,
        "last_chunk_seed_idx": seed_idx,  # For calculating absolute frame from tracked video
        "last_chunk_frames": chunk_frames,  # Number of frames in current tracked video
        "annotation_mode": get_annotation_mode(run_dir),
    }


@app.get("/frame0/{run_id}")
def frame0(run_id: str):

    jpeg0 = RUNS_ROOT / run_id / "xmem_generic" / "JPEGImages" / VIDEO_NAME / "00000.jpg"
    ann0  = RUNS_ROOT / run_id / "xmem_generic" / "Annotations" / VIDEO_NAME / "00000.png"

    if not jpeg0.exists():
        raise HTTPException(status_code=404, detail="frame0 jpg not found")

    frame = cv2.imread(str(jpeg0))
    if frame is None:
        raise HTTPException(status_code=500, detail="could not read frame0")

    if ann0.exists():
        labels = np.array(Image.open(ann0))
        max_id = int(labels.max())

        for cid in range(1, max_id + 1):
            m = (labels == cid)
            if not m.any():
                continue

            col = get_color_for_id(cid)
            overlay = frame.copy()
            overlay[m] = col
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

            ys, xs = np.where(m)
            cx, cy = int(xs.mean()), int(ys.mean())
            
            # Get text size to center it properly
            text = str(cid)
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Center the text (putText uses bottom-left corner, so adjust)
            text_x = cx - text_width // 2
            text_y = cy + text_height // 2
            
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

    # Encode frame to JPEG bytes for Response
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/tracked_frame/{run_id}/{relative_frame_idx}")
def get_tracked_frame(run_id: str, relative_frame_idx: int):
    """
    Get a frame from the current tracked chunk by relative frame index.
    Returns frame image with mask overlays from the chunk.
    """
    
    log.info(f"/tracked_frame run_id={run_id} relative_frame_idx={relative_frame_idx}")
    
    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found")
    
    # Get last chunk info
    last_chunk_meta = run_dir / "last_chunk_meta.txt"
    if not last_chunk_meta.exists():
        raise HTTPException(400, "No tracked chunk available. Run /track first.")
    
    chunk_kv = dict(line.split("=", 1) for line in last_chunk_meta.read_text(encoding="utf-8").splitlines())
    seed_idx = int(chunk_kv["seed_idx"])
    end_idx = int(chunk_kv["end_idx"])
    
    # Convert relative to absolute frame
    absolute_frame = seed_idx + relative_frame_idx
    
    if absolute_frame < seed_idx or absolute_frame > end_idx:
        raise HTTPException(400, f"Relative frame {relative_frame_idx} is out of chunk range [0, {end_idx-seed_idx}]")
    
    log.info(f"[TRACKED_FRAME] Relative frame {relative_frame_idx} -> absolute frame {absolute_frame} (chunk: {seed_idx}..{end_idx})")
    
    # Get frame image
    src_root = run_dir / "xmem_generic"
    jpeg_dir = src_root / "JPEGImages" / VIDEO_NAME
    frame_path = jpeg_dir / f"{absolute_frame:05d}.jpg"
    
    log.info(f"[TRACKED_FRAME] Loading frame image: {frame_path} (exists: {frame_path.exists()})")
    if not frame_path.exists():
        raise HTTPException(404, f"Frame {absolute_frame} not found")
    
    frame = load_frame_safely(frame_path, frame_idx=absolute_frame)
    log.info(f"[TRACKED_FRAME] Frame image loaded: shape={frame.shape}")
    
    # Find tracked mask for this frame (search in golden and all chunks)
    log.info(f"[TRACKED_FRAME] Searching for tracked mask for absolute frame {absolute_frame}")
    ann_path, ann_source = find_tracked_mask_for_frame(run_dir, absolute_frame)
    
    if ann_path and ann_path.exists():
        log.info(f"[TRACKED_FRAME] Found annotation: {ann_path} (source: {ann_source})")
        labels = np.array(Image.open(ann_path))
        max_id = int(labels.max())
        unique_ids = sorted(list(set(labels.flatten())))
        unique_ids = [id for id in unique_ids if id > 0]  # Remove background
        log.info(f"[TRACKED_FRAME] Annotation contains {len(unique_ids)} object IDs: {unique_ids}, max_id={max_id}")
        
        rendered_count = 0
        for cid in range(1, max_id + 1):
            m = (labels == cid)
            if not m.any():
                continue
            rendered_count += 1
            mask_pixels = int(m.sum())
            log.info(f"[TRACKED_FRAME] Rendering mask for ID {cid} ({mask_pixels} pixels)")
            
            col = get_color_for_id(cid)
            overlay = frame.copy()
            overlay[m] = col
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            
            ys, xs = np.where(m)
            cx, cy = int(xs.mean()), int(ys.mean())
            
            text = str(cid)
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            text_x = cx - text_width // 2
            text_y = cy + text_height // 2
            
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        log.info(f"[TRACKED_FRAME] Rendered {rendered_count} masks for frame {absolute_frame}")
    else:
        log.info(f"[TRACKED_FRAME] No annotation found for frame {absolute_frame} (path: {ann_path}, source: {ann_source})")
    
    # Encode frame to JPEG bytes for Response
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/frame/{run_id}/{frame_idx}")
def get_frame(run_id: str, frame_idx: int):
    """
    Get a specific frame with annotations (from golden or chunk).
    Returns frame image with mask overlays.
    """

    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found")
    
    meta = parse_meta_file(meta_path)
    n_total = int(meta["frames"])
    
    if frame_idx < 0 or frame_idx >= n_total:
        raise HTTPException(400, f"frame_idx {frame_idx} out of range [0, {n_total-1}]")
    
    # Try to get frame from golden first, then from chunk
    src_root = run_dir / "xmem_generic"
    jpeg_dir = src_root / "JPEGImages" / VIDEO_NAME
    frame_path = jpeg_dir / f"{frame_idx:05d}.jpg"
    
    if not frame_path.exists():
        raise HTTPException(404, f"Frame {frame_idx} not found")
    
    frame = load_frame_safely(frame_path, frame_idx=frame_idx)
    
    # Try golden annotation first
    golden_ann_dir = get_golden_ann_dir(run_dir)
    ann_path = golden_ann_dir / f"{frame_idx:05d}.png"
    
    # If not in golden, try chunk
    if not ann_path.exists():
        # Find which chunk contains this frame
        chunk_dirs = sorted((run_dir / "chunks").glob("*_*")) if (run_dir / "chunks").exists() else []
        for chunk_dir in chunk_dirs:
            name = chunk_dir.name
            try:
                start_idx, end_idx = map(int, name.split("_"))
                if start_idx <= frame_idx <= end_idx:
                    # Frame is in this chunk, but need to map to chunk's internal numbering
                    rel_idx = frame_idx - start_idx
                    chunk_ann_dir = chunk_dir / "Annotations" / VIDEO_NAME
                    chunk_ann_path = chunk_ann_dir / f"{rel_idx:05d}.png"
                    if chunk_ann_path.exists():
                        ann_path = chunk_ann_path
                        break
            except:
                continue
    
    if ann_path.exists():
        labels = np.array(Image.open(ann_path))
        max_id = int(labels.max())
        
        for cid in range(1, max_id + 1):
            m = (labels == cid)
            if not m.any():
                continue
            
            col = get_color_for_id(cid)
            overlay = frame.copy()
            overlay[m] = col
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            
            ys, xs = np.where(m)
            cx, cy = int(xs.mean()), int(ys.mean())
            
            # Get text size to center it properly
            text = str(cid)
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Center the text (putText uses bottom-left corner, so adjust)
            text_x = cx - text_width // 2
            text_y = cy + text_height // 2
            
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )
    
    # Encode frame to JPEG bytes for Response
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


def find_tracked_mask_for_frame(run_dir: Path, frame_idx: int) -> Tuple[Optional[Path], str]:
    """
    Find the tracked mask annotation for a given frame.
    Searches in golden first, then in all chunks.
    Returns (annotation_path, source_description) or (None, "not found") if not found.
    """
    
    log.info(f"[FIND_MASK] Searching for tracked mask for frame {frame_idx}")
    
    golden_ann_dir = get_golden_ann_dir(run_dir)
    ann_path = golden_ann_dir / f"{frame_idx:05d}.png"
    
    if ann_path.exists():
        log.info(f"[FIND_MASK] Found in golden: {ann_path}")
        return ann_path, "golden"
    else:
        log.info(f"[FIND_MASK] Not in golden: {ann_path} (exists: {ann_path.exists()})")
    
    # Search through all chunks
    chunk_dirs = sorted((run_dir / "chunks").glob("*_*")) if (run_dir / "chunks").exists() else []
    log.info(f"[FIND_MASK] Searching {len(chunk_dirs)} chunks: {[d.name for d in chunk_dirs]}")
    
    for chunk_dir in chunk_dirs:
        name = chunk_dir.name
        try:
            start_idx, end_idx = map(int, name.split("_"))
            log.info(f"[FIND_MASK] Checking chunk {name}: range {start_idx}..{end_idx}, frame {frame_idx} in range: {start_idx <= frame_idx <= end_idx}")
            if start_idx <= frame_idx <= end_idx:
                # Frame is in this chunk, map to chunk's internal numbering
                rel_idx = frame_idx - start_idx
                chunk_ann_dir = chunk_dir / "Annotations" / VIDEO_NAME
                chunk_ann_path = chunk_ann_dir / f"{rel_idx:05d}.png"
                log.info(f"[FIND_MASK] Chunk {name}: frame {frame_idx} -> local idx {rel_idx}, path: {chunk_ann_path} (exists: {chunk_ann_path.exists()})")
                if chunk_ann_path.exists():
                    log.info(f"[FIND_MASK] Found in chunk {name}: {chunk_ann_path}")
                    return chunk_ann_path, f"chunk_{name}"
        except Exception as e:
            log.warning(f"[FIND_MASK] Error parsing chunk {name}: {e}")
            continue
    
    log.info(f"[FIND_MASK] Not found for frame {frame_idx}")
    return None, "not found"


@app.post("/prepare_correction/{run_id}/{frame_idx}")
def prepare_correction(run_id: str, frame_idx: int):
    """
    Prepare frame for correction:
    1. Commit frames before frame_idx to golden
    2. Run SAM-3 on frame_idx
    3. Auto-assign IDs based on previous frame
    4. Return frame with masks, auto-assigned IDs, and list of existing IDs
    """
    
    log.info(f"/prepare_correction run_id={run_id} frame_idx={frame_idx}")
    
    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found")
    
    meta = parse_meta_file(meta_path)
    fps = float(meta["fps"])
    n_ids = int(meta["ids"])
    n_total = int(meta["frames"])
    prompt = meta.get("prompt", "object")
    
    if frame_idx < 1:
        raise HTTPException(400, "frame_idx must be >= 1 (cannot correct frame 0 this way)")
    if frame_idx >= n_total:
        raise HTTPException(400, f"frame_idx {frame_idx} >= total frames {n_total}")
    
    # Step 1: Commit frames up to (frame_idx - 1) - reuse logic from correct_frame
    processed, pct, max_idx = golden_progress(run_dir, n_total)
    if max_idx is None:
        raise HTTPException(500, "No golden frames found")
    
    log.info(f"[DEBUG] Golden progress: max_idx={max_idx}, frame_idx={frame_idx}")
    
    commit_up_to = frame_idx - 1
    log.info(f"[DEBUG] Need to commit frames up to: {commit_up_to}, current max_idx: {max_idx}")
    
    # Define golden_ann_dir early (used later for getting existing IDs)
    golden_ann_dir = get_golden_ann_dir(run_dir)
    
    committed_count = 0
    if commit_up_to > max_idx:
        # Use the same commit logic as /commit, but limit to commit_up_to (frame_idx - 1)
        # Get last chunk info (needed for fallback, same as /commit)
        last_chunk_file = run_dir / "last_chunk.txt"
        last_chunk_meta = run_dir / "last_chunk_meta.txt"
        if not last_chunk_file.exists() or not last_chunk_meta.exists():
            raise HTTPException(400, f"Cannot commit up to frame {commit_up_to}: no chunk available")
        
        chunk_root = Path(last_chunk_file.read_text(encoding="utf-8").strip())
        chunk_ann_dir = chunk_root / "Annotations" / VIDEO_NAME
        
        # Prefer deriving seed/end from the chunk folder name (same as /commit)
        seed_idx = None
        end_idx = None
        try:
            name = chunk_root.name
            if "_" in name:
                a, b = name.split("_", 1)
                seed_idx = int(a)
                end_idx = int(b)
        except Exception:
            seed_idx = None
            end_idx = None
        
        chunk_kv = dict(line.split("=", 1) for line in last_chunk_meta.read_text(encoding="utf-8").splitlines())
        seed_idx_meta = int(chunk_kv["seed_idx"])
        end_idx_meta = int(chunk_kv["end_idx"])
        
        if seed_idx is None or end_idx is None:
            seed_idx = seed_idx_meta
            end_idx = end_idx_meta
        
        # Find all chunks that need to be committed (same logic as /commit)
        chunks_dir = run_dir / "chunks"
        all_chunks_to_commit = []
        if chunks_dir.exists():
            all_chunk_folders = sorted([f for f in chunks_dir.iterdir() if f.is_dir()])
            
            # Use commit_up_to as the limit (instead of end_idx in regular commit)
            commit_end_limit = commit_up_to
            
            # Collect all candidate chunks first
            candidate_chunks = []
            for chunk_folder in all_chunk_folders:
                try:
                    name = chunk_folder.name
                    if "_" in name:
                        chunk_seed = int(name.split("_")[0])
                        chunk_end = int(name.split("_")[1])
                        # Include this chunk if:
                        # - it overlaps with our commit range (max_idx+1..commit_up_to)
                        # - AND it can fill at least one missing golden annotation in the overlapping range
                        # A chunk overlaps if: chunk_seed+1 <= commit_up_to AND chunk_end >= max_idx+1
                        overlaps_commit_range = (chunk_seed + 1) <= commit_up_to and chunk_end >= (max_idx + 1)
                        has_missing = False
                        if overlaps_commit_range:
                            # Check for missing frames only in the range we actually want to commit: (max_idx+1)..commit_up_to
                            check_start = max(chunk_seed + 1, max_idx + 1)
                            check_end = min(chunk_end, commit_up_to)
                            for gi in range(check_start, check_end + 1):
                                if not (golden_ann_dir / f"{gi:05d}.png").exists():
                                    has_missing = True
                                    break
                        should_include = overlaps_commit_range and has_missing
                        log.info(f"[PREPARE_CORRECTION] Chunk {name}: seed={chunk_seed}, end={chunk_end}, max_idx={max_idx}, commit_up_to={commit_up_to}, overlaps={overlaps_commit_range}, has_missing={has_missing}, include={should_include}")
                        if should_include:
                            candidate_chunks.append((chunk_seed, chunk_end, chunk_folder))
                except (ValueError, IndexError) as e:
                    log.warning(f"[PREPARE_CORRECTION] Failed to parse chunk folder {chunk_folder.name}: {e}")
                    continue
            
            # Sort chunks by seed descending (newest first) to prefer newer tracking over older
            # This ensures that when multiple chunks cover the same frame range, we use the newest one
            candidate_chunks.sort(key=lambda x: x[0], reverse=True)
            all_chunks_to_commit = candidate_chunks
        
        # If we found chunks, commit them all; otherwise fall back to the last chunk (same as /commit)
        if len(all_chunks_to_commit) > 0:
            log.info(f"[PREPARE_CORRECTION] Found {len(all_chunks_to_commit)} chunks to commit (sorted newest first)")
        else:
            all_chunks_to_commit = [(seed_idx, end_idx, chunk_root)]
        
        # Commit NEW frames only: seed+1..end (same as /commit)
        golden_jpeg_dir = get_golden_jpeg_dir(run_dir)
        ensure_dir(golden_jpeg_dir)
        
        src_root = run_dir / "xmem_generic"
        src_jpeg = src_root / "JPEGImages" / VIDEO_NAME
        
        committed = 0
        skipped_corrected = 0
        
        # Commit all chunks (same logic as /commit)
        for chunk_seed, chunk_end, chunk_folder in all_chunks_to_commit:
            chunk_ann_dir_this = chunk_folder / "Annotations" / VIDEO_NAME
            if not chunk_ann_dir_this.exists():
                log.warning(f"[PREPARE_CORRECTION] Skipping chunk {chunk_folder.name} - annotations missing")
                continue
            
            log.info(f"[PREPARE_CORRECTION] Processing chunk {chunk_folder.name}: seed={chunk_seed}, end={chunk_end}")
            
            # Determine actual range for this chunk: only commit frames from max_idx+1 to commit_up_to
            # We don't limit based on corrected frames here - we'll skip corrected frames individually during processing
            # This ensures we don't create gaps (e.g., if frame 199 is corrected, we still process frames 200-230)
            chunk_commit_start = max(chunk_seed + 1, max_idx + 1)
            chunk_commit_end = min(commit_up_to, chunk_end)
            log.info(f"[PREPARE_CORRECTION] Chunk {chunk_folder.name}: will commit frames {chunk_commit_start}..{chunk_commit_end} (out of chunk range {chunk_seed}..{chunk_end}, max_idx={max_idx})")
            
            # Skip this chunk if there's no overlap with the commit range
            if chunk_commit_start > chunk_commit_end:
                log.info(f"[PREPARE_CORRECTION] Chunk {chunk_folder.name}: skipping (no frames in range {max_idx+1}..{commit_up_to})")
                continue
            
            # First, check if the seed frame exists in golden - if not, copy it (same as /commit)
            # But only if seed frame is >= max_idx (we don't want to copy old seed frames)
            if chunk_seed >= max_idx:
                seed_mask_src = chunk_ann_dir_this / "00000.png"
                seed_mask_dst = golden_ann_dir / f"{chunk_seed:05d}.png"
                if seed_mask_src.exists() and not seed_mask_dst.exists():
                    log.debug(f"[PREPARE_CORRECTION] Seed frame {chunk_seed} not in golden, copying it")
                    shutil.copy2(seed_mask_src, seed_mask_dst)
                    # Also copy JPEG frame
                    seed_jpeg_src = src_jpeg / f"{chunk_seed:05d}.jpg"
                    if seed_jpeg_src.exists():
                        seed_jpeg_dst = golden_jpeg_dir / f"{chunk_seed:05d}.jpg"
                        shutil.copy2(seed_jpeg_src, seed_jpeg_dst)
                    committed += 1
            
            # Commit frames from this chunk: chunk_commit_start to chunk_commit_end
            files_to_copy = []
            jpeg_files_to_copy = []
            chunk_committed_frames = []
            
            for orig_idx in range(chunk_commit_start, chunk_commit_end + 1):
                rel = orig_idx - chunk_seed
                src = chunk_ann_dir_this / f"{rel:05d}.png"
                
                if not src.exists():
                    log.warning(f"[PREPARE_CORRECTION] Missing chunk mask for frame {orig_idx} (relative {rel}) - reached end of chunk, stopping")
                    break
                
                dst = golden_ann_dir / f"{orig_idx:05d}.png"
                
                # Check if this frame already has a corrected mask in golden (same as /commit)
                if dst.exists():
                    existing_mask = np.array(Image.open(dst))
                    chunk_mask = np.array(Image.open(src))
                    
                    if existing_mask.shape != chunk_mask.shape:
                        log.error(f"[PREPARE_CORRECTION] ⚠️  DIMENSION MISMATCH for frame {orig_idx}! Existing: {existing_mask.shape}, Chunk: {chunk_mask.shape}")
                        continue
                    
                    masks_different = not np.array_equal(existing_mask, chunk_mask)
                    if masks_different:
                        log.info(f"[PREPARE_CORRECTION] Frame {orig_idx} has corrected mask in golden (differs from chunk), skipping overwrite")
                        skipped_corrected += 1
                    else:
                        files_to_copy.append((src, dst))
                        chunk_committed_frames.append(orig_idx)
                else:
                    files_to_copy.append((src, dst))
                    chunk_committed_frames.append(orig_idx)
                
                # Always collect JPEG frame for copying
                src_jpeg_frame = src_jpeg / f"{orig_idx:05d}.jpg"
                if src_jpeg_frame.exists():
                    dst_jpeg_frame = golden_jpeg_dir / f"{orig_idx:05d}.jpg"
                    jpeg_files_to_copy.append((src_jpeg_frame, dst_jpeg_frame))
            
            # Copy all files in parallel (same as /commit)
            if files_to_copy:
                copied_count = copy_files_parallel(files_to_copy, max_workers=8)
                committed += copied_count
                if chunk_committed_frames:
                    first_frame = min(chunk_committed_frames)
                    last_frame = max(chunk_committed_frames)
                    log.info(f"[PREPARE_CORRECTION] Chunk {chunk_folder.name}: committed {copied_count} frames ({first_frame}..{last_frame})")
                else:
                    log.debug(f"[PREPARE_CORRECTION] Copied {copied_count} mask files in parallel")
            
            if jpeg_files_to_copy:
                copy_files_parallel(jpeg_files_to_copy, max_workers=8)
                log.debug(f"[PREPARE_CORRECTION] Copied {len(jpeg_files_to_copy)} JPEG files in parallel")
        
        # Calculate actual committed range (same format as /commit)
        if all_chunks_to_commit:
            first_chunk_seed = all_chunks_to_commit[0][0]
            last_chunk_end = min(commit_up_to, all_chunks_to_commit[-1][1])
            committed_range = f"{first_chunk_seed+1}..{last_chunk_end}"
        else:
            committed_range = f"{seed_idx+1}..{min(commit_up_to, end_idx)}"
        
        committed_count = committed
        log.info(f"[PREPARE_CORRECTION] Commit complete:")
        log.info(f"[PREPARE_CORRECTION]   Committed {committed} NEW frames to golden: {golden_ann_dir} ({committed_range})")
        if skipped_corrected > 0:
            log.info(f"[PREPARE_CORRECTION]   Skipped {skipped_corrected} frames that had corrected masks in golden")
        
        # Update golden preview video for committed frames
        # Extract from tracked.mp4 instead of rendering from golden (tracked video already has correct masks)
        try:
            golden_preview = run_dir / "golden" / "golden_preview.mp4"
            tracked_path = run_dir / "tracked.mp4"
            
            log.info(f"[PREPARE_CORRECTION] Video update check: committed_count={committed_count}, tracked_path.exists()={tracked_path.exists() if tracked_path else False}")
            log.info(f"[PREPARE_CORRECTION] Video update check: max_idx={max_idx}, commit_up_to={commit_up_to}")
            log.info(f"[PREPARE_CORRECTION] Video update check: all_chunks_to_commit count={len(all_chunks_to_commit) if all_chunks_to_commit else 0}")
            if all_chunks_to_commit:
                log.info(f"[PREPARE_CORRECTION] Video update check: first chunk={all_chunks_to_commit[0]}, last chunk={all_chunks_to_commit[-1]}")
            log.info(f"[PREPARE_CORRECTION] Video update check: seed_idx={seed_idx}, end_idx={end_idx}")
            
            # tracked.mp4 contains the full tracked video from the most recent tracking session
            # tracked.mp4 frame 0 = seed_idx (from last_chunk_meta.txt, where the tracking session started)
            # This is the max_idx at the time of tracking, which is where tracked.mp4 starts
            tracked_video_seed = seed_idx  # This is the seed of the tracking session that created tracked.mp4
            
            log.info(f"[PREPARE_CORRECTION] tracked.mp4 was created from tracking session starting at seed_idx={tracked_video_seed}")
            log.info(f"[PREPARE_CORRECTION] tracked.mp4 frame 0 = absolute frame {tracked_video_seed}")
            log.info(f"[PREPARE_CORRECTION] Current max_idx={max_idx}, commit_up_to={commit_up_to}")
            
            # We need to extract frames from max_idx+1 to commit_up_to
            # tracked.mp4 frame 0 = seed_idx (which equals max_idx when tracking started)
            # So we extract frames 1..(commit_up_to - max_idx) from tracked.mp4
            # This gives us absolute frames (max_idx+1)..commit_up_to
            if committed_count > 0 and tracked_path.exists():
                # Extract frames 1..(commit_up_to - max_idx) from tracked.mp4
                # This gives us exactly (commit_up_to - max_idx) frames: absolute frames (max_idx+1)..commit_up_to
                tracked_seg_start = 1  # Skip seed frame (frame 0 in tracked.mp4)
                tracked_seg_end = commit_up_to - max_idx  # Number of frames to extract
                
                log.info(f"[PREPARE_CORRECTION] Video extraction calculation:")
                log.info(f"[PREPARE_CORRECTION]   max_idx={max_idx}, commit_up_to={commit_up_to}")
                log.info(f"[PREPARE_CORRECTION]   tracked.mp4 frame 0 = absolute frame {tracked_video_seed} (should equal max_idx={max_idx})")
                log.info(f"[PREPARE_CORRECTION]   Extracting tracked.mp4 frames {tracked_seg_start}..{tracked_seg_end}")
                log.info(f"[PREPARE_CORRECTION]   This gives absolute frames {max_idx+1}..{commit_up_to} ({tracked_seg_end} frames)")
                
                if tracked_seg_end >= tracked_seg_start and tracked_seg_start >= 0:
                    log.info(f"[PREPARE_CORRECTION] ✅ Range is valid, proceeding with extraction")
                    seg_path = run_dir / "golden_segments" / f"tracked_{max_idx+1}_{commit_up_to}.mp4"
                    ensure_dir(seg_path.parent)
                    
                    # Extract frames using ffmpeg
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(tracked_path),
                        "-vf", f"select='gte(n,{tracked_seg_start})*lt(n,{tracked_seg_end+1})',setpts=N/({fps:.10f}*TB)",
                        "-r", f"{fps:.10f}",
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-crf", "20",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        str(seg_path),
                    ]
                    log.info(f"[PREPARE_CORRECTION] Running ffmpeg extraction: {' '.join(cmd)}")
                    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    
                    if p.returncode == 0 and seg_path.exists():
                        seg_size = seg_path.stat().st_size
                        log.info(f"[PREPARE_CORRECTION] ✅ Extracted tracked segment: {seg_path} (size={seg_size} bytes)")
                        # Append to golden preview
                        if golden_preview.exists():
                            golden_preview_size_before = golden_preview.stat().st_size
                            log.info(f"[PREPARE_CORRECTION] Appending to existing golden preview (size={golden_preview_size_before} bytes)")
                            _ffmpeg_concat(golden_preview, seg_path, golden_preview, fps)
                            golden_preview_size_after = golden_preview.stat().st_size
                            log.info(f"[PREPARE_CORRECTION] ✅ Updated golden preview video: {max_idx+1}..{commit_up_to} (size before={golden_preview_size_before}, after={golden_preview_size_after})")
                        else:
                            golden_preview.write_bytes(seg_path.read_bytes())
                            golden_preview_size = golden_preview.stat().st_size
                            log.info(f"[PREPARE_CORRECTION] ✅ Initialized golden preview video: {max_idx+1}..{commit_up_to} (size={golden_preview_size} bytes)")
                        if get_annotation_mode(run_dir) == "behavior":
                            append_masks_only_golden_segment(
                                run_dir, fps, n_ids, max_idx + 1, commit_up_to
                            )
                    else:
                        log.error(f"[PREPARE_CORRECTION] ❌ Failed to extract tracked segment (returncode={p.returncode}): {p.stdout[-500:] if p.stdout else 'no output'}")
                else:
                    log.warning(f"[PREPARE_CORRECTION] ⚠️  Invalid range: tracked_seg_start={tracked_seg_start}, tracked_seg_end={tracked_seg_end}")
                    log.warning(f"[PREPARE_CORRECTION] ⚠️  Conditions: tracked_seg_end >= tracked_seg_start = {tracked_seg_end >= tracked_seg_start}, tracked_seg_start >= 0 = {tracked_seg_start >= 0}")
                    log.warning(f"[PREPARE_CORRECTION] ⚠️  This means we cannot extract {commit_up_to - max_idx} frames from tracked.mp4")
            elif not tracked_path.exists():
                log.warning(f"[PREPARE_CORRECTION] tracked.mp4 not found at {tracked_path}, cannot update golden preview video")
            elif committed_count == 0:
                log.info(f"[PREPARE_CORRECTION] No committed frames (committed_count=0), skipping video update")
        except Exception as e:
            log.warning(f"[PREPARE_CORRECTION] Failed to update golden preview video (non-fatal): {e}", exc_info=True)
    
    # Step 2: Run SAM-3 on frame_idx
    src_root = run_dir / "xmem_generic"
    jpeg_dir = src_root / "JPEGImages" / VIDEO_NAME
    frame_path = jpeg_dir / f"{frame_idx:05d}.jpg"
    
    log.info(f"[PREPARE_CORRECTION] Loading frame image: {frame_path} (exists: {frame_path.exists()})")
    if not frame_path.exists():
        raise HTTPException(404, f"Frame {frame_idx} not found")
    
    log.info(f"[PREPARE_CORRECTION] Running SAM-3 on frame {frame_idx}")
    try:
        new_masks = run_sam3_on_frame(prompt, frame_path)
        log.info(f"[PREPARE_CORRECTION] SAM-3 found {len(new_masks)} masks for frame {frame_idx}")
    except RuntimeError as e:
        if "No valid masks from SAM-3" in str(e):
            log.warning(f"[PREPARE_CORRECTION] SAM-3 found no masks for frame {frame_idx} (this is OK - user can add masks with point prompts)")
            new_masks = []  # Empty list - user can add masks manually
        else:
            raise  # Re-raise other RuntimeErrors
    
    # Save masks temporarily for refinement (even if empty, so refinement can add masks)
    masks_file = get_correction_masks_file(run_dir, frame_idx)
    masks_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(masks_file, new_masks)
    log.info(f"[PREPARE_CORRECTION] Saved {len(new_masks)} masks to {masks_file} for refinement")
    
    # Step 3: Find tracked mask for this frame (for display and ID matching)
    log.info(f"[PREPARE_CORRECTION] Searching for tracked mask for frame {frame_idx}")
    tracked_mask_path, tracked_source = find_tracked_mask_for_frame(run_dir, frame_idx)
    tracked_label_map = None
    if tracked_mask_path and tracked_mask_path.exists():
        tracked_label_map = np.array(Image.open(tracked_mask_path))
        max_tracked_id = int(tracked_label_map.max())
        unique_ids = sorted(list(set(tracked_label_map.flatten())))
        unique_ids = [id for id in unique_ids if id > 0]  # Remove background
        log.info(f"[PREPARE_CORRECTION] Found tracked mask for frame {frame_idx} from {tracked_source}")
        log.info(f"[PREPARE_CORRECTION] Tracked mask path: {tracked_mask_path}")
        log.info(f"[PREPARE_CORRECTION] Tracked mask contains {len(unique_ids)} object IDs: {unique_ids}, max_id={max_tracked_id}")
    else:
        log.info(f"[PREPARE_CORRECTION] No tracked mask found for frame {frame_idx} (path: {tracked_mask_path}, source: {tracked_source})")
    
    # Step 4: Auto-assign IDs - prefer using the frame's tracked annotation if it exists
    # This allows re-correction while preserving previous ID assignments
    # If no masks found, skip ID matching (user will add masks manually)
    assignments = {}
    if new_masks:
        reference_label_map = tracked_label_map
        reference_source = f"frame {frame_idx} ({tracked_source})" if tracked_label_map is not None else None
        
        # Fall back to previous frame if no tracked annotation found
        if reference_label_map is None:
            prev_frame_idx = frame_idx - 1
            prev_mask_path, _ = find_tracked_mask_for_frame(run_dir, prev_frame_idx)
            
            if prev_mask_path and prev_mask_path.exists():
                reference_label_map = np.array(Image.open(prev_mask_path))
                reference_source = f"frame {prev_frame_idx} (previous frame)"
                log.info(f"Using previous frame {prev_frame_idx} as reference for ID matching")
            else:
                raise HTTPException(500, f"Previous frame annotation not found for frame {prev_frame_idx}")
        
        assignments = auto_assign_ids(new_masks, reference_label_map, iou_threshold=0.2)
        log.info(f"[PREPARE_CORRECTION] ID matching completed using {reference_source}")
        log.info(f"[PREPARE_CORRECTION] Assignments: {assignments}")
    else:
        log.info(f"[PREPARE_CORRECTION] No masks found by SAM-3, skipping ID matching (user can add masks with point prompts)")
    
    # Save assignments for use during refinement (to preserve IDs, even if empty)
    assignments_file = get_correction_assignments_file(run_dir, frame_idx)
    assignments_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(assignments_file, assignments)
    log.info(f"[PREPARE_CORRECTION] Saved ID assignments to {assignments_file} for refinement")
    
    # Get all existing IDs in golden sequence (for user to choose from)
    existing_ids = set()
    for ann_file in sorted(golden_ann_dir.glob("*.png")):
        ann = np.array(Image.open(ann_file))
        existing_ids.update(range(1, int(ann.max()) + 1))
    existing_ids = sorted(list(existing_ids))
    
    # Create preview image showing tracked masks (if frame was already processed) and new SAM masks
    log.info(f"[PREPARE_CORRECTION] Creating preview image for frame {frame_idx}")
    frame = load_frame_safely(frame_path, frame_idx=frame_idx)
    log.info(f"[PREPARE_CORRECTION] Frame image loaded: shape={frame.shape}")
    
    # First, render tracked masks (if frame was already processed) with lower opacity
    # Use the tracked_label_map we found earlier
    if tracked_label_map is not None:
        max_tracked_id = int(tracked_label_map.max())
        log.info(f"[PREPARE_CORRECTION] Rendering tracked masks: max_id={max_tracked_id}")
        
        # Render tracked masks with lower opacity (darker/more transparent)
        rendered_tracked_count = 0
        for obj_id in range(1, max_tracked_id + 1):
            tracked_mask = (tracked_label_map == obj_id)
            if not tracked_mask.any():
                continue
            rendered_tracked_count += 1
            col = get_color_for_id(obj_id)
            overlay = frame.copy()
            overlay[tracked_mask] = col
            frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)  # Very subtle overlay for tracked masks
            
            ys, xs = np.where(tracked_mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            
            text = f"prev:{obj_id}"
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            text_x = cx - text_width // 2
            text_y = cy + text_height // 2
            
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (200, 200, 200),  # Gray color for previous masks
                thickness,
                cv2.LINE_AA,
            )
        log.info(f"[PREPARE_CORRECTION] Rendered {rendered_tracked_count} tracked masks")
    else:
        log.info(f"[PREPARE_CORRECTION] No tracked masks to render (tracked_label_map is None)")
    
    # Then render new SAM masks with auto-assigned IDs (more prominent)
    if new_masks and assignments:
        log.info(f"[PREPARE_CORRECTION] Rendering {len(assignments)} new SAM masks")
        for mask_idx, assigned_id in assignments.items():
            mask = new_masks[mask_idx]
            mask_pixels = int(mask.sum())
            log.info(f"[PREPARE_CORRECTION] Rendering SAM mask {mask_idx} -> ID {assigned_id} ({mask_pixels} pixels)")
            col = get_color_for_id(assigned_id, min_val=0)
            overlay = frame.copy()
            overlay[mask] = col
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)  # More prominent overlay for new masks
            
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue  # Skip empty masks
            cx, cy = int(xs.mean()), int(ys.mean())
            
            text = str(assigned_id)
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            text_x = cx - text_width // 2
            text_y = cy + text_height // 2
            
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
    else:
        log.info(f"[PREPARE_CORRECTION] No new SAM masks to render (user can add masks with point prompts)")
    log.info(f"[PREPARE_CORRECTION] Preview rendering complete for frame {frame_idx}")
    
    # Frame encoding handled by encode_frame_to_base64 if needed
    
    # Prepare response
    mask_assignments = [
        {
            "mask_index": mask_idx,
            "auto_assigned_id": assigned_id,
            "is_new": assigned_id > max(existing_ids) if existing_ids else True,
        }
        for mask_idx, assigned_id in sorted(assignments.items())
    ]
    
    image_b64 = encode_frame_to_base64(frame, quality=90)
    
    # Get image dimensions for coordinate scaling
    img_height, img_width = frame.shape[:2]
    
    return JSONResponse(content={
        "frame_idx": frame_idx,
        "image": f"data:image/jpeg;base64,{image_b64}",
        "mask_assignments": mask_assignments,
        "existing_ids": existing_ids,
        "max_existing_id": max(existing_ids) if existing_ids else 0,
        "image_width": int(img_width),
        "image_height": int(img_height),
    })


class PointPrompt(BaseModel):
    x: int
    y: int
    is_positive: bool  # True = add to mask, False = remove from mask

class RefineMaskRequest(BaseModel):
    mask_index: int
    points: list[PointPrompt]  # Accumulated points for this mask

class AddMaskRequest(BaseModel):
    point: PointPrompt  # Single point to create a new mask

@app.post("/add_mask/{run_id}/{frame_idx}")
def add_mask(run_id: str, frame_idx: int, add_request: AddMaskRequest):
    """
    Add a new mask using a point prompt with SAM-3 video predictor (1-frame video session).
    This creates a new mask from scratch using a positive point prompt.
    """
    from sam3.visualization_utils import prepare_masks_for_visualization
    
    log.info(f"[ADD_MASK] ========== ADD NEW MASK ==========")
    log.info(f"/add_mask run_id={run_id} frame_idx={frame_idx} point=({add_request.point.x}, {add_request.point.y}, positive={add_request.point.is_positive})")
    
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"Run not found: {run_id}")
    
    # Load existing masks from prepare_correction (or create empty array if none exist)
    masks_file = get_correction_masks_file(run_dir, frame_idx)
    if masks_file.exists():
        masks = load_masks_safely(masks_file)
        log.info(f"[ADD_MASK] Loaded {len(masks)} existing masks from {masks_file}")
        for i, m in enumerate(masks[:3]):  # Log first 3 masks
            log.info(f"[ADD_MASK]   Mask {i}: type={type(m)}, dtype={getattr(m, 'dtype', 'N/A')}, shape={getattr(m, 'shape', 'N/A')}")
    else:
        masks = []
        log.info(f"[ADD_MASK] No existing masks found, starting with empty list")
    
    # Load ID assignments to identify deleted masks (ID <= 0 or missing from assignments)
    assignments_file = get_correction_assignments_file(run_dir, frame_idx)
    id_assignments = {}
    id_assignments = load_assignments_or_default(assignments_file, len(masks))
    log.info(f"[ADD_MASK] Using ID assignments: {id_assignments}")
    
    # Identify which masks are deleted:
    # 1. Masks with ID <= 0 in assignments
    # 2. Masks that exist in the masks file but are NOT in assignments (were deleted by removing from mapping)
    deleted_mask_indices = set()
    for idx in range(len(masks)):
        if idx in id_assignments:
            if id_assignments[idx] <= 0:
                deleted_mask_indices.add(idx)
        else:
            # Mask exists in file but not in assignments - it was deleted
            deleted_mask_indices.add(idx)
    
    if deleted_mask_indices:
        log.info(f"[ADD_MASK] Found {len(deleted_mask_indices)} deleted masks (indices: {sorted(deleted_mask_indices)})")
    
    # Load frame image
    meta = parse_meta_file(run_dir / "meta.txt")
    src_root = run_dir / "xmem_generic"
    jpeg_dir = src_root / "JPEGImages" / VIDEO_NAME
    frame_path = jpeg_dir / f"{frame_idx:05d}.jpg"
    
    if not frame_path.exists():
        raise HTTPException(404, f"Frame {frame_idx} not found")
    
    img = Image.open(frame_path).convert("RGB")
    W, H = img.size
    log.info(f"[ADD_MASK] Frame dimensions: {W}x{H}")
    
    # Validate point is within image bounds
    if not (0 <= add_request.point.x < W and 0 <= add_request.point.y < H):
        raise HTTPException(400, f"Point ({add_request.point.x}, {add_request.point.y}) is outside image bounds ({W}x{H})")
    
    # Use video predictor with 1-frame video session (exactly like refine_mask)
    predictor = get_video_predictor()
    
    # Create temporary 1-frame video folder
    tmpdir = tempfile.mkdtemp(prefix="sam3_add_mask_")
    try:
        # Load image and save as JPEG
        img = Image.open(frame_path).convert("RGB")
        image_np = np.array(img)
        frame_tmp_path = Path(tmpdir) / "00000.jpg"
        bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(frame_tmp_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError("Failed to write temporary frame for SAM3 session")
        
        # Start session
        resp = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(tmpdir),
            )
        )
        session_id = resp["session_id"]
        log.info(f"[ADD_MASK] Started session {session_id}")
        
        # Step 1: Add text prompt first to establish objects (like refine_mask does)
        meta = parse_meta_file(run_dir / "meta.txt")
        prompt = meta.get("prompt", "object")
        log.info(f"[ADD_MASK] Step 1: Adding text prompt '{prompt}' to establish objects")
        text_request_dict = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
        }
        predictor.handle_request(request=text_request_dict)
        log.info("[ADD_MASK] Text prompt added")

        # Step 2: Propagate to get initial masks
        log.info(f"[ADD_MASK] Step 2: Propagating after text prompt to get initial masks...")
        outputs0_initial = None
        for resp in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            if resp.get("frame_index") == 0:
                outputs0_initial = resp.get("outputs")
                break
        
        if outputs0_initial is None:
            raise RuntimeError("SAM3 propagate_in_video did not return frame 0 outputs")
        
        # Format and extract initial instances (we need this to determine a new obj_id)
        formatted0_initial = prepare_masks_for_visualization({0: outputs0_initial})[0]
        inst_list_initial = extract_instances_from_formatted(formatted0_initial)
        log.info(f"[ADD_MASK] Text prompt found {len(inst_list_initial)} initial instances (ignoring them - creating new mask)")
        
        # Step 3: Add point prompt to create NEW mask (always create new, don't check existing masks)
        if not add_request.point.is_positive:
            log.warning(f"[ADD_MASK] Point is negative, but converting to positive for new mask creation")
        
        points_xy = np.array([[add_request.point.x, add_request.point.y]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)  # Always positive for new mask
        
        # WORKAROUND: Duplicate single point (SAM-3 quirk)
        if len(points_xy) == 1:
            log.info(f"[ADD_MASK] WORKAROUND: Duplicating single point to work around SAM-3 quirk")
            points_xy = np.vstack([points_xy, points_xy])
            labels = np.append(labels, labels[0])
        
        # Convert to relative coordinates
        points_rel = points_xy.copy()
        points_rel[:, 0] /= float(W)
        points_rel[:, 1] /= float(H)
        
        log.info(f"[ADD_MASK] Step 3: Adding point prompt to create NEW mask (not touching existing masks)")
        
        # Always create a new obj_id (max from text prompt + 1)
        import torch
        points_tensor = torch.tensor(points_rel, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int32)
        
        max_obj_id = max([int(inst.get("obj_id", 0)) for inst in inst_list_initial], default=0)
        new_obj_id = max_obj_id + 1
        log.info(f"[ADD_MASK] Creating new mask with obj_id={new_obj_id} (not checking existing saved masks)")
        
        points_request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "points": points_tensor,
            "point_labels": labels_tensor,
            "obj_id": int(new_obj_id),
        }
        
        predictor.handle_request(request=points_request)
        log.info("[ADD_MASK] Point prompts added")

        # Step 4: Propagate to get the new/refined mask
        log.info(f"[ADD_MASK] Step 4: Propagating after point prompts to get new mask...")
        outputs0 = None
        for resp in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            if resp.get("frame_index") == 0:
                outputs0 = resp.get("outputs")
                break
        
        if outputs0 is None:
            raise RuntimeError("SAM3 propagate_in_video did not return frame 0 outputs")
        
        log.info(f"[ADD_MASK] Got outputs, type: {type(outputs0)}")
        
        # Format and extract instances
        formatted0 = prepare_masks_for_visualization({0: outputs0})[0]
        inst_list = extract_instances_from_formatted(formatted0)
        log.info(f"[ADD_MASK] Extracted {len(inst_list)} instances from output")
        
        if len(inst_list) == 0:
            raise RuntimeError("No masks found from point prompt. Try a different location.")
        
        # Find the instance that contains our point
        best_mask = None
        best_score = 0.0
        target_obj_id = points_request.get("obj_id")
        
        point_mask = np.zeros((H, W), dtype=bool)
        point_mask[add_request.point.y, add_request.point.x] = True
        
        for inst in inst_list:
            inst_obj_id = int(inst.get("obj_id", -1))
            mask_np = np.asarray(inst["mask"])
            mask_resized = safe_mask_hw(mask_np, H, W)
            
            # Prefer the mask with the target obj_id, or one that contains the point
            if target_obj_id is not None and inst_obj_id == target_obj_id:
                best_mask = mask_resized
                log.info(f"[ADD_MASK] Found mask with target obj_id={target_obj_id}")
                break
            elif 0 <= add_request.point.y < H and 0 <= add_request.point.x < W:
                if mask_resized[add_request.point.y, add_request.point.x]:
                    # Point is inside this mask - use IoU with point as score
                    iou = compute_iou(mask_resized, point_mask)
                    if iou > best_score:
                        best_score = iou
                        best_mask = mask_resized
        
        if best_mask is None:
            # Fallback: use the largest mask
            log.warning(f"[ADD_MASK] Point not in target mask, using largest mask as fallback")
            best_mask_size = 0
            for inst in inst_list:
                mask_np = np.asarray(inst["mask"])
                mask_resized = safe_mask_hw(mask_np, H, W)
                mask_size = int(mask_resized.sum())
                if mask_size > best_mask_size:
                    best_mask_size = mask_size
                    best_mask = mask_resized
        
        new_mask = best_mask
        # Ensure mask is boolean for indexing
        if new_mask.dtype != bool:
            new_mask = (new_mask > 0.5).astype(bool)
        new_mask_size = int(new_mask.sum())
        log.info(f"[ADD_MASK] Created new mask with {new_mask_size} pixels")
        
        # Close session
        try:
            predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
        except Exception as e:
            log.warning(f"[ADD_MASK] Error closing session: {e}")
        
    finally:
        # Clean up temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    # Add the new mask to the existing masks array
    # IMPORTANT: We keep all masks (including deleted ones) to preserve original indices
    # Deleted masks are marked with ID <= 0 in assignments, and will be skipped during apply_correction
    new_mask_index = len(masks)
    masks.append(new_mask)
    
    # Validate we're working with the correct frame
    log.info(f"[ADD_MASK] Saving masks with new mask for frame {frame_idx} to {masks_file}")
    log.info(f"[ADD_MASK] File path validation: expected frame_idx={frame_idx}, file contains 'correction_masks_{frame_idx}'")
    if f"correction_masks_{frame_idx}" not in str(masks_file):
        log.error(f"[ADD_MASK] ⚠️  FRAME INDEX MISMATCH! frame_idx={frame_idx} but file path is {masks_file}")
    
    # Save all masks (including deleted ones) to preserve original indices
    np.save(masks_file, np.array(masks, dtype=object))
    log.info(f"[ADD_MASK] ✓ Saved masks for frame {frame_idx} to {masks_file} (added new mask at index {new_mask_index}, total masks: {len(masks)})")
    
    # Update assignments: keep all original assignments, add new mask
    assignments = id_assignments.copy()
    
    # Assign a new ID to the newly added mask (max existing ID + 1)
    # Only consider non-deleted masks when finding max ID
    valid_ids = [v for k, v in assignments.items() if k not in deleted_mask_indices and v > 0]
    max_existing_id = max(valid_ids) if valid_ids else 0
    new_id = max_existing_id + 1
    assignments[new_mask_index] = new_id
    log.info(f"[ADD_MASK] Assigned new mask (index={new_mask_index}) to ID {new_id} (max existing was {max_existing_id})")
    
    # Save updated assignments (keeping deleted masks with ID <= 0, adding new mask)
    assignments_file = get_correction_assignments_file(run_dir, frame_idx)
    assignments_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(assignments_file, assignments)
    log.info(f"[ADD_MASK] Saved updated assignments to {assignments_file}")
    
    # Render preview image with all masks (including new one)
    frame = load_frame_safely(frame_path, frame_idx=frame_idx)
    
    # Render all masks (skip deleted ones)
    for mask_idx, mask in enumerate(masks):
        # Skip deleted masks in rendering
        if mask_idx in deleted_mask_indices:
            continue
        
        # Mask is already validated as boolean array by load_masks_safely()
        log.debug(f"[ADD_MASK] Rendering mask {mask_idx}: dtype={mask.dtype}, shape={mask.shape}, size={int(mask.sum())}px")
        
        assigned_id = assignments.get(mask_idx, mask_idx + 1)
        
        if mask_idx == new_mask_index:
            # Highlight the new mask in green
            col = (0, 255, 0)  # Green for new mask
            overlay = frame.copy()
            overlay[mask] = col
            frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        else:
            col = get_color_for_id(assigned_id, min_val=0)
            overlay = frame.copy()
            overlay[mask] = col
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw mask center with assigned ID
        ys, xs = np.where(mask)
        if len(ys) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(frame, str(assigned_id), (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw the point that created the new mask
    cv2.circle(frame, (add_request.point.x, add_request.point.y), 5, (0, 255, 0), -1)
    cv2.circle(frame, (add_request.point.x, add_request.point.y), 8, (255, 255, 255), 2)
    
    # Encode preview image
    image_b64 = encode_frame_to_base64(frame, quality=90)
    
    log.info(f"[ADD_MASK] ========== NEW MASK ADDED ==========")
    log.info(f"[ADD_MASK] New mask size: {new_mask_size}px")
    log.info(f"[ADD_MASK] Total masks: {len(masks)}")
    log.info(f"[ADD_MASK] ====================================")
    
    # Build mask_assignments array (same format as prepare_correction)
    # Use existing assignments for old masks, and assign a new ID for the new mask
    mask_assignments = []
    for mask_idx in range(len(masks)):
        assigned_id = assignments.get(mask_idx, mask_idx + 1)
        mask_assignments.append({
            "mask_index": mask_idx,
            "auto_assigned_id": assigned_id,
        })
    
    log.info(f"[ADD_MASK] Built mask_assignments: {len(mask_assignments)} entries")
    
    return JSONResponse(content={
        "image": f"data:image/jpeg;base64,{image_b64}",
        "new_mask_index": len(masks) - 1,
        "new_mask_size": new_mask_size,
        "total_masks": len(masks),
        "mask_assignments": mask_assignments,
        "image_width": int(W),
        "image_height": int(H),
    })

@app.post("/refine_mask/{run_id}/{frame_idx}")
def refine_mask(run_id: str, frame_idx: int, refine_request: RefineMaskRequest):
    """
    Refine a mask using point prompts with SAM-3 video predictor (1-frame video session).
    This follows the same approach as testing_backend.py: create a 1-frame video session,
    establish the mask as an object, then add point prompts to refine it.
    """
    from sam3.visualization_utils import prepare_masks_for_visualization
    
    log.info(f"[REFINE_MASK] ========== START REFINEMENT ==========")
    log.info(f"/refine_mask run_id={run_id} frame_idx={frame_idx} mask_index={refine_request.mask_index} points={len(refine_request.points)}")
    
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"Run not found: {run_id}")
    
    # Load saved masks from prepare_correction
    masks_file = get_correction_masks_file(run_dir, frame_idx)
    masks = load_masks_safely(masks_file)  # Will raise FileNotFoundError if missing
    log.info(f"[REFINE_MASK] Loaded {len(masks)} masks from {masks_file}")
    if refine_request.mask_index >= len(masks):
        raise HTTPException(400, f"Invalid mask_index {refine_request.mask_index} (max: {len(masks)-1})")
    
    # Load frame image
    meta = parse_meta_file(run_dir / "meta.txt")
    prompt = meta.get("prompt", "object")
    src_root = run_dir / "xmem_generic"
    jpeg_dir = src_root / "JPEGImages" / VIDEO_NAME
    frame_path = jpeg_dir / f"{frame_idx:05d}.jpg"
    
    if not frame_path.exists():
        raise HTTPException(404, f"Frame {frame_idx} not found")
    
    # Get the mask to refine (this might already be refined from a previous call)
    current_mask = masks[refine_request.mask_index]
    original_mask_size = int(current_mask.sum())
    log.info(f"[REFINE_MASK] Current mask {refine_request.mask_index} state:")
    log.info(f"  - Size: {original_mask_size} pixels")
    log.info(f"  - Shape: {current_mask.shape}")
    log.info(f"  - Dtype: {current_mask.dtype}")
    log.info(f"  - Min/Max: {current_mask.min()}/{current_mask.max()}")
    
    # Prepare point prompts for SAM-3
    # Convert absolute pixel coords to relative [0,1] coords
    img = Image.open(frame_path).convert("RGB")
    W, H = img.size
    
    log.info(f"[REFINE_MASK] Refining mask {refine_request.mask_index} with {len(refine_request.points)} points (current size: {original_mask_size} pixels)")
    
    # Track if object was removed (needs to be accessible after try block)
    object_removed = False
    
    # Use video predictor with 1-frame video session (exactly like testing_backend.py)
    predictor = get_video_predictor()
    
    # Create temporary 1-frame video folder (exactly like testing_backend.py start_single_image_session)
    tmpdir = tempfile.mkdtemp(prefix="sam3_refine_")
    try:
        # Load image and save as JPEG (exactly like testing_backend.py start_single_image_session)
        img = Image.open(frame_path).convert("RGB")
        image_np = np.array(img)
        frame_tmp_path = Path(tmpdir) / "00000.jpg"
        # Save as JPEG like the notebook expects for folders (exactly like testing_backend.py)
        bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(frame_tmp_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError("Failed to write temporary frame for SAM3 session")
        
        # Start session (exactly like testing_backend.py)
        resp = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(tmpdir),
            )
        )
        session_id = resp["session_id"]
        log.info(f"[REFINE_MASK] Started session {session_id}")
        
        # Add TEXT prompt on frame 0 (exactly like testing_backend.py /segment/init)
        # IMPORTANT: Create a completely fresh dict with ONLY text, no points variables in scope
        log.info(f"[REFINE_MASK] Step 1: Adding text prompt '{prompt}' to establish objects")
        text_request_dict = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
        }
        predictor.handle_request(request=text_request_dict)
        log.info("[REFINE_MASK] Text prompt added")

        # NOW prepare points (after text prompt is done, to avoid any scope issues)
        points_xy = np.array([[p.x, p.y] for p in refine_request.points], dtype=np.float32)
        # SAM3 convention: positive=1, negative=0
        labels = np.array([1 if p.is_positive else 0 for p in refine_request.points], dtype=np.int32)
        
        # WORKAROUND: SAM-3 quirk with single point prompts (both positive and negative)
        # Empirically discovered: single points (especially negative) cause incorrect behavior.
        # Duplicating the exact same point (providing no new information) fixes this behavior.
        # This suggests a technical quirk in SAM-3's point processing (possibly tensor shape
        # requirements or internal logic that expects multiple points) rather than a semantic issue.
        # Without duplication: single negative point → mask grows (+2%)
        # With duplication: same point duplicated → mask shrinks correctly (-10.9%)
        # We apply this workaround to ALL single points (positive and negative) to ensure consistent behavior.
        if len(points_xy) == 1:
            log.info(f"[REFINE_MASK] WORKAROUND: Duplicating single point (label={labels[0]}) to work around SAM-3 quirk")
            points_xy = np.vstack([points_xy, points_xy])  # Duplicate the point
            labels = np.append(labels, labels[0])  # Duplicate the label (preserve positive/negative)
            log.info(f"[REFINE_MASK] Duplicated point: now have 2 points at ({points_xy[0, 0]:.1f}, {points_xy[0, 1]:.1f}) with label={labels[0]}")
        
        # Convert to relative coordinates
        points_rel = points_xy.copy()
        points_rel[:, 0] /= float(W)
        points_rel[:, 1] /= float(H)
        
        log.info(f"[REFINE_MASK] Prepared {len(points_xy)} points ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")
        # Log point details for debugging (use points_xy after potential duplication)
        for i in range(len(points_xy)):
            label = labels[i]
            point_type = "positive" if label == 1 else "negative"
            px, py = int(points_xy[i, 0]), int(points_xy[i, 1])
            log.info(f"[REFINE_MASK] Point {i+1}: ({px}, {py}) - {point_type} (label={label})")
            # Check if point is inside current mask
            if 0 <= py < H and 0 <= px < W:
                is_inside = current_mask[py, px] if current_mask.dtype == bool else (current_mask[py, px] > 0.5)
                log.info(f"[REFINE_MASK] Point {i+1} is {'INSIDE' if is_inside else 'OUTSIDE'} current mask")
        
        # Propagate to get initial mask (exactly like testing_backend.py propagate_frame0)
        log.info(f"[REFINE_MASK] Step 2: Propagating after text prompt to get initial masks...")
        outputs0 = None
        for resp in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            if resp.get("frame_index") == 0:
                outputs0 = resp.get("outputs")
                break
        
        if outputs0 is None:
            raise RuntimeError("SAM3 propagate_in_video did not return frame 0 outputs")
        log.info(f"[REFINE_MASK] Got initial outputs from text prompt, type: {type(outputs0)}")
        if isinstance(outputs0, dict):
            log.info(f"[REFINE_MASK] Initial outputs keys: {list(outputs0.keys())[:20]}")
        
        # Format and extract instances (exactly like testing_backend.py format_frame0 + extract_instances_from_formatted)
        log.info(f"[REFINE_MASK] Step 3: Formatting and extracting instances from text prompt output...")
        formatted0 = prepare_masks_for_visualization({0: outputs0})[0]
        inst_list = extract_instances_from_formatted(formatted0)
        log.info(f"[REFINE_MASK] Extracted {len(inst_list)} instances from formatted output")
        
        # Find the obj_id that matches our original mask by IoU
        log.info(f"[REFINE_MASK] Matching current mask (size={original_mask_size}px) to text prompt instances...")
        obj_id = None
        best_iou = 0.0
        matched_mask_size = None
        
        for inst in inst_list:
            mask_np = np.asarray(inst["mask"])
            # Use safe_mask_hw to ensure correct size (exactly like testing_backend.py)
            mask_resized = safe_mask_hw(mask_np, H, W)
            inst_mask_size = int(mask_resized.sum())
            inst_obj_id = int(inst["obj_id"])

            # Compute IoU with current mask (which might already be refined)
            iou = compute_iou(mask_resized, current_mask)
            log.info(f"[REFINE_MASK]   - Instance obj_id={inst_obj_id}: size={inst_mask_size}px, IoU={iou:.3f}")
            
            if iou > best_iou:
                best_iou = iou
                obj_id = int(inst["obj_id"])
                matched_mask_size = inst_mask_size
        
        # If IoU is too low, try using point location as fallback (especially for very small masks)
        if obj_id is None or best_iou < 0.1:
            log.warning(f"[REFINE_MASK] IoU too low ({best_iou:.3f}), trying point-based fallback matching...")
            
            # Check which instance contains the most points
            point_matches = {}
            for inst in inst_list:
                mask_np = np.asarray(inst["mask"])
                mask_resized = safe_mask_hw(mask_np, H, W)
                inst_obj_id = int(inst.get("obj_id", -1))
                
                # Count how many points are inside this mask
                points_inside = 0
                for p in refine_request.points:
                    if 0 <= p.y < H and 0 <= p.x < W:
                        if mask_resized[p.y, p.x]:
                            points_inside += 1
                
                if points_inside > 0:
                    point_matches[inst_obj_id] = points_inside
            
            if point_matches:
                # Use the instance with the most points inside
                best_obj_id = max(point_matches.items(), key=lambda x: x[1])[0]
                log.info(f"[REFINE_MASK] Point-based fallback: obj_id={best_obj_id} contains {point_matches[best_obj_id]} point(s)")
                obj_id = best_obj_id
                
                # Find the matched mask size
                for inst in inst_list:
                    if int(inst.get("obj_id", -1)) == obj_id:
                        mask_np = np.asarray(inst["mask"])
                        mask_resized = safe_mask_hw(mask_np, H, W)
                        matched_mask_size = int(mask_resized.sum())
                        break
            else:
                # Last resort: use the instance with highest IoU even if < 0.1
                if obj_id is not None:
                    log.warning(f"[REFINE_MASK] No points in any mask, using best IoU match (obj_id={obj_id}, IoU={best_iou:.3f})")
                else:
                    raise RuntimeError(f"Could not find matching obj_id for mask {refine_request.mask_index} (best IoU: {best_iou:.3f}, no points in any mask)")
        
        log.info(f"[REFINE_MASK] Step 3: Matching complete")
        log.info(f"  - Matched current mask (size={original_mask_size}px) to obj_id={obj_id} (IoU={best_iou:.3f})")
        if matched_mask_size is not None:
            size_diff = matched_mask_size - original_mask_size
            size_diff_pct = (size_diff/original_mask_size*100) if original_mask_size > 0 else 0
            log.info(f"  - Text prompt mask size: {matched_mask_size}px")
            log.info(f"  - Size difference: {size_diff:+d}px ({size_diff_pct:+.1f}%)")
            if abs(size_diff) > original_mask_size * 0.05:  # More than 5% difference
                log.warning(f"[REFINE_MASK] ⚠️  WARNING: Text prompt mask differs significantly from current mask! This might cause refinement issues.")
            else:
                log.info(f"  - ✓ Text prompt mask matches current mask well (within 5%)")
        
        # Now add point prompts to refine this object (exactly like testing_backend.py /segment/refine)
        import torch
        points_tensor = torch.tensor(points_rel, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int32)
        
        # Ensure obj_id is valid (exactly like testing_backend.py)
        if obj_id is None:
            raise RuntimeError(f"obj_id is None after matching - cannot refine mask {refine_request.mask_index}")
        
        log.info(f"[REFINE_MASK] Step 4: Adding point prompts to refine obj_id={obj_id}")
        log.info(f"  - Number of points: {len(points_rel)}")
        log.info(f"  - Points tensor: shape={points_tensor.shape}, dtype={points_tensor.dtype}")
        log.info(f"  - Labels tensor: shape={labels_tensor.shape}, dtype={labels_tensor.dtype}, values={labels_tensor.tolist()}")
        log.info(f"  - Positive points: {int(labels_tensor.sum())}, Negative points: {len(labels_tensor) - int(labels_tensor.sum())}")
        
        # Make sure we explicitly pass obj_id as int (exactly like testing_backend.py)
        points_request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "points": points_tensor,
            "point_labels": labels_tensor,
            "obj_id": int(obj_id),
        }
        log.info(f"[REFINE_MASK] Points request: keys={list(points_request.keys())}, obj_id={points_request.get('obj_id')} (type: {type(points_request.get('obj_id'))})")
        log.info(f"[REFINE_MASK] Calling predictor.handle_request with point prompts...")
        _ = predictor.handle_request(request=points_request)
        log.info(f"[REFINE_MASK] ✓ Point prompts added successfully to SAM-3")
        
        # Propagate again to get refined mask (exactly like testing_backend.py propagate_frame0)
        log.info(f"[REFINE_MASK] Step 5: Propagating after point prompts to get refined mask...")
        log.info(f"  - Expecting refined output for obj_id={obj_id}")
        outputs0_refined = None
        for resp in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            if resp.get("frame_index") == 0:
                outputs0_refined = resp.get("outputs")
                log.info(f"[REFINE_MASK] Got refined outputs for frame 0, type: {type(outputs0_refined)}")
                if isinstance(outputs0_refined, dict):
                    log.info(f"[REFINE_MASK] Refined outputs keys: {list(outputs0_refined.keys())[:20]}")
                break
        
        if outputs0_refined is None:
            raise RuntimeError("SAM3 propagate_in_video did not return refined frame 0 outputs")
        
        # Format and extract instances (exactly like testing_backend.py)
        log.info(f"[REFINE_MASK] Step 6: Formatting and extracting refined instances...")
        formatted0_refined = prepare_masks_for_visualization({0: outputs0_refined})[0]
        inst_list_refined = extract_instances_from_formatted(formatted0_refined)
        log.info(f"[REFINE_MASK] Extracted {len(inst_list_refined)} instances from refined output")
        log.info(f"[REFINE_MASK] Available obj_ids in refined output: {[int(inst.get('obj_id', -1)) for inst in inst_list_refined]}")

        # Find the mask for our obj_id (exactly like testing_backend.py /segment/refine)
        found = None
        for inst in inst_list_refined:
            inst_obj_id = int(inst.get("obj_id", -1))
            inst_mask_size = int(np.asarray(inst.get("mask", np.array([]))).sum()) if inst.get("mask") is not None else 0
            log.info(f"[REFINE_MASK]   - Instance obj_id={inst_obj_id}, mask_size={inst_mask_size}px")
            if inst_obj_id == int(obj_id):
                found = inst
                log.info(f"[REFINE_MASK] ✓ Found matching instance with obj_id={inst_obj_id}, mask_size={inst_mask_size}px")
                break

        if found is None:
            # Object disappeared - likely removed by negative points
            log.warning(f"[REFINE_MASK] ⚠️  obj_id={obj_id} disappeared from refined output (likely removed by negative points)")
            log.warning(f"[REFINE_MASK] Available obj_ids: {[int(inst.get('obj_id', -1)) for inst in inst_list_refined]}")
            
            # Object was completely removed - keep original mask unchanged
            log.warning(f"[REFINE_MASK] Object was completely removed by refinement. Keeping original mask unchanged.")
            object_removed = True
            
            # Use the original mask (don't update it)
            refined_mask = current_mask.copy()
            # Ensure mask is boolean for indexing operations
            refined_mask = refined_mask.astype(bool) if refined_mask.dtype != bool else refined_mask
            refined_mask_size = original_mask_size
            size_change = 0
            size_change_pct = 0.0
            
            log.info(f"[REFINE_MASK] ========== REFINEMENT RESULT (OBJECT REMOVED) ==========")
            log.info(f"[REFINE_MASK] Original mask size: {original_mask_size}px")
            log.info(f"[REFINE_MASK] Refined mask size: {refined_mask_size}px (unchanged)")
            log.info(f"[REFINE_MASK] Size change: {size_change:+d}px ({size_change_pct:+.1f}%)")
            log.info(f"[REFINE_MASK] ⚠️  Object was removed by negative points. Mask preserved.")
            log.info(f"[REFINE_MASK] ======================================")
        else:
            # Get refined mask (exactly like testing_backend.py - uses safe_mask_hw)
            log.info(f"[REFINE_MASK] Step 7: Extracting refined mask...")
            mask = safe_mask_hw(np.asarray(found["mask"]), H, W)
            # Ensure mask is boolean for indexing operations
            refined_mask = mask.astype(bool) if mask.dtype != bool else mask
            refined_mask_size = int(refined_mask.sum())
            size_change = refined_mask_size - original_mask_size
            size_change_pct = (size_change / original_mask_size * 100) if original_mask_size > 0 else 0

            log.info(f"[REFINE_MASK] ========== REFINEMENT RESULT ==========")
            log.info(f"[REFINE_MASK] Original mask size: {original_mask_size}px")
            log.info(f"[REFINE_MASK] Refined mask size: {refined_mask_size}px")
            log.info(f"[REFINE_MASK] Size change: {size_change:+d}px ({size_change_pct:+.1f}%)")
            if len(refine_request.points) == 1 and labels[0] == 0 and size_change > 0:
                log.warning(f"[REFINE_MASK] ⚠️  WARNING: Single negative point caused mask to GROW (expected to shrink)!")
            elif len(refine_request.points) == 1 and labels[0] == 0 and size_change < 0:
                log.info(f"[REFINE_MASK] ✓ Single negative point correctly shrunk the mask")
            log.info(f"[REFINE_MASK] ======================================")
        
        # Close session
        try:
            predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
        except Exception as e:
            log.warning(f"[REFINE_MASK] Error closing session: {e}")
        
    finally:
        # Clean up temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    # Reload original masks to ensure we don't accidentally modify other masks
    # This ensures we only update the specific mask being refined
    original_masks = load_masks_safely(masks_file)
    
    # Validate we're working with the correct frame
    log.info(f"[REFINE_MASK] Saving refined masks for frame {frame_idx} to {masks_file}")
    log.info(f"[REFINE_MASK] File path validation: expected frame_idx={frame_idx}, file contains 'correction_masks_{frame_idx}'")
    if f"correction_masks_{frame_idx}" not in str(masks_file):
        log.error(f"[REFINE_MASK] ⚠️  FRAME INDEX MISMATCH! frame_idx={frame_idx} but file path is {masks_file}")
    
    # Only update the specific mask being refined - preserve all others exactly as they were
    original_masks[refine_request.mask_index] = refined_mask
    
    # Save updated masks (only the refined mask changed, all others are preserved)
    np.save(masks_file, np.array(original_masks, dtype=object))
    log.info(f"[REFINE_MASK] ✓ Saved refined masks for frame {frame_idx} to {masks_file} (updated mask {refine_request.mask_index}, preserved all other masks unchanged)")
    
    # Load ID assignments from prepare_correction to preserve IDs
    assignments_file = get_correction_assignments_file(run_dir, frame_idx)
    assignments = load_assignments_or_default(assignments_file, len(masks))
    log.info(f"[REFINE_MASK] Using ID assignments: {assignments}")
    
    # Re-render preview image with refined mask, using preserved ID assignments
    frame = load_frame_safely(frame_path, frame_idx=frame_idx)
    
    # Render all masks (with refined one) using preserved ID assignments
    # Skip deleted masks (ID <= 0 or missing from assignments) - only render active masks
    for mask_idx, mask in enumerate(original_masks):
        # Check if mask is deleted before processing
        # A mask is deleted if: 1) it's missing from assignments, or 2) its assigned_id <= 0
        if mask_idx not in assignments:
            # Mask is missing from assignments - it was deleted
            continue
        assigned_id = assignments[mask_idx]
        if assigned_id <= 0:
            # Skip deleted masks (ID <= 0)
            continue
        
        # Ensure mask is boolean for indexing operations
        mask_bool = mask.astype(bool) if mask.dtype != bool else mask
        if mask_idx == refine_request.mask_index:
            # Highlight the refined mask
            col = (0, 255, 0)  # Green for refined mask
            overlay = frame.copy()
            overlay[mask_bool] = col
            frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        else:
            col = get_color_for_id(assigned_id, min_val=0)  # Use assigned ID for color consistency
            overlay = frame.copy()
            overlay[mask_bool] = col
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw mask center with assigned ID (not mask index)
        ys, xs = np.where(mask_bool)
        if len(ys) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(frame, str(assigned_id), (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw point prompts on the image
    for p in refine_request.points:
        color = (0, 255, 0) if p.is_positive else (0, 0, 255)  # Green for positive, red for negative
        cv2.circle(frame, (p.x, p.y), 5, color, -1)
        cv2.circle(frame, (p.x, p.y), 8, (255, 255, 255), 2)
    
    # Encode preview image
    image_b64 = encode_frame_to_base64(frame, quality=90)
    
    log.info(f"[REFINE_MASK] Mask {refine_request.mask_index} refined, new size: {int(refined_mask.sum())} pixels")
    
    # Get image dimensions for coordinate validation
    img_height, img_width = frame.shape[:2]
    
    response_content = {
        "image": f"data:image/jpeg;base64,{image_b64}",
        "mask_index": refine_request.mask_index,
        "refined_mask_size": int(refined_mask.sum()),
        "image_width": int(img_width),
        "image_height": int(img_height),
    }
    
    # Add warning if object was removed
    if object_removed:
        response_content["warning"] = "Object was removed by negative points. Mask unchanged. Try adding positive points to restore it."
    
    return JSONResponse(content=response_content)

@app.post("/preview_correction_update/{run_id}/{frame_idx}")
def preview_correction_update(run_id: str, frame_idx: int, preview_update: PreviewUpdate):
    """
    Regenerate preview image with current ID mappings and deletions.
    Used for real-time preview updates as user edits the table.
    """
    
    log.info(f"/preview_correction_update run_id={run_id} frame_idx={frame_idx}")
    log.info(f"Preview update mapping: {preview_update.mapping}")
    
    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found")
    
    meta = parse_meta_file(meta_path)
    prompt = meta.get("prompt", "object")
    
    # Get frame and run SAM again (same as prepare_correction)
    src_root = run_dir / "xmem_generic"
    jpeg_dir = src_root / "JPEGImages" / VIDEO_NAME
    frame_path = jpeg_dir / f"{frame_idx:05d}.jpg"
    
    if not frame_path.exists():
        raise HTTPException(404, f"Frame {frame_idx} not found")
    
    # Load refined masks if they exist (from refine_mask/add_mask), otherwise run SAM-3 from scratch
    masks_file = get_correction_masks_file(run_dir, frame_idx)
    if masks_file.exists():
        log.info(f"[PREVIEW_CORRECTION_UPDATE] Loading refined masks from {masks_file}")
        new_masks = load_masks_safely(masks_file)
        log.info(f"[PREVIEW_CORRECTION_UPDATE] Loaded {len(new_masks)} refined masks")
    else:
        # Fall back to running SAM-3 from scratch if no refined masks exist
        log.info(f"[PREVIEW_CORRECTION_UPDATE] No refined masks found, running SAM-3 from scratch")
        new_masks = run_sam3_on_frame(prompt, frame_path)
        log.info(f"[PREVIEW_CORRECTION_UPDATE] Got {len(new_masks)} masks from SAM-3")
    
    # Load frame (make a copy so we don't modify the original)
    frame = load_frame_safely(frame_path, frame_idx=frame_idx)
    frame = frame.copy()  # Make a copy to avoid modifying original
    
    # Render masks with user's current ID mappings (skip deleted ones)
    rendered_count = 0
    for mask_idx_str, final_id in preview_update.mapping.items():
        mask_idx = int(mask_idx_str)
        if mask_idx >= len(new_masks):
            log.warning(f"Mask index {mask_idx} >= {len(new_masks)}, skipping")
            continue
        if final_id <= 0:  # 0 or negative means delete
            log.info(f"Skipping mask {mask_idx} (marked for deletion, final_id={final_id})")
            continue
        
        mask = new_masks[mask_idx]
        col = get_color_for_id(final_id, min_val=0)
        overlay = frame.copy()
        overlay[mask] = col
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        
        text = str(final_id)
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        text_x = cx - text_width // 2
        text_y = cy + text_height // 2
        
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        rendered_count += 1
    
    log.info(f"Rendered {rendered_count} masks in preview")
    
    # Save assignments file to preserve deletions (ID <= 0) for add_mask to use
    # Convert string keys to int keys for consistency
    assignments = {int(k): int(v) for k, v in preview_update.mapping.items()}
    assignments_file = get_correction_assignments_file(run_dir, frame_idx)
    assignments_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(assignments_file, assignments)
    deleted_count = sum(1 for v in assignments.values() if v <= 0)
    log.info(f"[PREVIEW_CORRECTION_UPDATE] Saved assignments to {assignments_file} (including {deleted_count} deleted masks)")
    
    image_b64 = encode_frame_to_base64(frame, quality=90)
    
    return JSONResponse(content={
        "image": f"data:image/jpeg;base64,{image_b64}",
    })

@app.post("/apply_correction/{run_id}/{frame_idx}")
def apply_correction(run_id: str, frame_idx: int, id_mapping: IDMapping):
    """
    Apply user's ID mapping to save corrected frame.
    id_mapping: dict mapping mask_index -> final_id
    """
    
    log.info(f"/apply_correction run_id={run_id} frame_idx={frame_idx} mapping={id_mapping}")
    
    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found")
    
    meta = parse_meta_file(meta_path)
    prompt = meta.get("prompt", "object")
    
    # Get frame path (needed for later operations)
    src_root = run_dir / "xmem_generic"
    jpeg_dir = src_root / "JPEGImages" / VIDEO_NAME
    frame_path = jpeg_dir / f"{frame_idx:05d}.jpg"
    
    log.info(f"[APPLY_CORRECTION] Frame path: {frame_path} (exists: {frame_path.exists()})")
    
    if not frame_path.exists():
        raise HTTPException(404, f"Frame {frame_idx} not found")
    
    # Check if refined masks exist (from point-based refinement)
    masks_file = get_correction_masks_file(run_dir, frame_idx)
    log.info(f"[APPLY_CORRECTION] Checking for refined masks: {masks_file} (exists: {masks_file.exists()})")
    log.info(f"[APPLY_CORRECTION] Frame index: {frame_idx}, expected file: {masks_file.name}")
    
    # Validate frame index in file path
    if f"correction_masks_{frame_idx}" not in str(masks_file):
        log.error(f"[APPLY_CORRECTION] ⚠️  FRAME INDEX MISMATCH! frame_idx={frame_idx} but file path is {masks_file}")
    
    if masks_file.exists():
        log.info(f"[APPLY_CORRECTION] Using refined masks from {masks_file} for frame {frame_idx}")
        new_masks = load_masks_safely(masks_file)
        log.info(f"[APPLY_CORRECTION] Loaded {len(new_masks)} refined masks from {masks_file.name} for frame {frame_idx}")
        
        # Validate that masks match the expected frame dimensions
        if new_masks:
            expected_img = Image.open(frame_path)
            expected_H, expected_W = expected_img.size[1], expected_img.size[0]
            actual_H, actual_W = new_masks[0].shape
            if (actual_H, actual_W) != (expected_H, expected_W):
                log.error(f"[APPLY_CORRECTION] ⚠️  DIMENSION MISMATCH! Frame {frame_idx} expects {expected_H}x{expected_W}, but masks are {actual_H}x{actual_W}")
                log.error(f"[APPLY_CORRECTION] This suggests masks might be from a different frame! Regenerating masks...")
                new_masks = run_sam3_on_frame(prompt, frame_path)
                log.info(f"[APPLY_CORRECTION] Regenerated {len(new_masks)} masks with correct dimensions")
            else:
                log.info(f"[APPLY_CORRECTION] ✓ Mask dimensions match frame: {actual_H}x{actual_W}")
    else:
        # Fall back to running SAM-3 from scratch if no refined masks exist
        log.info(f"[APPLY_CORRECTION] No refined masks found, running SAM-3 from scratch")
        new_masks = run_sam3_on_frame(prompt, frame_path)
        log.info(f"[APPLY_CORRECTION] SAM-3 returned {len(new_masks)} masks")
    
    # Apply user's ID mapping
    log.info(f"[APPLY_CORRECTION] Applying ID mapping: {id_mapping.mapping}")
    H, W = new_masks[0].shape
    log.info(f"[APPLY_CORRECTION] Mask dimensions: H={H}, W={W}, num_masks={len(new_masks)}")
    label_map = np.zeros((H, W), dtype=np.uint8)
    
    for mask_idx_str, final_id in id_mapping.mapping.items():
        mask_idx = int(mask_idx_str)
        if mask_idx >= len(new_masks):
            raise HTTPException(400, f"Invalid mask_index {mask_idx} (max: {len(new_masks)-1})")
        final_id = int(final_id)
        if final_id <= 0:  # Skip deleted masks (0 or negative)
            log.info(f"[APPLY_CORRECTION] Skipping mask {mask_idx} (deleted, final_id={final_id})")
            continue
        # Ensure mask is boolean for indexing
        mask_bool = new_masks[mask_idx].astype(bool) if new_masks[mask_idx].dtype != bool else new_masks[mask_idx]
        label_map[mask_bool] = final_id
        log.info(f"[APPLY_CORRECTION] Assigned mask {mask_idx} -> ID {final_id} (mask shape: {mask_bool.shape}, pixels: {mask_bool.sum()})")
    
    # NOTE: We do NOT renumber IDs during corrections. The user (or auto-assignment) has
    # explicitly chosen which IDs to use, and these IDs are meant to match existing IDs
    # from previous frames. Renumbering would break this continuity.
    # Gaps in IDs (e.g., [2,3,4,...,18] instead of [1,2,3,...,17]) are intentional and
    # should be preserved.
    
    # Save to golden
    golden_ann_dir = get_golden_ann_dir(run_dir)
    ensure_dir(golden_ann_dir)
    
    # Save original tracked mask (if it exists) before overwriting with corrected version
    tracked_mask_path, tracked_source = find_tracked_mask_for_frame(run_dir, frame_idx)
    if tracked_mask_path and tracked_mask_path.exists():
        # Save original tracked mask to a "before" folder for reference
        golden_before_dir = run_dir / "golden" / "Annotations_before" / VIDEO_NAME
        ensure_dir(golden_before_dir)
        before_ann_path = golden_before_dir / f"{frame_idx:05d}.png"
        shutil.copy2(tracked_mask_path, before_ann_path)
        log.info(f"Saved original tracked mask to {before_ann_path} (from {tracked_source})")
    
    # Save corrected annotation
    corrected_ann_path = golden_ann_dir / f"{frame_idx:05d}.png"
    log.info(f"[APPLY_CORRECTION] ========== SAVING CORRECTED FRAME ==========")
    log.info(f"[APPLY_CORRECTION] Frame index: {frame_idx}")
    log.info(f"[APPLY_CORRECTION] Source masks file: {masks_file.name if masks_file.exists() else 'N/A (regenerated)'}")
    log.info(f"[APPLY_CORRECTION] Target golden annotation: {corrected_ann_path.name}")
    log.info(f"[APPLY_CORRECTION] Label map shape: {label_map.shape}, dtype: {label_map.dtype}, max_id: {label_map.max()}")
    
    # Validate frame index in file name
    expected_filename = f"{frame_idx:05d}.png"
    if corrected_ann_path.name != expected_filename:
        log.error(f"[APPLY_CORRECTION] ⚠️  FILENAME MISMATCH! Expected {expected_filename}, got {corrected_ann_path.name}")
    
    Image.fromarray(label_map).save(corrected_ann_path)
    
    # Verify what was actually saved
    verify_saved = np.array(Image.open(corrected_ann_path))
    log.info(f"[APPLY_CORRECTION] ✓ Saved corrected annotation for frame {frame_idx}")
    log.info(f"[APPLY_CORRECTION] Verified saved: shape={verify_saved.shape}, max_id={verify_saved.max()}, file={corrected_ann_path.name}")
    log.info(f"[APPLY_CORRECTION] ===========================================")
    
    # Also copy JPEG frame
    log.info(f"[APPLY_CORRECTION] Copying JPEG frame, jpeg_dir={jpeg_dir}")
    golden_jpeg_dir = get_golden_jpeg_dir(run_dir)
    ensure_dir(golden_jpeg_dir)
    src_jpeg_frame = jpeg_dir / f"{frame_idx:05d}.jpg"
    log.info(f"[APPLY_CORRECTION] Source JPEG: {src_jpeg_frame} (exists: {src_jpeg_frame.exists()})")
    if src_jpeg_frame.exists():
        dst_jpeg_frame = golden_jpeg_dir / f"{frame_idx:05d}.jpg"
        shutil.copy2(src_jpeg_frame, dst_jpeg_frame)
        log.info(f"[APPLY_CORRECTION] Copied JPEG frame to {dst_jpeg_frame}")
    else:
        log.warning(f"[APPLY_CORRECTION] Source JPEG frame not found: {src_jpeg_frame}")
    
    new_max_id = int(label_map.max())
    unique_ids = sorted(list(set(label_map.flatten())))
    unique_ids = [id for id in unique_ids if id > 0]  # Remove background
    log.info(f"[APPLY_CORRECTION] Saved corrected annotation for frame {frame_idx} with {new_max_id} objects, IDs={unique_ids}")
    log.info(f"[APPLY_CORRECTION] Corrected mask path: {corrected_ann_path}")
    
    # Verify the saved mask
    verify_mask = np.array(Image.open(corrected_ann_path))
    verify_max_id = int(verify_mask.max())
    verify_ids = sorted(list(set(verify_mask.flatten())))
    verify_ids = [id for id in verify_ids if id > 0]
    log.info(f"[APPLY_CORRECTION] Verified saved mask: max_id={verify_max_id}, IDs={verify_ids}")
    if not np.array_equal(label_map, verify_mask):
        log.error(f"[APPLY_CORRECTION] ⚠️  WARNING: Saved mask doesn't match what we tried to save!")

    if get_annotation_mode(run_dir) == "behavior":
        activity_data = load_behavior_dimension(run_dir, "activity")
        known_cows = (
            {int(c) for c in activity_data.get("cow_ids", [])} if activity_data else set()
        )
        assigned_ids = {
            int(final_id) for final_id in id_mapping.mapping.values() if int(final_id) > 0
        }
        new_cow_ids = sorted(assigned_ids - known_cows)
        if new_cow_ids:
            register_late_behavior_cows(run_dir, new_cow_ids, frame_idx)
    
    # Update meta if needed
    n_ids = int(meta["ids"])
    if new_max_id > n_ids:
        meta["ids"] = str(new_max_id)
        meta_path.write_text("\n".join(f"{k}={v}" for k, v in meta.items()))
        log.info(f"[APPLY_CORRECTION] Updated meta: max_id={new_max_id}")
    
    # Update golden preview video
    fps = float(meta["fps"])
    log.info(f"[APPLY_CORRECTION] Updating golden preview video for corrected frame {frame_idx}")
    try:
        golden_preview = run_dir / "golden" / "golden_preview.mp4"
        log.info(f"[APPLY_CORRECTION] Golden preview path: {golden_preview} (exists: {golden_preview.exists()})")
        
        seg_path = run_dir / "golden_segments" / f"{frame_idx:05d}_{frame_idx:05d}.mp4"
        ensure_dir(seg_path.parent)
        log.info(f"[APPLY_CORRECTION] Rendering segment for frame {frame_idx} -> {seg_path}")
        _render_segment_from_golden(run_dir, fps, new_max_id, frame_idx, frame_idx, seg_path)
        
        if seg_path.exists():
            log.info(f"[APPLY_CORRECTION] Segment rendered: {seg_path} (size: {seg_path.stat().st_size} bytes)")
            seg_reencoded = run_dir / "golden_segments" / f"{frame_idx:05d}_{frame_idx:05d}_reencoded.mp4"
            if _ffmpeg_reencode_video(seg_path, seg_reencoded, fps):
                seg_reencoded.replace(seg_path)
                log.info(f"[APPLY_CORRECTION] Segment re-encoded")
            
            if golden_preview.exists():
                log.info(f"[APPLY_CORRECTION] Appending corrected frame segment to existing golden preview")
                log.info(f"[APPLY_CORRECTION] ⚠️  NOTE: This will append, not replace. Frame {frame_idx} may appear twice if already in preview.")
                _ffmpeg_concat(golden_preview, seg_path, golden_preview, fps)
                log.info(f"[APPLY_CORRECTION] Golden preview updated (appended)")
            else:
                log.info(f"[APPLY_CORRECTION] Golden preview doesn't exist, initializing from segment")
                golden_preview.write_bytes(seg_path.read_bytes())
                log.info(f"[APPLY_CORRECTION] Golden preview initialized")
            if get_annotation_mode(run_dir) == "behavior":
                append_masks_only_golden_segment(run_dir, fps, new_max_id, frame_idx, frame_idx)
        else:
            log.error(f"[APPLY_CORRECTION] Segment was not created: {seg_path}")
    except Exception as e:
        log.error(f"[APPLY_CORRECTION] Failed to update golden preview video: {e}", exc_info=True)
    
    return {"status": "success", "frame_idx": frame_idx, "max_id": new_max_id}


@app.post("/correct_frame")
def correct_frame(run_id: str, wrong_frame_idx: int):
    """
    Correction workflow:
    1. Commit all frames before wrong_frame_idx to golden
    2. Run SAM-3 on wrong_frame_idx
    3. Auto-assign IDs based on previous frame (wrong_frame_idx - 1)
    4. Save corrected annotation to golden
    5. Return frame image with masks and assigned IDs
    """
    
    log.info(f"/correct_frame run_id={run_id} wrong_frame_idx={wrong_frame_idx}")
    
    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found")
    
    meta = parse_meta_file(meta_path)
    fps = float(meta["fps"])
    n_ids = int(meta["ids"])
    n_total = int(meta["frames"])
    prompt = meta.get("prompt", "object")
    
    if wrong_frame_idx < 1:
        raise HTTPException(400, "wrong_frame_idx must be >= 1 (cannot correct frame 0 this way)")
    if wrong_frame_idx >= n_total:
        raise HTTPException(400, f"wrong_frame_idx {wrong_frame_idx} >= total frames {n_total}")
    
    # Step 1: Commit frames up to (wrong_frame_idx - 1)
    processed, pct, max_idx = golden_progress(run_dir, n_total)
    if max_idx is None:
        raise HTTPException(500, "No golden frames found")
    
    commit_up_to = wrong_frame_idx - 1
    if commit_up_to > max_idx:
        # Need to commit more frames from the last chunk
        last_chunk_file = run_dir / "last_chunk.txt"
        last_chunk_meta = run_dir / "last_chunk_meta.txt"
        
        if not last_chunk_file.exists() or not last_chunk_meta.exists():
            raise HTTPException(400, f"Cannot commit up to frame {commit_up_to}: no chunk available")
        
        chunk_root = Path(last_chunk_file.read_text(encoding="utf-8").strip())
        chunk_ann_dir = chunk_root / "Annotations" / VIDEO_NAME
        chunk_kv = dict(line.split("=", 1) for line in last_chunk_meta.read_text(encoding="utf-8").splitlines())
        seed_idx = int(chunk_kv["seed_idx"])
        end_idx = int(chunk_kv["end_idx"])
        
        golden_ann_dir = get_golden_ann_dir(run_dir)
        ensure_dir(golden_ann_dir)
        
        # Also copy JPEG frames
        golden_jpeg_dir = get_golden_jpeg_dir(run_dir)
        ensure_dir(golden_jpeg_dir)
        src_jpeg = src_root / "JPEGImages" / VIDEO_NAME
        
        # Commit frames seed+1..min(commit_up_to, end_idx)
        # Don't commit beyond what's in the chunk
        commit_end = min(commit_up_to, end_idx)
        log.info(f"[DEBUG] Committing frames {seed_idx+1}..{commit_end} (chunk has {seed_idx}..{end_idx}, requested up to {commit_up_to})")
        
        committed = 0
        skipped_corrected = 0
        for orig_idx in range(seed_idx + 1, commit_end + 1):
            rel = orig_idx - seed_idx
            src = chunk_ann_dir / f"{rel:05d}.png"
            if not src.exists():
                log.warning(f"Missing chunk mask for frame {orig_idx} (relative {rel} in chunk), skipping")
                continue
            dst = golden_ann_dir / f"{orig_idx:05d}.png"
            
            # Validate frame index alignment
            log.debug(f"[COMMIT] Copying frame: seed_idx={seed_idx}, orig_idx={orig_idx}, rel={rel}, src={src.name}, dst={dst.name}")
            
            # Check if this frame already has a corrected mask in golden
            if dst.exists():
                # Load both masks to compare
                existing_mask = np.array(Image.open(dst))
                chunk_mask = np.array(Image.open(src))
                
                # Validate dimensions match (safety check for frame alignment)
                if existing_mask.shape != chunk_mask.shape:
                    log.error(f"[COMMIT] ⚠️  DIMENSION MISMATCH for frame {orig_idx}! Existing: {existing_mask.shape}, Chunk: {chunk_mask.shape}")
                    log.error(f"[COMMIT] This suggests a frame index mismatch! Skipping this frame.")
                    continue
                
                # Check if masks are different (not just same IDs)
                masks_different = not np.array_equal(existing_mask, chunk_mask)
                
                if masks_different:
                    log.info(f"[COMMIT] Frame {orig_idx} has corrected mask in golden (differs from chunk), skipping overwrite")
                    skipped_corrected += 1
                    # Don't overwrite - keep the corrected mask
                else:
                    # Masks are identical, safe to overwrite
                    shutil.copy2(src, dst)
                    log.debug(f"[COMMIT] Copied frame {orig_idx}: {src.name} -> {dst.name}")
                    committed += 1
            else:
                # Frame doesn't exist in golden, safe to copy
                shutil.copy2(src, dst)
                log.debug(f"[COMMIT] Copied NEW frame {orig_idx}: {src.name} -> {dst.name}")
                committed += 1
            
            # Always copy JPEG frame (even if mask was skipped)
            src_jpeg_frame = src_jpeg / f"{orig_idx:05d}.jpg"
            if src_jpeg_frame.exists():
                dst_jpeg_frame = golden_jpeg_dir / f"{orig_idx:05d}.jpg"
                shutil.copy2(src_jpeg_frame, dst_jpeg_frame)
        
        if skipped_corrected > 0:
            log.info(f"✅ Committed {committed} frames to golden: {seed_idx+1}..{commit_end} (skipped {skipped_corrected} corrected frames)")
        else:
            log.info(f"✅ Committed {committed} frames to golden: {seed_idx+1}..{commit_end}")
        
        if commit_up_to > end_idx:
            log.warning(f"⚠️  Requested commit up to frame {commit_up_to}, but chunk only goes to {end_idx}. Only committed {seed_idx+1}..{end_idx}")
        
        # Update golden preview video for committed frames
        try:
            golden_preview = run_dir / "golden" / "golden_preview.mp4"
            if commit_up_to >= seed_idx + 1:
                # Render segment for committed frames
                seg_path = run_dir / "golden_segments" / f"{seed_idx+1:05d}_{commit_up_to:05d}.mp4"
                ensure_dir(seg_path.parent)
                _render_segment_from_golden(run_dir, fps, n_ids, seed_idx + 1, commit_up_to, seg_path)
                
                if seg_path.exists():
                    # Re-encode segment to ensure compatibility
                    seg_reencoded = run_dir / "golden_segments" / f"{seed_idx+1:05d}_{commit_up_to:05d}_reencoded.mp4"
                    if _ffmpeg_reencode_video(seg_path, seg_reencoded, fps):
                        seg_reencoded.replace(seg_path)
                    
                    # Append to golden preview
                    if golden_preview.exists():
                        _ffmpeg_concat(golden_preview, seg_path, golden_preview, fps)
                    else:
                        golden_preview.write_bytes(seg_path.read_bytes())
                    if get_annotation_mode(run_dir) == "behavior":
                        append_masks_only_golden_segment(
                            run_dir, fps, n_ids, seed_idx + 1, commit_up_to
                        )
                    log.info(f"Updated golden preview video with committed frames {seed_idx+1}..{commit_up_to}")
        except Exception as e:
            log.warning(f"Failed to update golden preview video (non-fatal): {e}")
    
    # Step 2: Run SAM-3 on wrong_frame_idx
    src_root = run_dir / "xmem_generic"
    jpeg_dir = src_root / "JPEGImages" / VIDEO_NAME
    frame_path = jpeg_dir / f"{wrong_frame_idx:05d}.jpg"
    
    if not frame_path.exists():
        raise HTTPException(404, f"Frame {wrong_frame_idx} not found")
    
    log.info(f"Running SAM-3 on frame {wrong_frame_idx}")
    new_masks = run_sam3_on_frame(prompt, frame_path)
    
    # Step 3: Auto-assign IDs based on previous frame
    prev_frame_idx = wrong_frame_idx - 1
    golden_ann_dir = get_golden_ann_dir(run_dir)
    prev_ann_path = golden_ann_dir / f"{prev_frame_idx:05d}.png"
    
    if not prev_ann_path.exists():
        raise HTTPException(500, f"Previous frame annotation not found: {prev_ann_path}")
    
    prev_label_map = np.array(Image.open(prev_ann_path))
    assignments = auto_assign_ids(new_masks, prev_label_map, iou_threshold=0.2)
    
    # Step 4: Create label map with assigned IDs
    H, W = new_masks[0].shape
    label_map = np.zeros((H, W), dtype=np.uint8)
    for new_idx, assigned_id in assignments.items():
        label_map[new_masks[new_idx]] = assigned_id
    
    # Save to golden
    golden_ann_dir = get_golden_ann_dir(run_dir)
    ensure_dir(golden_ann_dir)
    corrected_ann_path = golden_ann_dir / f"{wrong_frame_idx:05d}.png"
    Image.fromarray(label_map).save(corrected_ann_path)
    
    # Also copy JPEG frame to golden/JPEGImages/video1/
    golden_jpeg_dir = get_golden_jpeg_dir(run_dir)
    ensure_dir(golden_jpeg_dir)
    src_jpeg_frame = jpeg_dir / f"{wrong_frame_idx:05d}.jpg"
    if src_jpeg_frame.exists():
        dst_jpeg_frame = golden_jpeg_dir / f"{wrong_frame_idx:05d}.jpg"
        shutil.copy2(src_jpeg_frame, dst_jpeg_frame)
        log.info(f"Copied JPEG frame {wrong_frame_idx} to golden")
    
    log.info(f"Saved corrected annotation for frame {wrong_frame_idx} with {label_map.max()} objects")
    
    # Update golden preview video to include corrected frame
    new_max_id = int(label_map.max())
    try:
        golden_preview = run_dir / "golden_preview.mp4"
        # Render segment for corrected frame only
        seg_path = run_dir / "golden_segments" / f"{wrong_frame_idx:05d}_{wrong_frame_idx:05d}.mp4"
        ensure_dir(seg_path.parent)
        _render_segment_from_golden(run_dir, fps, new_max_id, wrong_frame_idx, wrong_frame_idx, seg_path)
        
        if seg_path.exists():
            # Re-encode segment to ensure compatibility
            seg_reencoded = run_dir / "golden_segments" / f"{wrong_frame_idx:05d}_{wrong_frame_idx:05d}_reencoded.mp4"
            if _ffmpeg_reencode_video(seg_path, seg_reencoded, fps):
                seg_reencoded.replace(seg_path)
            
            # Append to golden preview
            if golden_preview.exists():
                _ffmpeg_concat(golden_preview, seg_path, golden_preview, fps)
            else:
                golden_preview.write_bytes(seg_path.read_bytes())
            if get_annotation_mode(run_dir) == "behavior":
                append_masks_only_golden_segment(
                    run_dir, fps, new_max_id, wrong_frame_idx, wrong_frame_idx
                )
            log.info(f"Updated golden preview video with corrected frame {wrong_frame_idx}")
    except Exception as e:
        log.warning(f"Failed to update golden preview video with corrected frame (non-fatal): {e}")
    
    # Step 5: Return frame image with overlays (reuse get_frame logic)
    
    frame = load_frame_safely(frame_path, frame_idx=wrong_frame_idx)
    
    for cid in range(1, int(label_map.max()) + 1):
        m = (label_map == cid)
        if not m.any():
            continue
        
        col = get_color_for_id(cid)
        overlay = frame.copy()
        overlay[m] = col
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        ys, xs = np.where(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        
        # Get text size to center it properly
        text = str(cid)
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Center the text (putText uses bottom-left corner, so adjust)
        text_x = cx - text_width // 2
        text_y = cy + text_height // 2
        
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )
    
    # Encode frame to JPEG bytes for Response
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode frame")
    
    # Update meta to reflect new max ID if needed (already computed above)
    if new_max_id > n_ids:
        meta["ids"] = str(new_max_id)
        meta_path.write_text("\n".join(f"{k}={v}" for k, v in meta.items()))
    
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/result/{run_id}")
def result(run_id: str):
    path = RUNS_ROOT / run_id / "tracked.mp4"
    if not path.exists():
        raise HTTPException(404, "No result yet. Run /track first.")
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Cache-Control": "no-store, max-age=0", "Accept-Ranges": "bytes"},
    )


@app.get("/golden_video/{run_id}")
def golden_video(run_id: str):
    path = RUNS_ROOT / run_id / "golden" / "golden_preview.mp4"
    if not path.exists():
        raise HTTPException(404, "No golden preview yet.")

    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/golden_masks_video/{run_id}")
def golden_masks_video(run_id: str):
    """Golden preview with mask overlays only (no behaviour labels)."""
    path = RUNS_ROOT / run_id / "golden" / GOLDEN_PREVIEW_MASKS_ONLY
    if not path.exists():
        raise HTTPException(404, "No masks-only golden preview yet.")

    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/source/{run_id}")
def source(run_id: str):
    meta_path = RUNS_ROOT / run_id / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found")

    meta = parse_meta_file(meta_path)
    video_path = meta["video_path"]

    if not os.path.exists(video_path):
        raise HTTPException(404, f"Source video not found: {video_path}")

    return FileResponse(video_path, media_type="video/mp4")

@app.get("/paths/{run_id}")
def paths(run_id: str):
    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found")

    golden_root = run_dir / "golden"
    golden_ann = golden_root / "Annotations" / VIDEO_NAME
    golden_video = golden_root / "golden_preview.mp4"
    golden_jpeg = golden_root / "JPEGImages" / VIDEO_NAME

    return {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "golden_root": str(golden_root.resolve()),
        "golden_annotations": str(golden_ann.resolve()),
        "golden_preview_video": str(golden_video.resolve()) if golden_video.exists() else None,
        "golden_jpeg_images": str(golden_jpeg.resolve()) if golden_jpeg.exists() else None,
    }


def _golden_mask_presence_by_cow(
    run_dir: Path, cow_ids: List[int], max_frame: int
) -> Dict[int, set]:
    """Frames where each cow_id has at least one pixel in golden annotation masks."""
    golden_ann_dir = get_golden_ann_dir(run_dir)
    present: Dict[int, set] = {int(c): set() for c in cow_ids}
    for frame_idx in range(0, max_frame + 1):
        ann_path = golden_ann_dir / f"{frame_idx:05d}.png"
        if not ann_path.exists():
            continue
        arr = np.array(Image.open(ann_path))
        for cow_id in cow_ids:
            if np.any(arr == int(cow_id)):
                present[int(cow_id)].add(frame_idx)
    return present


def _segments_from_frame_labels(
    cow_id: int,
    frame_labels: Dict[int, str],
    dimension: str,
    max_frame: int,
) -> List[Dict[str, Any]]:
    """Run-length encode per-frame labels into contiguous segments."""
    segments: List[Dict[str, Any]] = []
    current_label: Optional[str] = None
    start: Optional[int] = None
    for frame_idx in range(0, max_frame + 1):
        label = frame_labels.get(frame_idx)
        if label is None:
            continue
        if label != current_label:
            if current_label is not None and start is not None:
                _append_behavior_segment(
                    segments, cow_id, start, frame_idx - 1, current_label, dimension
                )
            current_label = label
            start = frame_idx
    if current_label is not None and start is not None:
        _append_behavior_segment(segments, cow_id, start, None, current_label, dimension)
    return segments


def sync_activity_visibility_from_masks(run_dir: Path) -> bool:
    """
    Rebuild activity segments so frames without a golden mask for a cow use not_visible.
    Present frames keep the annotator's labels. Called before golden zip export.
    """
    if get_annotation_mode(run_dir) != "behavior":
        return False
    data = load_behavior_dimension(run_dir, "activity")
    if not data:
        return False

    golden_ann_dir = get_golden_ann_dir(run_dir)
    if not golden_ann_dir.exists():
        return False
    pngs = sorted(golden_ann_dir.glob("*.png"))
    if not pngs:
        return False

    max_frame = max(int(p.stem) for p in pngs)
    cow_ids = [int(c) for c in data.get("cow_ids", [])]
    if not cow_ids:
        return False

    before = json.dumps(data.get("segments", []), sort_keys=True)
    presence = _golden_mask_presence_by_cow(run_dir, cow_ids, max_frame)
    default_label = _default_label_for_dimension("activity")
    original_segments = list(data.get("segments", []))
    original_data = dict(data)
    original_data["segments"] = original_segments

    new_segments: List[Dict[str, Any]] = []
    for cow_id in cow_ids:
        frame_labels: Dict[int, str] = {}
        for frame_idx in range(0, max_frame + 1):
            if frame_idx not in presence.get(cow_id, set()):
                frame_labels[frame_idx] = NOT_VISIBLE_LABEL_ID
            else:
                label = get_behavior_label_at_frame(original_data, cow_id, frame_idx)
                frame_labels[frame_idx] = label if label is not None else default_label
        new_segments.extend(
            _segments_from_frame_labels(cow_id, frame_labels, "activity", max_frame)
        )

    data["segments"] = _normalize_behavior_segments(new_segments)
    after = json.dumps(data["segments"], sort_keys=True)
    if before == after:
        return False

    save_behavior_dimension(run_dir, "activity", data)
    mark_behavior_preview_out_of_sync(run_dir)
    log.info(
        f"[BEHAVIOR] sync_activity_visibility_from_masks run_dir={run_dir.name} "
        f"cows={cow_ids} frames=0..{max_frame}"
    )
    return True


@app.get("/download_golden/{run_id}")
def download_golden(run_id: str, background_tasks: BackgroundTasks):
    """
    Download the golden folder as a zip file.
    """
    log.info(f"/download_golden run_id={run_id}")
    
    run_dir = RUNS_ROOT / run_id
    if get_annotation_mode(run_dir) == "behavior":
        if sync_activity_visibility_from_masks(run_dir):
            log.info(f"[DOWNLOAD_GOLDEN] Applied not_visible segments from golden masks run_id={run_id}")
    golden_root = run_dir / "golden"
    
    # Create a temporary zip file in scratch space
    scratch_tmp = scratch_subdir("tmp")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=str(scratch_tmp)) as tmp_zip:
        zip_path = Path(tmp_zip.name)
    
    # Create zip file with golden folder contents
    log.info(f"[DOWNLOAD_GOLDEN] Creating zip file: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in golden_root.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(golden_root)
                zipf.write(file_path, arcname)
        for dim in BEHAVIOR_DIMENSIONS:
            bpath = behavior_file_path(run_dir, dim)
            if bpath.is_file():
                zipf.write(bpath, bpath.name)
                log.info(f"[DOWNLOAD_GOLDEN] Included {bpath.name} in zip")
    
    log.info(f"[DOWNLOAD_GOLDEN] Zip file created: {zip_path} ({zip_path.stat().st_size} bytes)")
    
    # Clean up temp file after download
    def cleanup_zip():
        zip_path.unlink()
        log.info(f"[DOWNLOAD_GOLDEN] Cleaned up temp zip file: {zip_path}")
    
    background_tasks.add_task(cleanup_zip)
    
    return FileResponse(
        path=str(zip_path),
        filename=f"{run_id}_golden.zip",
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={run_id}_golden.zip"},
        background=background_tasks
    )


@app.get("/health")
def health():
    """Health check endpoint for testing connectivity"""
    return {"status": "ok", "message": "Backend is running"}
