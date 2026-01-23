"""
testing_backend.py

FastAPI backend for:
- /segment/init   : upload ONE image + text prompt ("cow") -> initial instance masks
- /segment/refine : refine one instance using positive/negative points (SAM3 video session add_prompt)

This follows the SAME flow as the official SAM3 video notebook:
start_session -> add_prompt -> propagate_in_video -> use frame 0 outputs

Key design choice:
We use build_sam3_video_predictor() even for single images (treated as a 1-frame "video").
This is the most reliable way to get point prompting working in SAM3 right now.

Run:
  uvicorn testing_backend:app --host 0.0.0.0 --port 8000

Notes:
- Frontend should send points in PIXEL coordinates, and labels as +1 / -1.
- We convert to relative coords [0..1] and point_labels {1(pos),0(neg)} like the notebook.
"""

import io
import os
import uuid
import base64
import tempfile
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

import torch
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- SAM3 (video predictor + formatting utils)
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import prepare_masks_for_visualization


# -----------------------
# App + CORS
# -----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Utilities: mask encoding (PNG base64)
# -----------------------
def mask_to_png_base64(mask01: np.ndarray) -> str:
    """
    mask01: HxW float/bool mask in {0,1} or [0,1]
    returns base64 png (grayscale 0/255)
    """
    m = (mask01 > 0.5).astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", m)
    if not ok:
        raise RuntimeError("Failed to encode mask as PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def png_base64_to_mask(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.encode("utf-8"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Failed to decode mask PNG")
    return (img > 127).astype(np.float32)


def abs_points_to_rel(points_xy: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert Nx2 absolute pixel coords to Nx2 relative coords in [0,1]."""
    if points_xy.size == 0:
        return points_xy.astype(np.float32)
    rel = points_xy.astype(np.float32).copy()
    rel[:, 0] /= float(w)
    rel[:, 1] /= float(h)
    return rel


# -----------------------
# Predictor init (GPU IDs)
# -----------------------
def _gpu_ids() -> List[int]:
    if torch.cuda.is_available():
        return [torch.cuda.current_device()]
    return []


# Build predictor once (stateful sessions created per image)
PREDICTOR = build_sam3_video_predictor(gpus_to_use=_gpu_ids())


# -----------------------
# Session store
# -----------------------
@dataclass
class ImageSession:
    width: int
    height: int
    session_id: str
    tmpdir: str  # where we stored the 1-frame "video" folder
    # mapping from "instance_id" (frontend) to SAM3 obj_id
    obj_id_by_instance: Dict[str, int]


SESSIONS: Dict[str, ImageSession] = {}


# -----------------------
# API models
# -----------------------
class InitRequest(BaseModel):
    prompt: str = "cow"


class InitResponseInstance(BaseModel):
    instance_id: str
    obj_id: int
    score: float
    mask_png_b64: str


class InitResponse(BaseModel):
    image_id: str
    width: int
    height: int
    instances: List[InitResponseInstance]


class Point(BaseModel):
    x: float
    y: float
    label: int  # +1 or -1


class RefineRequest(BaseModel):
    image_id: str
    instance_id: str
    points: List[Point]


class RefineResponse(BaseModel):
    instance_id: str
    obj_id: int
    score: float
    mask_png_b64: str


# -----------------------
# Core SAM3 helpers
# -----------------------
def start_single_image_session(image_rgb: np.ndarray) -> Tuple[str, str]:
    """
    Create a temporary 1-frame "video folder" and start a SAM3 session on it.
    Returns (session_id, tmpdir).
    """
    tmpdir = tempfile.mkdtemp(prefix="sam3_img_")
    frame_path = os.path.join(tmpdir, "00000.jpg")
    # Save as JPEG like the notebook expects for folders
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(frame_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError("Failed to write temporary frame for SAM3 session")

    resp = PREDICTOR.handle_request(
        request=dict(
            type="start_session",
            resource_path=tmpdir,
        )
    )
    session_id = resp["session_id"]
    return session_id, tmpdir


def propagate_frame0(session_id: str) -> Optional[Any]:
    """
    Exactly like the notebook: iterate handle_stream_request(propagate_in_video)
    and return outputs for frame 0 if present.
    """
    out0 = None
    for resp in PREDICTOR.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        if resp.get("frame_index") == 0:
            out0 = resp.get("outputs")
    return out0


def format_frame0(outputs0: Any) -> Any:
    """
    Use SAM3's own visualization formatting utility to normalize output structure.
    This is the most robust way to parse masks without guessing keys.
    """
    formatted = prepare_masks_for_visualization({0: outputs0})
    # formatted is a dict-like: {frame_idx: formatted_outputs}
    return formatted[0]


def extract_instances_from_formatted(formatted0: Any) -> List[Dict[str, Any]]:
    """
    Turn formatted frame0 output into a list of dicts:
      [{"obj_id": int, "mask": HxW float(0/1), "score": float}, ...]
    This function is defensive because SAM3 formatting can vary.

    If this returns empty, print the formatted0 structure (see DEBUG block).
    """
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
                instances.append(
                    dict(obj_id=int(oid), mask=masks_np[i].astype(np.float32), score=float(scores_list[i] if i < len(scores_list) else 1.0))
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
                    score = v.get("score", v.get("iou", 1.0))
                    instances.append(dict(obj_id=int(oid), mask=np.asarray(m).astype(np.float32), score=float(score)))
                else:
                    # value itself is mask
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
                score = obj.get("score", obj.get("iou", 1.0))
                instances.append(dict(obj_id=int(oid), mask=np.asarray(m).astype(np.float32), score=float(score)))
            else:
                # list of masks
                instances.append(dict(obj_id=i, mask=np.asarray(obj).astype(np.float32), score=1.0))
        return instances

    return instances


def safe_mask_hw(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Ensure mask is HxW float32 in [0,1].
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


# -----------------------
# Endpoints
# -----------------------
@app.post("/segment/init", response_model=InitResponse)
async def segment_init(
    meta: InitRequest = Body(...),
    image: UploadFile = File(...),
):
    content = await image.read()
    pil = Image.open(io.BytesIO(content)).convert("RGB")
    image_np = np.array(pil)
    h, w = image_np.shape[:2]

    # Create a SAM3 session for this single image
    session_id, tmpdir = start_single_image_session(image_np)

    # Add a TEXT prompt on frame 0
    _ = PREDICTOR.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=meta.prompt,
        )
    )

    # Finalize via propagation (matches notebook)
    outputs0 = propagate_frame0(session_id)
    if outputs0 is None:
        raise RuntimeError("SAM3 propagate_in_video did not return frame 0 outputs")

    formatted0 = format_frame0(outputs0)

    # DEBUG (uncomment once if empty results)
    # print("RAW outputs0 type:", type(outputs0))
    # if isinstance(outputs0, dict): print("RAW outputs0 keys:", list(outputs0.keys())[:30])
    # print("FORMATTED0 type:", type(formatted0))
    # if isinstance(formatted0, dict): print("FORMATTED0 keys:", list(formatted0.keys())[:50])

    inst_list = extract_instances_from_formatted(formatted0)

    # Build response + mapping
    image_id = str(uuid.uuid4())
    obj_id_by_instance: Dict[str, int] = {}

    instances: List[InitResponseInstance] = []
    for i, inst in enumerate(inst_list):
        obj_id = int(inst["obj_id"])
        mask = safe_mask_hw(np.asarray(inst["mask"]), h, w)
        score = float(inst.get("score", 1.0))

        instance_id = str(i)  # frontend instance key
        obj_id_by_instance[instance_id] = obj_id

        instances.append(
            InitResponseInstance(
                instance_id=instance_id,
                obj_id=obj_id,
                score=score,
                mask_png_b64=mask_to_png_base64(mask),
            )
        )

    SESSIONS[image_id] = ImageSession(
        width=w,
        height=h,
        session_id=session_id,
        tmpdir=tmpdir,
        obj_id_by_instance=obj_id_by_instance,
    )

    return InitResponse(image_id=image_id, width=w, height=h, instances=instances)


@app.post("/segment/refine", response_model=RefineResponse)
async def segment_refine(req: RefineRequest):
    sess = SESSIONS.get(req.image_id)
    if sess is None:
        raise RuntimeError("Unknown image_id (session expired?)")

    obj_id = sess.obj_id_by_instance.get(req.instance_id)
    if obj_id is None:
        raise RuntimeError(f"Unknown instance_id {req.instance_id}")

    # Convert points: frontend sends pixel coords + label {+1,-1}
    pts_xy = np.array([[p.x, p.y] for p in req.points], dtype=np.float32)
    # SAM3 convention: positive=1, negative=0
    lbs = np.array([1 if p.label == 1 else 0 for p in req.points], dtype=np.int32)

    # Convert to relative coords as in notebook
    pts_rel = abs_points_to_rel(pts_xy, sess.width, sess.height)

    points_tensor = torch.tensor(pts_rel, dtype=torch.float32)
    labels_tensor = torch.tensor(lbs, dtype=torch.int32)

    # Add point prompt for that existing obj_id on frame 0 (refinement)
    _ = PREDICTOR.handle_request(
        request=dict(
            type="add_prompt",
            session_id=sess.session_id,
            frame_index=0,
            points=points_tensor,
            point_labels=labels_tensor,
            obj_id=int(obj_id),
        )
    )

    # Finalize via propagation (matches notebook)
    outputs0 = propagate_frame0(sess.session_id)
    if outputs0 is None:
        raise RuntimeError("SAM3 propagate_in_video did not return frame 0 outputs")

    formatted0 = format_frame0(outputs0)
    inst_list = extract_instances_from_formatted(formatted0)

    # Find the mask for our obj_id
    found = None
    for inst in inst_list:
        if int(inst.get("obj_id", -1)) == int(obj_id):
            found = inst
            break

    if found is None:
        # If formatting changed, dump minimal debug to logs
        print("WARN: Could not find obj_id in formatted outputs. obj_id=", obj_id)
        print("FORMATTED0 type:", type(formatted0))
        if isinstance(formatted0, dict):
            print("FORMATTED0 keys:", list(formatted0.keys())[:50])
        raise RuntimeError("Refine succeeded but could not locate refined object mask in outputs")

    mask = safe_mask_hw(np.asarray(found["mask"]), sess.height, sess.width)
    score = float(found.get("score", 1.0))

    return RefineResponse(
        instance_id=req.instance_id,
        obj_id=int(obj_id),
        score=score,
        mask_png_b64=mask_to_png_base64(mask),
    )


@app.get("/health")
def health():
    return {"ok": True, "sessions": len(SESSIONS)}


@app.post("/segment/close")
async def segment_close(image_id: str = Body(..., embed=True)):
    """
    Optional: close an image session and free resources.
    """
    sess = SESSIONS.pop(image_id, None)
    if sess is None:
        return {"ok": False, "reason": "unknown image_id"}

    try:
        _ = PREDICTOR.handle_request(
            request=dict(type="close_session", session_id=sess.session_id)
        )
    except Exception as e:
        print("WARN: close_session failed:", e)

    try:
        shutil.rmtree(sess.tmpdir, ignore_errors=True)
    except Exception:
        pass

    return {"ok": True}


# NOTE: don't call predictor.shutdown() while server runs;
# do it only on process exit if you want.
"""
testing_backend.py

FastAPI backend for:
- /segment/init   : upload ONE image + text prompt ("cow") -> initial instance masks
- /segment/refine : refine one instance using positive/negative points (SAM3 video session add_prompt)

This follows the SAME flow as the official SAM3 video notebook:
start_session -> add_prompt -> propagate_in_video -> use frame 0 outputs

Key design choice:
We use build_sam3_video_predictor() even for single images (treated as a 1-frame "video").
This is the most reliable way to get point prompting working in SAM3 right now.

Run:
  uvicorn testing_backend:app --host 0.0.0.0 --port 8000

Notes:
- Frontend should send points in PIXEL coordinates, and labels as +1 / -1.
- We convert to relative coords [0..1] and point_labels {1(pos),0(neg)} like the notebook.
"""

import io
import os
import uuid
import base64
import tempfile
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

import torch
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- SAM3 (video predictor + formatting utils)
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import prepare_masks_for_visualization


# -----------------------
# App + CORS
# -----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Utilities: mask encoding (PNG base64)
# -----------------------
def mask_to_png_base64(mask01: np.ndarray) -> str:
    """
    mask01: HxW float/bool mask in {0,1} or [0,1]
    returns base64 png (grayscale 0/255)
    """
    m = (mask01 > 0.5).astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", m)
    if not ok:
        raise RuntimeError("Failed to encode mask as PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def png_base64_to_mask(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.encode("utf-8"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Failed to decode mask PNG")
    return (img > 127).astype(np.float32)


def abs_points_to_rel(points_xy: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert Nx2 absolute pixel coords to Nx2 relative coords in [0,1]."""
    if points_xy.size == 0:
        return points_xy.astype(np.float32)
    rel = points_xy.astype(np.float32).copy()
    rel[:, 0] /= float(w)
    rel[:, 1] /= float(h)
    return rel


# -----------------------
# Predictor init (GPU IDs)
# -----------------------
def _gpu_ids() -> List[int]:
    if torch.cuda.is_available():
        return [torch.cuda.current_device()]
    return []


# Build predictor once (stateful sessions created per image)
PREDICTOR = build_sam3_video_predictor(gpus_to_use=_gpu_ids())


# -----------------------
# Session store
# -----------------------
@dataclass
class ImageSession:
    width: int
    height: int
    session_id: str
    tmpdir: str  # where we stored the 1-frame "video" folder
    # mapping from "instance_id" (frontend) to SAM3 obj_id
    obj_id_by_instance: Dict[str, int]


SESSIONS: Dict[str, ImageSession] = {}


# -----------------------
# API models
# -----------------------
class InitRequest(BaseModel):
    prompt: str = "cow"


class InitResponseInstance(BaseModel):
    instance_id: str
    obj_id: int
    score: float
    mask_png_b64: str


class InitResponse(BaseModel):
    image_id: str
    width: int
    height: int
    instances: List[InitResponseInstance]


class Point(BaseModel):
    x: float
    y: float
    label: int  # +1 or -1


class RefineRequest(BaseModel):
    image_id: str
    instance_id: str
    points: List[Point]


class RefineResponse(BaseModel):
    instance_id: str
    obj_id: int
    score: float
    mask_png_b64: str


# -----------------------
# Core SAM3 helpers
# -----------------------
def start_single_image_session(image_rgb: np.ndarray) -> Tuple[str, str]:
    """
    Create a temporary 1-frame "video folder" and start a SAM3 session on it.
    Returns (session_id, tmpdir).
    """
    tmpdir = tempfile.mkdtemp(prefix="sam3_img_")
    frame_path = os.path.join(tmpdir, "00000.jpg")
    # Save as JPEG like the notebook expects for folders
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(frame_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError("Failed to write temporary frame for SAM3 session")

    resp = PREDICTOR.handle_request(
        request=dict(
            type="start_session",
            resource_path=tmpdir,
        )
    )
    session_id = resp["session_id"]
    return session_id, tmpdir


def propagate_frame0(session_id: str) -> Optional[Any]:
    """
    Exactly like the notebook: iterate handle_stream_request(propagate_in_video)
    and return outputs for frame 0 if present.
    """
    out0 = None
    for resp in PREDICTOR.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        if resp.get("frame_index") == 0:
            out0 = resp.get("outputs")
    return out0


def format_frame0(outputs0: Any) -> Any:
    """
    Use SAM3's own visualization formatting utility to normalize output structure.
    This is the most robust way to parse masks without guessing keys.
    """
    formatted = prepare_masks_for_visualization({0: outputs0})
    # formatted is a dict-like: {frame_idx: formatted_outputs}
    return formatted[0]


def extract_instances_from_formatted(formatted0: Any) -> List[Dict[str, Any]]:
    """
    Turn formatted frame0 output into a list of dicts:
      [{"obj_id": int, "mask": HxW float(0/1), "score": float}, ...]
    This function is defensive because SAM3 formatting can vary.

    If this returns empty, print the formatted0 structure (see DEBUG block).
    """
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
                instances.append(
                    dict(obj_id=int(oid), mask=masks_np[i].astype(np.float32), score=float(scores_list[i] if i < len(scores_list) else 1.0))
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
                    score = v.get("score", v.get("iou", 1.0))
                    instances.append(dict(obj_id=int(oid), mask=np.asarray(m).astype(np.float32), score=float(score)))
                else:
                    # value itself is mask
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
                score = obj.get("score", obj.get("iou", 1.0))
                instances.append(dict(obj_id=int(oid), mask=np.asarray(m).astype(np.float32), score=float(score)))
            else:
                # list of masks
                instances.append(dict(obj_id=i, mask=np.asarray(obj).astype(np.float32), score=1.0))
        return instances

    return instances


def safe_mask_hw(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Ensure mask is HxW float32 in [0,1].
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


# -----------------------
# Endpoints
# -----------------------
@app.post("/segment/init", response_model=InitResponse)
async def segment_init(
    meta: InitRequest = Body(...),
    image: UploadFile = File(...),
):
    content = await image.read()
    pil = Image.open(io.BytesIO(content)).convert("RGB")
    image_np = np.array(pil)
    h, w = image_np.shape[:2]

    # Create a SAM3 session for this single image
    session_id, tmpdir = start_single_image_session(image_np)

    # Add a TEXT prompt on frame 0
    _ = PREDICTOR.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=meta.prompt,
        )
    )

    # Finalize via propagation (matches notebook)
    outputs0 = propagate_frame0(session_id)
    if outputs0 is None:
        raise RuntimeError("SAM3 propagate_in_video did not return frame 0 outputs")

    formatted0 = format_frame0(outputs0)

    # DEBUG (uncomment once if empty results)
    # print("RAW outputs0 type:", type(outputs0))
    # if isinstance(outputs0, dict): print("RAW outputs0 keys:", list(outputs0.keys())[:30])
    # print("FORMATTED0 type:", type(formatted0))
    # if isinstance(formatted0, dict): print("FORMATTED0 keys:", list(formatted0.keys())[:50])

    inst_list = extract_instances_from_formatted(formatted0)

    # Build response + mapping
    image_id = str(uuid.uuid4())
    obj_id_by_instance: Dict[str, int] = {}

    instances: List[InitResponseInstance] = []
    for i, inst in enumerate(inst_list):
        obj_id = int(inst["obj_id"])
        mask = safe_mask_hw(np.asarray(inst["mask"]), h, w)
        score = float(inst.get("score", 1.0))

        instance_id = str(i)  # frontend instance key
        obj_id_by_instance[instance_id] = obj_id

        instances.append(
            InitResponseInstance(
                instance_id=instance_id,
                obj_id=obj_id,
                score=score,
                mask_png_b64=mask_to_png_base64(mask),
            )
        )

    SESSIONS[image_id] = ImageSession(
        width=w,
        height=h,
        session_id=session_id,
        tmpdir=tmpdir,
        obj_id_by_instance=obj_id_by_instance,
    )

    return InitResponse(image_id=image_id, width=w, height=h, instances=instances)


@app.post("/segment/refine", response_model=RefineResponse)
async def segment_refine(req: RefineRequest):
    sess = SESSIONS.get(req.image_id)
    if sess is None:
        raise RuntimeError("Unknown image_id (session expired?)")

    obj_id = sess.obj_id_by_instance.get(req.instance_id)
    if obj_id is None:
        raise RuntimeError(f"Unknown instance_id {req.instance_id}")

    # Convert points: frontend sends pixel coords + label {+1,-1}
    pts_xy = np.array([[p.x, p.y] for p in req.points], dtype=np.float32)
    # SAM3 convention: positive=1, negative=0
    lbs = np.array([1 if p.label == 1 else 0 for p in req.points], dtype=np.int32)

    # Convert to relative coords as in notebook
    pts_rel = abs_points_to_rel(pts_xy, sess.width, sess.height)

    points_tensor = torch.tensor(pts_rel, dtype=torch.float32)
    labels_tensor = torch.tensor(lbs, dtype=torch.int32)

    # Add point prompt for that existing obj_id on frame 0 (refinement)
    _ = PREDICTOR.handle_request(
        request=dict(
            type="add_prompt",
            session_id=sess.session_id,
            frame_index=0,
            points=points_tensor,
            point_labels=labels_tensor,
            obj_id=int(obj_id),
        )
    )

    # Finalize via propagation (matches notebook)
    outputs0 = propagate_frame0(sess.session_id)
    if outputs0 is None:
        raise RuntimeError("SAM3 propagate_in_video did not return frame 0 outputs")

    formatted0 = format_frame0(outputs0)
    inst_list = extract_instances_from_formatted(formatted0)

    # Find the mask for our obj_id
    found = None
    for inst in inst_list:
        if int(inst.get("obj_id", -1)) == int(obj_id):
            found = inst
            break

    if found is None:
        # If formatting changed, dump minimal debug to logs
        print("WARN: Could not find obj_id in formatted outputs. obj_id=", obj_id)
        print("FORMATTED0 type:", type(formatted0))
        if isinstance(formatted0, dict):
            print("FORMATTED0 keys:", list(formatted0.keys())[:50])
        raise RuntimeError("Refine succeeded but could not locate refined object mask in outputs")

    mask = safe_mask_hw(np.asarray(found["mask"]), sess.height, sess.width)
    score = float(found.get("score", 1.0))

    return RefineResponse(
        instance_id=req.instance_id,
        obj_id=int(obj_id),
        score=score,
        mask_png_b64=mask_to_png_base64(mask),
    )


@app.get("/health")
def health():
    return {"ok": True, "sessions": len(SESSIONS)}


@app.post("/segment/close")
async def segment_close(image_id: str = Body(..., embed=True)):
    """
    Optional: close an image session and free resources.
    """
    sess = SESSIONS.pop(image_id, None)
    if sess is None:
        return {"ok": False, "reason": "unknown image_id"}

    try:
        _ = PREDICTOR.handle_request(
            request=dict(type="close_session", session_id=sess.session_id)
        )
    except Exception as e:
        print("WARN: close_session failed:", e)

    try:
        shutil.rmtree(sess.tmpdir, ignore_errors=True)
    except Exception:
        pass

    return {"ok": True}


# NOTE: don't call predictor.shutdown() while server runs;
# do it only on process exit if you want.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("testing_backend:app", host="0.0.0.0", port=8000, reload=False)
