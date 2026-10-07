"""Endpoints: behaviour labels, annotation mode and golden preview rebuild."""
from typing import Dict, Optional, Any
from fastapi import HTTPException, Query, APIRouter

from vos.behavior import (
    BEHAVIOR_DIMENSIONS,
    BEHAVIOR_DIMENSION_META,
    BEHAVIOR_LABELS_ACTIVITY,
    BEHAVIOR_LABELS_LABEL2,
    BEHAVIOR_LABELS_LABEL3,
    delete_label_from_frame,
    labels_at_frame,
    load_behavior_dimension,
    mark_behavior_preview_in_sync,
    mark_behavior_preview_out_of_sync,
    save_behavior_dimension,
    set_label_from_frame,
)
from vos.config import RUNS_ROOT, log
from vos.schemas import AnnotationModePayload, BehaviorDeleteLabelPayload, BehaviorSetLabelPayload
from vos.storage import get_annotation_mode, update_meta_key
from vos.video import rebuild_golden_preview_video


router = APIRouter()


@router.get("/behavior/labels")
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


@router.post("/run/{run_id}/annotation_mode")
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


@router.get("/behavior/{run_id}")
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


@router.post("/behavior/{run_id}/set_label")
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


@router.post("/behavior/{run_id}/delete_label")
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


@router.post("/golden/{run_id}/rebuild_preview")
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
