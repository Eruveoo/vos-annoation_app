"""Endpoints: SAM-3 initialization on the first frame and initial ID assignment."""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import cv2
from PIL import Image
from fastapi import HTTPException, UploadFile, File, APIRouter
from fastapi.responses import JSONResponse

from vos.behavior import (
    BEHAVIOR_DIMENSIONS,
    _default_label_for_dimension,
    create_initial_segments,
    save_behavior_dimension,
)
from vos.config import RUNS_ROOT, VIDEO_NAME, log
from vos.overlay import encode_frame_to_base64, get_color_for_id
from vos.schemas import ApplyInitPayload, PreviewUpdate
from vos.segmentation import auto_assign_ids, run_sam3_on_frame
from vos.storage import (
    ensure_dir,
    get_annotation_mode,
    get_golden_ann_dir,
    get_golden_jpeg_dir,
    get_init_masks_file,
    load_frame_safely,
    parse_meta_file,
)
from vos.video import _ffmpeg_reencode_video, _render_segment_from_golden


router = APIRouter()


@router.post("/init_sam/{run_id}")
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


@router.post("/match_init_ids/{run_id}")
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


@router.post("/preview_init_update/{run_id}")
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


@router.post("/apply_init_ids/{run_id}")
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

    # Clean up temporary masks file
    masks_file.unlink()

    log.info(f"[APPLY_INIT_IDS] Initialization complete: run_id={run_id}, n_ids={n_ids}")
    return {"run_id": run_id, "n_ids": n_ids}
