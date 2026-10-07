"""Endpoints: serving source and tracked frames."""
import numpy as np
import cv2
from PIL import Image
from fastapi import HTTPException, APIRouter
from fastapi.responses import Response

from vos.config import RUNS_ROOT, VIDEO_NAME, log
from vos.overlay import get_color_for_id
from vos.storage import get_golden_ann_dir, load_frame_safely, parse_meta_file
from vos.tracking import find_tracked_mask_for_frame


router = APIRouter()


@router.get("/frame0/{run_id}")
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


@router.get("/tracked_frame/{run_id}/{relative_frame_idx}")
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


@router.get("/frame/{run_id}/{frame_idx}")
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
