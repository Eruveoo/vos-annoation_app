"""Endpoints: video upload / prepare and frame extraction."""
import os
import uuid
import time
import threading
from datetime import datetime
from pathlib import Path
import cv2
from fastapi import HTTPException, UploadFile, File, BackgroundTasks, APIRouter
from fastapi.responses import JSONResponse

from vos.config import RUNS_ROOT, VIDEO_NAME, log
from vos.state import prepare_progress
from vos.storage import parse_meta_file
from vos.video import extract_frames


router = APIRouter()


# -------------------------
# API
# -------------------------
@router.post("/prepare")
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


@router.post("/prepare_upload")
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


@router.get("/prepare_upload_progress/{run_id}")
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
