import os
import shutil
import subprocess
import uuid
import time
from datetime import datetime
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, Request, Query, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Optional, Tuple

# IMPORTANT: use uvicorn logger (so logs show up in uvicorn output)
log = logging.getLogger("uvicorn.error")

app = FastAPI()

# -------------------------
# Request logging middleware
# -------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    log.info(f"{request.method} {request.url.path} -> {response.status_code} ({dt:.1f} ms)")
    return response


# -------------------------
# Config / paths
# -------------------------
XMEM_REPO = "./XMem"
XMEM_MODEL = os.path.join(XMEM_REPO, "saves", "XMem.pth")
VIDEO_NAME = "video1"
JPEG_QUALITY = 90
MASK_THRESHOLD = 0.5

RUNS_ROOT = Path("runs")

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


def masks_to_label_map(masks_bool):
    import numpy as np
    H, W = masks_bool[0].shape
    label = np.zeros((H, W), dtype=np.uint8)
    for i, m in enumerate(masks_bool, start=1):
        label[m] = i
    return label


def random_color(seed: int):
    import numpy as np
    rng = np.random.RandomState(seed)
    return tuple(int(x) for x in rng.randint(50, 255, size=3))


# -------------------------
# Model (lazy load)
# -------------------------
MODEL = None
PROCESSOR = None


def get_model():
    global MODEL, PROCESSOR
    if MODEL is None:
        log.info("Loading SAM-3 model...")
        t0 = time.perf_counter()
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        MODEL = build_sam3_image_model()
        PROCESSOR = Sam3Processor(MODEL)
        log.info(f"SAM-3 loaded in {time.perf_counter() - t0:.2f}s")
    return PROCESSOR

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
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    if p.returncode != 0:
        log.error(f"ffmpeg re-encode failed with return code {p.returncode}")
        log.error(f"ffmpeg output:\n{p.stdout[-2000:] if p.stdout else '(no output)'}")
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
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    if p.returncode != 0:
        log.error(f"ffmpeg drop-seed failed with return code {p.returncode}")
        log.error(f"ffmpeg output:\n{p.stdout[-2000:] if p.stdout else '(no output)'}")
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
def extract_frames(video_path: str, jpeg_dir: Path):
    import cv2

    log.info(f"Extracting frames from {video_path}")
    t0 = time.perf_counter()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
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
        if idx % LOG_EVERY_FRAMES_EXTRACT == 0:
            log.info(f"  extracted {idx} frames...")

    cap.release()
    log.info(f"Frame extraction done: {len(frames)} frames ({time.perf_counter()-t0:.2f}s)")
    if len(frames) == 0:
        raise RuntimeError("No frames extracted.")
    return frames, fps


def compute_iou(mask1, mask2) -> float:
    """Compute Intersection over Union (IoU) between two boolean masks."""
    import numpy as np
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_centroid_distance(mask1, mask2) -> float:
    """Compute distance between centroids of two masks."""
    import numpy as np
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
    import numpy as np
    
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


def run_sam3_on_frame(prompt: str, frame_path: Path) -> list:
    """
    Run SAM-3 on a specific frame and return list of boolean masks.
    """
    from PIL import Image
    import numpy as np
    
    log.info(f"SAM-3 on frame {frame_path}, prompt={prompt}")
    processor = get_model()
    
    img = Image.open(frame_path).convert("RGB")
    W, H = img.size
    
    state = processor.set_image(img)
    out = processor.set_text_prompt(state=state, prompt=prompt)
    
    log.info(f"SAM-3 raw masks: {len(out['masks'])}")
    
    masks = []
    for m in out["masks"]:
        mask = m.squeeze().cpu().numpy()
        mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST)) > MASK_THRESHOLD
        if mask.sum() > 500:
            masks.append(mask)
    
    if not masks:
        raise RuntimeError("No valid masks from SAM-3")
    
    log.info(f"SAM-3 kept {len(masks)} masks")
    return masks


def run_sam3_on_first_frame(prompt, jpeg_dir, ann_dir, frames):
    from PIL import Image
    import numpy as np

    log.info(f"SAM-3 on frame0, prompt={prompt}")
    processor = get_model()

    first_path = jpeg_dir / frames[0]
    img = Image.open(first_path).convert("RGB")
    W, H = img.size

    state = processor.set_image(img)
    out = processor.set_text_prompt(state=state, prompt=prompt)

    log.info(f"SAM-3 raw masks: {len(out['masks'])}")

    masks = []
    for m in out["masks"]:
        mask = m.squeeze().cpu().numpy()
        mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST)) > MASK_THRESHOLD
        if mask.sum() > 500:
            masks.append(mask)

    if not masks:
        raise RuntimeError("No valid masks from SAM-3")

    label_map = masks_to_label_map(masks)
    ann0 = ann_dir / frames[0].replace(".jpg", ".png")
    Image.fromarray(label_map).save(ann0)

    log.info(f"SAM-3 kept {label_map.max()} masks")
    return int(label_map.max()), str(first_path)


def golden_progress(run_dir: Path, n_total: int):
    golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
    if not golden_ann_dir.exists():
        log.info(f"[GOLDEN_PROGRESS] Golden annotations dir does not exist: {golden_ann_dir}")
        return 0, 0.0, None

    pngs = sorted(golden_ann_dir.glob("*.png"))
    if not pngs:
        log.info(f"[GOLDEN_PROGRESS] No PNG files found in {golden_ann_dir}")
        return 0, 0.0, None

    def idx_from_name(p: Path) -> int:
        return int(p.stem)

    frame_indices = [idx_from_name(p) for p in pngs]
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
    if seed_ann_path and seed_ann_path.exists():
        seed_ann = seed_ann_path
        log.info(f"Using auto-reset seed annotation: {seed_ann}")
    else:
        golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
        seed_ann = golden_ann_dir / f"{seed_idx:05d}.png"
        if not seed_ann.exists():
            raise RuntimeError(f"Missing golden seed annotation: {seed_ann}")

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
        if not src.exists():
            raise RuntimeError(f"Missing source frame: {src}")

        dst = dst_jpeg / f"{n:05d}.jpg"
        shutil.copy2(src, dst)
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

    proc = subprocess.Popen(
        cmd,
        cwd=XMEM_REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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


def render_video(jpeg_dir: Path, frames, found_pngs, out_video: Path, fps: float, n_ids: int):
    """
    Render all provided frames list (same length as found_pngs ideally).
    """
    import cv2
    import numpy as np
    from PIL import Image

    if not frames:
        raise RuntimeError("render_video got empty frames list")

    log.info(f"Rendering preview video: {out_video}")
    first = cv2.imread(str(jpeg_dir / frames[0]))
    if first is None:
        raise RuntimeError("Could not read first frame for rendering.")
    H, W = first.shape[:2]

    out_video.parent.mkdir(parents=True, exist_ok=True)
    # Use H.264 codec (avc1) for better browser compatibility
    # Fallback to mp4v if avc1 is not available
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (W, H))
    if not writer.isOpened():
        log.warning("avc1 codec not available, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter with codec mp4v for {out_video}")
    colors = {i: random_color(i) for i in range(1, n_ids + 1)}

    T = min(len(frames), len(found_pngs))
    for t in range(T):
        frame = cv2.imread(str(jpeg_dir / frames[t]))
        if frame is None:
            raise RuntimeError(f"Could not read frame {frames[t]}")
        mask = np.array(Image.open(found_pngs[t]))

        for cid, col in colors.items():
            m = (mask == cid)
            if not m.any():
                continue

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
                cv2.LINE_AA,
            )

        writer.write(frame)

        if (t + 1) % LOG_EVERY_FRAMES_RENDER == 0:
            log.info(f"  rendered {t+1}/{T}")

    writer.release()
    log.info("Rendering done")
    return T


def _render_segment_from_golden(run_dir: Path, fps: float, n_ids: int, start_idx: int, end_idx: int, out_path: Path):
    """
    Render golden overlay segment for frames start_idx..end_idx inclusive into out_path.
    Uses original JPEGs + golden label PNGs.
    """
    import numpy as np
    from PIL import Image
    
    log.info(f"[RENDER_GOLDEN] Rendering segment: frames {start_idx}..{end_idx} -> {out_path}")
    
    src_root = run_dir / "xmem_generic"
    src_jpeg = src_root / "JPEGImages" / VIDEO_NAME
    golden_ann = run_dir / "golden" / "Annotations" / VIDEO_NAME

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
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
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
            p2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
@app.post("/init")
def init(video_path: str, prompt: str):
    """
    Initialize a new annotation session:
    1. Extract frames from video
    2. Run SAM-3 on frame 0
    3. Return masks for manual ID assignment (don't save annotation yet)
    """
    import numpy as np
    import cv2
    from PIL import Image
    from fastapi.responses import JSONResponse
    
    log.info(f"/init video={video_path} prompt={prompt}")

    if not os.path.exists(video_path):
        raise HTTPException(400, f"Video not found: {video_path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = RUNS_ROOT / run_id

    custom_root = run_dir / "xmem_generic"
    jpeg_dir = custom_root / "JPEGImages" / VIDEO_NAME
    ann_dir = custom_root / "Annotations" / VIDEO_NAME

    ensure_clean_dir(run_dir)
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    frames, fps = extract_frames(video_path, jpeg_dir)
    
    # Run SAM-3 on frame 0 (but don't save annotation yet)
    log.info(f"[INIT] Running SAM-3 on frame 0, prompt={prompt}")
    processor = get_model()
    first_path = jpeg_dir / frames[0]
    img = Image.open(first_path).convert("RGB")
    W, H = img.size

    state = processor.set_image(img)
    out = processor.set_text_prompt(state=state, prompt=prompt)

    log.info(f"[INIT] SAM-3 raw masks: {len(out['masks'])}")

    masks = []
    for m in out["masks"]:
        mask = m.squeeze().cpu().numpy()
        mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST)) > MASK_THRESHOLD
        if mask.sum() > 500:
            masks.append(mask)

    if not masks:
        raise RuntimeError("No valid masks from SAM-3")

    n_masks = len(masks)
    log.info(f"[INIT] SAM-3 kept {n_masks} masks")
    
    # Save masks temporarily for later use
    masks_file = run_dir / "init_masks.npy"
    np.save(masks_file, masks)
    
    # Render preview image with auto-assigned IDs (1, 2, 3, ...)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    def color_for(id: int):
        np.random.seed(id * 42)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    log.info(f"[INIT] Rendering {n_masks} masks with auto-assigned IDs")
    for mask_idx, mask in enumerate(masks, start=1):
        assigned_id = mask_idx  # Auto-assign sequential IDs: 1, 2, 3, ...
        mask_pixels = int(mask.sum())
        log.info(f"[INIT] Rendering mask {mask_idx-1} -> ID {assigned_id} ({mask_pixels} pixels)")
        col = color_for(assigned_id)
        overlay = frame.copy()
        overlay[mask] = col
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
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
    
    # Encode preview image
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode frame")
    
    # Prepare response
    mask_assignments = [
        {
            "mask_index": mask_idx,
            "auto_assigned_id": mask_idx + 1,  # IDs start from 1
        }
        for mask_idx in range(n_masks)
    ]
    
    # Save metadata (without n_ids yet - will be updated after ID assignment)
    (run_dir / "meta.txt").write_text(
        f"video_path={video_path}\n"
        f"prompt={prompt}\n"
        f"fps={fps}\n"
        f"frames={len(frames)}\n"
        f"ids=0\n"  # Will be updated after ID assignment
    )
    
    import base64
    image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    
    log.info(f"[INIT] Returning {n_masks} masks for ID assignment")
    return JSONResponse(content={
        "run_id": run_id,
        "fps": fps,
        "n_frames_total": len(frames),
        "image": f"data:image/jpeg;base64,{image_b64}",
        "mask_assignments": mask_assignments,
    })


class IDMapping(BaseModel):
    mapping: Dict[str, int]


@app.post("/match_init_ids/{run_id}")
def match_init_ids(run_id: str, file: UploadFile = File(...)):
    """
    Match new SAM masks from init to IDs from a previous golden mask file.
    Uses the same IoU-based matching as auto_assign_ids.
    Accepts an uploaded PNG mask file.
    """
    import numpy as np
    import cv2
    from PIL import Image
    from fastapi.responses import JSONResponse
    import tempfile
    
    log.info(f"/match_init_ids run_id={run_id} uploaded_file={file.filename}")
    
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"Run not found: {run_id}")
    
    # Load saved masks from init
    masks_file = run_dir / "init_masks.npy"
    if not masks_file.exists():
        raise HTTPException(400, "Masks not found. Please run /init first.")
    
    new_masks = np.load(masks_file, allow_pickle=True)
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
    jpeg_dir = custom_root / "JPEGImages" / VIDEO_NAME
    first_path = jpeg_dir / "00000.jpg"
    
    if not first_path.exists():
        raise HTTPException(404, "Frame 0 not found")
    
    frame = cv2.imread(str(first_path))
    if frame is None:
        raise HTTPException(500, "Could not read frame 0")
    
    def color_for(id: int):
        np.random.seed(id * 42)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    # Render masks with matched IDs
    for mask_idx, mask in enumerate(new_masks):
        matched_id = assignments.get(mask_idx, mask_idx + 1)
        col = color_for(matched_id)
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
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode frame")
    
    # Prepare response
    mask_assignments = [
        {
            "mask_index": mask_idx,
            "auto_assigned_id": mask_idx + 1,  # Original auto-assigned (sequential)
            "matched_id": matched_id,  # Matched ID from previous mask
        }
        for mask_idx, matched_id in sorted(assignments.items())
    ]
    
    import base64
    image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    
    log.info(f"[MATCH_INIT_IDS] Returning {len(mask_assignments)} matched assignments ({matched_count}/{total_count} matched to previous IDs)")
    return JSONResponse(content={
        "mask_assignments": mask_assignments,
        "matched_count": matched_count,
        "total_count": total_count,
        "image": f"data:image/jpeg;base64,{image_b64}",
    })


@app.post("/apply_init_ids/{run_id}")
def apply_init_ids(run_id: str, id_mapping: IDMapping):
    """
    Apply user's ID mapping to frame 0 masks and complete initialization.
    This creates the annotation file, golden folder, and preview video.
    """
    import numpy as np
    from PIL import Image
    
    log.info(f"/apply_init_ids run_id={run_id} mapping={id_mapping.mapping}")
    
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"Run not found: {run_id}")
    
    # Load saved masks
    masks_file = run_dir / "init_masks.npy"
    if not masks_file.exists():
        raise HTTPException(400, "Masks not found. Please run /init first.")
    
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
    
    for mask_idx_str, final_id in id_mapping.mapping.items():
        mask_idx = int(mask_idx_str)
        if mask_idx >= n_masks:
            raise HTTPException(400, f"Invalid mask_index {mask_idx} (max: {n_masks-1})")
        final_id = int(final_id)
        if final_id <= 0:  # Skip deleted masks
            continue
        label_map[masks[mask_idx]] = final_id
    
    # Save annotation
    ann0 = ann_dir / "00000.png"
    Image.fromarray(label_map).save(ann0)
    n_ids = int(label_map.max())
    log.info(f"[APPLY_INIT_IDS] Saved annotation with {n_ids} objects, IDs={sorted([id for id in np.unique(label_map) if id > 0])}")
    
    # Create golden folder + seed frame0 annotation
    golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
    ensure_dir(golden_ann_dir)
    shutil.copy2(ann0, golden_ann_dir / "00000.png")
    
    # Also copy frame0 JPEG to golden/JPEGImages/video1/
    golden_jpeg_dir = run_dir / "golden" / "JPEGImages" / VIDEO_NAME
    ensure_dir(golden_jpeg_dir)
    shutil.copy2(jpeg_dir / "00000.jpg", golden_jpeg_dir / "00000.jpg")
    
    # Initialize golden preview video with frame0
    try:
        log.info("[APPLY_INIT_IDS] Initializing golden preview video with frame0...")
        seg0 = run_dir / "golden_segments" / "00000_00000.mp4"
        ensure_dir(seg0.parent)
        log.info(f"[APPLY_INIT_IDS] Rendering segment 0-0 to {seg0}")
        _render_segment_from_golden(run_dir, fps, n_ids, 0, 0, seg0)
        
        if not seg0.exists():
            raise RuntimeError(f"Segment file was not created: {seg0}")
        
        golden_preview_init = run_dir / "golden" / "golden_preview.mp4"
        ensure_dir(golden_preview_init.parent)
        golden_preview_init.write_bytes(seg0.read_bytes())
        
        # Re-encode to ensure browser-compatible format
        log.info("[APPLY_INIT_IDS] Re-encoding golden_preview.mp4 to browser-compatible format...")
        golden_preview_tmp = run_dir / "golden" / "golden_preview_tmp.mp4"
        if _ffmpeg_reencode_video(golden_preview_init, golden_preview_tmp, fps):
            golden_preview_tmp.replace(golden_preview_init)
            log.info("[APPLY_INIT_IDS] ✅ Golden preview initialized")
        else:
            log.warning("[APPLY_INIT_IDS] Re-encoding failed, keeping original")
    except Exception as e:
        log.error(f"[APPLY_INIT_IDS] ❌ Golden preview init failed (non-fatal): {e}", exc_info=True)
    
    # Update metadata with final n_ids
    meta_content = meta_path.read_text()
    meta_path.write_text(
        meta_content.replace("ids=0", f"ids={n_ids}")
    )
    
    # Clean up temporary masks file
    masks_file.unlink()
    
    log.info(f"[APPLY_INIT_IDS] Initialization complete: run_id={run_id}, n_ids={n_ids}")
    return {"run_id": run_id, "n_ids": n_ids}


@app.post("/resume")
def resume(run_id: str):
    """Resume an existing annotation session."""
    log.info(f"/resume run_id={run_id}")
    
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"Run not found: {run_id}")
    
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(400, f"Run metadata not found: {meta_path}")
    
    # Read metadata
    meta = parse_meta_file(meta_path)
    video_path = meta.get("video_path", "")
    prompt = meta.get("prompt", "")
    fps = float(meta.get("fps", "0"))
    n_frames_total = int(meta.get("frames", "0"))
    n_ids = int(meta.get("ids", "0"))
    
    # Verify video still exists
    if not os.path.exists(video_path):
        raise HTTPException(400, f"Original video not found: {video_path}")
    
    log.info(f"/resume done run_id={run_id}, video={video_path}, prompt={prompt}, fps={fps}, frames={n_frames_total}, ids={n_ids}")
    return {"run_id": run_id, "fps": fps, "n_frames_total": n_frames_total, "n_ids": n_ids, "video_path": video_path, "prompt": prompt}


@app.post("/save")
def save(run_id: str):
    """Create a backup/snapshot of the current run by copying it to a new run_id."""
    log.info(f"/save run_id={run_id}")
    
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"Run not found: {run_id}")
    
    # Create new backup run_id
    backup_run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    backup_run_dir = RUNS_ROOT / backup_run_id
    
    log.info(f"Creating backup: {run_id} -> {backup_run_id}")
    
    # Copy entire run directory (pure copy, no modifications)
    # Use dirs_exist_ok=False to ensure we don't overwrite anything
    shutil.copytree(run_dir, backup_run_dir, dirs_exist_ok=False)
    log.info(f"Backup created: {backup_run_dir}")
    
    # Store backup info in a separate file (don't modify meta.txt)
    backup_info_path = backup_run_dir / "backup_info.txt"
    backup_info_path.write_text(
        f"backup_of={run_id}\n"
        f"backup_created={datetime.now().isoformat()}\n"
    )
    log.info(f"Backup info saved to: {backup_info_path}")
    
    log.info(f"/save done: backup_run_id={backup_run_id}")
    return {"backup_run_id": backup_run_id, "original_run_id": run_id}


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
            
            # Check if we need SAM reset for this chunk (before processing)
            # If we have a seed from previous chunk, use it; otherwise check for SAM reset
            seed_ann_path = prev_chunk_seed_ann_path
            # Reset every auto_reset_interval frames from the initial seed_idx
            # So if seed_idx=9 and interval=10, reset at 9, 19, 29, etc.
            should_reset = (current_seed > seed_idx) and ((current_seed - seed_idx) % auto_reset_interval == 0)
            
            if should_reset:
                import numpy as np
                from PIL import Image
                
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
                        golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
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
                if seed_ann_path and seed_ann_path.exists():
                    shutil.copy2(seed_ann_path, chunk_ann_dir / "00000.png")
                else:
                    golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
                    golden_seed = golden_ann_dir / f"{current_seed:05d}.png"
                    if golden_seed.exists():
                        shutil.copy2(golden_seed, chunk_ann_dir / "00000.png")
                
                all_chunk_roots.append(chunk_root)
                log.info(f"✅ Chunk {current_seed} (single frame, no XMem)")
                
                # For single-frame chunk, the seed for next chunk is this frame's annotation
                single_frame_ann = chunk_ann_dir / "00000.png"
                if single_frame_ann.exists():
                    prev_chunk_seed_ann_path = single_frame_ann
                    log.info(f"Saved seed annotation for next chunk: {prev_chunk_seed_ann_path} (frame {current_seed})")
                else:
                    log.warning(f"Single-frame chunk annotation not found: {single_frame_ann}")
                
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
            
            # Run XMem on this chunk
            xmem_output = run_dir / "xmem_outputs" / f"{current_seed:05d}_{chunk_end:05d}"
            ensure_dir(xmem_output.parent)
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
        seed_ann_path = None
        chunk_ds = make_chunk_dataset(run_dir, seed_idx, end_idx, seed_ann_path=seed_ann_path)

        # Run XMem on this chunk dataset
        xmem_output = run_dir / "xmem_outputs" / f"{seed_idx:05d}_{end_idx:05d}"
        ensure_dir(xmem_output.parent)
        logs = run_xmem(chunk_ds, xmem_output)
        masks = find_xmem_pngs(xmem_output)

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
    )

    tracked_path = run_dir / "tracked.mp4"
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
    return {
        "run_id": run_id,
        "seed_idx": seed_idx,
        "end_idx": end_idx,
        "n_frames_rendered": used,
        "chunk_dir": str(chunk_root),
        "log_tail": logs[-30:],
    }

def _probe_duration(path: Path) -> float | None:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return None
    try:
        return float(p.stdout.strip())
    except:
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

    log.info(f"[COMMIT] Starting commit: chunk_root={chunk_root}, seed_idx={seed_idx}, end_idx={end_idx}")
    
    if not chunk_ann_dir.exists():
        raise HTTPException(500, f"Chunk annotations missing: {chunk_ann_dir}")

    golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
    ensure_dir(golden_ann_dir)
    
    # Find all chunks that need to be committed.
    # IMPORTANT: don't assume golden is contiguous (we've observed holes like missing 1..50 while having 51..99).
    # So we scan chunks and include any chunk that can "fill" at least one missing golden frame.
    processed, pct, max_golden_idx = golden_progress(run_dir, n_total)
    last_committed_frame = max_golden_idx if max_golden_idx is not None else -1
    
    chunks_dir = run_dir / "chunks"
    all_chunks_to_commit = []
    if chunks_dir.exists():
        log.info(f"[COMMIT] Scanning chunks directory: {chunks_dir}")
        all_chunk_folders = sorted([f for f in chunks_dir.iterdir() if f.is_dir()])
        log.info(f"[COMMIT] Found {len(all_chunk_folders)} chunk folders: {[f.name for f in all_chunk_folders]}")
        
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
        log.info(f"[COMMIT] Commit end limit for chunk scan: {commit_end_limit} (last_chunk end_idx={end_idx}, max_chunk_end_on_disk={max_chunk_end_on_disk})")

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
                    log.info(
                        f"[COMMIT] Chunk {name}: seed={chunk_seed}, end={chunk_end}, "
                        f"last_committed={last_committed_frame}, include={should_include} "
                        f"(within_limit={within_limit}, has_missing={has_missing})"
                    )
                    if should_include:
                        all_chunks_to_commit.append((chunk_seed, chunk_end, chunk_folder))
            except (ValueError, IndexError) as e:
                log.warning(f"[COMMIT] Failed to parse chunk folder {chunk_folder.name}: {e}")
                continue
    
    # If we found chunks, commit them all; otherwise fall back to the last chunk
    if len(all_chunks_to_commit) > 0:
        log.info(f"[COMMIT] Found {len(all_chunks_to_commit)} chunks to commit: {[(s, e) for s, e, _ in all_chunks_to_commit]}")
    else:
        # Fall back to single chunk commit (original behavior) - but log a warning
        log.warning(f"[COMMIT] No chunks found in chunks directory, falling back to last_chunk.txt: {seed_idx}..{end_idx}")
        all_chunks_to_commit = [(seed_idx, end_idx, chunk_root)]
        log.info(f"[COMMIT] Committing single chunk: {seed_idx}..{end_idx}")

    # Commit NEW frames only: seed+1..end
    # Also copy JPEG frames to golden/JPEGImages/video1/
    golden_jpeg_dir = run_dir / "golden" / "JPEGImages" / VIDEO_NAME
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
        
        log.info(f"[COMMIT] Processing chunk {chunk_folder.name}: seed={chunk_seed}, end={chunk_end}")
        
        # Check for corrected frames in this chunk's range
        chunk_last_corrected = None
        for orig_idx in range(chunk_seed + 1, chunk_end + 1):
            golden_mask = golden_ann_dir / f"{orig_idx:05d}.png"
            if golden_mask.exists():
                chunk_rel = orig_idx - chunk_seed
                chunk_mask_path = chunk_ann_dir_this / f"{chunk_rel:05d}.png"
                if chunk_mask_path.exists():
                    from PIL import Image
                    import numpy as np
                    golden_mask_data = np.array(Image.open(golden_mask))
                    chunk_mask_data = np.array(Image.open(chunk_mask_path))
                    if not np.array_equal(golden_mask_data, chunk_mask_data):
                        chunk_last_corrected = orig_idx
                        log.info(f"[COMMIT] Frame {orig_idx} is corrected (golden mask differs from chunk mask)")
        
        # Determine actual end for this chunk (respect corrected frames)
        chunk_commit_end = chunk_last_corrected if chunk_last_corrected is not None else chunk_end
        if chunk_last_corrected is not None:
            log.info(f"[COMMIT] Chunk {chunk_folder.name}: found corrected frame at {chunk_last_corrected}, will only commit up to this frame")
        
        # First, check if the seed frame exists in golden - if not, copy it
        # (The seed frame is at relative index 0 in the chunk)
        seed_mask_src = chunk_ann_dir_this / "00000.png"
        seed_mask_dst = golden_ann_dir / f"{chunk_seed:05d}.png"
        if seed_mask_src.exists() and not seed_mask_dst.exists():
            log.info(f"[COMMIT] Seed frame {chunk_seed} not in golden, copying it")
            shutil.copy2(seed_mask_src, seed_mask_dst)
            # Also copy JPEG frame
            seed_jpeg_src = src_jpeg / f"{chunk_seed:05d}.jpg"
            if seed_jpeg_src.exists():
                seed_jpeg_dst = golden_jpeg_dir / f"{chunk_seed:05d}.jpg"
                shutil.copy2(seed_jpeg_src, seed_jpeg_dst)
                log.info(f"[COMMIT]   Copied seed frame {chunk_seed} mask and JPEG to golden")
            committed += 1
        elif seed_mask_dst.exists():
            log.info(f"[COMMIT] Seed frame {chunk_seed} already exists in golden, skipping")
        
        # Commit frames from this chunk: seed+1 to commit_end (inclusive)
        log.info(f"[COMMIT] Committing chunk {chunk_folder.name}: frames {chunk_seed + 1} to {chunk_commit_end} (inclusive)")
        
        for orig_idx in range(chunk_seed + 1, chunk_commit_end + 1):
            rel = orig_idx - chunk_seed  # in chunk dataset, seed=0, next frame=1, ...
            src = chunk_ann_dir_this / f"{rel:05d}.png"
            
            log.info(f"[COMMIT] Processing frame {orig_idx} (relative {rel} in chunk)")
            log.info(f"[COMMIT]   Source mask: {src} (exists: {src.exists()})")
            
            if not src.exists():
                # If the file doesn't exist, it means we've reached the end of the chunk
                # This can happen if end_idx is one too high, or if the chunk is incomplete
                log.warning(f"[COMMIT] Missing chunk mask for frame {orig_idx} (relative {rel}) - reached end of chunk, stopping")
                break

            dst = golden_ann_dir / f"{orig_idx:05d}.png"
            log.info(f"[COMMIT]   Destination: {dst} (exists: {dst.exists()})")
            
            # Check if this frame already has a corrected mask in golden
            if dst.exists():
                # Load both masks to compare
                from PIL import Image
                import numpy as np
                existing_mask = np.array(Image.open(dst))
                chunk_mask = np.array(Image.open(src))
                
                existing_max_id = int(existing_mask.max())
                chunk_max_id = int(chunk_mask.max())
                existing_ids = sorted(list(set(existing_mask.flatten())))
                existing_ids = [id for id in existing_ids if id > 0]
                chunk_ids = sorted(list(set(chunk_mask.flatten())))
                chunk_ids = [id for id in chunk_ids if id > 0]
                
                # Check if masks are different (not just same IDs)
                masks_different = not np.array_equal(existing_mask, chunk_mask)
                
                log.info(f"[COMMIT]   Frame {orig_idx} already exists in golden!")
                log.info(f"[COMMIT]     Existing mask: max_id={existing_max_id}, IDs={existing_ids}")
                log.info(f"[COMMIT]     Chunk mask: max_id={chunk_max_id}, IDs={chunk_ids}")
                log.info(f"[COMMIT]     Masks are different: {masks_different}")
                
                if masks_different:
                    log.warning(f"[COMMIT]   ⚠️  Frame {orig_idx} has corrected mask in golden, but chunk mask is different!")
                    log.warning(f"[COMMIT]   ⚠️  SKIPPING overwrite - preserving corrected mask in golden")
                    skipped_corrected += 1
                    # Don't overwrite - keep the corrected mask
                else:
                    log.info(f"[COMMIT]   Masks are identical, overwriting is safe")
                    shutil.copy2(src, dst)
                    committed += 1
            else:
                # Frame doesn't exist in golden, safe to copy
                log.info(f"[COMMIT]   Copying new frame {orig_idx} to golden")
                shutil.copy2(src, dst)
                
                # Log mask info
                from PIL import Image
                import numpy as np
                mask = np.array(Image.open(dst))
                max_id = int(mask.max())
                unique_ids = sorted(list(set(mask.flatten())))
                unique_ids = [id for id in unique_ids if id > 0]
                log.info(f"[COMMIT]   Copied mask: max_id={max_id}, IDs={unique_ids}")
                committed += 1
            
            # Also copy JPEG frame (always, even if mask was skipped)
            src_jpeg_frame = src_jpeg / f"{orig_idx:05d}.jpg"
            if src_jpeg_frame.exists():
                dst_jpeg_frame = golden_jpeg_dir / f"{orig_idx:05d}.jpg"
                shutil.copy2(src_jpeg_frame, dst_jpeg_frame)
                log.info(f"[COMMIT]   Copied JPEG frame: {dst_jpeg_frame}")

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
    log.info("=" * 60)
    log.info("[COMMIT] UPDATING GOLDEN PREVIEW VIDEO")
    log.info("=" * 60)
    try:
        golden_preview = run_dir / "golden" / "golden_preview.mp4"
        chunk_new = chunk_root / "chunk_new.mp4"
        tracked_path = run_dir / "tracked.mp4"

        log.info(f"[COMMIT] golden_preview path: {golden_preview} (exists: {golden_preview.exists()})")
        log.info(f"[COMMIT] chunk_new path: {chunk_new} (exists: {chunk_new.exists()})")
        log.info(f"[COMMIT] tracked.mp4 path: {tracked_path} (exists: {tracked_path.exists()})")
        
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
                    from PIL import Image
                    import numpy as np
                    golden_mask_data = np.array(Image.open(golden_mask))
                    chunk_mask_data = np.array(Image.open(chunk_mask_path))
                    if not np.array_equal(golden_mask_data, chunk_mask_data):
                        last_corrected_frame = orig_idx  # Keep updating to find the LAST one
                        log.info(f"[COMMIT] Found corrected frame: {last_corrected_frame}")
        
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
                log.info(f"[COMMIT] ✅ Golden preview updated: tracked frames {seed_idx+1}..{last_corrected_frame-1} + corrected frame {last_corrected_frame}")
                log.info(f"[COMMIT] ⚠️  Frames {last_corrected_frame+1}..{int(chunk_kv.get('end_idx', end_idx))} were discarded (will be re-tracked from frame {last_corrected_frame})")
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
                        log.info(f"[COMMIT] ✅ Successfully appended chunk_new to golden_preview.mp4")
                    else:
                        log.error("[COMMIT] ❌ Concat failed (non-fatal).")
                else:
                    log.info("[COMMIT] golden_preview.mp4 does not exist, initializing from chunk_new.mp4...")
                    golden_preview.parent.mkdir(parents=True, exist_ok=True)
                    golden_preview.write_bytes(chunk_new.read_bytes())
                    init_size = golden_preview.stat().st_size
                    init_dur = _probe_duration(golden_preview)
                    log.info(f"[COMMIT] ✅ Initialized golden_preview.mp4 from chunk_new.mp4: size={init_size} bytes, duration={init_dur}s")
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
                    log.info(f"[COMMIT] ✅ Golden preview updated from rendered segment")
    except Exception as e:
        log.error(f"[COMMIT] ❌ Golden preview update failed (non-fatal): {e}", exc_info=True)
    
    log.info("=" * 60)


    processed, pct, max_idx = golden_progress(run_dir, n_total)
    return {
        "run_id": run_id,
        "committed_new_frames": committed,
        "golden_processed": processed,
        "golden_percent": pct,
        "golden_max_idx": max_idx,
        "seed_idx": seed_idx,
        "end_idx": end_idx,
    }


@app.get("/get_frame_from_time/{run_id}")
def get_frame_from_time(run_id: str, video_time: float):
    """
    Get frame number from video playback time for tracked video.
    Returns relative frame number (relative to chunk start).
    """
    import cv2
    
    log.info(f"/get_frame_from_time run_id={run_id} video_time={video_time}")
    
    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found")
    
    meta = parse_meta_file(meta_path)
    source_fps = float(meta.get("fps", 30.0))
    
    # Get tracked video path
    tracked_video_path = run_dir / "tracked.mp4"
    if not tracked_video_path.exists():
        raise HTTPException(404, "Tracked video not found. Run /track first.")
    
    # Get actual video properties
    cap = cv2.VideoCapture(str(tracked_video_path))
    if not cap.isOpened():
        raise HTTPException(500, "Could not open tracked video")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    log.info(f"[DEBUG] Video properties: fps={video_fps}, frame_count={video_frame_count}, source_fps={source_fps}")
    log.info(f"[DEBUG] Video time: {video_time}s")
    
    # Calculate relative frame from video time
    # Use video's actual FPS if available, otherwise fallback to source FPS
    if video_fps > 0:
        relative_frame = int(video_time * video_fps)
    else:
        relative_frame = int(video_time * source_fps)
    
    # Clamp to valid range
    relative_frame = max(0, min(relative_frame, video_frame_count - 1))
    
    # Get last golden frame to calculate absolute frame
    processed, pct, max_idx = golden_progress(run_dir, int(meta["frames"]))
    absolute_frame = max_idx + relative_frame if max_idx is not None else relative_frame
    
    log.info(f"[DEBUG] Calculated: relative_frame={relative_frame}, absolute_frame={absolute_frame}, max_idx={max_idx}")
    
    return {
        "relative_frame": relative_frame,
        "absolute_frame": absolute_frame,
        "video_time": video_time,
        "video_fps": float(video_fps) if video_fps > 0 else source_fps,
        "video_frame_count": video_frame_count,
    }


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
    }


@app.get("/frame0/{run_id}")
def frame0(run_id: str):
    import numpy as np
    import cv2
    from PIL import Image
    from fastapi.responses import Response

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

        def color_for(i: int):
            rng = np.random.RandomState(i)
            return tuple(int(x) for x in rng.randint(50, 255, size=3))

        for cid in range(1, max_id + 1):
            m = (labels == cid)
            if not m.any():
                continue

            col = color_for(cid)
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
    import numpy as np
    import cv2
    from PIL import Image
    from fastapi.responses import Response
    
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
    
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise HTTPException(500, f"Could not read frame {absolute_frame}")
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
        
        def color_for(i: int):
            rng = np.random.RandomState(i)
            return tuple(int(x) for x in rng.randint(50, 255, size=3))
        
        rendered_count = 0
        for cid in range(1, max_id + 1):
            m = (labels == cid)
            if not m.any():
                continue
            rendered_count += 1
            mask_pixels = int(m.sum())
            log.info(f"[TRACKED_FRAME] Rendering mask for ID {cid} ({mask_pixels} pixels)")
            
            col = color_for(cid)
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
    import numpy as np
    import cv2
    from PIL import Image
    from fastapi.responses import Response

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
    
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise HTTPException(500, f"Could not read frame {frame_idx}")
    
    # Try golden annotation first
    golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
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
        
        def color_for(i: int):
            rng = np.random.RandomState(i)
            return tuple(int(x) for x in rng.randint(50, 255, size=3))
        
        for cid in range(1, max_id + 1):
            m = (labels == cid)
            if not m.any():
                continue
            
            col = color_for(cid)
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
    from PIL import Image
    
    log.info(f"[FIND_MASK] Searching for tracked mask for frame {frame_idx}")
    
    golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
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
    import numpy as np
    import cv2
    from PIL import Image
    from fastapi.responses import JSONResponse
    
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
    
    committed_count = 0
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
        
        log.info(f"[DEBUG] Chunk info: seed_idx={seed_idx}, end_idx={end_idx}")
        log.info(f"[DEBUG] Will commit frames: {seed_idx+1}..{min(commit_up_to, end_idx)}")
        
        golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
        ensure_dir(golden_ann_dir)
        
        # Also copy JPEG frames
        golden_jpeg_dir = run_dir / "golden" / "JPEGImages" / VIDEO_NAME
        ensure_dir(golden_jpeg_dir)
        src_root = run_dir / "xmem_generic"
        src_jpeg = src_root / "JPEGImages" / VIDEO_NAME
        
        # Commit frames seed+1..min(commit_up_to, end_idx)
        # Don't commit beyond what's in the chunk
        commit_end = min(commit_up_to, end_idx)
        committed = 0
        for orig_idx in range(seed_idx + 1, commit_end + 1):
            rel = orig_idx - seed_idx
            src = chunk_ann_dir / f"{rel:05d}.png"
            if not src.exists():
                log.warning(f"Missing chunk mask for frame {orig_idx} (relative {rel}), skipping")
                continue
            dst = golden_ann_dir / f"{orig_idx:05d}.png"
            shutil.copy2(src, dst)
            
            # Also copy JPEG frame
            src_jpeg_frame = src_jpeg / f"{orig_idx:05d}.jpg"
            if src_jpeg_frame.exists():
                dst_jpeg_frame = golden_jpeg_dir / f"{orig_idx:05d}.jpg"
                shutil.copy2(src_jpeg_frame, dst_jpeg_frame)
            
            committed += 1
        
        committed_count = committed
        log.info(f"✅ Committed {committed} frames to golden: {seed_idx+1}..{commit_end}")
        
        if commit_up_to > end_idx:
            log.warning(f"⚠️  Requested commit up to frame {commit_up_to}, but chunk only goes to {end_idx}. Committed {seed_idx+1}..{end_idx}")
        
        # Update golden preview video for committed frames
        # Extract from tracked.mp4 instead of rendering from golden (tracked video already has correct masks)
        try:
            golden_preview = run_dir / "golden" / "golden_preview.mp4"
            tracked_path = run_dir / "tracked.mp4"
            
            if commit_end >= seed_idx + 1 and committed_count > 0 and tracked_path.exists():
                log.info(f"[PREPARE_CORRECTION] Extracting tracked segment {seed_idx+1}..{commit_end} from tracked.mp4")
                # Extract frames from tracked.mp4: frames seed+1..commit_end
                # tracked.mp4 contains frames seed..end_idx (seed is frame 0 in the video)
                # We want frames 1..(commit_end-seed_idx) from tracked.mp4
                tracked_seg_start = 1  # Skip seed frame (frame 0 in video)
                tracked_seg_end = commit_end - seed_idx  # Last frame to include
                seg_path = run_dir / "golden_segments" / f"tracked_{seed_idx+1}_{commit_end}.mp4"
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
                log.info(f"[PREPARE_CORRECTION] Running: {' '.join(cmd)}")
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                
                if p.returncode == 0 and seg_path.exists():
                    log.info(f"[PREPARE_CORRECTION] ✅ Extracted tracked segment: {seg_path}")
                    # Append to golden preview
                    if golden_preview.exists():
                        _ffmpeg_concat(golden_preview, seg_path, golden_preview, fps)
                    else:
                        golden_preview.write_bytes(seg_path.read_bytes())
                    log.info(f"[PREPARE_CORRECTION] ✅ Updated golden preview video with tracked frames {seed_idx+1}..{commit_end}")
                else:
                    log.error(f"[PREPARE_CORRECTION] Failed to extract tracked segment: {p.stdout[-500:] if p.stdout else 'no output'}")
            elif not tracked_path.exists():
                log.warning(f"[PREPARE_CORRECTION] tracked.mp4 not found, cannot update golden preview video")
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
    new_masks = run_sam3_on_frame(prompt, frame_path)
    log.info(f"[PREPARE_CORRECTION] SAM-3 found {len(new_masks)} masks for frame {frame_idx}")
    
    # Save masks temporarily for refinement
    masks_file = run_dir / f"correction_masks_{frame_idx}.npy"
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
    
    # Save assignments for use during refinement (to preserve IDs)
    assignments_file = run_dir / f"correction_assignments_{frame_idx}.npy"
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
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise HTTPException(500, f"Could not read frame {frame_idx}")
    log.info(f"[PREPARE_CORRECTION] Frame image loaded: shape={frame.shape}")
    
    def color_for(i: int):
        rng = np.random.RandomState(i)
        return tuple(int(x) for x in rng.randint(50, 255, size=3))
    
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
            col = color_for(obj_id)
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
    log.info(f"[PREPARE_CORRECTION] Rendering {len(assignments)} new SAM masks")
    for mask_idx, assigned_id in assignments.items():
        mask = new_masks[mask_idx]
        mask_pixels = int(mask.sum())
        log.info(f"[PREPARE_CORRECTION] Rendering SAM mask {mask_idx} -> ID {assigned_id} ({mask_pixels} pixels)")
        col = color_for(assigned_id)
        overlay = frame.copy()
        overlay[mask] = col
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)  # More prominent overlay for new masks
        
        ys, xs = np.where(mask)
        if len(ys) == 0:
            log.warning(f"[PREPARE_CORRECTION] Mask {mask_idx} has no pixels!")
            continue
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
    log.info(f"[PREPARE_CORRECTION] Preview rendering complete for frame {frame_idx}")
    
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode frame")
    
    # Prepare response
    mask_assignments = [
        {
            "mask_index": mask_idx,
            "auto_assigned_id": assigned_id,
            "is_new": assigned_id > max(existing_ids) if existing_ids else True,
        }
        for mask_idx, assigned_id in sorted(assignments.items())
    ]
    
    import base64
    image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    
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

class PreviewUpdate(BaseModel):
    mapping: Dict[str, int]  # mask_index -> final_id (0 means delete)

@app.post("/refine_mask/{run_id}/{frame_idx}")
def refine_mask(run_id: str, frame_idx: int, refine_request: RefineMaskRequest):
    """
    Refine a mask using point prompts.
    Takes a mask_index and list of points (positive/negative) and uses SAM-3 point prompts to refine the mask.
    """
    import numpy as np
    import cv2
    from PIL import Image
    from fastapi.responses import JSONResponse
    
    log.info(f"/refine_mask run_id={run_id} frame_idx={frame_idx} mask_index={refine_request.mask_index} points={len(refine_request.points)}")
    
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"Run not found: {run_id}")
    
    # Load saved masks from prepare_correction
    masks_file = run_dir / f"correction_masks_{frame_idx}.npy"
    if not masks_file.exists():
        raise HTTPException(400, f"Masks not found. Please run prepare_correction first.")
    
    masks = np.load(masks_file, allow_pickle=True)
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
    
    # Get the mask to refine
    original_mask = masks[refine_request.mask_index]
    
    # Prepare point prompts for SAM-3
    # SAM-3 expects points as numpy array with shape (N, 2) and labels as (N,) where 1=positive, 0=negative
    points_array = np.array([[p.x, p.y] for p in refine_request.points])
    labels_array = np.array([1 if p.is_positive else 0 for p in refine_request.points])
    
    log.info(f"[REFINE_MASK] Refining mask {refine_request.mask_index} with {len(refine_request.points)} points ({sum(labels_array)} positive, {len(labels_array)-sum(labels_array)} negative)")
    
    # Run SAM-3 with mask input + point prompts to refine the specific mask
    # This ensures we only refine the selected mask, not trigger a full detection
    processor = get_model()
    img = Image.open(frame_path).convert("RGB")
    W, H = img.size
    
    state = processor.set_image(img)
    
    # Convert original mask to tensor format for SAM-3
    # Resize mask to match SAM-3's expected input size
    import torch
    mask_tensor = torch.from_numpy(original_mask.astype(np.float32))
    # SAM-3 might expect mask in a specific format - try different approaches
    mask_input = mask_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Workaround: Since SAM-3 doesn't support mask+point refinement directly,
    # we use point prompts alone, then match the result back to the original mask
    # This prevents ID permutation while still allowing refinement
    refined_mask = original_mask  # Default to original mask
    out = None
    
    # First, try to find a method that accepts point prompts
    # Inspect processor to see what methods are available
    processor_methods = [m for m in dir(processor) if not m.startswith('_') and callable(getattr(processor, m))]
    log.info(f"[REFINE_MASK] Available processor methods: {processor_methods}")
    
    try:
        # Try different possible methods for point prompts
        if hasattr(processor, 'set_point_prompt'):
            # Try point prompt method
            out = processor.set_point_prompt(state=state, points=points_array, labels=labels_array)
            log.info("[REFINE_MASK] Used set_point_prompt method")
        elif hasattr(processor, 'predict'):
            # Try standard SAM predict with points (no mask input)
            try:
                out = processor.predict(
                    state=state,
                    point_coords=points_array,
                    point_labels=labels_array,
                    multimask_output=True,  # Get multiple candidates to match
                )
                log.info("[REFINE_MASK] Used predict method with points")
            except TypeError:
                # Try without multimask_output parameter
                out = processor.predict(
                    state=state,
                    point_coords=points_array,
                    point_labels=labels_array,
                )
                log.info("[REFINE_MASK] Used predict method with points (no multimask_output)")
        else:
            log.warning("[REFINE_MASK] No point prompt method found, trying to use text prompt as fallback")
            # Last resort: use text prompt (won't refine, but at least won't crash)
            out = None
    except Exception as e:
        log.error(f"[REFINE_MASK] Error calling point prompt method: {e}", exc_info=True)
        out = None
    
    # Extract masks from output and match to original mask
    if out is not None and "masks" in out and len(out["masks"]) > 0:
        # Get all candidate masks from SAM
        candidate_masks = []
        for m in out["masks"]:
            mask_tensor = m.squeeze().cpu().numpy()
            mask = np.array(Image.fromarray(mask_tensor).resize((W, H), Image.NEAREST)) > MASK_THRESHOLD
            if mask.sum() > 100:  # Filter out tiny masks
                candidate_masks.append(mask)
        
        if candidate_masks:
            # Match each candidate to the original mask using IoU
            best_mask = None
            best_iou = 0.0
            for candidate in candidate_masks:
                iou = compute_iou(candidate, original_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_mask = candidate
            
            if best_mask is not None and best_iou > 0.1:  # Require at least 10% overlap
                refined_mask = best_mask
                log.info(f"[REFINE_MASK] Matched refined mask: IoU={best_iou:.3f}, original size={int(original_mask.sum())}, refined size={int(refined_mask.sum())}")
            else:
                log.warning(f"[REFINE_MASK] Best match IoU too low ({best_iou:.3f}), using original mask")
        else:
            log.warning("[REFINE_MASK] No valid candidate masks from SAM output")
    else:
        if out is None:
            log.warning("[REFINE_MASK] Point prompt method not available, using original mask")
        else:
            log.warning("[REFINE_MASK] No masks in output, using original mask")
    
    # Update the mask in the masks array
    masks[refine_request.mask_index] = refined_mask
    np.save(masks_file, masks)  # Save updated masks
    
    # Load ID assignments from prepare_correction to preserve IDs
    assignments_file = run_dir / f"correction_assignments_{frame_idx}.npy"
    if assignments_file.exists():
        assignments = np.load(assignments_file, allow_pickle=True).item()
        log.info(f"[REFINE_MASK] Loaded ID assignments: {assignments}")
    else:
        # Fallback: create default assignments (mask_idx -> mask_idx + 1)
        assignments = {i: i + 1 for i in range(len(masks))}
        log.warning(f"[REFINE_MASK] Assignments file not found, using default: {assignments}")
    
    # Re-render preview image with refined mask, using preserved ID assignments
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise HTTPException(500, f"Could not read frame {frame_idx}")
    
    def color_for(i: int):
        rng = np.random.RandomState(i)
        return tuple(int(x) for x in rng.randint(50, 255, size=3))
    
    # Render all masks (with refined one) using preserved ID assignments
    for mask_idx, mask in enumerate(masks):
        assigned_id = assignments.get(mask_idx, mask_idx + 1)  # Use preserved ID
        
        if mask_idx == refine_request.mask_index:
            # Highlight the refined mask
            col = (0, 255, 0)  # Green for refined mask
            overlay = frame.copy()
            overlay[mask] = col
            frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        else:
            col = color_for(assigned_id)  # Use assigned ID for color consistency
            overlay = frame.copy()
            overlay[mask] = col
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw mask center with assigned ID (not mask index)
        ys, xs = np.where(mask)
        if len(ys) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(frame, str(assigned_id), (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw point prompts on the image
    for p in refine_request.points:
        color = (0, 255, 0) if p.is_positive else (0, 0, 255)  # Green for positive, red for negative
        cv2.circle(frame, (p.x, p.y), 5, color, -1)
        cv2.circle(frame, (p.x, p.y), 8, (255, 255, 255), 2)
    
    # Encode preview image
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode frame")
    
    import base64
    image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    
    log.info(f"[REFINE_MASK] Mask {refine_request.mask_index} refined, new size: {int(refined_mask.sum())} pixels")
    
    # Get image dimensions for coordinate validation
    img_height, img_width = frame.shape[:2]
    
    return JSONResponse(content={
        "image": f"data:image/jpeg;base64,{image_b64}",
        "mask_index": refine_request.mask_index,
        "refined_mask_size": int(refined_mask.sum()),
        "image_width": int(img_width),
        "image_height": int(img_height),
    })

@app.post("/preview_correction_update/{run_id}/{frame_idx}")
def preview_correction_update(run_id: str, frame_idx: int, preview_update: PreviewUpdate):
    """
    Regenerate preview image with current ID mappings and deletions.
    Used for real-time preview updates as user edits the table.
    """
    import numpy as np
    import cv2
    from PIL import Image
    from fastapi.responses import JSONResponse
    
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
    
    new_masks = run_sam3_on_frame(prompt, frame_path)
    log.info(f"Got {len(new_masks)} masks from SAM-3")
    
    # Load frame (make a copy so we don't modify the original)
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise HTTPException(500, f"Could not read frame {frame_idx}")
    frame = frame.copy()  # Make a copy to avoid modifying original
    
    def color_for(i: int):
        rng = np.random.RandomState(i)
        return tuple(int(x) for x in rng.randint(50, 255, size=3))
    
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
        col = color_for(final_id)
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
    
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode frame")
    
    import base64
    image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    
    return JSONResponse(content={
        "image": f"data:image/jpeg;base64,{image_b64}",
    })

@app.post("/apply_correction/{run_id}/{frame_idx}")
def apply_correction(run_id: str, frame_idx: int, id_mapping: IDMapping):
    """
    Apply user's ID mapping to save corrected frame.
    id_mapping: dict mapping mask_index -> final_id
    """
    import numpy as np
    from PIL import Image
    
    log.info(f"/apply_correction run_id={run_id} frame_idx={frame_idx} mapping={id_mapping}")
    
    run_dir = RUNS_ROOT / run_id
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        raise HTTPException(404, "run_id not found")
    
    meta = parse_meta_file(meta_path)
    prompt = meta.get("prompt", "object")
    
    # Get frame and run SAM again (or we could cache it, but simpler to rerun)
    src_root = run_dir / "xmem_generic"
    jpeg_dir = src_root / "JPEGImages" / VIDEO_NAME
    frame_path = jpeg_dir / f"{frame_idx:05d}.jpg"
    
    if not frame_path.exists():
        raise HTTPException(404, f"Frame {frame_idx} not found")
    
    new_masks = run_sam3_on_frame(prompt, frame_path)
    
    # Apply user's ID mapping
    H, W = new_masks[0].shape
    label_map = np.zeros((H, W), dtype=np.uint8)
    
    for mask_idx_str, final_id in id_mapping.mapping.items():
        mask_idx = int(mask_idx_str)
        if mask_idx >= len(new_masks):
            raise HTTPException(400, f"Invalid mask_index {mask_idx}")
        final_id = int(final_id)
        if final_id <= 0:  # Skip deleted masks (0 or negative)
            continue
        label_map[new_masks[mask_idx]] = final_id
    
    # NOTE: We do NOT renumber IDs during corrections. The user (or auto-assignment) has
    # explicitly chosen which IDs to use, and these IDs are meant to match existing IDs
    # from previous frames. Renumbering would break this continuity.
    # Gaps in IDs (e.g., [2,3,4,...,18] instead of [1,2,3,...,17]) are intentional and
    # should be preserved.
    
    # Save to golden
    golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
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
    Image.fromarray(label_map).save(corrected_ann_path)
    log.info(f"Saved corrected annotation to {corrected_ann_path}")
    
    # Also copy JPEG frame
    golden_jpeg_dir = run_dir / "golden" / "JPEGImages" / VIDEO_NAME
    ensure_dir(golden_jpeg_dir)
    src_jpeg_frame = jpeg_dir / f"{frame_idx:05d}.jpg"
    if src_jpeg_frame.exists():
        dst_jpeg_frame = golden_jpeg_dir / f"{frame_idx:05d}.jpg"
        shutil.copy2(src_jpeg_frame, dst_jpeg_frame)
    
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
    import numpy as np
    from PIL import Image
    
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
        
        golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
        ensure_dir(golden_ann_dir)
        
        # Also copy JPEG frames
        golden_jpeg_dir = run_dir / "golden" / "JPEGImages" / VIDEO_NAME
        ensure_dir(golden_jpeg_dir)
        src_jpeg = src_root / "JPEGImages" / VIDEO_NAME
        
        # Commit frames seed+1..min(commit_up_to, end_idx)
        # Don't commit beyond what's in the chunk
        commit_end = min(commit_up_to, end_idx)
        log.info(f"[DEBUG] Committing frames {seed_idx+1}..{commit_end} (chunk has {seed_idx}..{end_idx}, requested up to {commit_up_to})")
        
        committed = 0
        for orig_idx in range(seed_idx + 1, commit_end + 1):
            rel = orig_idx - seed_idx
            src = chunk_ann_dir / f"{rel:05d}.png"
            if not src.exists():
                log.warning(f"Missing chunk mask for frame {orig_idx} (relative {rel} in chunk), skipping")
                continue
            dst = golden_ann_dir / f"{orig_idx:05d}.png"
            shutil.copy2(src, dst)
            
            # Also copy JPEG frame
            src_jpeg_frame = src_jpeg / f"{orig_idx:05d}.jpg"
            if src_jpeg_frame.exists():
                dst_jpeg_frame = golden_jpeg_dir / f"{orig_idx:05d}.jpg"
                shutil.copy2(src_jpeg_frame, dst_jpeg_frame)
            
            committed += 1
        
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
    golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
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
    golden_ann_dir = run_dir / "golden" / "Annotations" / VIDEO_NAME
    ensure_dir(golden_ann_dir)
    corrected_ann_path = golden_ann_dir / f"{wrong_frame_idx:05d}.png"
    Image.fromarray(label_map).save(corrected_ann_path)
    
    # Also copy JPEG frame to golden/JPEGImages/video1/
    golden_jpeg_dir = run_dir / "golden" / "JPEGImages" / VIDEO_NAME
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
            log.info(f"Updated golden preview video with corrected frame {wrong_frame_idx}")
    except Exception as e:
        log.warning(f"Failed to update golden preview video with corrected frame (non-fatal): {e}")
    
    # Step 5: Return frame image with overlays (reuse get_frame logic)
    import cv2
    from fastapi.responses import Response
    
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise HTTPException(500, f"Could not read frame {wrong_frame_idx}")
    
    def color_for(i: int):
        rng = np.random.RandomState(i)
        return tuple(int(x) for x in rng.randint(50, 255, size=3))
    
    for cid in range(1, int(label_map.max()) + 1):
        m = (label_map == cid)
        if not m.any():
            continue
        
        col = color_for(cid)
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
    return FileResponse(path, media_type="video/mp4")


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
