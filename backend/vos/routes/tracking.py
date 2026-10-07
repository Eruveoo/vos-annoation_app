"""Endpoints: XMem tracking, committing chunks and progress."""
import shutil
import subprocess
import time
import threading
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from PIL import Image
from fastapi import HTTPException, Query, APIRouter

from vos.behavior import load_behavior_data, mark_behavior_preview_out_of_sync
from vos.config import RUNS_ROOT, VIDEO_NAME, log
from vos.overlay import masks_to_label_map
from vos.segmentation import auto_assign_ids, run_sam3_on_frame
from vos.state import track_progress
from vos.storage import (
    copy_files_parallel,
    ensure_clean_dir,
    ensure_dir,
    get_annotation_mode,
    get_golden_ann_dir,
    get_golden_jpeg_dir,
    parse_meta_file,
)
from vos.tracking import find_xmem_pngs, golden_progress, make_chunk_dataset, run_xmem
from vos.video import (
    _ffmpeg_concat,
    _ffmpeg_drop_seed_frame,
    _ffmpeg_reencode_video,
    _probe_duration,
    _render_segment_from_golden,
    render_video,
)


router = APIRouter()


    
@router.post("/track")
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



@router.post("/commit")
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
                else:
                    log.info("[COMMIT] golden_preview.mp4 does not exist, initializing from chunk_new.mp4...")
                    golden_preview.parent.mkdir(parents=True, exist_ok=True)
                    golden_preview.write_bytes(chunk_new.read_bytes())
                    init_size = golden_preview.stat().st_size
                    init_dur = _probe_duration(golden_preview)
                    log.debug(f"[COMMIT] Initialized golden_preview.mp4 from chunk_new.mp4: size={init_size} bytes, duration={init_dur}s")
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


@router.get("/get_frame_from_time/{run_id}")
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


@router.get("/track_progress/{run_id}")
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


@router.get("/progress/{run_id}")
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
