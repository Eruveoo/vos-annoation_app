"""Endpoints: preparing, previewing and applying mask corrections."""
import shutil
import subprocess
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse, Response

from vos.behavior import load_behavior_dimension, register_late_behavior_cows
from vos.config import RUNS_ROOT, VIDEO_NAME, log
from vos.overlay import encode_frame_to_base64, get_color_for_id
from vos.schemas import IDMapping, PreviewUpdate
from vos.segmentation import auto_assign_ids, run_sam3_on_frame
from vos.storage import (
    copy_files_parallel,
    ensure_dir,
    get_annotation_mode,
    get_correction_assignments_file,
    get_correction_masks_file,
    get_golden_ann_dir,
    get_golden_jpeg_dir,
    load_frame_safely,
    load_masks_safely,
    parse_meta_file,
)
from vos.tracking import find_tracked_mask_for_frame, golden_progress
from vos.video import _ffmpeg_concat, _ffmpeg_reencode_video, _render_segment_from_golden


router = APIRouter()


@router.post("/prepare_correction/{run_id}/{frame_idx}")
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

@router.post("/preview_correction_update/{run_id}/{frame_idx}")
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

@router.post("/apply_correction/{run_id}/{frame_idx}")
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
        else:
            log.error(f"[APPLY_CORRECTION] Segment was not created: {seg_path}")
    except Exception as e:
        log.error(f"[APPLY_CORRECTION] Failed to update golden preview video: {e}", exc_info=True)
    
    return {"status": "success", "frame_idx": frame_idx, "max_id": new_max_id}


@router.post("/correct_frame")
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
