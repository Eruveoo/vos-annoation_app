"""Endpoints: adding and refining masks with SAM-3 during correction."""
import shutil
import tempfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse

from vos.config import RUNS_ROOT, VIDEO_NAME, log
from vos.overlay import encode_frame_to_base64, get_color_for_id
from vos.schemas import AddMaskRequest, RefineMaskRequest
from vos.segmentation import (
    compute_iou,
    extract_instances_from_formatted,
    get_video_predictor,
    safe_mask_hw,
)
from vos.storage import (
    get_correction_assignments_file,
    get_correction_masks_file,
    load_assignments_or_default,
    load_frame_safely,
    load_masks_safely,
    parse_meta_file,
)


router = APIRouter()


@router.post("/add_mask/{run_id}/{frame_idx}")
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

@router.post("/refine_mask/{run_id}/{frame_idx}")
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
