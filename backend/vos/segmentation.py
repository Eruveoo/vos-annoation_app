"""SAM-3 models (lazy loaded), instance extraction and ID matching."""
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2
from PIL import Image
import torch
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
import sam3.model_builder
import pkg_resources

from vos.config import MASK_THRESHOLD, log
from vos.overlay import masks_to_label_map


# Fix sam3.__file__ if it's None (namespace package issue with editable installs)
import sam3
if not hasattr(sam3, '__file__') or sam3.__file__ is None:
    if hasattr(sam3.model_builder, '__file__') and sam3.model_builder.__file__:
        sam3.__file__ = str(Path(sam3.model_builder.__file__).parent.parent / "__init__.py")


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
