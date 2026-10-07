"""Run directory layout and loading/saving of frames, masks and meta files."""
import shutil
from pathlib import Path
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from fastapi import HTTPException

from vos.config import VIDEO_NAME, log


def ensure_clean_dir(path: Path):
    if path.exists():
        log.info(f"Removing directory: {path}")
        shutil.rmtree(path)
    log.info(f"Creating directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_files_parallel(src_dst_pairs: list, max_workers: int = 8):
    """
    Copy multiple files in parallel using ThreadPoolExecutor.
    
    Args:
        src_dst_pairs: List of (src_path, dst_path) tuples
        max_workers: Maximum number of parallel copy operations
    
    Returns:
        Number of successfully copied files
    """
    def copy_one(src_dst):
        src, dst = src_dst
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            log.warning(f"Failed to copy {src} to {dst}: {e}")
            return False
    
    if not src_dst_pairs:
        return 0
    
    copied = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(copy_one, pair): pair for pair in src_dst_pairs}
        for future in as_completed(futures):
            if future.result():
                copied += 1
    
    return copied


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


def get_annotation_mode(run_dir: Path) -> str:
    meta = parse_meta_file(run_dir / "meta.txt")
    mode = (meta.get("annotation_mode") or "standard").strip().lower()
    return mode if mode in ("standard", "behavior") else "standard"


def update_meta_key(meta_path: Path, key: str, value: str) -> None:
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    meta_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")


def load_frame_safely(frame_path: Path, frame_idx: Optional[int] = None) -> np.ndarray:
    """
    Load a frame from disk safely.
    
    Args:
        frame_path: Path to frame image
        frame_idx: Optional frame index for error messages
    
    Returns:
        Frame as numpy array (BGR format)
    
    Raises:
        HTTPException: If frame cannot be read
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        idx_msg = f" {frame_idx}" if frame_idx is not None else ""
        raise HTTPException(500, f"Could not read frame{idx_msg} from {frame_path}")
    return frame


def load_masks_safely(masks_file: Path) -> List[np.ndarray]:
    """
    Load masks from .npy file and ensure they are boolean numpy arrays.
    
    Args:
        masks_file: Path to .npy file containing masks
    
    Returns:
        List of boolean numpy arrays
    """
    masks_raw = np.load(masks_file, allow_pickle=True)
    masks = []
    for i, m in enumerate(masks_raw):
        if not isinstance(m, np.ndarray):
            log.warning(f"Loaded mask {i} is not a numpy array (type: {type(m)}), converting.")
            m = np.asarray(m)
        if m.dtype != bool:
            log.warning(f"Loaded mask {i} is not boolean (dtype: {m.dtype}), converting.")
            m = (m > 0.5).astype(bool)
        masks.append(m)
    return masks


def load_assignments_or_default(assignments_file: Path, n_masks: int) -> Dict[int, int]:
    """
    Load ID assignments from file or create default mapping.
    
    Args:
        assignments_file: Path to .npy file containing assignments dict
        n_masks: Number of masks (for default mapping)
    
    Returns:
        Dictionary mapping mask_index -> final_id
    """
    if assignments_file.exists():
        assignments = np.load(assignments_file, allow_pickle=True).item()
        return assignments
    else:
        # Default: mask_idx -> mask_idx + 1
        return {i: i + 1 for i in range(n_masks)}


# Path helper functions
def get_golden_ann_dir(run_dir: Path) -> Path:
    """Get golden annotations directory path."""
    return run_dir / "golden" / "Annotations" / VIDEO_NAME


def get_golden_jpeg_dir(run_dir: Path) -> Path:
    """Get golden JPEG images directory path."""
    return run_dir / "golden" / "JPEGImages" / VIDEO_NAME


def get_jpeg_dir(run_dir: Path) -> Path:
    """Get source JPEG images directory path."""
    return run_dir / "xmem_generic" / "JPEGImages" / VIDEO_NAME


def get_init_masks_file(run_dir: Path) -> Path:
    """Get init masks file path."""
    return run_dir / "init" / "init_masks.npy"


def get_correction_masks_file(run_dir: Path, frame_idx: int) -> Path:
    """Get correction masks file path for a specific frame."""
    return run_dir / "correction_masks" / f"correction_masks_{frame_idx}.npy"


def get_correction_assignments_file(run_dir: Path, frame_idx: int) -> Path:
    """Get correction assignments file path for a specific frame."""
    return run_dir / "correction_masks" / f"correction_assignments_{frame_idx}.npy"
