"""XMem tracking: chunk datasets, running XMem and locating its output masks."""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from vos.config import VIDEO_NAME, XMEM_MODEL, XMEM_REPO, config_dir, log
from vos.storage import ensure_clean_dir, get_golden_ann_dir


def golden_progress(run_dir: Path, n_total: int):
    golden_ann_dir = get_golden_ann_dir(run_dir)
    if not golden_ann_dir.exists():
        log.info(f"[GOLDEN_PROGRESS] Golden annotations dir does not exist: {golden_ann_dir}")
        return 0, 0.0, None

    pngs = sorted(golden_ann_dir.glob("*.png"))
    def idx_from_name(p: Path) -> int:
        return int(p.stem)

    frame_indices = [idx_from_name(p) for p in pngs]
    if not frame_indices:
        return 0, 0.0, None
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
    if seed_ann_path:
        seed_ann = seed_ann_path  # Will fail on copy if missing
        log.info(f"Using auto-reset seed annotation: {seed_ann}")
    else:
        golden_ann_dir = get_golden_ann_dir(run_dir)
        seed_ann = golden_ann_dir / f"{seed_idx:05d}.png"  # Will fail on copy if missing

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
        dst = dst_jpeg / f"{n:05d}.jpg"
        shutil.copy2(src, dst)  # Will raise FileNotFoundError if src missing
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
        "--model", str(XMEM_MODEL),
        "--output", os.path.abspath(xmem_output),
        "--dataset", "G",
        "--generic_path", os.path.abspath(dataset_root),
    ]

    # Set TORCH_HOME and TMPDIR to the configured dirs (avoid /tmp being full)
    # PyTorch will cache pretrained models (like ResNet50) here
    # TMPDIR is used for temporary extraction during download
    torch_cache_dir = config_dir("torch_cache_dir")
    tmp_dir = config_dir("tmp_dir")
    env = os.environ.copy()
    env["TORCH_HOME"] = str(torch_cache_dir)
    env["TMPDIR"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)  # Some tools use TMP instead of TMPDIR
    log.info(f"[XMem] Using TORCH_HOME={torch_cache_dir} for model cache")
    log.info(f"[XMem] Using TMPDIR={tmp_dir} for temporary files")

    proc = subprocess.Popen(
        cmd,
        cwd=str(XMEM_REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
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


def find_tracked_mask_for_frame(run_dir: Path, frame_idx: int) -> Tuple[Optional[Path], str]:
    """
    Find the tracked mask annotation for a given frame.
    Searches in golden first, then in all chunks.
    Returns (annotation_path, source_description) or (None, "not found") if not found.
    """
    
    log.info(f"[FIND_MASK] Searching for tracked mask for frame {frame_idx}")
    
    golden_ann_dir = get_golden_ann_dir(run_dir)
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
