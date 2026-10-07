"""ffmpeg helpers: frame extraction, re-encoding, rendering and concatenating videos."""
import shutil
import subprocess
import uuid
import time
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from PIL import Image

from vos.behavior import behavior_overlay_lines_at_frame
from vos.config import (
    JPEG_QUALITY,
    LOG_EVERY_FRAMES_EXTRACT,
    LOG_EVERY_FRAMES_RENDER,
    VIDEO_NAME,
    log,
)
from vos.overlay import draw_cow_overlay_with_behavior, random_color
from vos.storage import ensure_dir, get_annotation_mode, get_golden_ann_dir, parse_meta_file
from vos.tracking import golden_progress


def rebuild_golden_preview_video(run_dir: Path) -> bool:
    """Re-render golden_preview.mp4 from committed golden frames (includes behaviour overlays)."""
    meta_path = run_dir / "meta.txt"
    if not meta_path.exists():
        return False
    meta = parse_meta_file(meta_path)
    fps = float(meta["fps"])
    n_ids = int(meta.get("ids", 0) or 0)
    n_total = int(meta["frames"])
    if n_ids < 1:
        return False

    _, _, max_idx = golden_progress(run_dir, n_total)
    if max_idx is None:
        return False

    golden_preview = run_dir / "golden" / "golden_preview.mp4"
    tmp_out = run_dir / "golden" / "golden_preview_rebuild_tmp.mp4"
    ensure_dir(golden_preview.parent)

    log.info(f"[GOLDEN_PREVIEW] Rebuilding 0..{max_idx} for run {run_dir.name}")
    _render_segment_from_golden(run_dir, fps, n_ids, 0, int(max_idx), tmp_out)
    if not tmp_out.exists() or tmp_out.stat().st_size == 0:
        log.error("[GOLDEN_PREVIEW] Rebuild produced empty output")
        return False

    golden_preview_tmp = run_dir / "golden" / "golden_preview_tmp.mp4"
    if _ffmpeg_reencode_video(tmp_out, golden_preview_tmp, fps):
        golden_preview_tmp.replace(golden_preview)
    else:
        tmp_out.replace(golden_preview)
    if tmp_out.exists():
        tmp_out.unlink()
    log.info(f"[GOLDEN_PREVIEW] Rebuild complete: {golden_preview}")
    return True

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
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        if p.returncode != 0:
            log.error(f"ffmpeg re-encode failed with return code {p.returncode}")
            log.error(f"ffmpeg output:\n{p.stdout[-2000:] if p.stdout else '(no output)'}")
            return False
    except FileNotFoundError:
        log.warning(f"ffmpeg not found in PATH. Skipping re-encoding - video may not be browser-compatible.")
        log.warning(f"To fix: install ffmpeg or load the ffmpeg module (e.g., 'module load ffmpeg')")
        # Copy the original file as-is (may not be browser-compatible)
        try:
            shutil.copy2(in_mp4, out_mp4)
            log.warning(f"Copied original video without re-encoding: {out_mp4}")
            return True
        except Exception as e:
            log.error(f"Failed to copy video: {e}")
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
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        if p.returncode != 0:
            log.error(f"ffmpeg drop-seed failed with return code {p.returncode}")
            log.error(f"ffmpeg output:\n{p.stdout[-2000:] if p.stdout else '(no output)'}")
            return False
    except FileNotFoundError:
        log.warning(f"ffmpeg not found in PATH. Cannot drop seed frame - video may not be browser-compatible.")
        log.warning(f"To fix: install ffmpeg or load the ffmpeg module (e.g., 'module load ffmpeg')")
        # Copy the original file as-is (may not be browser-compatible)
        try:
            shutil.copy2(in_mp4, out_mp4)
            log.warning(f"Copied original video without dropping seed frame: {out_mp4}")
            return True
        except Exception as e:
            log.error(f"Failed to copy video: {e}")
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
def extract_frames(video_path: str, jpeg_dir: Path, progress_callback=None):
    """
    Extract frames from video using OpenCV, with ffmpeg fallback if OpenCV fails.
    
    Args:
        video_path: Path to video file
        jpeg_dir: Directory to save extracted frames
        progress_callback: Optional callback(progress: float, message: str) for progress updates
    """
    log.info(f"Extracting frames from {video_path}")
    t0 = time.perf_counter()

    # Try OpenCV first
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        
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
            
            # Update progress
            if progress_callback and total_frames:
                progress = min(100, (idx / total_frames) * 100)
                if idx % 50 == 0 or idx == total_frames:
                    progress_callback(progress, f"Extracted {idx}/{total_frames} frames...")
            elif progress_callback and (idx % 100 == 0 or idx % LOG_EVERY_FRAMES_EXTRACT == 0):
                estimated_total = 3000
                estimated_progress = min(95, (idx / estimated_total) * 100)
                progress_callback(estimated_progress, f"Extracted {idx} frames...")

        cap.release()
        
        if progress_callback:
            progress_callback(100, f"Extracted {len(frames)} frames")
        
        log.info(f"Frame extraction done: {len(frames)} frames ({time.perf_counter()-t0:.2f}s)")
        return frames, fps
        
    except Exception as e:
        log.warning(f"OpenCV extraction failed: {e}, trying ffmpeg fallback")
        return _extract_frames_ffmpeg(video_path, jpeg_dir, progress_callback)


def _extract_frames_ffmpeg(video_path: str, jpeg_dir: Path, progress_callback=None):
    """
    Extract frames using ffmpeg (more robust for problematic videos).
    
    Args:
        video_path: Path to video file
        jpeg_dir: Directory to save extracted frames
        progress_callback: Optional callback(progress: float, message: str) for progress updates
    """
    log.info(f"Using ffmpeg to extract frames from {video_path}")
    t0 = time.perf_counter()
    
    # Get FPS and frame count using ffprobe
    fps_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        fps_result = subprocess.run(fps_cmd, capture_output=True, text=True)
        if fps_result.returncode != 0:
            raise RuntimeError(f"ffprobe failed to get FPS: {fps_result.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found in PATH. Please install ffmpeg (which includes ffprobe).")
    
    fps_str = fps_result.stdout.strip()
    if "/" in fps_str:
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den > 0 else 30.0
    else:
        fps = float(fps_str) if fps_str else 30.0
    
    log.info(f"Detected FPS: {fps}")
    
    # Get total frame count for progress
    count_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    total_frames = None
    try:
        count_result = subprocess.run(count_cmd, capture_output=True, text=True, timeout=10)
        if count_result.returncode == 0:
            total_frames = int(count_result.stdout.strip()) if count_result.stdout.strip() else None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass  # Can't get frame count, will estimate
    
    if progress_callback:
        progress_callback(0, "Starting frame extraction with ffmpeg...")
    
    # Extract frames using ffmpeg
    output_pattern = str(jpeg_dir / "%05d.jpg")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-q:v", "2",  # High quality JPEG
        "-vsync", "0",  # Extract all frames
        output_pattern
    ]
    
    log.info(f"Running ffmpeg: {' '.join(cmd)}")
    try:
        # For ffmpeg, we can't easily track progress during extraction
        # We'll update progress after completion
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            log.error(f"ffmpeg extraction failed: {result.stderr}")
            raise RuntimeError(f"ffmpeg failed to extract frames: {result.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg or load the ffmpeg module (e.g., 'module load ffmpeg'). Frame extraction requires ffmpeg when OpenCV fails.")
    
    # List extracted frames
    frames = sorted([f.name for f in jpeg_dir.glob("*.jpg")])
    
    if len(frames) == 0:
        raise RuntimeError("ffmpeg extracted 0 frames")
    
    if progress_callback:
        progress_callback(100, f"Extracted {len(frames)} frames")
    
    log.info(f"Frame extraction done: {len(frames)} frames ({time.perf_counter()-t0:.2f}s)")
    return frames, fps


def render_video(
    jpeg_dir: Path,
    frames,
    found_pngs,
    out_video: Path,
    fps: float,
    n_ids: int,
    run_dir: Optional[Path] = None,
    behavior_frame_offset: int = 0,
    include_behavior: bool = True,
):
    """
    Render all provided frames list (same length as found_pngs ideally).
    Uses direct ffmpeg encoding from processed frame images (faster, more reliable).
    """
    if not frames:
        raise RuntimeError("render_video got empty frames list")

    log.info(f"Rendering preview video: {out_video}")
    first = cv2.imread(str(jpeg_dir / frames[0]))
    if first is None:
        raise RuntimeError("Could not read first frame for rendering.")
    H, W = first.shape[:2]

    out_video.parent.mkdir(parents=True, exist_ok=True)
    colors = {i: random_color(i) for i in range(1, n_ids + 1)}
    draw_behavior = (
        include_behavior
        and run_dir is not None
        and get_annotation_mode(run_dir) == "behavior"
    )

    T = min(len(frames), len(found_pngs))
    
    # Process frames and write to temporary directory, then encode with ffmpeg
    with tempfile.TemporaryDirectory(prefix="render_video_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        log.debug(f"Processing {T} frames to temporary directory: {tmpdir_path}")
        for t in range(T):
            frame = cv2.imread(str(jpeg_dir / frames[t]))
            if frame is None:
                raise RuntimeError(f"Could not read frame {frames[t]}")
            mask = np.array(Image.open(found_pngs[t]))
            try:
                abs_frame_idx = int(Path(frames[t]).stem) + int(behavior_frame_offset)
            except ValueError:
                abs_frame_idx = t + int(behavior_frame_offset)

            for cid, col in colors.items():
                m = (mask == cid)
                if not m.any():
                    continue
                behavior_lines = None
                if draw_behavior:
                    behavior_lines = behavior_overlay_lines_at_frame(
                        run_dir, cid, abs_frame_idx
                    )
                frame = draw_cow_overlay_with_behavior(
                    frame, m, cid, col, behavior_lines=behavior_lines
                )

            # Write processed frame to temp directory (use quality 85 for faster I/O)
            frame_path = tmpdir_path / f"frame_{t:05d}.jpg"
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            if (t + 1) % LOG_EVERY_FRAMES_RENDER == 0:
                log.debug(f"  processed {t+1}/{T} frames")

        # Encode video directly with ffmpeg (single pass, H.264)
        log.debug(f"Encoding video with ffmpeg from {T} frames...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", f"{fps:.10f}",
            "-i", str(tmpdir_path / "frame_%05d.jpg"),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",  # Good quality for preview videos
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_video),
        ]
        
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
            if p.returncode != 0:
                log.error(f"ffmpeg encoding failed with return code {p.returncode}")
                log.error(f"ffmpeg output:\n{p.stdout[-2000:] if p.stdout else '(no output)'}")
                raise RuntimeError(f"Failed to encode video with ffmpeg: {out_video}")
        except FileNotFoundError:
            log.error("ffmpeg not found in PATH. Cannot render video without ffmpeg.")
            raise RuntimeError("ffmpeg is required for video rendering but was not found in PATH")

    log.info(f"Rendering done: {out_video}")
    return T


def _render_segment_from_golden(
    run_dir: Path,
    fps: float,
    n_ids: int,
    start_idx: int,
    end_idx: int,
    out_path: Path,
    include_behavior: bool = True,
):
    """
    Render golden overlay segment for frames start_idx..end_idx inclusive into out_path.
    Uses original JPEGs + golden label PNGs.
    """
    
    log.info(f"[RENDER_GOLDEN] Rendering segment: frames {start_idx}..{end_idx} -> {out_path}")
    
    src_root = run_dir / "xmem_generic"
    src_jpeg = src_root / "JPEGImages" / VIDEO_NAME
    golden_ann = get_golden_ann_dir(run_dir)

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
        run_dir=run_dir,
        include_behavior=include_behavior,
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
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        log.error(f"ffmpeg not found in PATH. Cannot concatenate videos.")
        log.error(f"To fix: install ffmpeg or load the ffmpeg module (e.g., 'module load ffmpeg')")
        # Clean up temp files
        if tmp_list.exists():
            tmp_list.unlink()
        return False
    
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
            try:
                p2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            except FileNotFoundError:
                log.error(f"ffmpeg not found during retry. Cannot concatenate videos.")
                # Clean up temp files
                if tmp_list.exists():
                    tmp_list.unlink()
                if a_reencoded.exists():
                    a_reencoded.unlink()
                return False
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

def _probe_duration(path: Path) -> float | None:
    """Get video duration using ffprobe. Returns None if ffprobe is not available or fails."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return None
        try:
            return float(p.stdout.strip())
        except:
            return None
    except FileNotFoundError:
        # ffprobe not found in PATH - this is non-critical, just return None
        log.warning(f"ffprobe not found in PATH, cannot probe video duration for {path}")
        return None
    except Exception as e:
        log.warning(f"Error probing video duration: {e}")
        return None
