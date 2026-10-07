"""Behaviour annotation: label catalog (config/behavior_labels.json) and per-cow segment storage."""
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np
from PIL import Image

from vos.config import BACKEND_ROOT, log
from vos.storage import get_annotation_mode, get_golden_ann_dir


# -------------------------
# Behaviour annotation: labels + segment storage (3 dimensions)
# -------------------------
BEHAVIOR_LABEL_NONE = "none"
NOT_VISIBLE_LABEL_ID = "not_visible"
NOT_SEEN_LABEL_ID = "not_seen"

BEHAVIOR_DIMENSIONS = ("activity", "label2", "label3")
_BEHAVIOR_LABELS_LOCAL_PATH = BACKEND_ROOT / "config" / "behavior_labels.local.json"
BEHAVIOR_LABELS_CONFIG_PATH = Path(
    os.environ.get(
        "BEHAVIOR_LABELS_CONFIG",
        _BEHAVIOR_LABELS_LOCAL_PATH
        if _BEHAVIOR_LABELS_LOCAL_PATH.exists()
        else BACKEND_ROOT / "config" / "behavior_labels.json",
    )
)
_BEHAVIOR_STORAGE_FILES = {
    "activity": "behavior_labels_activity.json",
    "label2": "behavior_labels_label2.json",
    "label3": "behavior_labels_label3.json",
}


def _load_behavior_dimension_meta(config_path: Path) -> Dict[str, Dict[str, Any]]:
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    dims_cfg = config.get("dimensions", {})
    if set(dims_cfg) != set(BEHAVIOR_DIMENSIONS):
        raise ValueError(
            f"{config_path}: dimensions must be exactly {list(BEHAVIOR_DIMENSIONS)}, "
            f"got {list(dims_cfg)}"
        )

    required_ids = {
        "activity": {NOT_VISIBLE_LABEL_ID},
        "label2": {BEHAVIOR_LABEL_NONE, NOT_SEEN_LABEL_ID},
        "label3": {BEHAVIOR_LABEL_NONE},
    }
    meta: Dict[str, Dict[str, Any]] = {}
    for dim in BEHAVIOR_DIMENSIONS:
        cfg = dims_cfg[dim]
        labels = [
            {k: v for k, v in label.items() if k in ("id", "name_fi", "description_fi", "group_fi")}
            for label in cfg["labels"]
        ]
        ids = [label["id"] for label in labels]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{config_path}: duplicate label ids in {dim}")
        missing = required_ids[dim] - set(ids)
        if missing:
            raise ValueError(f"{config_path}: {dim} is missing required label ids {sorted(missing)}")
        if cfg["default_label"] not in ids:
            raise ValueError(f"{config_path}: {dim} default_label {cfg['default_label']!r} is not a label id")
        meta[dim] = {
            "file": _BEHAVIOR_STORAGE_FILES[dim],
            "labels": labels,
            "default_label": cfg["default_label"],
            "title_fi": cfg["title_fi"],
            "required": bool(cfg.get("required", False)),
            "affects_preview": True,
            "hidden_ids": set(),
        }
    return meta


BEHAVIOR_DIMENSION_META: Dict[str, Dict[str, Any]] = _load_behavior_dimension_meta(
    BEHAVIOR_LABELS_CONFIG_PATH
)
BEHAVIOR_LABELS_ACTIVITY: List[Dict[str, Any]] = BEHAVIOR_DIMENSION_META["activity"]["labels"]
BEHAVIOR_LABELS_LABEL2: List[Dict[str, Any]] = BEHAVIOR_DIMENSION_META["label2"]["labels"]
BEHAVIOR_LABELS_LABEL3: List[Dict[str, Any]] = BEHAVIOR_DIMENSION_META["label3"]["labels"]

DEFAULT_BEHAVIOR_LABEL_ID = BEHAVIOR_DIMENSION_META["activity"]["default_label"]
BEHAVIOR_FILE_NAME = _BEHAVIOR_STORAGE_FILES["activity"]


def _valid_label_ids_for_dimension(dimension: str) -> set:
    return {label["id"] for label in BEHAVIOR_DIMENSION_META[dimension]["labels"]}


def _default_label_for_dimension(dimension: str) -> str:
    return BEHAVIOR_DIMENSION_META[dimension]["default_label"]


def _segment_visible(label_id: str, dimension: str) -> bool:
    if dimension == "activity":
        return label_id != NOT_VISIBLE_LABEL_ID
    if dimension == "label2":
        return label_id not in (BEHAVIOR_LABEL_NONE, NOT_SEEN_LABEL_ID)
    # label3: only receiver behaviours count as active segments
    return label_id != BEHAVIOR_LABEL_NONE


def behavior_label_by_id(label_id: str, dimension: str = "activity") -> Dict[str, Any]:
    for label in BEHAVIOR_DIMENSION_META[dimension]["labels"]:
        if label["id"] == label_id:
            return label
    raise KeyError(label_id)


def behavior_file_path(run_dir: Path, dimension: str = "activity") -> Path:
    return run_dir / BEHAVIOR_DIMENSION_META[dimension]["file"]


def empty_behavior_data(cow_ids: Optional[List[int]] = None, dimension: str = "activity") -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "version": 1,
        "dimension": dimension,
        "segments": [],
        "cow_ids": sorted(cow_ids or []),
    }
    if dimension == "activity":
        data["preview_in_sync"] = True
    return data


def load_behavior_dimension(run_dir: Path, dimension: str) -> Optional[Dict[str, Any]]:
    path = behavior_file_path(run_dir, dimension)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _sort_behavior_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(segments, key=lambda s: (int(s["cow_id"]), int(s["start_frame"])))


def _prune_invalid_behavior_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pruned: List[Dict[str, Any]] = []
    for seg in segments:
        start = int(seg["start_frame"])
        end = seg.get("end_frame")
        if end is not None and int(end) < start:
            continue
        pruned.append(seg)
    return pruned


def _normalize_behavior_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _sort_behavior_segments(_prune_invalid_behavior_segments(segments))


def save_behavior_dimension(run_dir: Path, dimension: str, data: Dict[str, Any]) -> None:
    path = behavior_file_path(run_dir, dimension)
    data = dict(data)
    data["segments"] = _normalize_behavior_segments(data.get("segments", []))
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_behavior_data(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Activity dimension (Label 1); used for golden preview overlay."""
    return load_behavior_dimension(run_dir, "activity")


def save_behavior_data(run_dir: Path, data: Dict[str, Any]) -> None:
    save_behavior_dimension(run_dir, "activity", data)


def mark_behavior_preview_out_of_sync(run_dir: Path) -> None:
    data = load_behavior_dimension(run_dir, "activity")
    if data is None:
        return
    data["preview_in_sync"] = False
    save_behavior_dimension(run_dir, "activity", data)


def mark_behavior_preview_in_sync(run_dir: Path) -> None:
    data = load_behavior_dimension(run_dir, "activity")
    if data is None:
        return
    data["preview_in_sync"] = True
    save_behavior_dimension(run_dir, "activity", data)


def _validate_behavior_label_id(label_id: str, dimension: str) -> None:
    if label_id not in _valid_label_ids_for_dimension(dimension):
        raise ValueError(f"Unknown label_id for {dimension}: {label_id}")
    meta = BEHAVIOR_DIMENSION_META[dimension]
    if meta["required"] and label_id == BEHAVIOR_LABEL_NONE:
        raise ValueError(f"Label 1 (activity) cannot be '{BEHAVIOR_LABEL_NONE}'")


def create_initial_segments(
    cow_ids: List[int],
    start_frame: int,
    labels_by_cow: Dict[int, str],
    dimension: str = "activity",
) -> Dict[str, Any]:
    """One open-ended segment per cow starting at start_frame."""
    segments: List[Dict[str, Any]] = []
    default_label = _default_label_for_dimension(dimension)
    for cow_id in sorted(cow_ids):
        label_id = labels_by_cow.get(cow_id, default_label)
        _validate_behavior_label_id(label_id, dimension)
        segments.append(
            {
                "cow_id": int(cow_id),
                "start_frame": int(start_frame),
                "end_frame": None,
                "label_id": label_id,
                "visible": _segment_visible(label_id, dimension),
            }
        )
    return empty_behavior_data(cow_ids, dimension) | {"segments": segments}


def _pre_detection_label_for_dimension(dimension: str) -> str:
    """Label for frames before a cow first appears in masks (late add_mask / correction)."""
    if dimension == "activity":
        return NOT_VISIBLE_LABEL_ID
    if dimension == "label2":
        return NOT_SEEN_LABEL_ID
    return BEHAVIOR_LABEL_NONE


def register_late_behavior_cows(
    run_dir: Path, cow_ids: List[int], first_visible_frame: int
) -> List[int]:
    """
    Register cows that first appear after frame 0 (e.g. add_mask + apply_correction).
    Adds segments from frame 0 through first_visible_frame - 1 with a pre-detection label,
    then an open-ended segment from first_visible_frame with the dimension default.
    """
    if get_annotation_mode(run_dir) != "behavior":
        return []

    new_cow_ids = sorted({int(c) for c in cow_ids if int(c) > 0})
    if not new_cow_ids:
        return []

    first_visible_frame = int(first_visible_frame)
    registered: List[int] = []

    for dimension in BEHAVIOR_DIMENSIONS:
        data = load_behavior_dimension(run_dir, dimension)
        if data is None:
            data = empty_behavior_data([], dimension)

        existing = {int(c) for c in data.get("cow_ids", [])}
        to_add = [cid for cid in new_cow_ids if cid not in existing]
        if not to_add:
            continue

        segments: List[Dict[str, Any]] = list(data.get("segments", []))
        pre_label = _pre_detection_label_for_dimension(dimension)
        default_label = _default_label_for_dimension(dimension)

        for cow_id in to_add:
            if first_visible_frame > 0:
                segments.append(
                    {
                        "cow_id": cow_id,
                        "start_frame": 0,
                        "end_frame": first_visible_frame - 1,
                        "label_id": pre_label,
                        "visible": _segment_visible(pre_label, dimension),
                    }
                )
            segments.append(
                {
                    "cow_id": cow_id,
                    "start_frame": first_visible_frame,
                    "end_frame": None,
                    "label_id": default_label,
                    "visible": _segment_visible(default_label, dimension),
                }
            )
            registered.append(cow_id)

        data["cow_ids"] = sorted(existing | set(to_add))
        data["segments"] = _normalize_behavior_segments(segments)
        save_behavior_dimension(run_dir, dimension, data)

    registered_unique = sorted(set(registered))
    if registered_unique:
        mark_behavior_preview_out_of_sync(run_dir)
        log.info(
            f"[BEHAVIOR] Registered late cows {registered_unique} "
            f"first_visible_frame={first_visible_frame} run_dir={run_dir.name}"
        )
    return registered_unique


def _find_covering_behavior_segment(
    segments: List[Dict[str, Any]], cow_id: int, frame: int
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for seg in segments:
        if int(seg["cow_id"]) != cow_id:
            continue
        start = int(seg["start_frame"])
        end = seg.get("end_frame")
        if frame < start:
            continue
        if end is not None and frame > int(end):
            continue
        if best is None or start >= int(best["start_frame"]):
            best = seg
    return best


def _next_behavior_segment_start(
    segments: List[Dict[str, Any]], cow_id: int, frame: int
) -> Optional[int]:
    starts = [
        int(s["start_frame"])
        for s in segments
        if int(s["cow_id"]) == cow_id and int(s["start_frame"]) > frame
    ]
    return min(starts) if starts else None


def _append_behavior_segment(
    segments: List[Dict[str, Any]],
    cow_id: int,
    start_frame: int,
    end_frame: Optional[int],
    label_id: str,
    dimension: str,
) -> None:
    segments.append(
        {
            "cow_id": int(cow_id),
            "start_frame": int(start_frame),
            "end_frame": int(end_frame) if end_frame is not None else None,
            "label_id": label_id,
            "visible": _segment_visible(label_id, dimension),
        }
    )


def set_label_from_frame(
    data: Dict[str, Any],
    cow_id: int,
    frame: int,
    label_id: str,
    dimension: str = "activity",
) -> Dict[str, Any]:
    """
    Set label for cow_id from frame onward. Splits an existing segment when frame falls
    in the middle (e.g. A on 0–30, B on 30+, then C at 15 → 0–14 A, 15–29 C, 30+ B).
    """
    _validate_behavior_label_id(label_id, dimension)
    frame = int(frame)
    cow_id = int(cow_id)
    segments: List[Dict[str, Any]] = list(data.get("segments", []))

    for seg in segments:
        if int(seg["cow_id"]) == cow_id and int(seg["start_frame"]) == frame:
            seg["label_id"] = label_id
            seg["visible"] = _segment_visible(label_id, dimension)
            data["segments"] = _normalize_behavior_segments(segments)
            if cow_id not in data.get("cow_ids", []):
                data.setdefault("cow_ids", []).append(cow_id)
                data["cow_ids"] = sorted(data["cow_ids"])
            return data

    covering = _find_covering_behavior_segment(segments, cow_id, frame)
    next_start = _next_behavior_segment_start(segments, cow_id, frame)

    if covering is not None and int(covering["start_frame"]) < frame:
        cover_end = covering.get("end_frame")
        covering["end_frame"] = frame - 1
        if next_start is not None:
            new_end = next_start - 1
        elif cover_end is not None:
            new_end = int(cover_end)
        else:
            new_end = None
        _append_behavior_segment(segments, cow_id, frame, new_end, label_id, dimension)
    else:
        new_end = (next_start - 1) if next_start is not None else None
        _append_behavior_segment(segments, cow_id, frame, new_end, label_id, dimension)

    data["segments"] = _normalize_behavior_segments(segments)
    cow_ids_set = set(data.get("cow_ids", []))
    cow_ids_set.add(cow_id)
    data["cow_ids"] = sorted(cow_ids_set)
    return data


def delete_label_from_frame(
    data: Dict[str, Any],
    cow_id: int,
    frame: int,
    dimension: str = "activity",
) -> Dict[str, Any]:
    """
    Remove a behaviour change that starts at frame (undo split). Extends the previous
    segment to cover the deleted segment's range.
    """
    frame = int(frame)
    cow_id = int(cow_id)
    if frame <= 0:
        raise ValueError("Cannot delete the initial behaviour at frame 0")

    segments: List[Dict[str, Any]] = list(data.get("segments", []))
    target_idx: Optional[int] = None
    for i, seg in enumerate(segments):
        if int(seg["cow_id"]) == cow_id and int(seg["start_frame"]) == frame:
            target_idx = i
            break
    if target_idx is None:
        raise ValueError(f"No behaviour change at frame {frame} for cow {cow_id}")

    deleted = segments.pop(target_idx)
    deleted_end = deleted.get("end_frame")

    prev: Optional[Dict[str, Any]] = None
    for seg in segments:
        if int(seg["cow_id"]) != cow_id:
            continue
        if int(seg["start_frame"]) < frame:
            if prev is None or int(seg["start_frame"]) > int(prev["start_frame"]):
                prev = seg
    if prev is None:
        raise ValueError("Cannot delete: no preceding segment")

    prev["end_frame"] = int(deleted_end) if deleted_end is not None else None
    data["segments"] = _normalize_behavior_segments(segments)
    return data


def get_behavior_label_at_frame(data: Dict[str, Any], cow_id: int, frame: int) -> Optional[str]:
    frame = int(frame)
    cow_id = int(cow_id)
    best: Optional[Dict[str, Any]] = None
    for seg in data.get("segments", []):
        if seg["cow_id"] != cow_id:
            continue
        start = int(seg["start_frame"])
        end = seg.get("end_frame")
        if frame < start:
            continue
        if end is not None and frame > int(end):
            continue
        if best is None or start >= int(best["start_frame"]):
            best = seg
    return best["label_id"] if best else None


def labels_at_frame(data: Dict[str, Any], frame: int) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for cow_id in data.get("cow_ids", []):
        label = get_behavior_label_at_frame(data, int(cow_id), frame)
        if label is not None:
            out[int(cow_id)] = label
    return out


def behavior_overlay_lines_at_frame(run_dir: Path, cow_id: int, frame_idx: int) -> List[str]:
    """Finnish label lines for golden preview: activity, then label2, then label3."""
    if get_annotation_mode(run_dir) != "behavior":
        return []
    lines: List[str] = []
    for dim in BEHAVIOR_DIMENSIONS:
        data = load_behavior_dimension(run_dir, dim)
        if not data:
            continue
        label_id = get_behavior_label_at_frame(data, cow_id, frame_idx)
        if not label_id:
            continue
        if dim == "activity" and label_id == NOT_VISIBLE_LABEL_ID:
            continue
        if dim == "label2" and label_id in (BEHAVIOR_LABEL_NONE, NOT_SEEN_LABEL_ID):
            continue
        if dim == "label3" and label_id == BEHAVIOR_LABEL_NONE:
            continue
        try:
            lines.append(behavior_label_by_id(label_id, dim)["name_fi"])
        except KeyError:
            lines.append(label_id)
    return lines


def _golden_mask_presence_by_cow(
    run_dir: Path, cow_ids: List[int], max_frame: int
) -> Dict[int, set]:
    """Frames where each cow_id has at least one pixel in golden annotation masks."""
    golden_ann_dir = get_golden_ann_dir(run_dir)
    present: Dict[int, set] = {int(c): set() for c in cow_ids}
    for frame_idx in range(0, max_frame + 1):
        ann_path = golden_ann_dir / f"{frame_idx:05d}.png"
        if not ann_path.exists():
            continue
        arr = np.array(Image.open(ann_path))
        for cow_id in cow_ids:
            if np.any(arr == int(cow_id)):
                present[int(cow_id)].add(frame_idx)
    return present


def _segments_from_frame_labels(
    cow_id: int,
    frame_labels: Dict[int, str],
    dimension: str,
    max_frame: int,
) -> List[Dict[str, Any]]:
    """Run-length encode per-frame labels into contiguous segments."""
    segments: List[Dict[str, Any]] = []
    current_label: Optional[str] = None
    start: Optional[int] = None
    for frame_idx in range(0, max_frame + 1):
        label = frame_labels.get(frame_idx)
        if label is None:
            continue
        if label != current_label:
            if current_label is not None and start is not None:
                _append_behavior_segment(
                    segments, cow_id, start, frame_idx - 1, current_label, dimension
                )
            current_label = label
            start = frame_idx
    if current_label is not None and start is not None:
        _append_behavior_segment(segments, cow_id, start, None, current_label, dimension)
    return segments


def sync_activity_visibility_from_masks(run_dir: Path) -> bool:
    """
    Rebuild activity segments so frames without a golden mask for a cow use not_visible.
    Present frames keep the annotator's labels. Called before golden zip export.
    """
    if get_annotation_mode(run_dir) != "behavior":
        return False
    data = load_behavior_dimension(run_dir, "activity")
    if not data:
        return False

    golden_ann_dir = get_golden_ann_dir(run_dir)
    if not golden_ann_dir.exists():
        return False
    pngs = sorted(golden_ann_dir.glob("*.png"))
    if not pngs:
        return False

    max_frame = max(int(p.stem) for p in pngs)
    cow_ids = [int(c) for c in data.get("cow_ids", [])]
    if not cow_ids:
        return False

    before = json.dumps(data.get("segments", []), sort_keys=True)
    presence = _golden_mask_presence_by_cow(run_dir, cow_ids, max_frame)
    default_label = _default_label_for_dimension("activity")
    original_segments = list(data.get("segments", []))
    original_data = dict(data)
    original_data["segments"] = original_segments

    new_segments: List[Dict[str, Any]] = []
    for cow_id in cow_ids:
        frame_labels: Dict[int, str] = {}
        for frame_idx in range(0, max_frame + 1):
            if frame_idx not in presence.get(cow_id, set()):
                frame_labels[frame_idx] = NOT_VISIBLE_LABEL_ID
            else:
                label = get_behavior_label_at_frame(original_data, cow_id, frame_idx)
                frame_labels[frame_idx] = label if label is not None else default_label
        new_segments.extend(
            _segments_from_frame_labels(cow_id, frame_labels, "activity", max_frame)
        )

    data["segments"] = _normalize_behavior_segments(new_segments)
    after = json.dumps(data["segments"], sort_keys=True)
    if before == after:
        return False

    save_behavior_dimension(run_dir, "activity", data)
    mark_behavior_preview_out_of_sync(run_dir)
    log.info(
        f"[BEHAVIOR] sync_activity_visibility_from_masks run_dir={run_dir.name} "
        f"cows={cow_ids} frames=0..{max_frame}"
    )
    return True
