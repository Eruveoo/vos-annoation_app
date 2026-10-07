"""In-memory progress state shared between endpoints."""
from typing import Dict


# Progress tracking for upload/prepare operations
# Key: run_id, Value: {"stage": "upload"|"extract", "progress": 0-100, "message": str}
prepare_progress: Dict[str, Dict] = {}

# Progress tracking for tracking operations
# Key: run_id, Value: {"stage": "tracking"|"rendering", "progress": 0-100, "message": str, "current_frame": int, "total_frames": int}
track_progress: Dict[str, Dict] = {}
