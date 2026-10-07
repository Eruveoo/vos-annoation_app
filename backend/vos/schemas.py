"""Pydantic request payloads."""
from typing import Dict, Optional
from pydantic import BaseModel


class IDMapping(BaseModel):
    mapping: Dict[str, int]


class ApplyInitPayload(BaseModel):
    mapping: Dict[str, int]
    behavior_by_cow_id: Optional[Dict[str, str]] = None
    behavior_label2_by_cow_id: Optional[Dict[str, str]] = None
    behavior_label3_by_cow_id: Optional[Dict[str, str]] = None


class AnnotationModePayload(BaseModel):
    mode: str


class BehaviorSetLabelPayload(BaseModel):
    cow_id: int
    frame: int
    label_id: str
    dimension: str = "activity"


class BehaviorDeleteLabelPayload(BaseModel):
    cow_id: int
    frame: int
    dimension: str = "activity"


class PreviewUpdate(BaseModel):
    mapping: Dict[str, int]  # mask_index -> final_id (0 means delete)


class PointPrompt(BaseModel):
    x: int
    y: int
    is_positive: bool  # True = add to mask, False = remove from mask

class RefineMaskRequest(BaseModel):
    mask_index: int
    points: list[PointPrompt]  # Accumulated points for this mask

class AddMaskRequest(BaseModel):
    point: PointPrompt  # Single point to create a new mask
