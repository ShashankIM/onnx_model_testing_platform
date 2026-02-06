from typing import List, Optional
from pydantic import BaseModel


class ClassificationOut(BaseModel):
    model_name: str
    label: Optional[str] = None
    score: Optional[float] = None


class DetectionOut(BaseModel):
    bbox: List[int]
    score: float
    class_id: int
    class_name: str
    crop_image_base64: Optional[str] = None
    classifications: List[ClassificationOut]


class ImageResult(BaseModel):
    filename: str
    detections: List[DetectionOut]
    annotated_image_base64: str


class PredictResponse(BaseModel):
    results: List[ImageResult]
