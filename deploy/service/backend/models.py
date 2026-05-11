from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path


class VideoUploadResponse(BaseModel):
    video_id: str
    message: str
    status: str


class AnomalyFrame(BaseModel):
    frame_idx: int
    timestamp: float
    timestamp_str: Optional[str] = None
    score: float
    caption: Optional[str] = None
    summary: Optional[str] = None
    image_path: Optional[str] = None
    image_base64: Optional[str] = None


class DetectionResult(BaseModel):
    video_id: str
    video_name: str
    total_frames: int
    fps: float
    duration: float
    anomaly_frames: List[AnomalyFrame]
    normal_frames: int
    abnormal_frames: int
    anomaly_ratio: float
    overall_score: float
    max_score: float = 0.0
    threshold: float = 0.0
    mean_score: float = 0.0
    std_score: float = 0.0
    summary: str
    top_anomaly_captions: List[str]
    processing_time: float
    steps_completed: List[str] = []


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_count: int
    model_loaded: bool


class ModelStatus(BaseModel):
    blip2_loaded: List[str]
    blip2_available: List[str]
    qwen_loaded: bool
    imagebind_loaded: bool


class VideoInfo(BaseModel):
    video_id: str
    video_name: str
    duration: float
    fps: float
    total_frames: int
    resolution: str