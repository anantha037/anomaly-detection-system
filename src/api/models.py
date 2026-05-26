from typing import List

from pydantic import BaseModel, Field, field_validator


class WindowResult(BaseModel):
    window_id: int
    reconstruction_error: float
    is_anomaly: bool
    severity: str


class PredictRequest(BaseModel):
    windows: List[List[List[float]]] = Field(..., description="Batch of windows, shape [n_windows, 30, 8]")

    @field_validator("windows")
    @classmethod
    def check_shape(cls, v):
        if not v:
            raise ValueError("Input windows list cannot be empty")
        for i, window in enumerate(v):
            if len(window) != 30:
                raise ValueError(f"Window at index {i} must have length 30, got {len(window)}")
            for j, step in enumerate(window):
                if len(step) != 8:
                    raise ValueError(f"Timestep {j} in window {i} must have length 8, got {len(step)}")
        return v


class PredictResponse(BaseModel):
    results: List[WindowResult]
    anomaly_count: int
    total_windows: int
    alert_rate: float


class ThresholdResponse(BaseModel):
    threshold: float
    multiplier: float
    method: str


class MetricsResponse(BaseModel):
    model: str
    f1_score: float
    precision: float
    recall: float
    best_multiplier: float
    total_test_windows: int
    anomalous_windows: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold_loaded: bool
