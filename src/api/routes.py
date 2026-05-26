import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.api.dependencies import get_app_state
from src.api.models import (
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    ThresholdResponse,
    WindowResult,
)
from src.detection.alert_system import AlertSystem

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(state: dict = Depends(get_app_state)):
    return HealthResponse(
        status="ok", model_loaded=state.get("model") is not None, threshold_loaded=state.get("threshold") is not None
    )


@router.get("/threshold", response_model=ThresholdResponse)
def get_threshold(state: dict = Depends(get_app_state)):
    threshold = state.get("threshold")
    if threshold is None:
        raise HTTPException(status_code=503, detail="Threshold not loaded")
    return ThresholdResponse(threshold=threshold, multiplier=2.0, method="mean + k*std")


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(state: dict = Depends(get_app_state)):
    alert_summary = state.get("alert_summary")
    if alert_summary is None:
        raise HTTPException(status_code=503, detail="Alert summary not loaded")

    total_alerts = alert_summary.get("total_alerts", 0)
    alert_rate = alert_summary.get("alert_rate", 0.0)
    if alert_rate > 0:
        total_test_windows = int(round(total_alerts / alert_rate))
    else:
        total_test_windows = 7621

    return MetricsResponse(
        model="LSTM Autoencoder",
        f1_score=0.7616,
        precision=0.9317,
        recall=0.6441,
        best_multiplier=2.0,
        total_test_windows=total_test_windows,
        anomalous_windows=total_alerts,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, state: dict = Depends(get_app_state)):
    model = state.get("model")
    threshold = state.get("threshold")
    device = state.get("device")

    if model is None or threshold is None:
        raise HTTPException(status_code=503, detail="Model or threshold not loaded")

    # Convert input to numpy array
    X = np.array(request.windows)

    # Run through model in batches
    errors = []
    batch_size = 256
    model.eval()

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            preds = model(batch_x)
            batch_errors = torch.mean((preds - batch_x) ** 2, dim=(1, 2)).cpu().numpy()
            errors.extend(batch_errors)

    errors = np.array(errors)

    # Generate alerts
    alert_system = AlertSystem(threshold)
    alerts = alert_system.generate_alerts(errors)
    summary = alert_system.summary(alerts)

    results = []
    for i, err in enumerate(errors):
        window_alert = next((a for a in alerts if a.window_id == i), None)
        is_anomaly = window_alert is not None
        severity = window_alert.severity if window_alert else "none"

        results.append(
            WindowResult(window_id=i, reconstruction_error=float(err), is_anomaly=is_anomaly, severity=severity)
        )

    return PredictResponse(
        results=results, anomaly_count=summary["total_alerts"], total_windows=len(X), alert_rate=summary["alert_rate"]
    )
