import os
import sys
from pathlib import Path
import json
import torch
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.autoencoder import LSTMAutoencoder

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing API lifespan...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    app_state["device"] = device
    
    model_path = os.getenv("MODEL_PATH", "data/processed/autoencoder.pth")
    threshold_path = os.getenv("THRESHOLD_PATH", "data/processed/threshold.npy")
    scaler_path = os.getenv("SCALER_PATH", "data/processed/scaler.pkl")
    alert_summary_path = os.getenv("ALERT_SUMMARY_PATH", "data/processed/alert_summary.json")
    
    # Load Model
    if os.path.exists(model_path):
        model = LSTMAutoencoder(n_features=8, latent_dim=32, n_layers=2)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        app_state["model"] = model
        print(f"Loaded model from {model_path}")
    else:
        app_state["model"] = None
        print(f"Warning: Model not found at {model_path}")

    # Load Threshold
    if os.path.exists(threshold_path):
        app_state["threshold"] = float(np.load(threshold_path))
        print(f"Loaded threshold from {threshold_path}: {app_state['threshold']}")
    else:
        app_state["threshold"] = None
        print(f"Warning: Threshold not found at {threshold_path}")

    # Load Scaler
    if os.path.exists(scaler_path):
        app_state["scaler"] = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    else:
        app_state["scaler"] = None
        print(f"Warning: Scaler not found at {scaler_path}")

    # Load Alert Summary
    if os.path.exists(alert_summary_path):
        with open(alert_summary_path, 'r') as f:
            app_state["alert_summary"] = json.load(f)
        print(f"Loaded alert summary from {alert_summary_path}")
    else:
        app_state["alert_summary"] = None
        print(f"Warning: Alert summary not found at {alert_summary_path}")

    yield
    print("Shutting down API...")
    app_state.clear()

def get_app_state():
    return app_state
