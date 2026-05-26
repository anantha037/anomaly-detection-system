import os
import sys
from pathlib import Path

import numpy as np
import requests
import torch

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_test_data():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"

    try:
        X_test = np.load(data_dir / "X_test.npy")
        y_test = np.load(data_dir / "y_test.npy")
        return X_test, y_test
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None


def fetch_predictions(X_test, batch_size=500):
    api_url = os.getenv("API_URL", "http://localhost:8000")
    url = f"{api_url}/predict"
    all_results = []

    total_windows = len(X_test)
    for i in range(0, total_windows, batch_size):
        batch = X_test[i : i + batch_size].tolist()
        try:
            response = requests.post(url, json={"windows": batch}, timeout=30)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                for j, res in enumerate(results):
                    res["window_id"] = i + j
                all_results.extend(results)
            else:
                print(f"Error from API: {response.status_code}")
                return []
        except requests.RequestException as e:
            print(f"Connection error: {e}")
            return []

    return all_results


def compute_sensor_errors(X_test, predictions, model, device):
    """
    For anomalous windows only, compute per-sensor MSE between input and reconstruction.
    """
    anomalous_indices = [res["window_id"] for res in predictions if res["is_anomaly"]]
    if not anomalous_indices:
        return {}

    X_anom = X_test[anomalous_indices]
    X_tensor = torch.tensor(X_anom, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)
        # Compute MSE per feature across batch and seq_len
        mse_per_feature = torch.mean((preds - X_tensor) ** 2, dim=(0, 1)).cpu().numpy()

    sensor_names = [
        "Accelerometer1RMS",
        "Accelerometer2RMS",
        "Current",
        "Pressure",
        "Temperature",
        "Thermocouple",
        "Voltage",
        "Volume Flow RateRMS",
    ]

    return {name: float(err) for name, err in zip(sensor_names, mse_per_feature)}
