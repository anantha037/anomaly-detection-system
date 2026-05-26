import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.detection.alert_system import AlertSystem
from src.detection.threshold_engine import ThresholdEngine
from src.models.autoencoder import LSTMAutoencoder


def evaluate():
    processed_dir = Path("data/processed")

    print("Loading data and model artifacts...")
    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    scaler_path = processed_dir / "scaler.pkl"
    model_path = processed_dir / "autoencoder.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate model
    model = LSTMAutoencoder(n_features=8, latent_dim=32, n_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Initialize Engine
    engine = ThresholdEngine(model, device, scaler_path)

    print("Computing train errors...")
    train_errors = engine.compute_errors(X_train)

    print("Tuning threshold...")
    multipliers = [2.0, 2.5, 3.0, 3.5, 4.0]
    best_multiplier, best_threshold, metrics_df = engine.find_best_threshold(X_test, y_test, train_errors, multipliers)

    print("\n--- Threshold Tuning Metrics ---")
    print(metrics_df.to_string(index=False))

    print(f"\nBest Multiplier: {best_multiplier}")
    print(f"Best Threshold: {best_threshold:.6f}")

    engine.save_threshold(best_threshold, processed_dir / "threshold.npy")
    print(f"Saved best threshold to {processed_dir / 'threshold.npy'}")

    print("\nGenerating alerts for test set...")
    alert_system = AlertSystem(best_threshold)
    test_errors = engine.compute_errors(X_test)
    alerts = alert_system.generate_alerts(test_errors)

    summary = alert_system.summary(alerts)
    print("\n--- Alert Summary ---")
    print(json.dumps(summary, indent=4))

    summary_path = processed_dir / "alert_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved alert summary to {summary_path}")


if __name__ == "__main__":
    evaluate()
