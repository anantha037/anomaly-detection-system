import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score


def train_isolation_forest():
    load_dotenv()
    processed_dir = Path("data/processed")

    print("Loading data...")
    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    # Flatten windows: (n, 30, 8) -> (n, 240)
    print(f"Original X_train shape: {X_train.shape}")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    print(f"Flattened X_train shape: {X_train_flat.shape}")

    contamination = 0.05
    n_estimators = 100

    # MLflow setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("anomaly-detection")

    with mlflow.start_run(run_name="isolation_forest_baseline"):
        print("Training Isolation Forest...")
        model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42, n_jobs=-1)
        model.fit(X_train_flat)

        print("Predicting on test set...")
        # Returns -1 for anomalies, 1 for normal
        preds = model.predict(X_test_flat)

        # Convert to 1 for anomalies, 0 for normal
        preds_binary = np.where(preds == -1, 1, 0)

        print("Evaluating...")
        precision = precision_score(y_test, preds_binary)
        recall = recall_score(y_test, preds_binary)
        f1 = f1_score(y_test, preds_binary)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Save model
        model_path = processed_dir / "isolation_forest.pkl"
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

        # Log to MLflow
        mlflow.log_params(
            {
                "contamination": contamination,
                "n_estimators": n_estimators,
                "window_size": X_train.shape[1],
                "n_features": X_train.shape[2],
                "input_shape": str(X_train_flat.shape[1]),
            }
        )
        mlflow.log_metrics({"precision": precision, "recall": recall, "f1_score": f1})
        mlflow.log_artifact(str(model_path))
        print("Logged to MLflow successfully.")


if __name__ == "__main__":
    train_isolation_forest()
