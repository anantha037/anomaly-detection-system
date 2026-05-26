import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import os
import sys
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

# Add src to path so we can import autoencoder
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.autoencoder import LSTMAutoencoder
from src.detection.threshold_engine import ThresholdEngine

def train():
    load_dotenv()
    processed_dir = Path("data/processed")
    
    print("Loading data...")
    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    # Hyperparams
    N_FEATURES = 8
    LATENT_DIM = 32
    N_LAYERS = 2
    WINDOW_SIZE = 30
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    
    # MLflow setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("anomaly-detection")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMAutoencoder(N_FEATURES, LATENT_DIM, N_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    with mlflow.start_run(run_name="lstm_autoencoder") as run:
        print("Training Autoencoder...")
        model.train()
        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            mean_epoch_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("train_loss", mean_epoch_loss, step=epoch)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch [{epoch}/{EPOCHS}], Loss: {mean_epoch_loss:.6f}")
                
        print("Computing training errors...")
        model.eval()
        with torch.no_grad():
            X_train_gpu = X_train_tensor.to(device)
            train_preds = model(X_train_gpu)
            train_errors = torch.mean((train_preds - X_train_gpu)**2, dim=(1, 2)).cpu().numpy()
            
        print("Tuning threshold using ThresholdEngine...")
        scaler_path = processed_dir / "scaler.pkl"
        engine = ThresholdEngine(model, device, scaler_path)
        
        # Test multipliers from 1.0 to 4.0 in steps of 0.1
        multipliers = np.arange(1.0, 4.1, 0.1).tolist()
        best_multiplier, best_threshold, metrics_df = engine.find_best_threshold(
            X_test, y_test, train_errors, multipliers
        )
        
        # Extract metrics for best threshold
        best_row = metrics_df[metrics_df["multiplier"] == best_multiplier].iloc[0]
        precision = float(best_row["precision"])
        recall = float(best_row["recall"])
        f1 = float(best_row["f1"])
        
        print(f"Best Multiplier: {best_multiplier:.2f}")
        print(f"Best Threshold: {best_threshold:.6f}")
        print(f"F1 Score: {f1:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
        
        print("Saving model and threshold locally...")
        model_path = processed_dir / "autoencoder.pth"
        torch.save(model.state_dict(), model_path)
        
        threshold_path = processed_dir / "threshold.npy"
        np.save(threshold_path, best_threshold)
        
        # Log params & metrics to MLflow
        mlflow.log_params({
            "latent_dim": LATENT_DIM,
            "n_layers": N_LAYERS,
            "window_size": WINDOW_SIZE,
            "n_features": N_FEATURES,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "best_multiplier": best_multiplier,
            "device": str(device)
        })
        
        mlflow.log_metrics({
            "threshold": float(best_threshold),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
        # Log model artifacts & register the model in the MLflow Model Registry
        print("Logging PyTorch model and registering to Model Registry...")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="LSTM_Autoencoder"
        )
        
        # Log threshold and scaler files as artifacts
        mlflow.log_artifact(str(threshold_path))
        mlflow.log_artifact(str(scaler_path))
        
        print("Logged run to MLflow and registered Model successfully.")

if __name__ == "__main__":
    train()
