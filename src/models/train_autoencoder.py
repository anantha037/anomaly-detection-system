import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import os
import sys

# Add src to path so we can import autoencoder
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.autoencoder import LSTMAutoencoder

def train():
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMAutoencoder(N_FEATURES, LATENT_DIM, N_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.6f}")
            
    print("Computing threshold...")
    model.eval()
    with torch.no_grad():
        X_train_gpu = X_train_tensor.to(device)
        train_preds = model(X_train_gpu)
        # Compute mean MSE per window
        train_errors = torch.mean((train_preds - X_train_gpu)**2, dim=(1, 2)).cpu().numpy()
        
    threshold = np.mean(train_errors) + 2 * np.std(train_errors)
    print(f"Threshold computed: {threshold:.6f}")
    
    print("Saving model and threshold...")
    model_path = processed_dir / "autoencoder.pth"
    torch.save(model.state_dict(), model_path)
    
    threshold_path = processed_dir / "threshold.npy"
    np.save(threshold_path, threshold)
    
    print(f"Saved model to {model_path}")
    print(f"Saved threshold to {threshold_path}")

if __name__ == "__main__":
    train()
