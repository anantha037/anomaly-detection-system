import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path
import os

def create_windows(data, labels=None, window_size=30, stride=1):
    windows = []
    window_labels = []
    
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i : i + window_size])
        if labels is not None:
            # A window is anomalous if any timestep in it is anomalous
            window_labels.append(1 if np.any(labels[i : i + window_size]) else 0)
            
    if labels is not None:
        return np.array(windows), np.array(window_labels)
    return np.array(windows)

def preprocess():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading training data (anomaly-free)...")
    train_df = pd.read_csv(raw_dir / "anomaly-free.csv", sep=";")
    train_df = train_df.drop(columns=["datetime", "anomaly", "changepoint"], errors="ignore")
    train_data = train_df.values
    
    print(f"Original train shape: {train_data.shape}")
    
    print("Loading test data (valve1 and valve2)...")
    test_dfs = []
    for valve, count in [("valve1", 4), ("valve2", 3)]:
        for i in range(1, count + 1):
            filepath = raw_dir / f"{valve}_{i}.csv"
            df = pd.read_csv(filepath, sep=";")
            test_dfs.append(df)
            
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Keep anomaly column for labels
    labels = test_df["anomaly"].values
    
    test_df = test_df.drop(columns=["datetime", "anomaly", "changepoint"], errors="ignore")
    test_data = test_df.values
    
    print(f"Original test shape: {test_data.shape}")
    print(f"Original labels shape: {labels.shape}")
    
    print("Scaling data...")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    joblib.dump(scaler, processed_dir / "scaler.pkl")
    print(f"Saved scaler to {processed_dir / 'scaler.pkl'}")
    
    window_size = int(os.getenv("WINDOW_SIZE", 30))
    stride = int(os.getenv("STRIDE", 1))
    
    print(f"Applying sliding window (size={window_size}, stride={stride})...")
    X_train = create_windows(train_scaled, window_size=window_size, stride=stride)
    X_test, y_test = create_windows(test_scaled, labels=labels, window_size=window_size, stride=stride)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "X_test.npy", X_test)
    np.save(processed_dir / "y_test.npy", y_test)
    print(f"Saved windowed data to {processed_dir}")

if __name__ == "__main__":
    preprocess()
