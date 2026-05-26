import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import os
import pandera.pandas as pa

# Define Pandera schema for data quality validation
SKAB_SCHEMA = pa.DataFrameSchema({
    "Accelerometer1RMS": pa.Column(float, coerce=True),
    "Accelerometer2RMS": pa.Column(float, coerce=True),
    "Current": pa.Column(float, coerce=True),
    "Pressure": pa.Column(float, coerce=True),
    "Temperature": pa.Column(float, coerce=True),
    "Thermocouple": pa.Column(float, coerce=True),
    "Voltage": pa.Column(float, coerce=True),
    "Volume Flow RateRMS": pa.Column(float, coerce=True),
    "anomaly": pa.Column(int, coerce=True, checks=pa.Check.isin([0, 1])),
    "changepoint": pa.Column(float, coerce=True, required=False)
})

def create_windows(data, labels=None, window_size=30, stride=1):
    windows = []
    window_labels = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
        if labels is not None:
            window_labels.append(1 if np.any(labels[i:i + window_size]) else 0)
    if labels is not None:
        return np.array(windows), np.array(window_labels)
    return np.array(windows)

def preprocess():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load all valve files, split into normal (train) and all (test)
    print("Loading valve data...")
    all_dfs = []
    for valve, count in [("valve1", 4), ("valve2", 3)]:
        for i in range(1, count + 1):
            df = pd.read_csv(raw_dir / f"{valve}_{i}.csv", sep=";")
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Validate data quality using Pandera
    print("Validating dataset quality...")
    combined = SKAB_SCHEMA.validate(combined)

    feature_cols = ["Accelerometer1RMS", "Accelerometer2RMS", "Current",
                    "Pressure", "Temperature", "Thermocouple",
                    "Voltage", "Volume Flow RateRMS"]

    labels = combined["anomaly"].values
    all_features = combined[feature_cols].values

    # Train = only normal rows (anomaly==0)
    normal_mask = labels == 0
    train_data = all_features[normal_mask]
    print(f"Normal (train) timesteps: {train_data.shape}")
    print(f"All (test) timesteps:     {all_features.shape}")
    print(f"Anomalous timesteps:      {labels.sum()} ({labels.mean()*100:.1f}%)")

    # Fit scaler ONLY on normal data
    print("\nFitting StandardScaler on normal data only...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(all_features)
    joblib.dump(scaler, processed_dir / "scaler.pkl")
    print(f"Saved scaler to {processed_dir / 'scaler.pkl'}")

    # Sliding windows
    window_size = int(os.getenv("WINDOW_SIZE", 30))
    stride = int(os.getenv("STRIDE", 1))
    print(f"\nApplying sliding window (size={window_size}, stride={stride})...")

    X_train = create_windows(train_scaled, window_size=window_size, stride=stride)
    X_test, y_test = create_windows(test_scaled, labels=labels,
                                     window_size=window_size, stride=stride)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(f"Anomalous windows in test: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "X_test.npy", X_test)
    np.save(processed_dir / "y_test.npy", y_test)
    print(f"\nSaved windowed data to {processed_dir}")

if __name__ == "__main__":
    preprocess()