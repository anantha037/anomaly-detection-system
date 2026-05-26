# AnomalyGuard: Multivariate Time Series Anomaly Detection

> A production-grade anomaly detection system using PyTorch LSTM Autoencoders, FastAPI, and Plotly Dash.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/mlflow-%23d9ea3f.svg?style=flat&logo=mlflow&logoColor=black)](https://mlflow.org/)
[![Pre-Commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## 1. Overview
AnomalyGuard is a robust, production-ready time series anomaly detection platform built on the Skoltech Anomaly Benchmark (SKAB). It implements an LSTM-based Autoencoder to reconstruct multi-sensor telemetry readings. Anomalies are flagged dynamically when the reconstruction error (MSE) exceeds a statistical threshold optimized via F1-score tuning.

### Core Architecture Flow

```text
       Raw Data (SKAB)
             │
             ▼
      ┌───────────────┐
      │  Preprocess   │  <─── Pandera Schema Quality Guard (Coercion, shape, type checks)
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │  Autoencoder  │  <─── PyTorch LSTM Encoder-Decoder Model
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │ Threshold Eng │  <─── Optimal multiplier selection via F1 tuning
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │ FastAPI Server│  <─── Live HTTP predictions, health, metrics (Loguru JSON logging)
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │  Plotly Dash  │  <─── Frontend visualization, alert statistics, telemetry charts
      └───────────────┘
```

---

## 2. Tech Stack & MLOps Standards

* **Deep Learning Framework:** PyTorch (LSTM Autoencoder sequence-to-sequence model).
* **Data Validation:** Pandera (enforces strict types, bounds, and column requirements on telemetry ingestion).
* **Experiment Tracking & Registry:** MLflow (logs hyperparams, epoch loss, threshold results, and version-controls models).
* **Serving Layer:** FastAPI powered by Uvicorn.
* **Telemetry UI:** Interactive Plotly Dash with Bootstrap styling.
* **Structured Logging:** Loguru (outputs colorized logs for development and clean JSON format logs for production aggregators).
* **Linting & Code Quality:** Ruff static analysis and formatting configuration.
* **CI/CD Pipeline:** GitHub Actions workflow executing formatting, linting, and unit tests on commit.

---

## 3. Setup and Installation

### Option A: Running via Docker (Recommended)
Launch the entire containerized stack using Docker Compose:
```bash
# 1. Build and run the containers in detached mode
docker-compose up -d --build

# 2. Verify both containers are up
docker-compose ps
```
* **Dashboard Interface:** [http://localhost:8050](http://localhost:8050)
* **API Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option B: Local Development Setup
Ensure Python 3.11+ is installed on your machine.

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Install and Initialize Pre-commit Hooks
Register the git hook configuration to execute before git commits:
```bash
pre-commit install
```

#### 3. Run Preprocessing & Training
Prepare the dataset (validating against Pandera) and train the Autoencoder tracking with MLflow:
```bash
# Ingest and pre-validate raw dataset
python src/data/preprocess.py

# Train the LSTM model, optimize threshold, and register model
python src/models/train_autoencoder.py
```

#### 4. Launch the Applications
Run each command in a separate terminal:
```bash
# Start FastAPI backend
python src/api/main.py

# Start Dash frontend (local config checks API_URL env)
python dashboard/app.py
```

---

## 4. API Documentation & Prediction Instructions

### Endpoints Overview

| Method | Endpoint | Description | Sample Response |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Ingests server dependencies and tells status | `{"status": "ok", "model_loaded": true, ...}` |
| `GET` | `/threshold` | Fetches the optimal computed MSE threshold | `{"threshold": 0.750671, "multiplier": 1.0}` |
| `GET` | `/metrics` | Returns model accuracy metrics on test data | `{"f1_score": 0.7397, "precision": 0.7295, ...}` |
| `POST` | `/predict` | Predicts anomalies on a batch of windows | `{"results": [{"is_anomaly": false, ...}]}` |

---

### How to Make Predictions (`POST /predict`)

To submit batches of telemetry windows to the model, send a `POST` request to `http://localhost:8000/predict` with a JSON payload of shape `[N_windows, 30, 8]`. 

Each window consists of **30 timesteps** across **8 telemetry features** in this order:
`[Accelerometer1RMS, Accelerometer2RMS, Current, Pressure, Temperature, Thermocouple, Voltage, Volume Flow RateRMS]`

#### Test Request (using cURL):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "windows": [
      [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
        [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
        [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3],
        [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
        [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
        [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
        [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
        [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
        [2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        [2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1],
        [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2],
        [2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3],
        [2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4],
        [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
        [2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
        [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7]
      ]
    ]
  }'
```

#### Desired Output:
```json
{
  "results": [
    {
      "window_id": 0,
      "reconstruction_error": 0.4281,
      "is_anomaly": false,
      "severity": "NORMAL"
    }
  ],
  "anomaly_count": 0,
  "total_windows": 1,
  "alert_rate": 0.0
}
```

---

## 5. Verification & Testing

Verify that your local changes run cleanly by invoking the testing suite:

```bash
# Run pytest test suite
pytest

# Run pre-commit checks manually across files
pre-commit run --all-files
```

---

## 6. Benchmarks & Results
The LSTM Autoencoder drastically outperforms traditional baseline models on the SKAB dataset:

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| **LSTM Autoencoder** | **0.7295** | **0.7502** | **0.7397** |
| **Isolation Forest** | 0.3698 | 1.0000 | 0.5399 |

*Note: Isolation Forest's CONTAMINATION parameter defaults assume a high recall profile but cause high false-alarm rates (low precision).*
