# AnomalyGuard: Multivariate Time Series Anomaly Detection

> A production-grade anomaly detection system using PyTorch LSTM Autoencoders, FastAPI, and Plotly Dash.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/mlflow-%23d9ea3f.svg?style=flat&logo=mlflow&logoColor=black)

## Overview
This project implements a complete end-to-end pipeline for detecting anomalies in multivariate time series data. It uses an LSTM-based Autoencoder to reconstruct sensor readings. Anomalies are flagged when the reconstruction error exceeds an optimal dynamic threshold.

## Architecture

```text
Raw Data (SKAB)
      │
      ▼
┌───────────────┐
│ Preprocessor  │ (Windowing, MinMaxScaler)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Autoencoder   │ (PyTorch LSTM, 2 Layers)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Threshold Eng │ (Tuning Reconstruction Error Multiplier)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ FastAPI       │ (Serving Layer, Batch Predictions)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Plotly Dash   │ (Interactive UI, Diagnostics)
└───────────────┘
```

## Tech Stack
| Component | Technology |
|---|---|
| **Deep Learning** | PyTorch |
| **Baseline Model** | Scikit-learn (Isolation Forest) |
| **Experiment Tracking** | MLflow |
| **API Serving** | FastAPI, Uvicorn |
| **Dashboard** | Plotly Dash, Bootstrap |
| **Containerization** | Docker, Docker Compose |

## Project Structure
```text
anomaly-detection-system/
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
├── src/
│   ├── api/          # FastAPI application
│   ├── data/         # Download and preprocessing scripts
│   ├── detection/    # Threshold tuning and alert system
│   └── models/       # LSTM Autoencoder & Isolation Forest
├── dashboard/        # Interactive Plotly Dash frontend
├── data/
│   └── processed/    # Local volume for models and processed data
├── docker-compose.yml
└── requirements.txt
```

## Quickstart (Docker)
The easiest way to run the entire stack is via Docker Compose.

1. Ensure `.env` is properly configured (you can copy `.env.example`).
2. Build and start the containers:
```bash
docker-compose up -d --build
```
3. Access the services:
   - Dashboard: http://localhost:8050
   - FastAPI Docs: http://localhost:8000/docs

## Quickstart (Local)
To run the services directly on your host machine:

1. Start the API:
```bash
python src/api/main.py
```
2. Start the Dashboard (in a separate terminal):
```bash
python dashboard/app.py
```

## Results
The LSTM Autoencoder drastically outperforms traditional baseline models on the SKAB dataset:

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| **LSTM Autoencoder** | 0.9317 | 0.6441 | **0.7616** |
| **Isolation Forest** | 0.3698 | 1.0000 | 0.5399 |

## Dataset
This project uses the [Skoltech Anomaly Benchmark (SKAB)](https://github.com/waico/SKAB), designed for evaluating algorithms for anomaly detection in multivariate time series from physical systems.
