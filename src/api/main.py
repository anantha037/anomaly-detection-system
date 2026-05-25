import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Ensure src is in the path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.api.dependencies import lifespan
from src.api.routes import router

app = FastAPI(
    title="Anomaly Detection API",
    version="1.0.0",
    description="LSTM Autoencoder based multivariate time series anomaly detection",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
