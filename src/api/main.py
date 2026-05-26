import sys
import time
from pathlib import Path

# ruff: noqa: E402
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Ensure src is in the path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Setup structured logging
from src.api.logging_config import setup_logging

setup_logging()

from src.api.dependencies import lifespan
from src.api.routes import router

app = FastAPI(
    title="Anomaly Detection API",
    version="1.0.0",
    description="LSTM Autoencoder based multivariate time series anomaly detection",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000

    logger.bind(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round(process_time, 2),
        client_ip=request.client.host if request.client else "unknown",
    ).info(
        f"Request handled: {request.method} {request.url.path} -> Status {response.status_code} ({process_time:.2f}ms)"
    )

    return response


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
