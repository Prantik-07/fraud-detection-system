"""
main.py — FastAPI backend for Credit Card Fraud Detection.

Run: uvicorn api.main:app --reload --port 8000
"""

import os, sys, json, time, logging
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    Transaction, PredictionResponse,
    BatchRequest, BatchResponse,
    HealthResponse, StatsResponse,
)
from fastapi.staticfiles import StaticFiles
from src.predict import load_model_and_scaler, predict_one

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fraud_api")

_model = None
_scaler = None
_metrics_data = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _scaler, _metrics_data
    logger.info("Loading model and scaler …")
    try:
        _model, _scaler = load_model_and_scaler()
        logger.info("Model loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}. Run `python -m src.train` first.")

    metrics_path = os.path.join("models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _metrics_data = json.load(f)
    yield
    logger.info("Shutting down …")


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud scoring via XGBoost + SMOTE.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    return {
        "status": "ok" if _model else "model_not_loaded",
        "model": type(_model).__name__ if _model else "none",
        "version": "1.0.0",
    }


@app.get("/metrics", tags=["Meta"])
def metrics():
    if not _metrics_data:
        raise HTTPException(503, "Metrics not loaded. Run training first.")
    return _metrics_data


@app.get("/stats", response_model=StatsResponse, tags=["Meta"])
def stats():
    if not _metrics_data:
        raise HTTPException(503, "Metrics not loaded. Run training first.")
    best_name = _metrics_data.get("best_model", "XGBoost")
    best = next(
        (m for m in _metrics_data["metrics"] if m["model"] == best_name and m["condition"] == "after_smote"), {}
    )
    return {
        "total_transactions": 284_807,
        "fraud_transactions": 492,
        "fraud_rate": round(492 / 284_807, 6),
        "best_model": best_name,
        "best_model_f1": best.get("f1", 0.0),
        "best_model_recall": best.get("recall", 0.0),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(transaction: Transaction):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")
    t0 = time.perf_counter()
    result = predict_one(transaction.to_feature_list(), model=_model, scaler=_scaler)
    logger.info(f"PREDICT | amount={transaction.Amount:.2f} | {result['prediction']} | {result['confidence']:.4f} | {(time.perf_counter()-t0)*1000:.1f}ms")
    return result


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(batch: BatchRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")
    results = [predict_one(t.to_feature_list(), model=_model, scaler=_scaler) for t in batch.transactions]
    fraud_count = sum(1 for r in results if r["prediction"] == "FRAUD")
    logger.info(f"BATCH | size={len(results)} | fraud={fraud_count}")
    return {"results": results, "total": len(results), "fraud_count": fraud_count}


# ── Serve React Frontend ──────────────────────────────────────────────────────
# Mount the React build directory if it exists
dist_path = os.path.join(os.path.dirname(__file__), "..", "client", "dist")
if os.path.exists(dist_path):
    app.mount("/", StaticFiles(directory=dist_path, html=True), name="static")
else:
    logger.warning(f"Static files directory not found at {dist_path}. React frontend will not be served.")
