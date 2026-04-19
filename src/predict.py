"""
predict.py — Standalone prediction utility (no API dependency).
Load the saved model and scaler, score a single transaction or a CSV.

Usage:
    python -m src.predict --amount 150.0 --time 40000 --v1 -1.36 ...
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("models", "best_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

V_COLS  = [f"V{i}" for i in range(1, 29)]
ALL_COLS = ["Time"] + V_COLS + ["Amount"]   # 30 features total


# ── Load artefacts ────────────────────────────────────────────────────────────
def load_model_and_scaler():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ── Risk level helper ─────────────────────────────────────────────────────────
def get_risk_level(prob: float) -> str:
    if prob < 0.40:
        return "LOW"
    elif prob < 0.70:
        return "MEDIUM"
    else:
        return "HIGH"


# ── Single prediction ─────────────────────────────────────────────────────────
def predict_one(features: list, model=None, scaler=None) -> dict:
    """
    Predict a single transaction.

    Args:
        features: List of 30 floats → [Time, V1..V28, Amount]
        model:    Pre-loaded model (optional — loads from disk if None)
        scaler:   Pre-loaded scaler (optional)

    Returns:
        dict with keys: prediction, confidence, risk_level
    """
    if model is None or scaler is None:
        model, scaler = load_model_and_scaler()

    arr = np.array(features).reshape(1, -1)
    df  = pd.DataFrame(arr, columns=ALL_COLS)

    # Scale only Amount and Time
    df[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])

    X   = df.values
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return {
        "prediction":  "FRAUD" if pred == 1 else "LEGITIMATE",
        "confidence":  round(float(prob), 4),
        "risk_level":  get_risk_level(prob),
    }


# ── Batch prediction ──────────────────────────────────────────────────────────
def predict_batch(csv_path: str, output_path: str = "predictions.csv"):
    """Predict all transactions in a CSV and write results."""
    model, scaler = load_model_and_scaler()
    df = pd.read_csv(csv_path)

    # Drop Class if present
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    df_scaled = df.copy()
    df_scaled[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])

    preds = model.predict(df_scaled.values)
    probs = model.predict_proba(df_scaled.values)[:, 1]

    df["prediction"] = ["FRAUD" if p == 1 else "LEGITIMATE" for p in preds]
    df["confidence"] = np.round(probs, 4)
    df["risk_level"] = [get_risk_level(p) for p in probs]

    df.to_csv(output_path, index=False)
    print(f"[predict] Results saved → {output_path}")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud Detection — Single Prediction")
    parser.add_argument("--amount", type=float, default=100.0)
    parser.add_argument("--time",   type=float, default=50000.0)
    for i in range(1, 29):
        parser.add_argument(f"--v{i}", type=float, default=0.0)

    args = parser.parse_args()

    features = [args.time] + [getattr(args, f"v{i}") for i in range(1, 29)] + [args.amount]
    result = predict_one(features)

    print("\n" + "─"*40)
    print(f"  Prediction : {result['prediction']}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print(f"  Risk Level : {result['risk_level']}")
    print("─"*40)
