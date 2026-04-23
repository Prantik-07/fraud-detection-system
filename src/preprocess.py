"""
preprocess.py — Data loading, scaling, splitting, and SMOTE resampling.
Run this module from the project root:
    python -m src.preprocess
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "creditcard.csv")
MODEL_DIR   = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Load ───────────────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the Credit Card Fraud CSV and return as DataFrame."""
    print(f"[preprocess] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"[preprocess] Shape: {df.shape}")
    print(f"[preprocess] Fraud rate: {df['Class'].mean():.4%}")
    return df


# ── Scale ──────────────────────────────────────────────────────────────────────
def scale_features(df: pd.DataFrame, fit: bool = True, scaler=None):
    """
    Scale 'Amount' and 'Time' with StandardScaler.
    V1–V28 are already PCA-transformed — leave them as-is.

    Args:
        df:     Input DataFrame.
        fit:    If True, fit a new scaler and save it. Else transform only.
        scaler: Pre-fitted scaler (used when fit=False).

    Returns:
        Scaled DataFrame, fitted/provided scaler.
    """
    df = df.copy()

    if fit:
        scaler = StandardScaler()
        df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])
        joblib.dump(scaler, SCALER_PATH, protocol=5)
        print(f"[preprocess] Scaler saved → {SCALER_PATH}")
    else:
        assert scaler is not None, "Pass a fitted scaler when fit=False"
        df[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])

    return df, scaler


# ── Split ──────────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified 80/20 train-test split.

    Returns:
        X_train, X_test, y_train, y_test (all as NumPy arrays)
    """
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[preprocess] Train size : {X_train.shape[0]:,} samples")
    print(f"[preprocess] Test size  : {X_test.shape[0]:,} samples")
    print(f"[preprocess] Train fraud: {y_train.sum():,} ({y_train.mean():.4%})")
    return X_train, X_test, y_train, y_test


# ── SMOTE ──────────────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE **only on training data** to avoid data leakage.

    Returns:
        X_resampled, y_resampled
    """
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print(f"[preprocess] After SMOTE → {X_res.shape[0]:,} samples")
    print(f"[preprocess] Fraud after SMOTE: {y_res.sum():,} ({y_res.mean():.4%})")
    return X_res, y_res


# ── Full pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(path: str = DATA_PATH):
    """
    End-to-end preprocessing pipeline.

    Returns:
        X_train, X_test, y_train, y_test,          ← raw (no SMOTE)
        X_train_res, y_train_res,                   ← after SMOTE
        scaler
    """
    df = load_data(path)
    df, scaler = scale_features(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    return X_train, X_test, y_train, y_test, X_train_res, y_train_res, scaler


if __name__ == "__main__":
    run_pipeline()
