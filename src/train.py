"""
train.py — Train Logistic Regression, Random Forest, and XGBoost.
Evaluates before and after SMOTE, saves all models + metrics.json.

Run from project root:
    python -m src.train
"""

import os
import json
import time
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)
from xgboost import XGBClassifier

from src.preprocess import run_pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = "models"
PLOTS_DIR   = "plots"
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Model definitions ─────────────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=577,  # ~legit/fraud ratio
            eval_metric="logloss", random_state=42,
            use_label_encoder=False
        ),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, model_name: str, condition: str) -> dict:
    """Compute full metrics and return as dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = {
        "model":     model_name,
        "condition": condition,       # "before_smote" | "after_smote"
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": cm,
        "roc_fpr":   fpr.tolist(),
        "roc_tpr":   tpr.tolist(),
    }

    print(f"\n{'─'*55}")
    print(f"  {model_name}  [{condition}]")
    print(f"{'─'*55}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}  ← key for fraud")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix:\n  {np.array(cm)}")

    return metrics


# ── Save confusion matrix plot ─────────────────────────────────────────────────
def save_cm_plot(cm, model_name, condition):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name}\n[{condition}]")
    thresh = np.array(cm).max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i][j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black")
    plt.tight_layout()
    fname = f"{model_name.replace(' ', '_').lower()}_{condition}_cm.png"
    fpath = os.path.join(PLOTS_DIR, fname)
    plt.savefig(fpath, dpi=120)
    plt.close()
    return fpath


# ── Save ROC curve ────────────────────────────────────────────────────────────
def save_roc_plot(all_metrics):
    """Overlay ROC curves for all after-SMOTE models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"Logistic Regression": "#e74c3c",
              "Random Forest": "#2ecc71",
              "XGBoost": "#3498db"}

    for m in all_metrics:
        if m["condition"] == "after_smote":
            ax.plot(m["roc_fpr"], m["roc_tpr"],
                    label=f"{m['model']} (AUC={m['roc_auc']:.3f})",
                    color=colors.get(m["model"], "gray"), lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — After SMOTE", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"\n[train] ROC curve saved → {fpath}")


# ── Main training loop ────────────────────────────────────────────────────────
def train_all():
    print("\n[train] Loading and preprocessing data …")
    (X_train, X_test, y_train, y_test,
     X_train_res, y_train_res, _) = run_pipeline()

    all_metrics = []
    best_model  = None
    best_f1     = 0.0
    best_name   = ""

    for name, model in get_models().items():
        # ── Before SMOTE ──────────────────────────────────────────────────────
        print(f"\n[train] Training {name} WITHOUT SMOTE …")
        t0 = time.time()
        model.fit(X_train, y_train)
        print(f"[train] Done in {time.time()-t0:.1f}s")
        m = evaluate(model, X_test, y_test, name, "before_smote")
        save_cm_plot(m["confusion_matrix"], name, "before_smote")
        all_metrics.append(m)

        # ── After SMOTE ───────────────────────────────────────────────────────
        print(f"\n[train] Training {name} WITH SMOTE …")
        model_smote = get_models()[name]  # fresh instance
        t0 = time.time()
        model_smote.fit(X_train_res, y_train_res)
        print(f"[train] Done in {time.time()-t0:.1f}s")
        m_s = evaluate(model_smote, X_test, y_test, name, "after_smote")
        save_cm_plot(m_s["confusion_matrix"], name, "after_smote")
        all_metrics.append(m_s)

        # Save all models
        fname = name.replace(" ", "_").lower() + "_model.pkl"
        joblib.dump(model_smote, os.path.join(MODEL_DIR, fname))
        print(f"[train] Saved {fname}")

        # Track best model
        if m_s["f1"] > best_f1:
            best_f1    = m_s["f1"]
            best_model = model_smote
            best_name  = name

    # ── Save best model ────────────────────────────────────────────────────────
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"), protocol=5)
    print(f"\n[train] ✅ Best model: {best_name} (F1={best_f1:.4f})")
    print(f"[train] Saved → models/best_model.pkl")

    # ── Save test probabilities for dashboard threshold tuner ──────────────────
    test_probs  = best_model.predict_proba(X_test)[:, 1]
    np.save(os.path.join(MODEL_DIR, "test_probs.npy"),  test_probs)
    np.save(os.path.join(MODEL_DIR, "test_labels.npy"), y_test)
    print(f"[train] Test probs/labels saved → models/test_probs.npy, test_labels.npy")

    # ── Save metrics ───────────────────────────────────────────────────────────
    # Strip large arrays (fpr/tpr) before saving to keep JSON small
    metrics_for_json = []
    for m in all_metrics:
        entry = {k: v for k, v in m.items() if k not in ("roc_fpr", "roc_tpr")}
        metrics_for_json.append(entry)

    with open(METRICS_PATH, "w") as f:
        json.dump({"metrics": metrics_for_json, "best_model": best_name}, f, indent=2)
    print(f"[train] Metrics saved → {METRICS_PATH}")

    # ── ROC plot ───────────────────────────────────────────────────────────────
    save_roc_plot(all_metrics)

    return all_metrics, best_model


if __name__ == "__main__":
    train_all()
