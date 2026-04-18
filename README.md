---
title: Credit Card Fraud Detection
emoji: 🛡️
colorFrom: indigo
colorTo: red
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# 🛡️ Credit Card Fraud Detection System

> End-to-end ML system: EDA → SMOTE → Model Comparison → FastAPI → Streamlit Dashboard

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.ai)
[![React](https://img.shields.io/badge/React-19-blue?logo=react)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Vite](https://img.shields.io/badge/Vite-8-purple?logo=vite)](https://vite.dev)
[![HF Spaces](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/fraud-detection)

---

## 📋 Problem Statement

Credit card fraud causes **over $30 billion in annual losses** globally. With only **0.17% of transactions being fraudulent**, a naive model that labels everything as "legitimate" achieves 99.83% accuracy — yet catches **zero fraud**.

This project demonstrates how to:
- Handle extreme class imbalance using **SMOTE**
- Evaluate models on **Recall** (catching fraud) rather than raw accuracy
- Deploy a real-time scoring API + interactive dashboard

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total Transactions | 284,807 |
| Fraudulent | 492 (0.17%) |
| Features | V1–V28 (PCA), Time, Amount |
| Missing Values | None |

> Download `creditcard.csv` from Kaggle and place it in the `data/` directory.

---

## 🛠️ Approach

### 1. EDA
- Class imbalance visualization
- Amount & Time distribution (fraud vs legit)
- Correlation analysis (V14, V12, V10 most negatively correlated with fraud)

### 2. Preprocessing
- **StandardScaler** on `Amount` and `Time` (V1–V28 are already PCA-transformed)
- **80/20 stratified** train-test split
- **SMOTE** applied **only on training data** (prevents data leakage)

### 3. Model Training
Trained 3 models × 2 conditions (before/after SMOTE):

| Model | Recall Before SMOTE | Recall After SMOTE | F1 After SMOTE |
|---|---|---|---|
| Logistic Regression | ~58% | ~91% | ~0.88 |
| Random Forest | ~75% | ~89% | ~0.90 |
| **XGBoost ✅** | ~80% | **~92%** | **~0.91** |

### 4. Key Finding
> SMOTE improved fraud recall by **+12–33%** across all models. XGBoost with SMOTE achieved the best overall performance.

---

## 🏗️ Project Structure

```
fraud-detection/
├── data/
│   └── creditcard.csv          # Download from Kaggle
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory analysis
│   └── 02_modeling.ipynb       # Training + comparison
├── src/
│   ├── preprocess.py           # Scaling + SMOTE pipeline
│   ├── train.py                # Train all 3 models
│   └── predict.py              # Prediction utility
├── api/
│   ├── main.py                 # FastAPI (serves React build in prod)
│   └── schemas.py              # Pydantic models
├── client/                     # React Frontend
│   ├── src/                    # App logic (Lucide-React + Recharts)
│   ├── dist/                   # Built production files
│   └── package.json
├── models/                     # Saved PKLs + metrics.json
├── tests/
│   ├── test_preprocess.py      # pytest unit tests
│   └── test_api.py             # FastAPI integration tests
├── plots/                      # Auto-generated EDA plots
├── Dockerfile                  # Multi-stage build (Node + Python)
└── requirements.txt
```

---

## 🚀 How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Place `creditcard.csv` in the `data/` folder (download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).

### 3. Run notebooks (EDA + Modeling)
```bash
jupyter lab notebooks/
```
Open `01_eda.ipynb` then `02_modeling.ipynb` and run all cells.

### 4. Or train directly via script
```bash
python -m src.train
```

### 5. Start the FastAPI backend
```bash
uvicorn api.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### 6. Start the React dashboard (Dev mode)
```bash
cd client
npm install
npm run dev
```
Dashboard: http://localhost:5173

### 7. Run tests
```bash
pytest tests/ -v
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/stats` | Dataset + model stats |
| POST | `/predict` | Score a single transaction |
| POST | `/predict/batch` | Score multiple transactions |

### Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 406, "V1": -2.31, "V2": 1.95, ..., "Amount": 149.62}'
```

### Example response
```json
{
  "prediction": "FRAUD",
  "confidence": 0.9821,
  "risk_level": "HIGH"
}
```

---

## 🐳 Docker (Optional)

```bash
docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection
```

---

## 🤗 Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Set SDK to **Streamlit**
3. Push this repo (include trained model files — use Git LFS for pkl files)
4. HF Spaces will auto-detect `app.py` and run the dashboard

---

## 💬 Interview Talking Points

- *"I handled class imbalance using SMOTE and showed a 12–33% recall improvement across all models"*
- *"I compared 3 models and chose XGBoost based on recall, not just accuracy"*
- *"I deployed a real-time scoring API that returns predictions in milliseconds"*
- *"I built an interactive dashboard with a threshold tuner to visualize the precision-recall tradeoff live"*
- *"I wrote pytest unit tests to validate no data leakage — SMOTE is applied only on training data"*

---

## 📄 License

MIT © 2024

