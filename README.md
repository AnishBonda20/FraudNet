# FraudNet — Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end fraud detection platform that combines a stacking ensemble of gradient boosting models with SHAP explainability and a full-stack web interface.

> Built as a B.Tech final-year project at VIT — School of Computer Science and Engineering.

---

## Key Results

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| XGBoost | 0.8197 | 0.8571 | 0.8380 | 0.8221 |
| LightGBM | 0.8088 | 0.8673 | 0.8370 | 0.8334 |
| CatBoost | 0.8354 | 0.8571 | 0.8461 | 0.8267 |
| **Stacking Ensemble** | **0.8534** | **0.8878** | **0.8703** | **0.8499** |

- Catches **88.78%** of all fraudulent transactions
- Average inference latency: **~180ms**
- Dataset: Kaggle ULB (284,807 transactions, 0.172% fraud)

---

## Tech Stack

**Backend:** Python 3.11 · FastAPI · SQLAlchemy · SQLite · XGBoost · LightGBM · CatBoost · SHAP · scikit-learn

**Frontend:** React 18 · Vite · Recharts · React Router · Axios

---

## Project Structure

```
FraudNet/
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI routes (11 endpoints)
│   │   ├── auth.py         # JWT + bcrypt authentication
│   │   ├── predictor.py    # Ensemble inference + SHAP
│   │   ├── schemas.py      # Pydantic request/response models
│   │   ├── database.py     # SQLAlchemy ORM (SQLite)
│   │   └── config.py       # Environment config via .env
│   ├── ml/
│   │   ├── train.py        # Model training script
│   │   ├── preprocess.py   # Feature engineering + SMOTEENN
│   │   └── models/         # Serialized .pkl files (post-training)
│   ├── data/               # Place creditcard.csv here (not tracked)
│   ├── requirements.txt
│   └── run.py
└── frontend/
    ├── src/
    │   ├── pages/          # Dashboard, Predict, Analyze, Models, History
    │   ├── components/     # Layout, Navbar, LogoIcon
    │   └── api.js          # Axios client with JWT interceptor
    ├── package.json
    └── vite.config.js
```

---

## Getting Started

### 1 — Dataset

Download the Kaggle Credit Card Fraud dataset and place it at:

```
backend/data/creditcard.csv
```

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

### 2 — Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env — set a strong SECRET_KEY and optionally add SMTP credentials

# Train the models (takes ~5–10 minutes)
python ml/train.py --data data/creditcard.csv

# Start the API server (port 8001)
python run.py
```

API: http://localhost:8001  
Swagger docs: http://localhost:8001/docs

---

### 3 — Frontend

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:5173

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | /auth/register | — | Register with OTP verification |
| POST | /auth/verify-otp | — | Verify OTP and activate account |
| POST | /auth/login | — | Login, receive JWT token |
| GET | /auth/me | JWT | Current user info |
| POST | /predict | JWT | Analyse transaction + SHAP |
| GET | /history | JWT | Prediction history (last 50) |
| GET | /stats | JWT | Dashboard statistics |
| GET | /models | JWT | Model comparison metrics |
| GET | /health | — | Health check |

---

## ML Pipeline

### Models
| Model | Notes |
|-------|-------|
| XGBoost | 400 trees, depth 6, eval_metric=aucpr |
| LightGBM | Fast GBDT, class_weight="balanced" |
| CatBoost | Auto class weights, 500 iterations |
| Stacking Ensemble | XGB + LGB + CAT → Logistic Regression meta-learner (5-fold CV) |

### Resampling
**SMOTEENN** — SMOTE synthetic oversampling + Edited Nearest Neighbours cleaning. Addresses the 578:1 class imbalance in the raw dataset.

### Explainability
**SHAP TreeExplainer** — per-prediction feature attributions showing exactly which features pushed the model toward fraud or legitimate.

### Evaluation Metric
Primary: **PR-AUC** (not ROC-AUC) — with only 0.17% fraud rate, ROC-AUC is misleadingly optimistic. PR-AUC directly measures performance on the minority class.

---

## Features

The model uses 18 engineered features:

- `amount` — transaction amount (RobustScaler normalized)
- `hour_of_day` — extracted from raw Time column (0–23)
- `day_of_week` — extracted from raw Time column (0–6)
- `v1, v2, v3, v4, v7, v9, v10, v11, v12, v14, v16, v17, v18, v19, v20` — top PCA components by importance

---

## Security

- **JWT** (HS256, 60-min expiry) for all protected endpoints
- **bcrypt** password hashing
- **OTP email verification** on registration (Gmail SMTP / console fallback for dev)
- **AES-256-GCM** client-side encryption of raw transaction inputs before API call
