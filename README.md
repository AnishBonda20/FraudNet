# FraudNet — Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-FF6600?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-FFD700)
![Status](https://img.shields.io/badge/Status-Active-00C853)

> An end-to-end fraud detection platform built on a stacking ensemble of gradient boosting models with SHAP explainability, JWT-secured REST API, and a React dashboard

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Endpoints](#api-endpoints)
- [ML Pipeline](#ml-pipeline)
- [Security](#security)
- [Why This Project](#why-this-project)

---

## Overview

Credit card fraud causes over **$32 billion** in annual losses globally. Traditional rule-based systems generate high false-positive rates and lack explainability. FraudNet addresses this with:

- A **stacking ensemble** of XGBoost + LightGBM + CatBoost → Logistic Regression meta-learner
- **SMOTEENN resampling** to handle extreme class imbalance (0.172% fraud rate)
- **SHAP TreeExplainer** for per-prediction feature attribution
- A **full-stack web application** with analyst-friendly dashboard and transaction analyzer
- **Production-grade authentication** — JWT + bcrypt + OTP email verification

---

## Key Results

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Cost ($) |
|---|---|---|---|---|---|---|
| XGBoost | 0.8197 | 0.8571 | 0.8380 | 0.9701 | 0.8221 | 6,105 |
| LightGBM | 0.8088 | 0.8673 | 0.8370 | 0.9712 | 0.8334 | 6,735 |
| CatBoost | 0.8354 | 0.8571 | 0.8461 | 0.9698 | 0.8267 | 5,920 |
| **Stacking Ensemble** | **0.8534** | **0.8878** | **0.8703** | **0.9760** | **0.8499** | **5,710** |

- Detects **88.78%** of all fraudulent transactions (Recall)
- **PR-AUC 0.8499** — best across all models
- Confusion matrix: TP=87, FP=42, FN=11, TN=56,822
- Average inference latency: **~180ms**
- Cost metric: FN × $500 + FP × $5 (real-world asymmetry)

---

## Features

### Web Application
- **Smart Transaction Analyzer** — user-friendly form with 18 inputs, AES-256-GCM client-side encryption, real-time risk gauge + SHAP bar chart
- **Dashboard** — fraud rate trends, total predictions, average risk score, live statistics
- **Model Comparison** — side-by-side metrics table for all 4 models
- **Prediction History** — last 50 predictions, sortable table
- **Algorithm Explorer** — visual walkthrough of the full ML pipeline
- **Batch Analysis** — analyze multiple transactions at once

### Backend
- 11 REST API endpoints with full Swagger docs
- JWT authentication (60-min expiry) + bcrypt password hashing
- OTP email verification via Gmail SMTP (console fallback for dev)
- SQLite persistence via SQLAlchemy ORM
- SHAP explanations returned with every prediction

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI 0.111 + Uvicorn (ASGI) |
| **ML Models** | XGBoost 2.0.3 · LightGBM 4.3.0 · CatBoost 1.2.5 |
| **ML Utilities** | scikit-learn 1.4.2 · imbalanced-learn · SHAP 0.45 · joblib |
| **Database** | SQLite via SQLAlchemy 2.0 |
| **Auth** | python-jose (JWT) · bcrypt · secrets (OTP) |
| **Frontend** | React 18.3 · Vite 5.3 · React Router 6.23 |
| **Charts** | Recharts 2.12 |
| **HTTP Client** | Axios 1.7 |

---

## Project Structure

```
FraudNet/
├── backend/
│   ├── app/
│   │   ├── main.py           # 11 FastAPI routes
│   │   ├── auth.py           # JWT + bcrypt helpers
│   │   ├── predictor.py      # Ensemble inference + SHAP
│   │   ├── schemas.py        # Pydantic request/response models
│   │   ├── database.py       # SQLAlchemy ORM — users, predictions tables
│   │   ├── otp.py            # OTP generation + Gmail SMTP
│   │   └── config.py         # Environment config via .env
│   ├── ml/
│   │   ├── train.py          # Full training pipeline
│   │   ├── preprocess.py     # Feature engineering + SMOTEENN + scaler
│   │   └── models/
│   │       ├── ensemble.pkl  # Trained StackingClassifier
│   │       ├── scaler.pkl    # Fitted RobustScaler
│   │       └── metrics.json  # Saved evaluation metrics
│   ├── data/                 # Place creditcard.csv here (gitignored)
│   ├── requirements.txt
│   ├── run.py                # Uvicorn entry point (port 8001)
│   └── reset_db.py           # Dev utility — clears all DB tables
└── frontend/
    ├── src/
    │   ├── pages/
    │   │   ├── Analyze.jsx       # Smart Transaction Analyzer
    │   │   ├── Dashboard.jsx     # Stats overview
    │   │   ├── Predict.jsx       # Raw 18-feature predict form
    │   │   ├── Models.jsx        # Model comparison table
    │   │   ├── History.jsx       # Prediction history
    │   │   ├── Algorithm.jsx     # ML pipeline visualizer
    │   │   ├── Batch.jsx         # Batch transaction analysis
    │   │   ├── Landing.jsx       # Public landing page
    │   │   ├── Login.jsx         # Login page
    │   │   └── Register.jsx      # Registration + OTP verification
    │   ├── components/
    │   │   ├── Layout.jsx        # App shell + sidebar navigation
    │   │   └── LogoIcon.jsx      # SVG logo component
    │   ├── api.js                # Axios instance with JWT interceptor
    │   ├── App.jsx               # Router + route guards
    │   └── index.css             # Global styles (dark navy theme)
    ├── package.json
    └── vite.config.js            # Vite + proxy to backend (port 8001)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git

### 1 — Clone the repo

```bash
git clone https://github.com/AnishBonda20/FraudNet.git
cd FraudNet
```

### 2 — Dataset

Download the Kaggle Credit Card Fraud dataset and place it at:

```
backend/data/creditcard.csv
```

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### 3 — Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set a strong SECRET_KEY (min 32 chars)
# Optionally add SMTP_EMAIL and SMTP_PASSWORD for OTP emails

# Train the models (takes ~5–10 minutes)
python ml/train.py --data data/creditcard.csv

# Start the API server
python run.py
```

API: http://localhost:8001  
Swagger docs: http://localhost:8001/docs

### 4 — Frontend

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:5173

> The Vite dev server proxies `/api/*` requests to `http://localhost:8001` automatically — no CORS configuration needed.

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/register` | — | Register with OTP email verification |
| `POST` | `/auth/verify-otp` | — | Verify OTP and activate account |
| `POST` | `/auth/login` | — | Login, receive JWT access token |
| `GET` | `/auth/me` | JWT | Get current user info |
| `POST` | `/predict` | JWT | Analyse transaction — returns verdict + confidence + SHAP |
| `GET` | `/predict/sample` | JWT | Get a random sample transaction |
| `GET` | `/history` | JWT | Last 50 predictions (sortable) |
| `GET` | `/stats` | JWT | Dashboard statistics |
| `GET` | `/models` | JWT | All 4 model comparison metrics |
| `GET` | `/compare` | JWT | Head-to-head model comparison |
| `GET` | `/health` | — | Health check |

---

## ML Pipeline

### 1 — Feature Engineering
18 features extracted from the raw Kaggle dataset:
- `amount` — RobustScaler normalized (resistant to outliers)
- `hour_of_day` — extracted from raw `Time` column (0–23)
- `day_of_week` — extracted from raw `Time` column (0–6)
- `v1, v2, v3, v4, v7, v9, v10, v11, v12, v14, v16, v17, v18, v19, v20` — top 15 PCA components selected by feature importance

### 2 — Resampling (SMOTEENN)
Raw class ratio: **578:1** (284,315 legitimate vs 492 fraud)

- **SMOTE** — generates synthetic minority samples via k-NN interpolation
- **ENN** — removes borderline majority samples near the decision boundary
- Result: balanced training set → improved recall without excessive false positives

### 3 — Stacking Ensemble
```
Base learners (5-fold CV):
  XGBoost  ──┐
  LightGBM ──┼──► [p_xgb, p_lgb, p_cat] ──► Logistic Regression ──► Final probability
  CatBoost ──┘         meta-features              meta-learner
```

### 4 — SHAP Explainability
TreeExplainer runs on the XGBoost base learner and returns per-feature SHAP values with every prediction — showing exactly why a transaction was flagged.

**Top fraud indicators:**
1. `v14` — strongest negative fraud signal
2. `v4` — high positive association
3. `v12` — discriminative PCA component
4. `v10` — moderate fraud signal
5. `amount` — transaction size matters

---

## Security

| Feature | Implementation |
|---|---|
| Authentication | JWT HS256, 60-minute expiry |
| Password storage | bcrypt (salted hash, never plaintext) |
| Registration | 6-digit OTP via Gmail SMTP, 10-min expiry |
| Input validation | Pydantic schemas on all endpoints |
| Client-side privacy | AES-256-GCM encryption of raw inputs before API call |
| Environment secrets | `.env` file (gitignored) — never hardcoded |

---

## Why This Project

| Challenge | Solution |
|---|---|
| 0.172% fraud rate (extreme imbalance) | SMOTEENN resampling |
| Single model limitations | Stacking ensemble — captures complementary patterns |
| Black-box predictions | SHAP TreeExplainer — per-prediction feature attribution |
| High cost of missed fraud | Cost-sensitive evaluation (FN × $500 + FP × $5) |
| End-to-end deployment gap | Full-stack FastAPI + React web application |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
