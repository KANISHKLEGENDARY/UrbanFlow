# URBANFLOW 🚕
### AI-Powered Ride Demand Prediction & Surge Pricing Intelligence — New York City

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)
![React](https://img.shields.io/badge/React-18-purple)
![Vite](https://img.shields.io/badge/Vite-7.3.1-blueviolet)
![Model](https://img.shields.io/badge/Model-XGBoost%20%2B%20LightGBM-orange)
![Score](https://img.shields.io/badge/Test%20R%C2%B2-0.8239-brightgreen)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL%20%2B%20SQLAlchemy-blue)

A full-stack, machine learning-powered prediction platform and pricing engine that forecasts 15-minute interval ride-hailing demand across NYC's 263 TLC zones, featuring route-aware graph routing via OSMnx, dynamic three-policy surge pricing, and SHAP-based model explainability.

---

## Table of Contents
* [Overview](#overview)
* [Problem Statement](#problem-statement)
* [Live Demo](#live-demo)
* [Features](#features)
* [System Architecture](#system-architecture)
* [Backend Pipeline](#backend-pipeline)
* [Machine Learning Model](#machine-learning-model)
* [Surge Pricing & ETA Engine](#surge-pricing--eta-engine)
* [API Reference](#api-reference)
* [Tech Stack](#tech-stack)
* [Project Structure](#project-structure)
* [Installation & Setup](#installation--setup)
* [Environment Variables](#environment-variables)
* [Data Sources](#data-sources)
* [Limitations & Future Work](#limitations--future-work)

---

## Overview

**UrbanFlow** is a production-scale ride demand forecasting and pricing intelligence platform. At its core, the project runs a stacked machine learning ensemble of **XGBoost** and **LightGBM** trained on over **65 million raw Uber/Lyft (HVFHV) trip records** (aggregated to ~2.25M time-slot zones). It predicts localized demand for 15-minute intervals across NYC's 263 taxi zones.

The application serves these predictions via an asynchronous **FastAPI** backend, persisted in a **PostgreSQL** database, and visualizes the results on a modern, dark-themed **React** dashboard. In addition to demand prediction, UrbanFlow features:
* **Route-Aware ETA calculations** by traversing physical road graphs using **OSMnx** and **NetworkX**.
* **A Three-Policy Pricing Engine** that simulates and compares flat, reactive, and predictive (anticipatory) surge multipliers.
* **SHAP Explainability** to make black-box ML predictions transparent to users.

---

## Problem Statement

Modern ride-hailing services (like Uber and Lyft) face extreme spatial and temporal volatility. In NYC alone, millions of rides occur daily, and demand changes rapidly by the minute. Traditional ride-hailing networks suffer from two main gaps:
1. **Inefficient Distribution:** Drivers are often poorly distributed because dispatch systems cannot anticipate demand spikes 15–30 minutes in advance, resulting in high passenger wait times.
2. **Opacity:** Surge pricing algorithms are "black boxes"—riders and drivers see a multiplier but never understand the underlying features pushing prices up.
3. **Inaccurate ETAs:** Standard routing models estimate travel times based on straight-line distances (Haversine calculations) rather than traversing actual road networks.

UrbanFlow solves these issues by predicting fine-grained demand beforehand, serving route-aware road graph ETAs, and breaking down pricing factors transparently using SHAP values.

---

## Live Demo

* **Frontend Dashboard:** [http://localhost:5173](http://localhost:5173) (after local setup)
* **Backend API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs) (FastAPI auto-generated interactive Swagger UI)

---

## Features

* 🗺️ **Interactive Heatmap & Time Slider**
  Render NYC's 263 TLC taxi zones using a Leaflet map. A 96-slot slider allows users to scrub through a full 24-hour cycle (in 15-minute increments) to visualize anticipated demand hotspots.
  
* 📈 **Advanced Blended ML Ensemble**
  Combines XGBoost and LightGBM to yield a test $R^2$ of **0.8239**. Features are engineered using GroupKFold validation to guarantee no spatial leakage across training splits.

* 📍 **Route-Aware road Graph ETA**
  Computes precise route travel times by snapping pickup and dropoff points to NYC's physical OpenStreetMap network nodes, calculating the shortest route via Dijkstra's algorithm.

* 💸 **Dynamic Surge Pricing & Policy Comparison**
  Applies surge multipliers based on forecasted demand, time-of-day rush hours, nightlife factors, and weekends. Compares flat, reactive, and anticipatory predictive policies in real time.

* 🔍 **SHAP Explanations**
  Bridges the explainability gap by breaking down exactly how features (like past hour lags, zone averages, or weekend flags) contribute to shifting a zone's demand prediction away from the baseline.

* 🔐 **Secure JWT Authentication**
  Fully functional JSON Web Token authentication system featuring registration, login, token refresh, and route security.

* 📊 **Prometheus Performance Monitoring**
  Contains built-in middleware tracking requests, latency, and system health metrics, exposed via `/metrics`.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND (React)                       │
│      React-Leaflet Heatmap — Recharts — Playback Slider     │
└──────────────┬───────────────────────────────▲──────────────┘
               │ HTTP API Requests             │ JSON Response
               ▼                               │
┌─────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                      │
│                                                             │
│  ┌───────────────────────┐       ┌───────────────────────┐  │
│  │     auth_routes       │       │     demand_routes     │  │
│  │   /api/v1/auth/*      │       │    /api/v1/demand/*   │  │
│  └───────────────────────┘       └───────────┬───────────┘  │
│  ┌───────────────────────┐                   │              │
│  │     explain_routes    │◄──────────────────┼──────────────┤
│  │    /api/v1/explain/*  │                   ▼              │
│  └───────────────────────┘          ┌─────────────────┐     │
│  ┌───────────────────────┐          │ DemandService   │     │
│  │     pricing_routes    │◄─────────┤  - XGBoost Final│     │
│  │    /api/v1/pricing/*  │          │  - LightGBM     │     │
│  └───────────────────────┘          │  - SHAP Explains│     │
│  ┌───────────────────────┐          └─────────────────┘     │
│  │       eta_routes      │                   ▲              │
│  │     /api/v1/eta/*     │                   │              │
│  └───────────┬───────────┘          ┌─────────────────┐     │
│              ▼                      │ pyarrow / pandas│     │
│      ┌───────────────┐              │ preprocessed data│    │
│      │  ETAService   │              └─────────────────┘     │
│      │ - OSMnx Graph │                                      │
│      │ - NetworkX    │                                      │
│      └───────┬───────┘                                      │
└──────────────┼──────────────────────────────────────────────┘
               │ SQLAlchemy ORM (asyncpg)
               ▼
┌─────────────────────────────────────────────────────────────┐
│                    POSTGRESQL DATABASE                      │
│      Log tables: predictions, pricing runs, user auth       │
└─────────────────────────────────────────────────────────────┘
```

---

## Backend Pipeline

For every inference cycle, the backend performs the following pipeline:

1. **Feature Construction:** Takes the target parameters (zone, hour, minute, day of week, day of month, month) and merges them with computed historical statistics:
   * **Spatial stats:** Zone historical demand mean, max, and standard deviation.
   * **Lag features:** Shifted values representing demand 15 min ago (`lag_1`), 1 hour ago (`lag_4`), and 24 hours ago (`lag_96`).
   * **Rolling averages:** Average demand over the last 2 hours (`rolling_8`) and 24 hours (`rolling_96`).
   * **Cyclical Encodings:** Sine and cosine representations of hour, weekday, and time slot.
2. **Model Evaluation:** Feeds the feature vector into the serialized XGBoost and LightGBM models to output the predicted demand score.
3. **Surge Multiplier Calculation:** Evaluates the predicted demand against active pricing policies.
4. **ETA & Route Generation:** Snap coordinates to the cached OSMnx road graph to calculate path travel distance and travel time. Snaps back to zone centroid metrics if the road network is unavailable.
5. **Database Logging & Return:** Async log statistics written to PostgreSQL, returning the results to the React dashboard.

---

## Machine Learning Model

### Algorithm & Strategy
UrbanFlow trains an **XGBoost** model and a **LightGBM** model, blending the two through a stacked regression layout.
* **Spatial Validation Safety:** Rather than a random train/test split (which causes severe spatial leakage), the models are validated using a **5-fold GroupKFold cross-validation grouped by `zone_id`**. This ensures the validation sets consist of zones the model has never trained on, yielding a robust generalized score.
* **Test Performance:** 
  * **Test $R^2$ Score:** `0.8239`
  * **Test RMSE:** `0.0706`

### Hyperparameters
```python
XGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 200,
    "eval_metric": "rmse",
}

LGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
}
```

### Feature Grouping (29 Features Total)
* **Time Features:** `hour`, `minute_of_hour`, `time_slot` (0-95), `day_of_week`, `day_of_month`, `month`, `is_weekend`, `is_peak`, `is_night`, cyclical sine/cosine columns.
* **Spatial Features:** `zone_id`, `borough_enc`, `zone_lat`, `zone_lng`.
* **Demand Features:** `zone_demand_mean`, `zone_demand_std`, `zone_demand_max`, `zone_slot_demand`, `borough_demand_mean`.
* **Lags:** `demand_lag_1` (15m), `demand_lag_4` (1h), `demand_lag_96` (24h).
* **Rolling averages:** `demand_rolling_8` (2h), `demand_rolling_96` (24h).

---

## Surge Pricing & ETA Engine

### Surge Pricing Policies
The backend contains a `PricingService` comparing three active business logic pricing methods:
1. **Flat Policy:** Baseline price calculation without surge adjustments (constant `1.0x` multiplier).
2. **Reactive Policy:** Triggers immediately based on the current demand level.
   $$\text{Surge} = 1.0 + (\text{predicted\_demand} - 0.5) \times 4.0 \quad (\text{capped at } 3.0\text{x})$$
3. **Predictive Policy:** Anticipatory pricing. It calculates the reactive surge rate, then increases the multiplier by `+0.15x` if the query falls within the hour leading up to standard morning/evening rush hours (e.g., between 6:00 AM and 7:00 AM, or 4:00 PM and 5:00 PM), prompting drivers to navigate to high-demand zones before the rush begins.

### ETA Routing Engine
* **Graph Mode:** Loads a cached `.graphml` file containing NYC's physical road network nodes and edges built using **OSMnx**. Snap coordinate locations to the closest network node and invoke `networkx.shortest_path` using the `travel_time` edge weight.
* **Fallback Mode:** In case of missing local graphs, calculates Haversine great-circle distance between zone centroids, applying a time-of-day traffic speed multiplier to approximate travel duration.

---

## API Reference

All endpoints run on `http://localhost:8000` by default. Complete Swagger API docs are accessible at `http://localhost:8000/docs`.

### Authentication Endpoints
* **`POST /api/v1/auth/register`**  
  Registers a new user profile. Returns access + refresh tokens.
* **`POST /api/v1/auth/login`**  
  Authenticates credentials and returns JWT bearer tokens.
* **`POST /api/v1/auth/refresh`**  
  Refreshes an expired access token using a valid refresh token.

### Demand Prediction Endpoints
* **`POST /api/v1/demand/predict`**  
  Predict demand for a specific zone ID at a specific timestamp.
* **`POST /api/v1/demand/heatmap`**  
  Predict demand across all 263 zones for the map. Used by the React dashboard heatmap.
* **`GET /api/v1/demand/stats`**  
  Returns historical logging statistics (e.g., query volume, avg demand, top queried zones).

### Pricing & ETA Endpoints
* **`POST /api/v1/pricing/calculate`**  
  Calculates trip surge multiplier, adjusted fare, and details pricing factors (e.g., Morning Rush, Weekend Nightlife).
* **`POST /api/v1/pricing/compare`**  
  Returns comparative estimates between Flat, Reactive, and Predictive policies.
* **`POST /api/v1/eta/estimate`**  
  Computes route distance (miles) and travel ETA (minutes) between pickup and dropoff zone IDs.

### Explainability Endpoints
* **`GET /api/v1/explain/zone/{zone_id}`**  
  Provides SHAP feature attribution breakdowns explaining *why* the model predicted a specific demand.
* **`GET /api/v1/explain/feature-importance`**  
  Returns global feature importance rankings across all features.

---

## Tech Stack

### Backend
* **Python 3.11** - Core backend runtime.
* **FastAPI 0.111** - High-performance asynchronous API framework.
* **Uvicorn** - ASGI web server.
* **Scikit-Learn** - Modeling, GroupKFold CV, stacking logic.
* **XGBoost & LightGBM** - Ensemble models.
* **Joblib** - Serializing and loading models.
* **SQLAlchemy & asyncpg** - Asynchronous database interaction.
* **OSMnx & NetworkX** - Graph routing and shortest path calculation.
* **SHAP** - Local model explainability.

### Frontend
* **React 18** - UI development.
* **Vite 7.3.1** - Build automation and dev server.
* **React-Leaflet & Leaflet** - Interactive spatial maps.
* **Recharts** - Charts, lines, and heat plots.
* **CSS-in-JS & Vanilla CSS** - User interface theme customization.

---

## Project Structure

```
urbanflow/
│
├── config.py                  # Central configuration (paths, hyperparameters, features)
├── requirements.txt           # Python backend dependencies
├── Dockerfile                 # Docker setup
├── docker-compose.yml         # Local database and API containers
│
├── data/
│   ├── download_nyc_data.py   # Script to download raw data and zones
│   ├── preprocess.py          # Preprocesses raw trips -> 15-min interval files
│   └── processed/             # Directory for preprocessed Parquet splits
│
├── ml/
│   ├── features.py            # Featurization logic (lags, rollings, cyclical)
│   ├── xgboost_model.py       # XGBoost training configuration
│   ├── lightgbm_model.py      # LightGBM training configuration
│   ├── ensemble.py            # Stacking and blending logic
│   ├── explainer.py           # SHAP analysis and explanations
│   ├── eta_model.py           # Graph routing travel time calculations
│   └── train.py               # ML training pipeline execution script
│
├── api/
│   ├── main.py                # App launcher
│   ├── schemas.py             # Pydantic schemas
│   ├── routes/                # Endpoint files (demand, pricing, explain, eta)
│   ├── services/              # Business logic (demand, pricing, eta, database)
│   └── auth/                  # JWT security and authentication
│
├── db/
│   ├── database.py            # Database connections
│   └── models.py              # Database models (User, logs)
│
└── frontend/
    ├── src/
    │   ├── App.jsx            # Main app component
    │   └── main.jsx           # Vite entrypoint
    ├── package.json           # npm dependencies
    └── vite.config.js         # Vite configuration
```

---

## Installation & Setup

### Prerequisites
* **Python 3.11+**
* **Node.js 18+**
* **PostgreSQL** (or running via Docker Compose)

### 1. Clone & Environment Configuration
```bash
git clone https://github.com/your-username/urbanflow.git
cd urbanflow

# Copy environment template
copy .env.example .env   # On Windows
# cp .env.example .env   # On Linux/macOS
```

### 2. Set Up the Database
You can spin up a PostgreSQL instance locally or using Docker:
```bash
# Option A: Start database via Docker
docker compose up db -d

# Option B: Run on a local PostgreSQL instance and edit DATABASE_URL in .env:
# DATABASE_URL=postgresql+asyncpg://<username>:<password>@localhost:5432/urbanflow
```

### 3. Initialize Python Environment & Run Preprocessing
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Create tables
python scripts/init_database.py

# Download Zone Lookup & Parquet files
python data/download_nyc_data.py

# Aggregate raw trips to 15-min intervals
python data/preprocess.py
```

### 4. Train the ML Models
To train the demand forecasting ensemble models and generate the SHAP explainer objects:
```bash
python ml/train.py
```
This produces final model binaries inside `outputs/models/`.

### 5. Launch Backend & Frontend
Start the FastAPI server:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Launch the React dashboard:
```bash
cd frontend
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

---

## Environment Variables

The project uses `.env` in the root folder to manage runtime settings:
```ini
DATABASE_URL=postgresql+asyncpg://urbanflow:urbanflow@localhost:5432/urbanflow
AUTH_ENABLED=true
SECRET_KEY=urbanflow-dev-secret-key-change-in-production
```
* Set `AUTH_ENABLED=false` to bypass authentication headers for development.

---

## Data Sources

* **NYC TLC trip data:** High Volume For-Hire Vehicle (HVFHV) Parquet records representing taxi, Uber, and Lyft trips, obtained from the NYC Taxi and Limousine Commission.
* **OpenStreetMap (OSM):** Road network graphs of New York City, downloaded using the OSMnx library.

---

## Limitations & Future Work

* **Real-time Ingestion:** Current demand forecasts depend on offline batch calculations. Future updates will leverage Kafka/Flink streaming to update lag variables in real time.
* **Weather Integration:** Incorporating weather forecasts (rainfall, snow, temperature) as features to enhance demand modeling.
* **Multi-City Support:** Generalizing the pipeline to run forecasts for London, Chicago, and San Francisco.

---
Built with ❤️ for urban transportation intelligence. If this repository is helpful, please give it a ⭐ on GitHub!
