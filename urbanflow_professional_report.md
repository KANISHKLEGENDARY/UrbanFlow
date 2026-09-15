# URBANFLOW: AI-Powered Ride Demand Prediction & Surge Pricing Intelligence
## Executive Technical & Operations Report
**Prepared for:** Enterprise Integration & Operations Evaluation  
**Subject:** Full-Stack Spatial-Temporal Ride forecasting, Routing, and Pricing System

---

## 1. Executive Summary

In modern urban mobility networks (such as Uber, Lyft, and Ola), efficiency hinges on the ability to balance driver supply with passenger demand in real time. **UrbanFlow** is an enterprise-grade, full-stack predictive intelligence platform designed to forecast ride-hailing demand, compute route-aware travel times, and simulate dynamic pricing strategies. 

At the core of the system is a stacked machine learning ensemble of **XGBoost** and **LightGBM** trained on **over 65 million raw High-Volume For-Hire Vehicle (HVFHV) trip records** from the New York City Taxi and Limousine Commission (TLC). The model predicts localized demand at 15-minute intervals across NYC's 263 TLC zones. 

This report provides a detailed, professional breakdown of the system’s architecture, machine learning models, database structure, and real-world applications.

---

## 2. Table of Contents
1. [Overview & System Purpose](#3-overview--system-purpose)
2. [Problem Statement & Real-World Implications](#4-problem-statement--real-world-implications)
3. [Visual Interface & Live Demo Environment](#5-visual-interface--live-demo-environment)
4. [Platform Core Features](#6-platform-core-features)
5. [System Architecture](#7-system-architecture)
6. [Backend Pipeline & Execution Lifecycle](#8-backend-pipeline--execution-lifecycle)
7. [Machine Learning Pipeline & Feature Engineering](#9-machine-learning-pipeline--feature-engineering)
8. [Surge Pricing & ETA Engines](#10-surge-pricing--eta-engines)
9. [API Reference & Schema Specifications](#11-api-reference--schema-specifications)
10. [Technology Stack Breakdown](#12-technology-stack-breakdown)
11. [Project Directory Structure](#13-project-directory-structure)
12. [Installation, Preprocessing, and Model Training](#14-installation-preprocessing-and-model-training)
13. [Environment Configuration](#15-environment-configuration)
14. [Data Sources & Acquisition](#16-data-sources--acquisition)
15. [Limitations & Future Development Roadmap](#17-limitations--future-development-roadmap)

---

## 3. Overview & System Purpose

UrbanFlow is designed to act as the "operating system" for ride-hailing fleet management. The system transitions from raw historical taxi logs to real-time predictive analytics and automated pricing actions.

```mermaid
graph TD
    A[Raw NYC TLC Parquet Data] --> B[Data Cleaning & Feature Engineering]
    B --> C[XGBoost & LightGBM Training]
    C --> D[Serialized Model Binaries]
    D --> E[FastAPI Engine]
    F[OSMnx Road Graph] --> E
    E --> G[Interactive React Dashboard]
    E --> H[PostgreSQL Logging & Auditing]
```

### Real-World Analogy
Think of a traditional ride-hailing service as a **reactive** fire station: it only responds when a fire alarm (a ride request) rings. This causes long wait times during peak hours. UrbanFlow acts as a **predictive** weather station: it forecasts where the "demand storm" will hit 15–30 minutes before it arrives. This allows operators to position drivers in advance, smoothing out supply-demand imbalances before they create delays.

---

## 4. Problem Statement & Real-World Implications

Ride-hailing networks are subject to intense spatial and temporal volatility. UrbanFlow resolves three primary industry challenges:

### 1. Inefficient Driver Distribution (The Supply-Demand Gap)
Without predictive capabilities, drivers default to cruising popular areas (like Midtown Manhattan) or returning to where they last dropped off a passenger. 
* **Real-World Impact:** When a major concert ends at Madison Square Garden (Zone 161) at 10:30 PM, thousands of attendees open their apps simultaneously. If drivers are scattered across Brooklyn or Queens, passengers face wait times exceeding 30 minutes, and the operator loses thousands of dollars in unfulfilled bookings. UrbanFlow predicts this spike 15 minutes in advance, enabling proactive driver dispatch.

### 2. Lack of Transparency in Surge Pricing (The "Black Box" Problem)
Surge pricing is critical for balancing markets, but it faces rider and regulator backlash when multipliers appear arbitrary.
* **Real-World Impact:** A user charged a 2.5x multiplier on a rainy Tuesday morning often feels exploited. By integrating **SHAP (SHapley Additive exPlanations)**, UrbanFlow breaks down the exact variables driving the price (e.g., "+0.6x because the local zone is at 90% capacity, +0.3x due to morning commute congestion, +0.2x due to a 24-hour historical demand spike"). This transparency builds trust and helps satisfy regulatory compliance.

### 3. Inaccurate ETAs (The Routing Problem)
Many dispatch engines rely on simple straight-line (Haversine) distance calculations scaled by an arbitrary speed factor to estimate arrival times.
* **Real-World Impact:** If a driver is in Midtown Manhattan and a passenger is across the East River in Long Island City, a straight-line calculation might show them as 1.5 miles apart, predicting a 5-minute ETA. In reality, the driver must navigate heavy traffic to access the Queensboro Bridge, taking 25 minutes. UrbanFlow Snaps locations to New York's physical OpenStreetMap road network nodes to calculate route-aware ETAs based on Dijkstra’s shortest path algorithm.

---

## 5. Visual Interface & Live Demo Environment

UrbanFlow provides two user-facing interfaces to support operations teams and developers:

### 1. Frontend Operations Dashboard (`http://localhost:5173`)
A dark-themed, responsive React dashboard designed for operations dispatchers.
* **Interactive Heatmap:** Renders NYC's 263 TLC zones using Leaflet. The color of each zone transitions from green (low demand) to red (critical demand) based on predictions.
* **96-Slot Time Slider:** Allows users to scrub through a 24-hour cycle in 15-minute increments (representing slots 0 to 95) to watch demand patterns shift across the city.
* **Zone Detail & Explainability Panel:** Clicking a zone opens a side panel displaying the predicted demand percentage, the live surge fare breakdown, and a bar chart showing the SHAP feature contributions explaining the prediction.

### 2. Interactive Swagger API Docs (`http://localhost:8000/docs`)
FastAPI auto-generates interactive documentation for developers. Developers can test API requests, view JSON schemas, and inspect performance metrics directly in the browser.

---

## 6. Platform Core Features

* **Interactive Heatmap & Time Slider:** NYC's geography is rendered as a vector grid. Operations teams can preview tomorrow's expected morning rush hour by sliding the time control to 8:15 AM.
* **Stacked ML Ensemble:** Combines XGBoost (highly efficient tree structure) and LightGBM (leaf-wise growth algorithm) to produce a combined test $R^2$ of **0.8239**.
* **Route-Aware Graph Routing:** Estimates travel times by loading New York's actual road network using OSMnx and NetworkX.
* **Three-Policy Pricing Engine:** Compares flat, reactive, and anticipatory predictive pricing models.
* **SHAP Explainability:** Translates complex machine learning logic into simple, feature-by-feature explanations.
* **JWT Authentication:** Implements a security layer utilizing JSON Web Tokens with encrypted password hashing.
* **Prometheus Monitoring:** Standardizes observability by exposing core application metrics (request rates, prediction latency, memory usage) via a `/metrics` endpoint.

---

## 7. System Architecture

UrbanFlow is structured as a decoupled, three-tier architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND (React)                       │
│    Leaflet Map Grid ── Playback Slider ── Recharts Panels   │
└──────────────┬───────────────────────────────▲──────────────┘
               │ HTTP API Requests             │ JSON Response
               ▼                               │ (Surge, SHAP, ETA)
┌─────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                      │
│                                                             │
│   ┌────────────────────┐          ┌────────────────────┐    │
│   │    auth_routes     │          │    demand_routes   │    │
│   └────────────────────┘          └──────────┬─────────┘    │
│   ┌────────────────────┐                     │              │
│   │    explain_routes  │◄────────────────────┼──────────────┤
│   └────────────────────┘                     ▼              │
│   ┌────────────────────┐            ┌─────────────────┐     │
│   │    pricing_routes  │◄───────────┤  DemandService  │     │
│   └────────────────────┘            │  - XGBoost/LGBM │     │
│   ┌────────────────────┐            │  - SHAP Explains│     │
│   │     eta_routes     │            └─────────────────┘     │
│   └──────────┬─────────┘                     ▲              │
│              ▼                               │              │
│       ┌──────────────┐                       │              │
│       │  ETAService  │              ┌─────────────────┐     │
│       │  - OSMnx /   │              │  pandas Parquet │     │
│       │   NetworkX   │              │   Stats Cache   │     │
│       └──────┬───────┘              └─────────────────┘     │
└──────────────┼──────────────────────────────────────────────┘
               │ SQLAlchemy ORM (asyncpg)
               ▼
┌─────────────────────────────────────────────────────────────┐
│                    POSTGRESQL DATABASE                      │
│   Tables: users, predictions_log, pricing_history, eta_log  │
└─────────────────────────────────────────────────────────────┘
```

### Real-World Data Flow Walkthrough
1. **Request:** An operations manager selects Zone 132 (JFK Airport) at 5:30 PM on a Friday.
2. **API Call:** The React frontend triggers a POST request to `/api/v1/pricing/calculate` with JWT credentials.
3. **Authentication:** FastAPI intercepts the request, verifies the bearer token, and rejects it if expired.
4. **Demand Inference:** `DemandService` compiles the 29-feature vector, feeds it to the LightGBM/XGBoost models, and obtains a predicted demand of **0.88 (88% capacity)**.
5. **Surge & Routing Calculation:** `PricingService` reads the 0.88 demand, adds evening rush factors (+0.3x) and critical demand factors (+1.2x), outputting a **2.5x surge**. If an ETA was requested, `ETAService` computes the travel time across the road graph.
6. **Logging:** The request and output parameters are written to the `predictions_log` and `pricing_history` tables in PostgreSQL using asynchronous database workers.
7. **Response:** The frontend receives the JSON response and updates the heatmap, surge panel, and SHAP charts.

---

## 8. Backend Pipeline & Execution Lifecycle

For every inference cycle, the backend performs a five-step pipeline:

```
[Request Input] ──► 1. Feature Construction ──► 2. Model Evaluation ──► 3. Surge Calculation ──► 4. ETA Calculation ──► 5. Async Database Log ──► [Response]
```

1. **Feature Construction:** The target inputs (zone, hour, minute, day of week) are combined with precomputed spatial-temporal characteristics, including lag variables (demand at 15 minutes, 1 hour, and 24 hours ago) and rolling averages.
2. **Model Evaluation:** The 29-feature vector is passed to the ensemble models. XGBoost and LightGBM generate individual scores, which are blended using optimized weights.
3. **Surge Multiplier Calculation:** The demand output is processed by `PricingService` to evaluate active pricing policies (Flat, Reactive, Predictive).
4. **ETA & Route Generation:** Snaps pickup and dropoff coordinates to the cached OSMnx graph. If the graph is missing, it falls back to centroid-based Haversine distance, adjusted by borough speeds and traffic multipliers.
5. **Database Logging & Return:** Async PostgreSQL tasks store audit logs, and the API returns the JSON response.

---

## 9. Machine Learning Pipeline & Feature Engineering

### 1. Algorithm & Stacking Strategy
UrbanFlow trains an XGBoost model and a LightGBM model. 
* **LightGBM** uses a leaf-wise growth algorithm, making it highly effective at handling large-scale spatial datasets.
* **XGBoost** builds trees level-wise, offering strong regularization to prevent overfitting.
* The system blends their predictions. In the Phase 2 retrained pipeline, the optimized blending weights allocate a weight of `1.0` to LightGBM due to its superior performance on the 15-minute interval dataset.

### 2. Spatial Validation Safety (Preventing Spatial Leakage)
A common mistake in spatial-temporal modeling is using a random train/test split. Because traffic patterns in neighboring zones are highly correlated, a random split allows the model to memorize local trends, leading to artificially high training scores but poor real-world performance.
* **Solution:** UrbanFlow uses **5-fold GroupKFold cross-validation grouped by `zone_id`**. This ensures the model is validated on zones it has never seen during training, forcing it to learn generalizable spatial-temporal features.

| Metric | Out-of-Fold (OOF) $R^2$ | Test $R^2$ (Unseen Future Month) | Test RMSE |
| :--- | :---: | :---: | :---: |
| **XGBoost** | 0.7454 | - | - |
| **LightGBM** | 0.7637 | - | - |
| **Optimized Blend** | 0.7637 | **0.8239** | **0.0706** |

The high test $R^2$ score (**0.8239**) indicates that the model explains approximately 82.4% of the variance in ride demand on unseen data.

### 3. Feature Engineering (29 Features Total)
The models rely on 29 engineered features, grouped into five categories:

* **Time Features (15):** `hour`, `minute_of_hour`, `time_slot` (0-95), `day_of_week`, `day_of_month`, `month`, `is_weekend`, `is_peak`, `is_night`, and cyclical encodings.
* **Spatial Features (4):** `zone_id`, `borough_enc` (encoded borough names), `zone_lat`, `zone_lng`.
* **Demand Statistics (5):** `zone_demand_mean` (historical mean), `zone_demand_std` (variance), `zone_demand_max` (peak volume), `zone_slot_demand` (historical slot average), `borough_demand_mean`.
* **Lag Features (3):** `demand_lag_1` (15 mins ago), `demand_lag_4` (1 hour ago), `demand_lag_96` (24 hours ago).
* **Rolling Averages (2):** `demand_rolling_8` (past 2-hour average), `demand_rolling_96` (past 24-hour average).

> [!NOTE]
> **Cyclical Time Encoding:**
> Hours (0-23) and slots (0-95) are cyclical. To prevent the model from treating midnight (0) and 11:45 PM (95) as distant values, they are mapped to sine and cosine space:
> $$\text{slot\_sin} = \sin\left(\frac{2\pi \cdot \text{slot}}{96}\right), \quad \text{slot\_cos} = \cos\left(\frac{2\pi \cdot \text{slot}}{96}\right)$$
> This projects time onto a circle, helping the model learn that 11:45 PM and midnight are temporally adjacent.

---

## 10. Surge Pricing & ETA Engines

### Surge Pricing Policies
`PricingService` implements three distinct pricing policies:

```
[Flat Policy] ────────► Always 1.0x (Baseline price, no surge)
[Reactive Policy] ────► Surge scales immediately based on current demand capacity
[Predictive Policy] ──► Anticipates demand spikes and adjusts multipliers 1 hour early
```

1. **Flat Policy:** Baseline pricing. The multiplier is always **1.0x**.
2. **Reactive Policy:** Adjusts fares based on current demand capacity:
   $$\text{Surge} = 1.0 + (\text{predicted\_demand} - 0.5) \times 4.0 \quad (\text{capped at } 3.0\text{x})$$
3. **Predictive Policy (Anticipatory):** Anticipates upcoming rush hours. It calculates the reactive surge rate, then adds **+0.15x** if the target time falls within the hour leading up to standard commutes (6:00 AM–7:00 AM or 4:00 PM–5:00 PM).
   * **Real-World Benefit:** This early price signal encourages drivers to navigate to high-demand zones *before* the rush begins, reducing passenger wait times.

### ETA Routing Engine
* **Graph Mode:** Snaps coordinates to the closest road network node in the cached OSMnx graph. It calculates the shortest path using Dijkstra's algorithm, using travel time as the edge weight.
* **Fallback Mode:** If the road network graph is unavailable, the engine calculates the Haversine distance between zone centroids, adjusts for street network winding (+35%), divides by borough-specific average speeds, and applies a time-of-day traffic multiplier:

| Borough Location | Base Speed Assumption |
| :--- | :--- |
| **Manhattan Only** | 11.0 mph (high congestion) |
| **Manhattan Mixed** | 15.0 mph (medium congestion) |
| **EWR Airport Corridor** | 24.0 mph (highway transit) |
| **Default/Other Boroughs** | 18.5 mph |

---

## 11. API Reference & Schema Specifications

Below are the key endpoints exposed by the FastAPI backend:

### Authentication Endpoints
* **`POST /api/v1/auth/register`**
  * **Purpose:** Registers a new user profile.
  * **Payload:** `{ "username": "ops_user", "email": "user@company.com", "password": "securepassword" }`
  * **Response:** Returns JWT access and refresh tokens.
* **`POST /api/v1/auth/login`**
  * **Purpose:** Authenticates credentials and returns a JWT bearer token.

### Demand Prediction Endpoints
* **`POST /api/v1/demand/predict`**
  * **Purpose:** Predicts demand for a specific zone and time.
  * **Request Schema:**
    ```json
    {
      "zone_id": 161,
      "hour": 17,
      "minute": 15,
      "day_of_week": 4,
      "day_of_month": 15,
      "month": 3
    }
    ```
  * **Response Schema:**
    ```json
    {
      "zone_id": 161,
      "predicted_demand": 0.884,
      "demand_raw": 530.4,
      "surge_multiplier": 2.38,
      "demand_level": "critical",
      "zone_name": "Midtown Center",
      "borough": "Manhattan"
    }
    ```
* **`POST /api/v1/demand/heatmap`**
  * **Purpose:** Returns predictions for all 249 valid zones for map rendering.

### Pricing & ETA Endpoints
* **`POST /api/v1/pricing/calculate`**
  * **Purpose:** Calculates the adjusted fare and provides a factor breakdown.
  * **Response Sample:**
    ```json
    {
      "surge_multiplier": 2.2,
      "adjusted_fare": 33.0,
      "factors": [
        { "factor": "Critical Demand", "impact": "+1.2x", "description": "Demand at 88% capacity" },
        { "factor": "Evening Rush", "impact": "+0.3x", "description": "Peak commute window" }
      ],
      "demand_level": "critical"
    }
    ```
* **`POST /api/v1/pricing/compare`**
  * **Purpose:** Returns comparative estimates for Flat, Reactive, and Predictive policies.
* **`POST /api/v1/eta/estimate`**
  * **Purpose:** Computes route distance and travel ETA between zone IDs.

### Explainability Endpoints
* **`GET /api/v1/explain/zone/{zone_id}`**
  * **Purpose:** Provides SHAP feature attribution breakdowns explaining the prediction.

---

## 12. Technology Stack Breakdown

The system is built on a modern, decoupled technology stack:

### Backend Architecture
* **Python 3.11:** Core execution runtime.
* **FastAPI (v0.111):** High-performance ASGI framework for asynchronous endpoints.
* **SQLAlchemy & asyncpg:** Asynchronous Object-Relational Mapper (ORM) and PostgreSQL database driver.
* **XGBoost & LightGBM:** Machine learning ensemble frameworks.
* **OSMnx & NetworkX:** Libraries for spatial graph processing and route optimization.
* **SHAP:** Explains prediction factors using cooperative game theory concepts.

### Frontend Dashboard
* **React 18 & Vite:** Frontend framework and build automation tool.
* **Leaflet & React-Leaflet:** Library for rendering maps and demand heatmaps.
* **Recharts:** Formats and displays SHAP attribution charts.
* **Vanilla CSS:** Custom styling utilizing CSS variables and responsive grid systems.

---

## 13. Project Directory Structure

```
urbanflow/
│
├── config.py                  # System configuration, hyperparameters, and feature definitions
├── requirements.txt           # Python backend dependencies
├── Dockerfile                 # Container deployment specification
├── docker-compose.yml         # Container configuration for PostgreSQL and FastAPI
│
├── data/
│   ├── download_nyc_data.py   # Script to download raw TLC Parquet files and taxi zone lookups
│   ├── preprocess.py          # Data pipeline to clean and aggregate raw trips into 15-minute grids
│   └── processed/             # Directory containing processed training and testing datasets
│
├── ml/
│   ├── features.py            # Feature engineering logic (time, lag, and rolling statistics)
│   ├── xgboost_model.py       # XGBoost model training configuration
│   ├── lightgbm_model.py      # LightGBM model training configuration
│   ├── ensemble.py            # Logic for blending and stacking models
│   ├── explainer.py           # Prepares SHAP explainers and generates attribution charts
│   ├── eta_model.py           # Estimator for OSMnx road routing and centroid fallbacks
│   └── train.py               # ML training pipeline execution script
│
├── api/
│   ├── main.py                # FastAPI entry point
│   ├── schemas.py             # Pydantic schemas for request and response validation
│   ├── routes/                # Route definitions (demand, pricing, explain, eta)
│   ├── services/              # Core business services (demand, pricing, routing)
│   └── auth/                  # JWT security and authentication services
│
├── db/
│   ├── database.py            # Database connections and engine setup
│   └── models.py              # SQLAlchemy ORM models (users, prediction logs, etc.)
│
└── frontend/
    ├── src/                   # React source files (App.jsx, main.jsx)
    └── package.json           # Frontend package dependencies
```

---

## 14. Installation, Preprocessing, and Model Training

To deploy the platform locally, follow these steps:

### 1. Initialize the Python Environment
```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Preprocess the Data
```bash
# Download raw TLC trip data and taxi zone maps
python data/download_nyc_data.py

# Clean and aggregate raw trips into 15-minute intervals
python data/preprocess.py
```

### 3. Train the ML Models
```bash
# Run the training pipeline to generate model binaries and SHAP explainers
python ml/train.py
```
This script trains the XGBoost and LightGBM models, runs the GroupKFold cross-validation, and saves the serialized models (`.pkl`) to the `outputs/models/` directory.

### 4. Initialize the Database & Start the Services
```bash
# Create PostgreSQL database tables
python scripts/init_database.py

# Start the FastAPI backend
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 5. Launch the Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser to view the interactive dashboard.

---

## 15. Environment Configuration

UrbanFlow uses a `.env` file in the root directory to manage runtime settings:

```ini
# Database Connection URL (PostgreSQL + asyncpg)
DATABASE_URL=postgresql+asyncpg://urbanflow:urbanflow@localhost:5432/urbanflow

# Authentication Configuration
AUTH_ENABLED=true

# Security Key for JWT Token Generation
SECRET_KEY=urbanflow-dev-secret-key-change-in-production
```
* **Authentication Bypass:** Set `AUTH_ENABLED=false` during development to bypass JWT validation.

---

## 16. Data Sources & Acquisition

UrbanFlow integrates two data streams:
1. **NYC TLC High-Volume For-Hire Vehicle (HVFHV) Trip Data:** Represents ride-hailing records (Uber/Lyft) from the NYC Taxi and Limousine Commission. The dataset contains fields such as pickup/dropoff times, location zone IDs, trip distances, and passenger fares.
2. **OpenStreetMap (OSM) Road Graphs:** Extracted via the OSMnx library. The road graph represents NYC's street grid, including node coordinates and segment lengths. It is cached locally as `nyc_road_graph.graphml`.

---

## 17. Limitations & Future Development Roadmap

While UrbanFlow is fully functional, we have identified several opportunities for production improvements:

### 1. Shift from Batch to Real-Time Streaming Ingestion
* **Current Limitation:** Lag features (`demand_lag_1`, `demand_lag_4`) rely on precomputed historical zone averages rather than real-time metrics.
* **Proposed Solution:** Integrate **Apache Kafka** and **Apache Flink** to process live pickup logs. This will allow the system to update lag features in real time, making the model responsive to unexpected incidents (e.g., a sudden subway breakdown).

### 2. Integration of External Contextual Data (Weather & Events)
* **Current Limitation:** The model does not account for weather conditions or scheduled events.
* **Proposed Solution:** Integrate weather APIs to supply real-time rainfall, snowfall, and temperature features. During a rainstorm, taxi demand spikes, and integrating this data will improve forecast accuracy.

### 3. Scalability to Multi-City Operations
* **Current Limitation:** The routing grid and zone lookups are configured specifically for New York City.
* **Proposed Solution:** Modularize the spatial indexing system to accept configurations for other major cities (e.g., Chicago, London, San Francisco).

---
*Report compiled for corporate review and deployment authorization. For access inquiries or source-code reviews, please contact the development lead.*
