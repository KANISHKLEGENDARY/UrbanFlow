# UrbanFlow — Project Architecture Blueprint

> Every file explained BEFORE a single line of code is written.

---

## Project Location

```
C:\Users\91931\.gemini\antigravity\scratch\urbanflow\
```

---

## What This Project Does (One Paragraph)

UrbanFlow predicts **ride demand** across NYC's 263 taxi zones using real Yellow Taxi trip data (3M+ trips, Jan 2023). It uses an **ensemble of XGBoost + LightGBM** models with SHAP explainability, serves predictions through a **FastAPI backend**, and visualizes everything on a **React dashboard** — showing demand heatmaps, surge pricing recommendations, and AI-powered explanations for why demand is high in each zone.

---

## The Build Order

We build files in this exact sequence. Each step is independently testable.

```
Phase 1: ML Pipeline (you can run and verify each file independently)
Phase 2: Data Pipeline (download + preprocess NYC TLC data)  
Phase 3: API Backend (FastAPI serving predictions)
Phase 4: Frontend Dashboard (React — what the user actually sees)
Phase 5: Production Polish (Docker, monitoring, docs)
```

---

## Complete Folder Structure

```
urbanflow/
│
├── config.py                    ← Central settings (paths, hyperparams)
├── requirements.txt             ← All Python dependencies
├── README.md                    ← Project documentation
│
├── data/                        ← Data storage (raw + processed)
│   ├── download_nyc_data.py     ← Downloads taxi data + zone shapefile
│   └── preprocess.py            ← Cleans raw data → demand aggregation
│
├── ml/                          ← Machine learning pipeline
│   ├── features.py              ← Feature engineering functions
│   ├── xgboost_model.py         ← XGBoost model wrapper
│   ├── lightgbm_model.py        ← LightGBM model wrapper
│   ├── ensemble.py              ← Combines XGB + LGB predictions
│   ├── explainer.py             ← SHAP analysis and visualizations
│   └── train.py                 ← Main training script (run this)
│
├── api/                         ← FastAPI backend
│   ├── main.py                  ← API entry point
│   ├── schemas.py               ← Request/response data models
│   ├── routes/
│   │   ├── demand.py            ← Demand prediction endpoints
│   │   ├── pricing.py           ← Surge pricing endpoints
│   │   └── explain.py           ← SHAP explanation endpoints
│   └── services/
│       ├── demand_service.py    ← Loads models, makes predictions
│       └── pricing_service.py   ← Computes surge multipliers
│
├── frontend/                    ← React dashboard
│   └── (created via npx later)
│
└── outputs/                     ← Generated artifacts
    ├── models/                  ← Saved trained models (.pkl)
    ├── shap/                    ← SHAP plots (PNG images)
    └── reports/                 ← Training metrics, logs
```

---

## File-by-File Explanation

### Root Files

---

#### `config.py` — Central Configuration

**What it does**: Stores ALL configurable settings in one place — file paths, model hyperparameters, feature lists, API ports. Every other file imports from here instead of hardcoding values.

**Why it exists**: If you want to change the learning rate from 0.01 to 0.02, or switch from 5-fold to 3-fold CV, you change ONE file — not hunt through 10 files. Interviewers love this pattern because it shows you think about maintainability.

**Key contents**:
- `DATA_DIR` — path to your NYC parquet file
- `MODEL_DIR` — where trained models get saved
- `XGB_PARAMS` — XGBoost hyperparameters (same as your hackathon + tuning)
- `LGB_PARAMS` — LightGBM hyperparameters
- `FEATURE_LIST` — which columns the model uses
- `N_FOLDS` — number of cross-validation folds

---

#### `requirements.txt` — Dependencies

**What it does**: Lists every Python package needed. You run `pip install -r requirements.txt` once.

**Key packages**:
```
pandas, numpy          → data manipulation
xgboost, lightgbm      → gradient boosted tree models  
scikit-learn           → cross-validation, metrics
shap                   → model explainability
fastapi, uvicorn       → API server
geopandas, shapely     → geographic zone data
matplotlib, seaborn    → plotting
joblib                 → model serialization
```

---

### `data/` — Data Pipeline

---

#### `data/download_nyc_data.py` — Data Downloader

**What it does**: Downloads the NYC taxi zone lookup CSV (maps LocationID → zone name + borough) from NYC's public data server. You already have the main trip Parquet file, so it only downloads the zone metadata.

**Why it exists**: The trip data has `PULocationID = 161` but you need to know that means "Midtown Center, Manhattan". This file fetches that mapping.

**What it produces**: `data/raw/taxi_zone_lookup.csv`

---

#### `data/preprocess.py` — Data Preprocessor

**What it does**: Takes the raw 3M-row trip Parquet file and transforms it into a **demand table** — the format our ML models actually consume.

**Step-by-step**:
1. **Load** the Parquet file (3,066,766 rows)
2. **Clean** — Remove invalid rows (negative fares, zero distance, bad dates, LocationID 264/265)
3. **Aggregate** — Count pickups per zone per 1-hour window:
   ```
   Zone 161 | 2023-01-15 08:00 | 47 pickups (= demand)
   Zone 161 | 2023-01-15 09:00 | 62 pickups (= demand)
   Zone 237 | 2023-01-15 08:00 | 12 pickups (= demand)
   ```
4. **Normalize** — Scale demand to [0, 1] range (just like your hackathon)
5. **Add zone metadata** — Borough, zone name from the lookup CSV
6. **Save** as `data/processed/demand_hourly.parquet`

**This is the bridge between raw taxi data and your hackathon-style ML pipeline.** After this step, the data looks conceptually similar to your Flipkart GRiD data — zones instead of geohashes, hours instead of timestamps, demand as a normalized score.

**What it produces**: `data/processed/demand_hourly.parquet` (~263 zones × 744 hours × features)

---

### `ml/` — Machine Learning Pipeline

This is where your hackathon expertise gets upgraded.

---

#### `ml/features.py` — Feature Engineering

**What it does**: Contains ALL feature engineering functions as clean, reusable functions. This is your hackathon's feature code — refactored from a flat script into callable functions.

**Functions**:
- `compute_time_features(df)` — Extracts hour, day_of_week, is_weekend, is_peak, is_night, cyclical sin/cos encodings. *Same logic as your hackathon.*
- `compute_zone_demand_stats(target_df, ref_df)` — Mean/std/max demand per zone. *Equivalent to your `geo_demand_mean` etc., but using LocationID instead of geohash.*
- `compute_zone_hour_demand(target_df, ref_df)` — Demand per zone per hour. *Equivalent to your `geo_hour_demand`.*
- `compute_lag_features(df)` — **NEW** — demand at t-1, t-2, t-24 (same zone, previous hours/previous day). *Your hackathon didn't have proper lag features because it only had 2 days of data. NYC has 31 days — lags become very powerful.*
- `compute_rolling_features(df)` — **NEW** — 7-day rolling average demand per zone.
- `compute_borough_demand(target_df, ref_df)` — Borough-level average demand. *Equivalent to your `geo_prefix` features — Manhattan zones share patterns.*

**Why it's separate**: Every function takes `ref_df` (reference data to compute stats from) so they're **fold-safe by design**. You pass training fold as `ref_df` → no leakage. This is the lesson from your hackathon codified into architecture.

---

#### `ml/xgboost_model.py` — XGBoost Wrapper

**What it does**: Wraps XGBoost into a clean class with `fit()` and `predict()` methods. Same hyperparameters as your best hackathon submission.

**Why it's a class**: So the training script can treat XGBoost and LightGBM identically — `model.fit(X, y)` works the same way for both. This makes the ensemble code simple.

**Key settings** (from your hackathon):
```python
n_estimators=5000, learning_rate=0.01, max_depth=8
subsample=0.8, colsample_bytree=0.8
early_stopping_rounds=200
```

---

#### `ml/lightgbm_model.py` — LightGBM Wrapper

**What it does**: Same interface as the XGBoost wrapper, but using LightGBM. LightGBM is Microsoft's gradient boosting library — it's faster than XGBoost and uses a different tree-building algorithm (leaf-wise vs. level-wise).

**Why we need BOTH**: Different algorithms make different errors. When you average their predictions, the errors partially cancel out → better accuracy. This is the core principle of ensembling.

**Key settings**:
```python
num_leaves=127, learning_rate=0.01, n_estimators=5000
subsample=0.8, colsample_bytree=0.8
early_stopping_rounds=200
```

---

#### `ml/ensemble.py` — Model Combiner

**What it does**: Takes predictions from XGBoost and LightGBM and combines them into a single, better prediction.

**Two strategies**:

1. **Weighted Blending** (simple, reliable):
   ```
   final = 0.55 × XGB_pred + 0.45 × LGB_pred
   ```
   Weights are optimized by testing different combinations on the validation set.

2. **Stacking** (more powerful):
   - Use XGB and LGB out-of-fold predictions as **input features**
   - Train a simple Ridge Regression model on top
   - The meta-model learns "XGB is better for peak hours, LGB is better for low-demand zones"

**Why this matters for interviews**: You can say *"I used a stacked ensemble with a Ridge meta-learner, optimizing the blend weights on out-of-fold predictions to prevent second-level overfitting."*

---

#### `ml/explainer.py` — SHAP Explainability

**What it does**: Generates visual explanations of WHY the model makes each prediction. Uses SHAP (SHapley Additive exPlanations) — a game theory approach that assigns each feature a contribution score.

**What it produces**:
1. **Feature importance bar chart** — "Which features matter most globally?" (e.g., zone_hour_demand > hour > is_weekend)
2. **Summary beeswarm plot** — "How does each feature value affect predictions?" (e.g., high hour values → high demand)
3. **Single prediction waterfall** — "Why is Zone 161 predicted at 0.85 demand?" (hour=17 pushes +0.3, zone_mean=0.7 pushes +0.2, is_weekend=0 pushes -0.1)
4. **JSON-serializable explanations** — For the API to serve to the frontend

**Why this exists**: This is what makes your project stand out in interviews. Any student can train XGBoost. Few can explain WHY their model predicts what it predicts. SHAP gives you that superpower.

---

#### `ml/train.py` — Training Orchestrator (THE MAIN SCRIPT)

**What it does**: This is the file you run. It orchestrates everything:

```
1. Load processed demand data
2. Compute base features (time, zone metadata)
3. GroupKFold split by LocationID (5 folds)
4. For each fold:
   a. Compute fold-safe demand features from training fold only
   b. Train XGBoost → collect out-of-fold predictions
   c. Train LightGBM → collect out-of-fold predictions
5. Ensemble: optimize blend weights on OOF predictions
6. Evaluate: R², RMSE on original scale
7. Retrain on full data → save final models
8. Run SHAP analysis → save plots
9. Print summary report
```

**Why it's separate from the model files**: Separation of concerns. The model files define HOW to train. This file defines WHAT to train on and WHEN. If you want to switch from GroupKFold to TimeSeriesSplit, you change only this file.

---

### `api/` — FastAPI Backend

---

#### `api/main.py` — API Entry Point

**What it does**: Creates the FastAPI app, registers all route modules, and starts the server. When you run `uvicorn api.main:app`, this is what launches.

**Endpoints it mounts**:
- `/api/v1/demand/*` — Demand predictions
- `/api/v1/pricing/*` — Surge pricing
- `/api/v1/explain/*` — SHAP explanations

---

#### `api/schemas.py` — Data Models

**What it does**: Defines the exact shape of every API request and response using Pydantic. This is the "contract" between frontend and backend.

**Example**:
```python
class DemandRequest:
    zone_id: int        # 1-263
    hour: int           # 0-23  
    day_of_week: int    # 0-6
    
class DemandResponse:
    zone_id: int
    zone_name: str      # "Midtown Center"
    borough: str        # "Manhattan"
    predicted_demand: float  # 0.0 to 1.0
    surge_multiplier: float  # 1.0 to 3.0
```

---

#### `api/routes/demand.py` — Demand Endpoints

**What it does**: Handles demand prediction requests.

**Endpoints**:
- `POST /predict` — Predict demand for a specific zone + time
- `GET /heatmap` — Get demand for ALL 263 zones at a given hour (for the map)
- `GET /zones` — List all zones with current demand level

---

#### `api/routes/pricing.py` — Pricing Endpoints

**What it does**: Computes surge pricing multipliers based on predicted demand.

**Logic**: High demand → high surge. The formula accounts for:
- Predicted demand level (from ML model)
- Time of day (peak vs. off-peak)
- Borough (Manhattan naturally has higher base demand)

---

#### `api/routes/explain.py` — Explainability Endpoints

**What it does**: Returns SHAP-based explanations for any prediction.

**Endpoints**:
- `GET /explain/{zone_id}?hour=17` — "Why is demand high in zone 161 at 5 PM?"
- `GET /feature-importance` — Global feature ranking

**This is the innovation** — most ride-hailing platforms show you a surge price but never explain WHY. Our API explains every prediction.

---

#### `api/services/demand_service.py` — Demand Business Logic

**What it does**: Loads the trained ensemble model from disk, prepares features for a given zone+time, runs the prediction, and returns the result. This is the bridge between the API routes (which handle HTTP) and the ML models (which handle math).

---

#### `api/services/pricing_service.py` — Pricing Business Logic

**What it does**: Takes a demand prediction and converts it into a surge multiplier using a configurable pricing policy. Supports three modes:
1. **Flat** — Always 1.0× (baseline)
2. **Reactive** — Simple demand-based scaling
3. **Predictive** — Uses demand forecast to anticipate surges before they happen

---

### `outputs/` — Generated Artifacts

All generated files go here. Nothing in this folder is committed to git — it's all generated by running the code.

- `outputs/models/` — Saved `.pkl` files (XGBoost, LightGBM, ensemble weights)
- `outputs/shap/` — SHAP plots as PNG images  
- `outputs/reports/` — Training logs, fold metrics, feature importance CSVs

---

## How You Run & Verify Each Phase

### Phase 1: ML Pipeline
```bash
pip install -r requirements.txt           # Install deps (you approve)
python data/preprocess.py                  # Process NYC data → see output stats
python ml/train.py                         # Train models → see R² scores, SHAP plots
```
**You verify**: R² score prints to terminal. SHAP plots appear in `outputs/shap/`. Open them in your image viewer.

### Phase 2: API Backend
```bash
uvicorn api.main:app --reload              # Start API server
# Open http://localhost:8000/docs          # Swagger UI — test endpoints in browser
```
**You verify**: Open `localhost:8000/docs` in your browser. Click "Try it out" on any endpoint. See real predictions.

### Phase 3: Frontend Dashboard
```bash
cd frontend && npm run dev                 # Start React dev server
# Open http://localhost:3000               # See the dashboard
```
**You verify**: Open `localhost:3000` in your browser. See the heatmap, charts, pricing panels.

---

## What's NOT Included (Keeping It Clean)

- ❌ No Docker (added later in Step 3 if needed)
- ❌ No CI/CD pipelines
- ❌ No database (everything runs from files)
- ❌ No authentication
- ❌ No monitoring service (added later)

**Why**: These are production concerns. We build the core product first, then layer on production features. Clean architecture > feature bloat.
