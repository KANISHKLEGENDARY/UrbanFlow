import os
from pathlib import Path

from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()

# Load .env from project root (DATABASE_URL, AUTH_ENABLED, SECRET_KEY, etc.)
load_dotenv(PROJECT_ROOT / ".env")

# Directory Paths
RAW_DATA_DIR = PROJECT_ROOT  
RAW_DATA_FILES = [
    RAW_DATA_DIR / "fhvhv_tripdata_2026-02.parquet",
    RAW_DATA_DIR / "fhvhv_tripdata_2026-03.parquet",
    RAW_DATA_DIR / "fhvhv_tripdata_2026-04.parquet",
]

# Processed data
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DEMAND_PATH = PROCESSED_DIR / "demand_15min.parquet"
ZONE_LOOKUP_PATH = DATA_DIR / "raw" / "taxi_zone_lookup.csv"
ROAD_GRAPH_PATH = DATA_DIR / "raw" / "nyc_road_graph.graphml"

# Model outputs
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
SHAP_DIR = OUTPUT_DIR / "shap"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories on import
for d in [PROCESSED_DIR, DATA_DIR / "raw", MODEL_DIR, SHAP_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# HVFHV Column Mapping
HVFHV_COLUMNS = {
    "pickup_datetime": "pickup_datetime",       # was tpep_pickup_datetime in Yellow Taxi
    "dropoff_datetime": "dropoff_datetime",     # was tpep_dropoff_datetime
    "pickup_location": "PULocationID",          # same in both datasets
    "dropoff_location": "DOLocationID",         # same in both datasets
    "fare": "base_passenger_fare",              # was fare_amount
    "distance": "trip_miles",                   # was trip_distance
    "duration_sec": "trip_time",                # HVFHV has this natively (seconds)
    "provider": "hvfhs_license_num",            # HV0003=Uber, HV0005=Lyft
    "shared_request": "shared_request_flag",    # unique to HVFHV
}

# Columns to actually load from Parquet (saves memory — skip 15+ unused columns)
HVFHV_LOAD_COLUMNS = [
    "hvfhs_license_num",
    "pickup_datetime", "dropoff_datetime",
    "PULocationID", "DOLocationID",
    "trip_miles", "trip_time",
    "base_passenger_fare",
]


# Data Preprocessing Settings

# 15-minute windows for finer-grained predictions
AGGREGATION_FREQ = "15min"
SLOTS_PER_HOUR = 4       # 60 / 15 = 4
SLOTS_PER_DAY = 96        # 24 × 4

# Filters for cleaning raw trip data (adjusted for HVFHV / Uber/Lyft)
MIN_TRIP_DISTANCE = 0.3     # miles — Uber/Lyft trips are rarely < 0.3 mi
MAX_TRIP_DISTANCE = 100.0   # miles — filter out extreme outliers
MIN_FARE = 5.0              # dollars — Uber/Lyft base fares start ~$5
MAX_FARE = 300.0            # dollars — filter out recording errors
MIN_TRIP_DURATION_SEC = 60  # 1 minute minimum
MAX_TRIP_DURATION_SEC = 10800  # 3 hours maximum
INVALID_LOCATION_IDS = [264, 265]  # "Unknown" and "N/A" zones

# Train/test split — month-based (3 months of data)
# Train: February + March 2026 (months 2, 3)
# Test:  April 2026 (month 4)
TRAIN_MONTHS = [2, 3]
TEST_MONTHS = [4]


# Cross-Validation Settings
N_FOLDS = 5
CV_GROUP_COLUMN = "zone_id"  
RANDOM_STATE = 42


# XGBoost Hyperparameters
XGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "early_stopping_rounds": 200,
    "eval_metric": "rmse",
    "verbosity": 0,  
}


# LightGBM Hyperparameters
LGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "num_leaves": 127,        
    "max_depth": -1,          
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,             
    "verbosity": -1,          
}

EARLY_STOPPING_ROUNDS = 200


# Feature List
TIME_FEATURES = [
    "hour", "minute_of_hour",  # minute_of_hour: 0, 15, 30, or 45
    "time_slot",               # 0–95: unique 15-min slot in the day (hour*4 + minute/15)
    "day_of_week", "day_of_month", "month",
    "is_weekend", "is_peak", "is_night",
    "hour_sin", "hour_cos", 
    "dow_sin", "dow_cos",      
    "slot_sin", "slot_cos",    
]

ZONE_FEATURES = [
    "zone_id", "borough_enc",
    "zone_lat", "zone_lng",
]

DEMAND_FEATURES = [
    "zone_demand_mean", "zone_demand_std", "zone_demand_max",
    "zone_slot_demand",       
    "borough_demand_mean",
]

LAG_FEATURES = [
    "demand_lag_1",   
    "demand_lag_4",   
    "demand_lag_96",  
]

ROLLING_FEATURES = [
    "demand_rolling_8",    
    "demand_rolling_96",  
]

ALL_FEATURES = TIME_FEATURES + ZONE_FEATURES + DEMAND_FEATURES + LAG_FEATURES + ROLLING_FEATURES


# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# ETA configuration
ETA_FREEFLOW_SPEED_MPH = 18.0
ETA_MIN_BASE_MINUTES = 3.0

# PostgreSQL database configuration
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/urbanflow",
)

# Auto-detection if inside a Docker container to adjust local port routing
if Path('/.dockerenv').exists():
    import re
    if "@localhost:" in DATABASE_URL or "@127.0.0.1:" in DATABASE_URL:
        DATABASE_URL = re.sub(r'@(localhost|127\.0\.0\.1):\d+', '@db:5432', DATABASE_URL)

DATABASE_ENABLED = os.environ.get("DATABASE_ENABLED", "true").lower() not in {"0", "false", "no"}

# Surge pricing configuration
SURGE_BASE = 1.0        # minimum multiplier (no surge)
SURGE_MAX = 3.0          # maximum multiplier cap
SURGE_THRESHOLD = 0.5    


# JWT Authentication
SECRET_KEY = os.environ.get("SECRET_KEY", "urbanflow-dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "false").lower() not in {"0", "false", "no"}
