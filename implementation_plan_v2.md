# UrbanFlow v2.0 — Implementation Roadmap

6 major upgrades to transform UrbanFlow from a portfolio project into a production-grade system.

---

## Execution Order (Recommended)

```mermaid
graph LR
    A["Phase 1\nHVFHV Data\n+ 15-min"] --> B["Phase 2\nRetrain\nModels"]
    B --> C["Phase 3\nETA Module\nOSMNX"]
    C --> D["Phase 4\nPostgreSQL\nDatabase"]
    D --> E["Phase 5\nAuth\nJWT"]
    E --> F["Phase 6\nDocker +\nMonitoring"]
```

> [!IMPORTANT]
> **Do phases 1-3 first.** They add new ML/engineering features. Phases 4-6 are infrastructure — impressive but won't change the dashboard's functionality. If you're short on time, phases 1-3 alone will cover every missing claim from your description.

---

## Phase 1: Switch to HVFHV Data + 15-Minute Aggregation

**Goal:** Replace Yellow Taxi data with Uber/Lyft (HVFHV) data and switch from 1-hour to 15-minute prediction windows.

**Effort:** ~3-4 hours

### Why HVFHV is different from Yellow Taxi

| Aspect | Yellow Taxi (current) | HVFHV (target) |
|--------|----------------------|----------------|
| What it is | Traditional medallion cabs | Uber, Lyft, Via |
| Trips/month | ~3M | ~20M (6-7x more) |
| File size | ~400 MB | ~700 MB – 1 GB |
| Pickup time column | `tpep_pickup_datetime` | `pickup_datetime` |
| Dropoff time column | `tpep_dropoff_datetime` | `dropoff_datetime` |
| Fare column | `fare_amount` | `base_passenger_fare` |
| Distance column | `trip_distance` | `trip_miles` |
| Duration column | (computed) | `trip_time` (seconds, built-in!) |
| Extra columns | — | `request_datetime`, `on_scene_datetime`, `hvfhs_license_num`, `shared_ride_flag` |
| Download URL | `yellow_tripdata_YYYY-MM.parquet` | `fhvhv_tripdata_YYYY-MM.parquet` |

> [!TIP]
> HVFHV has `trip_time` (duration in seconds) as a native column — no need to compute it from pickup/dropoff. It also has `request_datetime` (when rider opened the app) which is gold for ETA modeling.

### Data download

```
https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-01.parquet
```

### Files to modify

#### [MODIFY] `config.py`
```python
# Change raw data path
RAW_DATA_PATH = Path(r"...\fhvhv_tripdata_2023-01.parquet")

# Change aggregation window
AGGREGATION_FREQ = "15min"  # was "1h"

# Update column mappings (new section)
COLUMN_MAP = {
    "pickup_datetime": "pickup_datetime",      # was tpep_pickup_datetime
    "dropoff_datetime": "dropoff_datetime",    # was tpep_dropoff_datetime
    "pickup_location": "PULocationID",          # same
    "dropoff_location": "DOLocationID",         # same
    "fare": "base_passenger_fare",              # was fare_amount
    "distance": "trip_miles",                   # was trip_distance
    "duration": "trip_time",                    # NEW: native seconds column
}

# Update data filters for HVFHV
MIN_FARE = 5.0          # Uber/Lyft minimum fares are higher
MAX_FARE = 300.0
MIN_TRIP_DISTANCE = 0.5  # HVFHV has fewer ultra-short trips
```

#### [MODIFY] `data/download_nyc_data.py`
- Change the download URL to `fhvhv_tripdata_2023-01.parquet`
- Add a function to download the file if it doesn't exist (it's ~700 MB)

#### [MODIFY] `data/preprocess.py`
This is the biggest change. Every column reference needs updating:

| Current code | New code |
|-------------|----------|
| `df["tpep_pickup_datetime"]` | `df["pickup_datetime"]` |
| `df["tpep_dropoff_datetime"]` | `df["dropoff_datetime"]` |
| `df["fare_amount"]` | `df["base_passenger_fare"]` |
| `df["trip_distance"]` | `df["trip_miles"]` |
| Computing `trip_duration_sec` from timestamps | Using native `df["trip_time"]` directly |
| `AGGREGATION_FREQ = "1h"` | `AGGREGATION_FREQ = "15min"` |

Additional preprocessing needed for HVFHV:
- Filter by `hvfhs_license_num` if you want only Uber (`HV0003`) or Lyft (`HV0005`)
- Handle `shared_ride_flag` — shared rides should count as demand but affect pricing differently
- HVFHV has ~20M rows per month vs. 3M for Yellow Taxi — consider sampling or using only 1 week for faster iteration during development

#### Impact on downstream

| File | Impact |
|------|--------|
| `ml/features.py` | **No changes needed** — features are computed from `demand`, `zone_id`, `hour`, etc. which remain the same shape |
| `ml/train.py` | **No changes needed** — reads `train.parquet` which has the same column names after preprocessing |
| `api/` | **No changes needed** — API operates on model predictions, not raw data |

> [!WARNING]
> **15-minute aggregation will 4x the row count** (from ~65K to ~260K rows). Training time will increase proportionally. Also, the `hour` feature in `config.py` should be replaced or augmented with a `time_slot` (0-95 for 15-min windows) or kept as `hour` with an additional `minute_of_hour` (0, 15, 30, 45).

### New features to add for 15-min windows

Add to `config.py`:
```python
TIME_FEATURES = [
    "hour", "minute_of_hour",  # NEW: 0, 15, 30, 45
    "time_slot",               # NEW: 0-95 (hour*4 + minute/15)
    "day_of_week", "day_of_month", "is_weekend", "is_peak", "is_night",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "slot_sin", "slot_cos",    # NEW: cyclical encoding for 96 slots
]
```

---

## Phase 2: Retrain Models

**Goal:** Retrain XGBoost + LightGBM on the new HVFHV data with 15-min windows.

**Effort:** ~1-2 hours (mostly waiting for training)

### Steps
1. Run `python data/preprocess.py` with new HVFHV data
2. Verify output shape (should be ~260K rows for 15-min windows, ~250 zones)
3. Update `config.ALL_FEATURES` if you added `minute_of_hour`, `time_slot`, `slot_sin`, `slot_cos`
4. Run `python ml/train.py`
5. Expect: Training time ~15-20 min (4x more data)
6. Expect: R² might change — HVFHV data has different patterns than Yellow Taxi
7. Update SHAP analysis — save new plots
8. Update dashboard time slider to support 15-min granularity (96 slots instead of 24 hours)

### Frontend update needed

The hour slider in `App.jsx` currently goes 0-23. For 15-min windows, change to:
- Option A: Keep hour slider (0-23) + add minute dropdown (0, 15, 30, 45)
- Option B: Single slider 0-95 with label showing "HH:MM"

---

## Phase 3: ETA Prediction Module (OSMNX/NetworkX)

**Goal:** Given pickup zone + dropoff zone + time of day, predict how long the trip will take using real NYC road network data.

**Effort:** ~6-8 hours (most complex new feature)

### How it works

```mermaid
graph TD
    A["User selects\npickup zone + dropoff zone"] --> B["Look up zone centroids\n(lat, lng)"]
    B --> C["Find nearest road\nnodes via OSMNX"]
    C --> D["Compute shortest path\nvia NetworkX"]
    D --> E["Sum edge travel_time\n= base ETA"]
    E --> F["Apply time-of-day\nmultiplier from model"]
    F --> G["Return ETA\nin minutes"]
```

### New dependencies

Add to `requirements.txt`:
```
osmnx>=1.9.0
networkx>=3.1
```

### New files to create

#### [NEW] `data/download_road_network.py`

```python
"""
Downloads and caches the NYC road network graph.
First run takes ~2-5 minutes. Subsequent runs load from cache.

Produces:
    data/raw/nyc_road_graph.graphml  (~50-100 MB)
"""
import osmnx as ox

def download_nyc_graph():
    # Download drivable road network for all 5 boroughs
    G = ox.graph_from_place("New York City, New York, USA", network_type="drive")

    # Add speed and travel time to each edge
    G = ox.add_edge_speeds(G)           # imputes speed_kph from highway type
    G = ox.add_edge_travel_times(G)      # travel_time = length / speed

    # Save to disk (cache for future use)
    ox.save_graphml(G, "data/raw/nyc_road_graph.graphml")
    return G
```

**Graph stats (approximate):**
- NYC "drive" network: ~100K-200K nodes, ~250K-500K edges
- File size: ~50-100 MB as GraphML
- Memory: ~500 MB - 1 GB when loaded
- Download time: 2-5 minutes (one-time)

#### [NEW] `ml/eta_model.py`

This module has TWO approaches combined:

**Approach A — Graph-based base ETA (OSMNX/NetworkX):**
```python
class GraphETAEstimator:
    def __init__(self, graph_path):
        self.G = ox.load_graphml(graph_path)

    def estimate_eta(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
        # Find nearest road nodes
        orig_node = ox.nearest_nodes(self.G, pickup_lng, pickup_lat)
        dest_node = ox.nearest_nodes(self.G, dropoff_lng, dropoff_lat)

        # Shortest path by travel_time
        route = ox.shortest_path(self.G, orig_node, dest_node, weight="travel_time")

        # Sum edge travel times along route
        total_seconds = sum(
            self.G[u][v][0]["travel_time"]
            for u, v in zip(route[:-1], route[1:])
        )
        return total_seconds / 60  # return minutes
```

**Approach B — ML-based time-of-day adjustment:**

The graph gives you "free-flow" ETA (no traffic). But at 5 PM rush hour, a 10-minute trip actually takes 25 minutes. Train a small model on HVFHV's `trip_time` to learn the multiplier:

```python
# Training data: actual_duration / graph_duration = time_multiplier
# Features: hour, day_of_week, pickup_zone, dropoff_zone, is_peak
# Model: Simple XGBoost or even a lookup table
```

**Combined ETA = graph_base_ETA × time_of_day_multiplier**

#### [NEW] `api/routes/eta.py`

New API endpoints:
```
POST /api/v1/eta/estimate
    Request:  { pickup_zone: 161, dropoff_zone: 132, hour: 17, day_of_week: 2 }
    Response: {
        pickup_zone: "Midtown Center",
        dropoff_zone: "JFK Airport",
        base_eta_minutes: 42.5,        # free-flow graph ETA
        adjusted_eta_minutes: 68.3,     # after traffic adjustment
        traffic_multiplier: 1.61,       # "rush hour adds 61% to travel time"
        distance_miles: 18.7,
        route_summary: "via I-678 S / Van Wyck Expy"
    }
```

#### [NEW] `api/services/eta_service.py`

Singleton service (like `demand_service.py`) that loads the graph on startup and serves ETA predictions.

#### [MODIFY] `api/main.py`

Register the new ETA router:
```python
from api.routes import demand, pricing, explain, eta
app.include_router(eta.router)
```

#### [MODIFY] `frontend/src/App.jsx`

Add an ETA panel to the dashboard:
- Two zone selectors: "From" and "To" dropdowns
- Display: estimated travel time, distance, traffic multiplier
- Visual: show the estimated route on a simple map (optional, requires Leaflet/Mapbox)

> [!WARNING]
> **Memory consideration:** The NYC road graph uses ~500 MB - 1 GB RAM. Combined with the existing models (~145 MB), your API server will need ~1.5-2 GB RAM minimum. This is fine for development but important for Docker resource limits.

---

## Phase 4: PostgreSQL Database

**Goal:** Replace flat Parquet files with a proper database for zone metadata, prediction logs, and user data.

**Effort:** ~4-5 hours

### New dependencies

```
sqlalchemy>=2.0
asyncpg>=0.29.0        # async PostgreSQL driver
alembic>=1.13.0        # database migrations
```

### What goes in the database

| Table | Purpose | Currently stored as |
|-------|---------|-------------------|
| `zones` | Zone ID, name, borough, lat, lng | In-memory dict from CSV |
| `predictions_log` | Every prediction made (for analytics) | Not stored at all |
| `users` | User accounts (for auth - Phase 5) | Doesn't exist |
| `zone_stats` | Pre-computed zone statistics | In-memory dict from train.parquet |
| `pricing_history` | Surge pricing log | Not stored at all |

### New files to create

#### [NEW] `db/database.py`
- SQLAlchemy async engine setup
- Session factory
- `DATABASE_URL` from environment variable

#### [NEW] `db/models.py`
- SQLAlchemy ORM models for each table

#### [NEW] `db/migrations/`
- Alembic migration scripts for creating/updating tables

#### [MODIFY] `config.py`
```python
# Database
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/urbanflow")
```

#### [MODIFY] `api/services/demand_service.py`
- On startup: load zone metadata from PostgreSQL instead of CSV
- On each prediction: log the prediction to `predictions_log` table asynchronously (fire-and-forget, don't slow down the response)

### What stays as files (NOT in DB)
- **Model PKL files** — stay on disk (they're 145 MB binary blobs, not relational data)
- **SHAP plots** — stay as PNG files
- **Training data** — stays as Parquet (it's a one-time batch input, not live data)

---

## Phase 5: JWT Authentication

**Goal:** Protect API endpoints with user login and token-based authentication.

**Effort:** ~3-4 hours

### New dependencies

```
python-jose[cryptography]>=3.3.0   # JWT token creation/verification
passlib[bcrypt]>=1.7.4             # password hashing
python-dotenv>=1.0.0               # environment variables
```

### New files to create

#### [NEW] `api/auth/security.py`
- Password hashing (bcrypt)
- JWT token creation (access token + refresh token)
- Token verification middleware
- `SECRET_KEY` from environment variable (never hardcoded)

#### [NEW] `api/auth/routes.py`
```
POST /api/v1/auth/register  → create user account
POST /api/v1/auth/login     → returns JWT access token
POST /api/v1/auth/refresh   → refresh expired token
GET  /api/v1/auth/me        → get current user info
```

#### [MODIFY] `api/routes/demand.py`, `pricing.py`, `explain.py`
- Add `Depends(get_current_user)` to each endpoint
- Health check remains public (no auth)

### Auth flow
```
1. User registers → password hashed with bcrypt → stored in PostgreSQL
2. User logs in → server verifies password → returns JWT (valid 30 min)
3. Frontend stores JWT in localStorage
4. Every API request includes: Authorization: Bearer <token>
5. Server verifies token on each request → allows or rejects
```

#### [MODIFY] `frontend/src/App.jsx`
- Add login/register modal
- Store JWT in localStorage
- Add `Authorization` header to all fetch() calls
- Show user info in header

---

## Phase 6: Docker + Monitoring

**Goal:** Containerize everything and add observability.

**Effort:** ~4-5 hours

### Docker files to create

#### [NEW] `Dockerfile`
```dockerfile
# Multi-stage build for small production image
FROM python:3.12-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Download road network on build (cached in image)
RUN python data/download_road_network.py

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### [NEW] `docker-compose.yml`
```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql+asyncpg://urbanflow:password@db:5432/urbanflow
      - SECRET_KEY=${SECRET_KEY}
    depends_on: [db]
    deploy:
      resources:
        limits:
          memory: 2G    # models + graph need ~1.5 GB

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: urbanflow
      POSTGRES_USER: urbanflow
      POSTGRES_PASSWORD: password
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports: ["5432:5432"]

  frontend:
    build: ./frontend
    ports: ["5173:5173"]
    depends_on: [api]

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    depends_on: [prometheus]

volumes:
  pgdata:
```

#### [NEW] `frontend/Dockerfile`
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]
```

### Monitoring files

#### [NEW] `monitoring/prometheus.yml`
- Scrape FastAPI metrics endpoint every 15 seconds

#### [NEW] `api/middleware/metrics.py`
- Track: request count, latency histogram, error rate, prediction count per zone
- Expose at `GET /metrics` (Prometheus format)

#### [NEW] `monitoring/grafana/dashboard.json`
- Pre-built Grafana dashboard with panels:
  - Request rate (req/sec)
  - P50/P95/P99 latency
  - Error rate
  - Top predicted zones
  - Model prediction distribution
  - Surge pricing distribution

#### [MODIFY] `api/main.py`
- Add Prometheus metrics middleware
- Add structured logging (JSON format)

---

## Updated Project Architecture (After All Phases)

```
urbanflow/
├── docker-compose.yml              # [NEW] orchestrates all services
├── Dockerfile                      # [NEW] API container
├── .env                            # [NEW] secrets (SECRET_KEY, DB_URL)
├── config.py                       # [MODIFIED] HVFHV paths, 15-min, DB URL
├── requirements.txt                # [MODIFIED] new dependencies
│
├── data/
│   ├── download_nyc_data.py        # [MODIFIED] HVFHV download
│   ├── download_road_network.py    # [NEW] OSMNX graph download
│   ├── preprocess.py               # [MODIFIED] HVFHV columns, 15-min
│   └── raw/
│       ├── nyc_road_graph.graphml  # [NEW] cached road network
│       └── taxi_zone_lookup.csv
│
├── db/                             # [NEW] entire directory
│   ├── database.py                 # async SQLAlchemy engine
│   ├── models.py                   # ORM table definitions
│   └── migrations/                 # Alembic migrations
│
├── ml/
│   ├── features.py                 # [MODIFIED] add time_slot features
│   ├── xgboost_model.py            # unchanged
│   ├── lightgbm_model.py           # unchanged
│   ├── ensemble.py                 # unchanged
│   ├── explainer.py                # unchanged
│   ├── train.py                    # [MODIFIED] new feature list
│   └── eta_model.py                # [NEW] graph + ML ETA estimator
│
├── api/
│   ├── main.py                     # [MODIFIED] new routers, metrics
│   ├── schemas.py                  # [MODIFIED] ETA + auth schemas
│   ├── auth/                       # [NEW] entire directory
│   │   ├── security.py             # JWT + bcrypt
│   │   └── routes.py               # login/register endpoints
│   ├── middleware/                  # [NEW]
│   │   └── metrics.py              # Prometheus metrics
│   ├── routes/
│   │   ├── demand.py               # [MODIFIED] add auth dependency
│   │   ├── pricing.py              # [MODIFIED] add auth dependency
│   │   ├── explain.py              # [MODIFIED] add auth dependency
│   │   └── eta.py                  # [NEW] ETA endpoints
│   └── services/
│       ├── demand_service.py       # [MODIFIED] DB integration
│       ├── pricing_service.py      # unchanged
│       └── eta_service.py          # [NEW] graph loading + ETA logic
│
├── monitoring/                     # [NEW] entire directory
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboard.json
│
└── frontend/
    ├── Dockerfile                  # [NEW]
    └── src/
        ├── App.jsx                 # [MODIFIED] ETA panel, auth, 15-min slider
        ├── App.css                 # [MODIFIED] new component styles
        └── index.css               # unchanged
```

---

## Effort Summary

| Phase | What | New Files | Modified Files | Effort |
|-------|------|-----------|----------------|--------|
| 1 | HVFHV + 15-min | 0 | 3 (config, download, preprocess) | 3-4 hrs |
| 2 | Retrain models | 0 | 2 (config features, App.jsx slider) | 1-2 hrs |
| 3 | ETA module | 4 (download_road, eta_model, eta routes, eta service) | 2 (main, schemas) | 6-8 hrs |
| 4 | PostgreSQL | 3+ (database, models, migrations) | 2 (config, demand_service) | 4-5 hrs |
| 5 | JWT Auth | 2 (security, auth routes) | 4 (3 route files, App.jsx) | 3-4 hrs |
| 6 | Docker + Monitoring | 5 (Dockerfiles, compose, prometheus, metrics, grafana) | 1 (main) | 4-5 hrs |
| | **Total** | **~14 new files** | **~10 modified files** | **~22-28 hrs** |

---

## Open Questions for You

> [!IMPORTANT]
> These decisions affect the implementation. Please answer before I start building.

1. **HVFHV month**: Should I use January 2023 (same month as current Yellow Taxi) or a more recent month like January 2024? Newer data = more relevant, but 2023 matches your current setup.

2. **Uber only or all HVFHV?**: HVFHV includes Uber (`HV0003`), Lyft (`HV0005`), Via (`HV0004`). Do you want all three, or filter to Uber-only for a cleaner narrative?

3. **15-min vs keep 1-hour for API?**: The model will train on 15-min windows, but should the dashboard slider also switch to 15-min granularity (96 time slots)? Or keep the hourly slider and aggregate 15-min predictions into hourly for display?

4. **ETA visual**: Should the ETA panel be a simple text display (pickup zone → dropoff zone → estimated time), or do you want a map visualization showing the actual route? Map requires adding Leaflet.js or Mapbox — adds another ~4 hours.

5. **Database hosting**: For PostgreSQL, are you okay with running it locally via Docker, or do you want a cloud-hosted option (like Supabase or Neon free tier)?

6. **Priority order**: Do you agree with the recommended phase order (1→2→3→4→5→6), or do you want to reorder? For example, if Docker is more urgent for deployment, we could do it earlier.
