"""
UrbanFlow -- FastAPI Application Entry Point

This is the file that launches the API server.
Run with: uvicorn api.main:app --reload

Endpoints:
    /api/v1/demand/*   -- Demand prediction (single zone, heatmap)
    /api/v1/pricing/*  -- Surge pricing (calculate, compare policies)
    /api/v1/explain/*  -- SHAP explainability (per-zone, global)
    /api/v1/eta/*      -- ETA prediction (route travel time)
    /api/v1/auth/*     -- Authentication (register, login, refresh)
    /health            -- Health check
    /metrics           -- Prometheus metrics
    /docs              -- Swagger UI (auto-generated interactive API docs)
"""

import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from api.routes import demand, pricing, explain, eta
from api.auth import routes as auth_routes
from api.services.demand_service import demand_service
from api.services.database_service import database_service
from api.services.eta_service import eta_service
from api.middleware.metrics import MetricsMiddleware, get_metrics_text
from api.schemas import HealthResponse
import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, clean up on shutdown."""
    print("\n[UrbanFlow] Loading ML models...")
    success = demand_service.load_models()
    if success:
        zones = len(demand_service.zone_stats.get("zone_meta", {}))
        eta_service.load(demand_service)
        await database_service.initialize(demand_service)
        print(f"[UrbanFlow] Ready! {zones} zones available")
        if config.AUTH_ENABLED:
            print("[UrbanFlow] JWT authentication is ENABLED")
        else:
            print("[UrbanFlow] JWT authentication is DISABLED (development mode)")
    else:
        print("[UrbanFlow] WARNING: Models not loaded. Run 'python ml/train.py' first.")
    yield
    await database_service.shutdown()
    print("[UrbanFlow] Shutting down...")


app = FastAPI(
    title="UrbanFlow API",
    description=(
        "AI-powered ride demand prediction and surge pricing optimization "
        "for NYC, built with XGBoost + LightGBM ensemble and SHAP explainability. "
        "Features JWT authentication, ETA prediction, and PostgreSQL persistence."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS -- allow frontend dev servers to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics middleware
app.add_middleware(MetricsMiddleware)

# Register route modules
app.include_router(demand.router)
app.include_router(pricing.router)
app.include_router(explain.router)
app.include_router(eta.router)
app.include_router(auth_routes.router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the API is running and models are loaded."""
    zones = len(demand_service.zone_stats.get("zone_meta", {})) if demand_service.zone_stats else 0
    return HealthResponse(
        status="healthy" if demand_service.is_loaded else "degraded",
        model_loaded=demand_service.is_loaded,
        zones_available=zones,
        eta_loaded=eta_service.is_loaded,
        database_available=database_service.available,
        auth_enabled=config.AUTH_ENABLED,
    )


@app.get("/metrics", tags=["Monitoring"], response_class=PlainTextResponse)
async def prometheus_metrics():
    """Expose Prometheus metrics."""
    return get_metrics_text()


@app.get("/api/v1/debug/diagnose", tags=["Health"])
async def run_diagnostics():
    from api.services.database_service import database_service
    from api.services.demand_service import demand_service
    await database_service.initialize(demand_service)
    from pathlib import Path
    diag_file = Path(config.PROJECT_ROOT) / "db_diagnostic.txt"
    if diag_file.exists():
        return {
            "status": "done",
            "database_available": database_service.available,
            "database_url": config.DATABASE_URL,
            "logs": diag_file.read_text()
        }
    return {"status": "failed", "logs": "Diagnostic file not created"}


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "UrbanFlow API",
        "version": "2.0.0",
        "description": "Ride demand prediction and surge pricing for NYC",
        "docs": "/docs",
        "endpoints": {
            "demand": "/api/v1/demand/",
            "pricing": "/api/v1/pricing/",
            "explain": "/api/v1/explain/",
            "eta": "/api/v1/eta/",
            "auth": "/api/v1/auth/",
            "health": "/health",
            "metrics": "/metrics",
        },
    }
