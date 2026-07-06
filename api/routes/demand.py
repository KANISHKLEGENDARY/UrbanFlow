"""
UrbanFlow -- Demand Prediction API Routes

Endpoints:
    POST /api/v1/demand/predict     -- Predict demand for a specific zone + time
    POST /api/v1/demand/heatmap     -- Get demand for ALL 263 zones (for the map)
    GET  /api/v1/demand/zones       -- List all available zones with metadata
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, Depends, HTTPException
from api.auth.security import get_current_user
from api.schemas import (
    DemandRequest, DemandResponse,
    HeatmapRequest, HeatmapResponse, ZoneDemand,
    PredictionStatsResponse,
)
from api.services.database_service import database_service
from api.services.demand_service import demand_service

router = APIRouter(prefix="/api/v1/demand", tags=["Demand Prediction"])


@router.get("/stats", response_model=PredictionStatsResponse)
async def get_prediction_stats():
    """Return prediction analytics from the database.

    Aggregates total prediction count, breakdown by type, top queried zones,
    average demand, and hourly distribution.
    """
    stats = await database_service.get_prediction_stats()
    return PredictionStatsResponse(**stats)


@router.post("/predict", response_model=DemandResponse)
async def predict_demand(req: DemandRequest, _user: dict = Depends(get_current_user)):
    """
    Predict ride demand for a specific zone at a specific time.

    Example: "What's the demand at zone 161 (Midtown) at 5 PM on a Tuesday?"
    """
    if not demand_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    try:
        result = demand_service.predict_zone(
            zone_id=req.zone_id,
            hour=req.hour,
            day_of_week=req.day_of_week,
            day_of_month=req.day_of_month,
            minute=req.minute,
            month=req.month,
        )
        await database_service.log_prediction("single", req, result)
        return DemandResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/heatmap", response_model=HeatmapResponse)
async def get_heatmap(req: HeatmapRequest, _user: dict = Depends(get_current_user)):
    """
    Get demand predictions for ALL zones at a given time.
    Used by the frontend to render the demand heatmap on the map.
    """
    if not demand_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    try:
        zones = demand_service.predict_all_zones(
            hour=req.hour,
            day_of_week=req.day_of_week,
            day_of_month=req.day_of_month,
            minute=req.minute,
            month=req.month,
        )

        avg_demand = sum(z["predicted_demand"] for z in zones) / len(zones) if zones else 0
        await database_service.log_prediction(
            "heatmap",
            req,
            {"avg_demand": round(avg_demand, 4), "predicted_demand": round(avg_demand, 4)},
        )

        return HeatmapResponse(
            hour=req.hour,
            minute=req.minute,
            day_of_week=req.day_of_week,
            total_zones=len(zones),
            avg_demand=round(avg_demand, 4),
            zones=[ZoneDemand(**z) for z in zones],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/zones")
async def list_zones():
    """List all available NYC taxi zones with metadata."""
    if not demand_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return {"zones": demand_service.get_zone_list()}
