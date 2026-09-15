# This file is responsible for transferring the predicitions from backend to frontend. The requests made by frintend are entertained in this file and different functions are responsible for making sure that those requests get completed. The first function gets the status of overall dataset i.e which zone is the busiest, how much demand is there etc. The data comes in the form of a json object and then is automatically parsed into python pydantic object. That pydantic object is then sent further for making predictions and a json object in turn is returned in the response for the particular request. This file also handles the request for the predictions and data of a particular place in NYC selected by user and also for the display of heatmap on the dashboard telling the percentages of predictions zone by zone. The predictions are also logged in the database for future use and analysis.The heatmap is displayed in the form of color and the prediction done by the model in each zone is multiplied by 4000 to get the actual prediction value. The prediction and heatmap endpoints are protected by JWT authentication via FastAPI dependency injection, ensuring only logged-in users can access demand forecasts. Explicit error handling guardrails are also setup in this file to return errors in something wents wrong. 

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
    stats = await database_service.get_prediction_stats()
    return PredictionStatsResponse(**stats)


@router.post("/predict", response_model=DemandResponse)
async def predict_demand(req: DemandRequest, _user: dict = Depends(get_current_user)):
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
    if not demand_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return {"zones": demand_service.get_zone_list()}
