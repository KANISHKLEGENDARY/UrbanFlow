# This file is responsible for the surge pricing calculations. It takes the request from the frontend and calculates the surge pricing based on the demand prediction. It also logs the pricing in the database for future use. It also handles the request for the history of pricing for a particular zone. The requests made by frontend are entertained in this file and different functions are responsible for making sure that those requests get completed. The first function gets the history of pricing for a particular zone. The second function calculates the surge pricing for a trip. Explicit error handling guardrails are also setup in this file to return errors if something wents wrong. 

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, Depends, HTTPException, Query
from api.auth.security import get_current_user
from api.schemas import PricingRequest, PricingResponse, PricingFactor, PricingHistoryItem
from api.services.database_service import database_service
from api.services.demand_service import demand_service
from api.services.pricing_service import pricing_service

router = APIRouter(prefix="/api/v1/pricing", tags=["Surge Pricing"])


@router.get("/history/{zone_id}", response_model=list[PricingHistoryItem])
async def get_pricing_history(
    zone_id: int,
    limit: int = Query(20, ge=1, le=100, description="Max records to return"),
):
    # This function is responsible for bringing some historical prices of a particular zone from the database. Some limits are being setup to ensure the request made by the user lies within certain range otherwise the API won't work.
    
    history = await database_service.get_pricing_history(zone_id=zone_id, limit=limit)
    return [PricingHistoryItem(**item) for item in history]


@router.post("/calculate", response_model=PricingResponse)
async def calculate_pricing(req: PricingRequest, _user: dict = Depends(get_current_user)):

    # Responsible for calculating surge pricing for a trip with explainable breakdown. Returns the surge multiplier AND the factors contributing to it, so users understand WHY the price is what it is.

    if not demand_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    prediction = demand_service.predict_zone(
        zone_id=req.zone_id,
        hour=req.hour,
        day_of_week=req.day_of_week,
        day_of_month=req.day_of_month,
        minute=req.minute,
        month=req.month,
    )

    pricing = pricing_service.calculate_surge(
        demand=prediction["predicted_demand"],
        hour=req.hour,
        minute=req.minute,
        day_of_week=req.day_of_week,
        zone_name=prediction["zone_name"],
        borough=prediction["borough"],
    )

    response = PricingResponse(
        zone_id=req.zone_id,
        zone_name=prediction["zone_name"],
        surge_multiplier=pricing["surge_multiplier"],
        adjusted_fare=round(req.base_fare * pricing["surge_multiplier"], 2),
        demand_level=pricing["demand_level"],
        factors=[PricingFactor(**f) for f in pricing["factors"]],
    )
    await database_service.log_pricing(req, response.model_dump())
    return response


@router.post("/compare")
async def compare_policies(req: PricingRequest, _user: dict = Depends(get_current_user)):

    # It compares all three pricing policies for the same scenario. It is useful for the dashboard to show policy trade-offs.

    if not demand_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    prediction = demand_service.predict_zone(
        zone_id=req.zone_id,
        hour=req.hour,
        day_of_week=req.day_of_week,
        day_of_month=req.day_of_month,
        minute=req.minute,
        month=req.month,
    )

    policies = pricing_service.compare_policies(
        demand=prediction["predicted_demand"],
        hour=req.hour,
        minute=req.minute,
    )

    for key in policies:
        policies[key]["adjusted_fare"] = round(
            req.base_fare * policies[key]["multiplier"], 2
        )

    return {
        "zone": prediction["zone_name"],
        "borough": prediction["borough"],
        "predicted_demand": prediction["predicted_demand"],
        "base_fare": req.base_fare,
        "policies": policies,
    }
