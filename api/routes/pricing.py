"""
UrbanFlow -- Surge Pricing API Routes

Endpoints:
    POST /api/v1/pricing/calculate  -- Calculate surge pricing for a trip
    POST /api/v1/pricing/compare    -- Compare pricing policies side by side
"""

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
    """Return recent pricing history for a zone.

    Returns up to `limit` most recent pricing calculations for the specified
    zone, ordered newest first.
    """
    history = await database_service.get_pricing_history(zone_id=zone_id, limit=limit)
    return [PricingHistoryItem(**item) for item in history]


@router.post("/calculate", response_model=PricingResponse)
async def calculate_pricing(req: PricingRequest, _user: dict = Depends(get_current_user)):
    """
    Calculate surge pricing for a trip with explainable breakdown.

    Returns the surge multiplier AND the factors contributing to it,
    so users understand WHY the price is what it is.
    """
    if not demand_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Get demand prediction first
    prediction = demand_service.predict_zone(
        zone_id=req.zone_id,
        hour=req.hour,
        day_of_week=req.day_of_week,
        day_of_month=req.day_of_month,
        minute=req.minute,
        month=req.month,
    )

    # Calculate surge with explanation
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
    """
    Compare all three pricing policies for the same scenario.
    Useful for the dashboard to show policy trade-offs.
    """
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

    # Add adjusted fares
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
