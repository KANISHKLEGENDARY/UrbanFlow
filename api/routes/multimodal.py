"""
UrbanFlow -- Multi-Modal Choice API Route
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, Depends, HTTPException
from api.auth.security import get_current_user
from api.schemas import MultimodalRequest, MultimodalResponse
from api.services.demand_service import demand_service
from api.services.eta_service import eta_service
from api.services.multimodal_service import multimodal_service

router = APIRouter(prefix="/api/v1/multimodal", tags=["Multi-Modal Choice"])


@router.post("/choice", response_model=MultimodalResponse)
async def calculate_mode_choice(
    req: MultimodalRequest,
    dropoff_zone: Optional[int] = None,
    _user: dict = Depends(get_current_user),
):
    """
    Calculate transport mode choice probabilities (Car, Auto, Bike) for a scenario.
    
    If `dropoff_zone` is provided, computes exact travel distance and duration using the road network.
    Otherwise, uses borough-specific representative distances and durations.
    """
    if not demand_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Get zone metadata to find the borough
    meta = demand_service.zone_stats["zone_meta"].get(req.zone_id, {})
    borough = meta.get("borough", "Unknown").lower()

    # Determine distance and duration
    if dropoff_zone is not None:
        try:
            # Leverage existing road network routing
            eta_res = eta_service.estimate_eta(
                pickup_zone=req.zone_id,
                dropoff_zone=dropoff_zone,
                hour=req.hour,
                minute=req.minute,
                day_of_week=req.day_of_week,
            )
            distance_miles = eta_res["distance_miles"]
            duration_minutes = eta_res["adjusted_eta_minutes"]
        except Exception:
            # Fall back to borough-based averages if routing fails
            distance_miles, duration_minutes = _get_borough_defaults(borough)
    else:
        # Default representative values per borough
        distance_miles, duration_minutes = _get_borough_defaults(borough)

    # Compute choice share probabilities
    shares = multimodal_service.predict_mode_share(
        distance_miles=distance_miles,
        travel_time_minutes=duration_minutes,
        surge_multiplier=req.surge_multiplier,
        hour=req.hour,
        base_fare=req.base_fare,
    )

    return MultimodalResponse(
        probabilities=shares["probabilities"],
        costs=shares["costs"],
        durations=shares["durations"],
    )


def _get_borough_defaults(borough: str) -> tuple[float, float]:
    """Helper returning standard trip distances and travel times per borough."""
    if "manhattan" in borough:
        return 2.2, 12.0  # Short distance, high traffic density
    elif "brooklyn" in borough or "queens" in borough or "bronx" in borough:
        return 4.5, 18.0  # Medium distance
    elif "staten island" in borough:
        return 7.5, 24.0  # Long distance
    elif "ewr" in borough or "newark" in borough:
        return 16.0, 32.0  # Long distance airport trip
    else:
        return 3.5, 15.0  # Global fallback
