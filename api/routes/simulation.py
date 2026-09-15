"""
UrbanFlow -- Fleet Simulation & Rebalancing API Routes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, Depends, HTTPException
from api.auth.security import get_current_user
from api.schemas import RebalanceRequest, RebalanceResponse, RelocationOrder
from api.services.demand_service import demand_service
from api.services.simulation_service import simulation_service

router = APIRouter(prefix="/api/v1/simulation", tags=["Fleet Simulation"])


@router.post("/rebalance", response_model=RebalanceResponse)
async def compute_rebalance(
    req: RebalanceRequest,
    _user: dict = Depends(get_current_user),
):
    """
    Run a simulation step for the 15-minute slot and compute rebalancing recommendations.
    
    Relocates vehicles from surplus zones to deficit zones.
    """
    if not demand_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    results = simulation_service.get_rebalancing_orders(
        hour=req.hour,
        minute=req.minute,
        day_of_week=req.day_of_week,
        fleet_size=req.fleet_size,
    )

    # Convert dictionary items into RelocationOrder pydantic objects
    orders = [RelocationOrder(**o) for o in results["orders"]]

    return RebalanceResponse(
        hour=req.hour,
        minute=req.minute,
        day_of_week=req.day_of_week,
        orders=orders,
        metrics=results["metrics"],
    )
