"""
UrbanFlow -- ETA API Routes

Endpoints:
    POST /api/v1/eta/estimate -- Estimate travel time between two TLC zones
    GET  /api/v1/eta/status   -- ETA service status
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, Depends, HTTPException

from api.auth.security import get_current_user

from api.schemas import ETARequest, ETAResponse, ETAStatsResponse
from api.services.database_service import database_service
from api.services.eta_service import eta_service

router = APIRouter(prefix="/api/v1/eta", tags=["ETA Prediction"])


@router.get("/stats", response_model=ETAStatsResponse)
async def get_eta_stats():
    """Return ETA usage statistics.

    Aggregates total queries, most popular pickup→dropoff routes,
    average ETA, and graph vs fallback usage ratio.
    """
    stats = await database_service.get_eta_stats()
    return ETAStatsResponse(**stats)


@router.post("/estimate", response_model=ETAResponse)
async def estimate_eta(req: ETARequest, _user: dict = Depends(get_current_user)):
    """Estimate ETA between pickup and dropoff zones."""
    if not eta_service.is_loaded:
        raise HTTPException(status_code=503, detail="ETA service not loaded yet")

    try:
        result = eta_service.estimate(
            pickup_zone=req.pickup_zone,
            dropoff_zone=req.dropoff_zone,
            hour=req.hour,
            minute=req.minute,
            day_of_week=req.day_of_week,
        )
        await database_service.log_eta(req, result)
        return ETAResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status")
async def get_eta_status():
    """Return ETA service status."""
    return eta_service.status()
