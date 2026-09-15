# This file is responsible for sending the data of ripple spillover if there is traffic in a city. It basically catches the request from the frontend and sends it to the congestion service to process. This files answers the questions that if a certain place is jammed in a city then what is the effect on the surrounding area and the city as a whole. It also tells the user about the top bottleneck spots in the city. It also provides a way to check the ripple effect in a city for a specific time and day.  

from fastapi import APIRouter, HTTPException, Query, status
from api.schemas import (
    RippleCongestionRequest,
    RippleCongestionResponse,
    CitywideBottlenecksResponse,
)
from api.services.congestion_service import congestion_service

router = APIRouter(prefix="/api/v1/congestion", tags=["Congestion & Interconnected Flow"])

# This function will calculate the ripple effect spreading in other zones due to traffic in a particular zone. It takes the zone_id, hour, minute, day_of_week and threshold as input and returns the ripple effect in the zone. If there is some error like models are laoded properly then the mentioned error will be returned according to their types.
@router.post(
    "/ripple",
    response_model=RippleCongestionResponse,
    summary="Analyze Ripple Congestion Spillover",
    description="Calculates spatial spillover intensity and travel delay propagation originating from a target zone into 1-hop and 2-hop adjacent cells."
)
def get_ripple_congestion(req: RippleCongestionRequest):
    try:
        res = congestion_service.get_zone_ripple(
            zone_id=req.zone_id,
            hour=req.hour,
            minute=req.minute,
            day_of_week=req.day_of_week,
            threshold=req.threshold
        )
        if "error" in res:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=res["error"])
        return res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ripple analysis error: {e}")


# This function is responsible for returning the top 5 places which have demand greater than the threshold. These are refferred to as bottlenecks here. The function takes inputs as query parameters in th URL not as a json body. The function returns the response in the format of CitywideBottlenecksResponse.
@router.get(
    "/bottlenecks",
    response_model=CitywideBottlenecksResponse,
    summary="Get Citywide Primary Bottleneck Hotspots",
    description="Identifies top primary root bottleneck zones exceeding capacity threshold and triggering cascading ripple congestion across NYC."
)
def get_citywide_bottlenecks(
    hour: int = Query(..., ge=0, le=23, description="Hour of day (0-23)"),
    minute: int = Query(0, ge=0, le=45, description="Minute of hour (0, 15, 30, 45)"),
    day_of_week: int = Query(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)"),
    threshold: float = Query(0.50, ge=0.10, le=0.95, description="Capacity threshold (default 0.50)")
):
    try:
        return congestion_service.get_citywide_bottlenecks(
            hour=hour,
            minute=minute,
            day_of_week=day_of_week,
            threshold=threshold
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Bottleneck fetch error: {e}")
