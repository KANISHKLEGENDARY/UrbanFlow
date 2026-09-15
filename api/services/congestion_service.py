# This file mainly meant for the manager of the platform to see the congested places or the places with high traffic are affecting which more places leading to the increases in the travel time of the customer. Its main feature is to take the predictions of the demands of the rider by the customers and then orchestrate spillover intensity calculation and make sure that these get delievered to the dashboard. This feature is only meant for the managerial decision making purposes of the platform though the service layer itself is role-agnostic.

import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.congestion_model import spatial_congestion_engine, CONGESTION_THRESHOLD
from api.services.demand_service import demand_service


class CongestionService:
    def __init__(self):
        self.engine = spatial_congestion_engine

    def get_zone_ripple(
        self,
        zone_id: int,
        hour: int,
        minute: int,
        day_of_week: int,
        threshold: float = CONGESTION_THRESHOLD
    ) -> Dict[str, Any]:
        if not demand_service.is_loaded:
            return {
                "error": "Demand service not loaded. Run model training first.",
                "root_zone_id": zone_id,
                "is_congested": False
            }

        predictions = demand_service.predict_all_zones(
            hour=hour,
            minute=minute,
            day_of_week=day_of_week
        )

        return self.engine.compute_ripple_spillover(
            target_zone_id=zone_id,
            all_zone_predictions=predictions,
            threshold=threshold
        )

    def get_citywide_bottlenecks(
        self,
        hour: int,
        minute: int,
        day_of_week: int,
        threshold: float = CONGESTION_THRESHOLD
    ) -> Dict[str, Any]:
        if not demand_service.is_loaded:
            return {"bottlenecks": [], "total_congested_zones": 0}

        predictions = demand_service.predict_all_zones(
            hour=hour,
            minute=minute,
            day_of_week=day_of_week
        )

        congested = [p for p in predictions if p.get("predicted_demand", 0.0) >= threshold]
        congested.sort(key=lambda x: x.get("predicted_demand", 0.0), reverse=True)

        top_bottlenecks = []
        for p in congested[:5]: 
            ripple = self.engine.compute_ripple_spillover(
                target_zone_id=p["zone_id"],
                all_zone_predictions=predictions,
                threshold=threshold
            )
            top_bottlenecks.append({
                "zone_id": p["zone_id"],
                "zone_name": p["zone_name"],
                "borough": p["borough"],
                "demand_pct": round(p["predicted_demand"] * 100, 1),
                "surge_multiplier": p.get("surge_multiplier", 1.0),
                "total_affected_neighbors": ripple["total_affected_zones"],
                "max_added_delay_min": ripple["max_delay_added_minutes"],
                "severity": ripple["severity_level"]
            })

        return {
            "hour": hour,
            "minute": minute,
            "day_of_week": day_of_week,
            "threshold_used_pct": int(threshold * 100),
            "total_congested_zones": len(congested),
            "bottlenecks": top_bottlenecks
        }


congestion_service = CongestionService()
