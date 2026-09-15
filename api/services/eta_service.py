# This file is responsibel for handing over the json report of the ETA calculation between the 2 zones which the user has requested for. This file calls the ZoneETAEstimator class instance which further takes the inputs of the of the zones and then runs the necessary function required for ETA calculation. The ZoneETAEstimator instance then returns the ETA report to this file. This file in turn sends that ETA report to the concerned routing file for displaying on the dashboard. This file initializes the estimator, which internally attempts to load the road graph and falls back to centroid-based estimation if unavailable.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.eta_model import ZoneETAEstimator
import config


class ETAService:

    def __init__(self):
        self.estimator = None
        self.zone_meta = {}
        self.is_loaded = False

    def load(self, demand_service) -> bool:
        if not demand_service.zone_stats:
            self.is_loaded = False
            return False

        self.zone_meta = demand_service.zone_stats.get("zone_meta", {})
        self.estimator = ZoneETAEstimator(config.ROAD_GRAPH_PATH)
        self.is_loaded = True

        if self.estimator.graph_available:
            print("[ETAService] Loaded cached OSMNX road graph")
        else:
            print("[ETAService] Road graph unavailable; using centroid fallback")
        return True

    def estimate(
        self,
        pickup_zone: int,
        dropoff_zone: int,
        hour: int,
        minute: int,
        day_of_week: int,
    ) -> dict:
        if not self.is_loaded or self.estimator is None:
            raise RuntimeError("ETA service not loaded.")

        pickup_meta = self.zone_meta.get(pickup_zone)
        dropoff_meta = self.zone_meta.get(dropoff_zone)
        if not pickup_meta:
            raise ValueError(f"Unknown pickup zone: {pickup_zone}")
        if not dropoff_meta:
            raise ValueError(f"Unknown dropoff zone: {dropoff_zone}")

        eta = self.estimator.estimate(
            pickup_meta=pickup_meta,
            dropoff_meta=dropoff_meta,
            hour=hour,
            minute=minute,
            day_of_week=day_of_week,
        )

        return {
            "pickup_zone": pickup_zone,
            "pickup_zone_name": pickup_meta.get("zone_name", f"Zone {pickup_zone}"),
            "dropoff_zone": dropoff_zone,
            "dropoff_zone_name": dropoff_meta.get("zone_name", f"Zone {dropoff_zone}"),
            "hour": hour,
            "minute": minute,
            "day_of_week": day_of_week,
            "base_eta_minutes": round(eta["base_eta_minutes"], 1),
            "adjusted_eta_minutes": round(eta["adjusted_eta_minutes"], 1),
            "traffic_multiplier": eta["traffic_multiplier"],
            "distance_miles": round(eta["distance_miles"], 1),
            "route_summary": eta["route_summary"],
            "graph_available": eta["graph_available"],
        }

    def status(self) -> dict:
        return {
            "loaded": self.is_loaded,
            "graph_available": bool(self.estimator and self.estimator.graph_available),
            "road_graph_path": str(config.ROAD_GRAPH_PATH),
            "zones_available": len(self.zone_meta),
        }

eta_service = ETAService()
