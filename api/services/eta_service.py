"""
UrbanFlow -- ETA Service

Loads the Phase 3 ETA estimator and serves pickup/dropoff zone travel-time
estimates. The service uses the cached OSMNX road graph when available and
falls back to centroid-distance estimates when the graph has not been cached.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.eta_model import ZoneETAEstimator
import config


class ETAService:
    """Singleton service for route ETA estimates."""

    def __init__(self):
        self.estimator = None
        self.zone_meta = {}
        self.is_loaded = False

    def load(self, demand_service) -> bool:
        """Load zone metadata and initialize the ETA estimator."""
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
        """Estimate ETA between two zones."""
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
        """Return ETA service status."""
        return {
            "loaded": self.is_loaded,
            "graph_available": bool(self.estimator and self.estimator.graph_available),
            "road_graph_path": str(config.ROAD_GRAPH_PATH),
            "zones_available": len(self.zone_meta),
        }


eta_service = ETAService()
