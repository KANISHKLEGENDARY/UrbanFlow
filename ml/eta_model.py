# This file is responsible for the ETA (Estimated Time of Arrival) prediction. It uses OSMNX to find the shortest path between two points and calculates the travel time. It also has a fallback mechanism to calculate the ETA using the haversine distance and the speed of the vehicle. First this file checks if the system is having OSMNX downloaded road graphnetwork of NYC. But that is not available due to memory constraints of the sytem. So this calculates the distance between the two zones of NYC by using havesine formula. Also it calculates the speed of the vehicle by using the fallback_mechanism only. This is also responsible for calculating the traffic multiplier i.e. the extra time taken to travel between the 2 zones due to traffic at last of this file. For ex:- Today is tuesday and the time is 08:30 AM. It is in the category of rush hour so the multiplier will be 0.35, also it is in the slot 00:30 min so the 0.05 multiplier would also be considered. Hence the total multiplier would be 1.40x.This is the actual logic.

import math
from pathlib import Path

import config


def haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_miles = 3958.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_miles * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class GraphETAEstimator:

    def __init__(self, graph_path: Path):
        try:
            import osmnx as ox
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError("osmnx and networkx are required for graph ETA.") from exc

        if not graph_path.exists():
            raise FileNotFoundError(f"Road graph not found: {graph_path}")

        self.ox = ox
        self.nx = nx
        self.graph_path = graph_path
        self.graph = ox.load_graphml(graph_path)

    def estimate_eta(
        self,
        pickup_lat: float,
        pickup_lng: float,
        dropoff_lat: float,
        dropoff_lng: float,
    ) -> dict:
        orig_node = self.ox.nearest_nodes(self.graph, pickup_lng, pickup_lat)
        dest_node = self.ox.nearest_nodes(self.graph, dropoff_lng, dropoff_lat)

        route = self.nx.shortest_path(
            self.graph,
            orig_node,
            dest_node,
            weight="travel_time",
        )

        total_seconds = 0.0
        total_meters = 0.0
        for u, v in zip(route[:-1], route[1:]):
            edge_options = self.graph.get_edge_data(u, v)
            best_edge = min(
                edge_options.values(),
                key=lambda edge: float(edge.get("travel_time", float("inf"))),
            )
            total_seconds += float(best_edge.get("travel_time", 0))
            total_meters += float(best_edge.get("length", 0))

        return {
            "base_eta_minutes": max(total_seconds / 60, config.ETA_MIN_BASE_MINUTES),
            "distance_miles": total_meters / 1609.344,
            "route_summary": "Shortest path on cached NYC OSM road graph",
            "graph_available": True,
        }


class ZoneETAEstimator:

    def __init__(self, graph_path: Path | None = None):
        self.graph_estimator = None
        self.graph_error = None
        graph_path = graph_path or config.ROAD_GRAPH_PATH

        try:
            self.graph_estimator = GraphETAEstimator(graph_path)
        except Exception as exc:
            self.graph_error = str(exc)

    @property
    def graph_available(self) -> bool:
        return self.graph_estimator is not None

    def estimate(
        self,
        pickup_meta: dict,
        dropoff_meta: dict,
        hour: int,
        minute: int,
        day_of_week: int,
    ) -> dict:
        pickup_lat = float(pickup_meta.get("zone_lat", 40.7128))
        pickup_lng = float(pickup_meta.get("zone_lng", -74.0060))
        dropoff_lat = float(dropoff_meta.get("zone_lat", 40.7128))
        dropoff_lng = float(dropoff_meta.get("zone_lng", -74.0060))

        if self.graph_estimator:
            try:
                base = self.graph_estimator.estimate_eta(
                    pickup_lat,
                    pickup_lng,
                    dropoff_lat,
                    dropoff_lng,
                )
            except Exception as exc:
                self.graph_error = str(exc)
                base = self._fallback_eta(
                    pickup_lat,
                    pickup_lng,
                    dropoff_lat,
                    dropoff_lng,
                    pickup_meta,
                    dropoff_meta,
                )
        else:
            base = self._fallback_eta(
                pickup_lat,
                pickup_lng,
                dropoff_lat,
                dropoff_lng,
                pickup_meta,
                dropoff_meta,
            )

        multiplier = self.traffic_multiplier(hour, minute, day_of_week)
        adjusted = base["base_eta_minutes"] * multiplier

        return {
            **base,
            "traffic_multiplier": multiplier,
            "adjusted_eta_minutes": adjusted,
            "graph_error": self.graph_error,
        }

    def _fallback_eta(
        self,
        pickup_lat: float,
        pickup_lng: float,
        dropoff_lat: float,
        dropoff_lng: float,
        pickup_meta: dict,
        dropoff_meta: dict,
    ) -> dict:
        straight_line = haversine_miles(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
        road_distance = max(straight_line * 1.35, 0.25)
        speed = self._fallback_speed_mph(pickup_meta, dropoff_meta)
        base_minutes = max((road_distance / speed) * 60, config.ETA_MIN_BASE_MINUTES)

        return {
            "base_eta_minutes": base_minutes,
            "distance_miles": road_distance,
            "route_summary": "Centroid-distance fallback; run data/download_road_network.py for graph routing",
            "graph_available": False,
        }

    def _fallback_speed_mph(self, pickup_meta: dict, dropoff_meta: dict) -> float:
        boroughs = {pickup_meta.get("borough"), dropoff_meta.get("borough")}
        if "EWR" in boroughs:
            return 24.0
        if boroughs == {"Manhattan"}:
            return 11.0
        if "Manhattan" in boroughs:
            return 15.0
        return config.ETA_FREEFLOW_SPEED_MPH

    def traffic_multiplier(self, hour: int, minute: int, day_of_week: int) -> float:
        multiplier = 1.0

        if hour in [7, 8, 9, 16, 17, 18, 19]:
            multiplier += 0.35
        if day_of_week in [5, 6]:
            multiplier += 0.10
        if day_of_week in [4, 5] and hour in [20, 21, 22, 23, 0, 1, 2]:
            multiplier += 0.20
        if hour in [0, 1, 2, 3, 4, 5]:
            multiplier -= 0.10
        if minute in [30, 45] and hour in [7, 8, 17, 18]:
            multiplier += 0.05

        return round(max(multiplier, 0.8), 2)
