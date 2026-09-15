# The SimulationService acts as a virtual fleet dispatcher. It initializes vehicle distribution proportional to historical zone demand, then periodically predicts demand for all zones and identifies supply-demand mismatches. Surplus and deficit zones are matched using a greedy bipartite algorithm that prioritizes the shortest haversine distance pairs to minimize deadhead miles. For each relocation order, it estimates travel duration assuming 18 mph average speed plus dispatch overhead. Finally, it computes operational metrics including total deadhead miles, estimated wait time reduction, and fleet utilization percentage, returning the top 30 relocation orders for dashboard visualization.

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.services.demand_service import demand_service

class SimulationService:
    def __init__(self):
        self.fleet_distribution = {}  
        self.total_fleet_size = 0

    def initialize_fleet(self, fleet_size: int = 500) -> None:
        if not demand_service.is_loaded:
            return

        self.total_fleet_size = fleet_size
        self.fleet_distribution = {}

        zone_means = demand_service.zone_stats.get("zone_means", {})
        total_mean = sum(zone_means.values()) if zone_means else 1.0

        all_zone_ids = sorted(demand_service.zone_stats["zone_meta"].keys())

        allocated = 0
        for zid in all_zone_ids:
            mean_demand = zone_means.get(zid, 0)
            share = mean_demand / total_mean if total_mean > 0 else 1.0 / len(all_zone_ids)
            vehicles = int(round(share * fleet_size))
            self.fleet_distribution[zid] = vehicles
            allocated += vehicles

        diff = fleet_size - allocated
        if diff != 0 and all_zone_ids:
            highest_zone = max(zone_means, key=zone_means.get)
            self.fleet_distribution[highest_zone] += diff

    def get_rebalancing_orders(
        self,
        hour: int,
        minute: int,
        day_of_week: int,
        fleet_size: int = 500,
    ) -> dict:
        if not demand_service.is_loaded:
            return {"orders": [], "metrics": {}}

        if self.total_fleet_size != fleet_size or not self.fleet_distribution:
            self.initialize_fleet(fleet_size)

        predictions = demand_service.predict_all_zones(
            hour=hour,
            minute=minute,
            day_of_week=day_of_week,
        )

        surpluses = []
        deficits = []
        expected_demand_total = 0.0

        for pred in predictions:
            zid = pred["zone_id"]
            predicted_demand = pred["predicted_demand"]
            raw_max = demand_service.zone_stats["raw_maxs"].get(zid, 50.0)
            
            needed_vehicles = int(round(predicted_demand * (raw_max / 4.0)))
            if pred["demand_level"] in ["high", "critical"] and needed_vehicles == 0:
                needed_vehicles = 1
                
            expected_demand_total += needed_vehicles
            current_supply = self.fleet_distribution.get(zid, 0)
            diff = current_supply - needed_vehicles

            if diff > 1:
                surpluses.append({
                    "zone_id": zid,
                    "zone_name": pred["zone_name"],
                    "lat": pred["lat"],
                    "lng": pred["lng"],
                    "available": int(diff),
                })
            elif diff < -1:
                deficits.append({
                    "zone_id": zid,
                    "zone_name": pred["zone_name"],
                    "lat": pred["lat"],
                    "lng": pred["lng"],
                    "needed": int(abs(diff)),
                })

        orders = []
        total_deadhead_miles = 0.0
        total_relocated_vehicles = 0
        total_deficits_resolved = 0

        pairs = []
        for s in surpluses:
            for d in deficits:
                dist = self._haversine(s["lat"], s["lng"], d["lat"], d["lng"])
                pairs.append((dist, s, d))

        pairs.sort(key=lambda x: x[0])

        surplus_map = {s["zone_id"]: s["available"] for s in surpluses}
        deficit_map = {d["zone_id"]: d["needed"] for d in deficits}

        for dist, s, d in pairs:
            s_id = s["zone_id"]
            d_id = d["zone_id"]

            avail = surplus_map.get(s_id, 0)
            need = deficit_map.get(d_id, 0)

            if avail > 0 and need > 0:
                move_count = min(avail, need)
                surplus_map[s_id] -= move_count
                deficit_map[d_id] -= move_count

                duration_min = (dist / 18.0) * 60.0 + 2.0  

                orders.append({
                    "from_zone_id": s_id,
                    "from_zone_name": s["zone_name"],
                    "to_zone_id": d_id,
                    "to_zone_name": d["zone_name"],
                    "vehicle_count": move_count,
                    "distance_miles": round(dist, 2),
                    "duration_minutes": round(duration_min, 1),
                })

                self.fleet_distribution[s_id] -= move_count
                self.fleet_distribution[d_id] += move_count

                total_deadhead_miles += dist * move_count
                total_relocated_vehicles += move_count
                total_deficits_resolved += move_count

        wait_reduction = 0.0
        if expected_demand_total > 0:
            wait_reduction = (total_deficits_resolved * 6.5) / max(1.0, expected_demand_total)
            wait_reduction = min(6.5, wait_reduction)  

        utilization = 0.0
        if fleet_size > 0:
            active_vehicles = sum(
                min(self.fleet_distribution.get(pred["zone_id"], 0), int(round(pred["predicted_demand"] * (raw_max / 4.0))))
                for pred in predictions
                if (raw_max := demand_service.zone_stats["raw_maxs"].get(pred["zone_id"], 50.0))
            )
            utilization = (active_vehicles / fleet_size) * 100.0
            utilization = min(95.0, max(20.0, utilization))  

        metrics = {
            "total_fleet_size": fleet_size,
            "vehicles_relocated": total_relocated_vehicles,
            "total_deadhead_miles": round(total_deadhead_miles, 1),
            "wait_time_reduction_minutes": round(wait_reduction, 1),
            "fleet_utilization_pct": round(utilization, 1),
            "deficit_zones_count": len(deficits),
            "surplus_zones_count": len(surpluses),
        }

        orders.sort(key=lambda x: x["vehicle_count"], reverse=True)

        return {
            "hour": hour,
            "minute": minute,
            "day_of_week": day_of_week,
            "orders": orders[:30],  
            "metrics": metrics,
        }

    def _haversine(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        R = 3958.8  # Earth radius in miles
        d_lat = math.radians(lat2 - lat1)
        d_lng = math.radians(lng2 - lng1)
        a = (math.sin(d_lat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(d_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

simulation_service = SimulationService()
