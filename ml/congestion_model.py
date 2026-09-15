# This file is responsible for calculating the ripple effect i.e. the spilling of traffic from one place to another place. For ex:- if a cricket match is going on in a stadium in new delhi and the match finishesat 9pm then the people from the stadium will leave all together from the stadium and will head to their homes. The model will predict high rider demands in that zone already and the riders will be placed there. But there will be traffic in new Delhi and that traffic will get spilled to neighbouring district like Faridabd, Gurgaon etc. This spilling percentage of traffic and the time increased in travel will be calculated in this file only in the below mentioned formulas. It is being done for 2 levels:- First for the immediate neighbours of that particular zone, and then for the neighbours of the neighbours of that zone. The hardcoded values in these formulas is being used just to show the concept of traffic spillover without rquiring traffic engineering datasets which are hard to obtain. In the real ride demand prediction system, these willl be replaced by BPR volume-delay function and GPS probe data. 

import json
from pathlib import Path
from typing import Dict, List, Any
import config

CONGESTION_THRESHOLD = 0.50

class SpatialCongestionEngine:

    def __init__(self, adjacency_path: Path = None):
        self.adjacency_path = adjacency_path or (config.DATA_DIR / "raw" / "zone_adjacency.json")
        self.graph: Dict[int, Dict[str, Any]] = {}
        self._load_adjacency()

    def _load_adjacency(self):
        if not self.adjacency_path.exists():
            print(f"[CongestionEngine] Warning: Adjacency file missing at {self.adjacency_path}")
            return

        try:
            with open(self.adjacency_path, "r") as f:
                data = json.load(f)
                for zid_str, info in data.items():
                    zid = int(zid_str)
                    self.graph[zid] = {
                        "zone_id": zid,
                        "zone_name": info.get("zone_name", f"Zone {zid}"),
                        "borough": info.get("borough", "Unknown"),
                        "neighbors": [int(n) for n in info.get("neighbors", [])]
                    }
            print(f"[CongestionEngine] Loaded spatial graph for {len(self.graph)} zones.")
        except Exception as e:
            print(f"[CongestionEngine] Failed to load adjacency graph: {e}")

    def compute_ripple_spillover(
        self,
        target_zone_id: int,
        all_zone_predictions: List[Dict[str, Any]],
        threshold: float = CONGESTION_THRESHOLD
    ) -> Dict[str, Any]:
        pred_map = {p["zone_id"]: p for p in all_zone_predictions}
        target_pred = pred_map.get(target_zone_id)

        if not target_pred:
            return {
                "root_zone_id": target_zone_id,
                "root_zone_name": f"Zone {target_zone_id}",
                "borough": "Unknown",
                "capacity_load_pct": 0.0,
                "is_congested": False,
                "threshold_used": threshold,
                "severity_level": "normal",
                "direct_neighbors_impacted": [],
                "secondary_neighbors_impacted": [],
                "total_affected_zones": 0,
                "max_delay_added_minutes": 0.0,
                "mitigation_advice": "No congestion detected in origin cell."
            }

        root_demand = target_pred.get("predicted_demand", 0.0)
        root_name = target_pred.get("zone_name", f"Zone {target_zone_id}")
        root_borough = target_pred.get("borough", "Unknown")

        is_congested = root_demand >= threshold
        excess = max(0.0, root_demand - threshold)

        severity_level = "normal"
        if root_demand >= 0.80:
            severity_level = "critical"
        elif root_demand >= 0.65:
            severity_level = "high"
        elif root_demand >= threshold: 
            severity_level = "moderate"

        direct_impacts = []
        secondary_impacts = []
        seen_zones = {target_zone_id}

        root_info = self.graph.get(target_zone_id, {"neighbors": []})
        direct_neighbors = root_info.get("neighbors", [])

        if is_congested and excess > 0:
            for n_id in direct_neighbors:
                if n_id in seen_zones:
                    continue
                seen_zones.add(n_id)

                n_pred = pred_map.get(n_id, {})
                n_name = n_pred.get("zone_name", self.graph.get(n_id, {}).get("zone_name", f"Zone {n_id}"))
                n_borough = n_pred.get("borough", self.graph.get(n_id, {}).get("borough", "Unknown"))
                base_demand = n_pred.get("predicted_demand", 0.2)

                #Main formulas for calculating spillover and delay..
                spillover_intensity = min(0.95, round(excess * 1.85 + (base_demand * 0.3), 2))
                added_delay_min = round(spillover_intensity * 8.5, 1)

                direct_impacts.append({
                    "zone_id": n_id,
                    "zone_name": n_name,
                    "borough": n_borough,
                    "hop_distance": 1,
                    "base_demand_pct": round(base_demand * 100, 1),
                    "spillover_intensity_pct": round(spillover_intensity * 100, 1),
                    "delay_added_minutes": added_delay_min,
                    "status": "high_spillover" if spillover_intensity > 0.4 else "moderate_spillover"
                })

            direct_impacts.sort(key=lambda x: x["spillover_intensity_pct"], reverse=True)

            for d_item in direct_impacts[:4]:
                d_id = d_item["zone_id"]
                d_neighbors = self.graph.get(d_id, {}).get("neighbors", [])
                
                for s_id in d_neighbors:
                    if s_id in seen_zones:
                        continue
                    seen_zones.add(s_id)

                    s_pred = pred_map.get(s_id, {})
                    s_name = s_pred.get("zone_name", self.graph.get(s_id, {}).get("zone_name", f"Zone {s_id}"))
                    s_borough = s_pred.get("borough", self.graph.get(s_id, {}).get("borough", "Unknown"))
                    s_base_demand = s_pred.get("predicted_demand", 0.15)

                    sec_intensity = min(0.70, round(d_item["spillover_intensity_pct"] / 100.0 * 0.55, 2))
                    sec_delay_min = round(sec_intensity * 4.5, 1)

                    if sec_intensity > 0.10:
                        secondary_impacts.append({
                            "zone_id": s_id,
                            "zone_name": s_name,
                            "borough": s_borough,
                            "hop_distance": 2,
                            "base_demand_pct": round(s_base_demand * 100, 1),
                            "spillover_intensity_pct": round(sec_intensity * 100, 1),
                            "delay_added_minutes": sec_delay_min,
                            "status": "secondary_ripple"
                        })

            secondary_impacts.sort(key=lambda x: x["spillover_intensity_pct"], reverse=True)

        max_delay = max([i["delay_added_minutes"] for i in direct_impacts + secondary_impacts], default=0.0)
        total_affected = len(direct_impacts) + len(secondary_impacts)

        mitigation_advice = "Traffic flow operating normally. No spillover rerouting required."
        if severity_level == "critical":
            mitigation_advice = f"CRITICAL BOTTLE-NECK in {root_name}: Divert 25% incoming fleet to perimeter 1-hop corridors immediately."
        elif severity_level == "high":
            mitigation_advice = f"HIGH SPILLOVER: Reroute cross-borough dispatches around {root_name} to avoid +{max_delay}m adjacent delay."
        elif is_congested:
            mitigation_advice = f"MODERATE LOAD (>50%): Monitor 1-hop neighbors ({len(direct_impacts)} adjacent cells affected)."

        return {
            "root_zone_id": target_zone_id,
            "root_zone_name": root_name,
            "borough": root_borough,
            "capacity_load_pct": round(root_demand * 100, 1),
            "is_congested": is_congested,
            "threshold_used_pct": int(threshold * 100),
            "severity_level": severity_level,
            "direct_neighbors_impacted": direct_impacts[:6],
            "secondary_neighbors_impacted": secondary_impacts[:6],
            "total_affected_zones": total_affected,
            "max_delay_added_minutes": max_delay,
            "mitigation_advice": mitigation_advice
        }

spatial_congestion_engine = SpatialCongestionEngine()
