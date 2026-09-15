# This file is mainly responsible for the calculation of the surge multiplier and demand level at a particular zone inside the NYC city. The surge multiplier is being decided by percentages of demands in the zone and also the hour of time in that particular zone. According to that the code adjusts the surge mutiplier and returns it along with the proper explanation behind it. Also according to the percentages of the demand it is being decided in this code file only that the demand in particular zone id critical, moderate, or low. At last an intelligent strategy of calculating the surge multiplier is being used to compare three strategies of pricing policy which are as follows: flat, reactive, and predictive. These are being used to compare surge pricing multiplier strategy. The compare_policies utility, along with its helper methods _reactive_surge and _predictive_surge, is an analytics tool that returns side-by-side pricing scenarios for internal dashboards, A/B test planning, or business intelligence. It does not override the main price it provides visibility into alternative strategies.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


class PricingService:

    def calculate_surge(
        self,
        demand: float,
        hour: int,
        minute: int = 0,
        day_of_week: int = 0,
        zone_name: str = "",
        borough: str = "",
    ) -> dict:
        factors = []
        surge = config.SURGE_BASE

        if demand > 0.75:
            demand_add = 1.2
            factors.append({
                "factor": "Critical Demand",
                "impact": f"+{demand_add:.1f}x",
                "description": f"Demand at {demand:.0%} capacity -- extremely busy",
            })
        elif demand > 0.5:
            demand_add = 0.6
            factors.append({
                "factor": "High Demand",
                "impact": f"+{demand_add:.1f}x",
                "description": f"Demand at {demand:.0%} capacity -- above average",
            })
        elif demand > config.SURGE_THRESHOLD:
            demand_add = 0.3
            factors.append({
                "factor": "Moderate Demand",
                "impact": f"+{demand_add:.1f}x",
                "description": f"Demand at {demand:.0%} capacity -- slightly elevated",
            })
        else:
            demand_add = 0.0
            factors.append({
                "factor": "Normal Demand",
                "impact": "+0.0x",
                "description": f"Demand at {demand:.0%} capacity -- no surge needed",
            })
        surge += demand_add

        if hour in [7, 8, 9]:
            rush_add = 0.3
            factors.append({
                "factor": "Morning Rush",
                "impact": f"+{rush_add:.1f}x",
                "description": f"Peak commute window ({hour:02d}:{minute:02d})",
            })
            surge += rush_add
        elif hour in [17, 18, 19]:
            rush_add = 0.3
            factors.append({
                "factor": "Evening Rush",
                "impact": f"+{rush_add:.1f}x",
                "description": f"Peak commute window ({hour:02d}:{minute:02d})",
            })
            surge += rush_add

        if hour in [0, 1, 2, 3, 4]:
            night_add = 0.2
            factors.append({
                "factor": "Late Night",
                "impact": f"+{night_add:.1f}x",
                "description": "Limited driver availability after midnight",
            })
            surge += night_add

        if day_of_week in [4, 5] and hour in [21, 22, 23, 0, 1, 2]:
            weekend_add = 0.2
            factors.append({
                "factor": "Weekend Nightlife",
                "impact": f"+{weekend_add:.1f}x",
                "description": "Friday/Saturday night high demand",
            })
            surge += weekend_add

        surge = min(surge, config.SURGE_MAX)

        return {
            "surge_multiplier": round(surge, 2),
            "factors": factors,
            "demand_level": self._demand_level(demand),
        }

    def _demand_level(self, demand: float) -> str:
        if demand < 0.25:
            return "low"
        elif demand < 0.5:
            return "moderate"
        elif demand < 0.75:
            return "high"
        return "critical"

    def compare_policies(self, demand: float, hour: int, minute: int = 0) -> dict:
        return {
            "flat": {"multiplier": 1.0, "policy": "Flat (no surge)"},
            "reactive": {
                "multiplier": round(self._reactive_surge(demand), 2),
                "policy": "Reactive Surge",
            },
            "predictive": {
                "multiplier": round(self._predictive_surge(demand, hour, minute), 2),
                "policy": "Predictive Surge",
            },
        }

    def _reactive_surge(self, demand: float) -> float:
        if demand <= 0.5:
            return 1.0
        return 1.0 + (demand - 0.5) * 4.0  # max 3.0x at demand=1.0

    def _predictive_surge(self, demand: float, hour: int, minute: int = 0) -> float:
        base = self._reactive_surge(demand)
        if hour in [6, 16] or (hour in [7, 17] and minute == 0):
            base += 0.15
        return min(base, config.SURGE_MAX)


pricing_service = PricingService()
