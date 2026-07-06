"""
UrbanFlow -- Pricing Service

Computes surge pricing multipliers based on predicted demand.
Supports three pricing policies:
    1. Flat    -- Always 1.0x (no surge, baseline)
    2. Reactive -- Simple demand-threshold based scaling
    3. Predictive -- Uses demand forecast + time-of-day adjustments

Also provides SHAP-powered explanations for WHY the surge is what it is.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config


class PricingService:
    """Computes surge pricing and provides pricing explanations."""

    def calculate_surge(
        self,
        demand: float,
        hour: int,
        minute: int = 0,
        day_of_week: int = 0,
        zone_name: str = "",
        borough: str = "",
    ) -> dict:
        """
        Calculate surge multiplier with breakdown of contributing factors.

        Args:
            demand: Predicted normalized demand [0-1]
            hour: Hour of day
            minute: Minute of hour (0, 15, 30, 45)
            day_of_week: 0=Mon, 6=Sun
            zone_name: For display purposes
            borough: For display purposes

        Returns:
            Dict with surge_multiplier and factor breakdown
        """
        factors = []
        surge = config.SURGE_BASE

        # Factor 1: Base demand level
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

        # Factor 2: Rush hour
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

        # Factor 3: Late night
        if hour in [0, 1, 2, 3, 4]:
            night_add = 0.2
            factors.append({
                "factor": "Late Night",
                "impact": f"+{night_add:.1f}x",
                "description": "Limited driver availability after midnight",
            })
            surge += night_add

        # Factor 4: Weekend nights
        if day_of_week in [4, 5] and hour in [21, 22, 23, 0, 1, 2]:
            weekend_add = 0.2
            factors.append({
                "factor": "Weekend Nightlife",
                "impact": f"+{weekend_add:.1f}x",
                "description": "Friday/Saturday night high demand",
            })
            surge += weekend_add

        # Cap at maximum
        surge = min(surge, config.SURGE_MAX)

        return {
            "surge_multiplier": round(surge, 2),
            "factors": factors,
            "demand_level": self._demand_level(demand),
        }

    def _demand_level(self, demand: float) -> str:
        """Classify demand into human-readable level."""
        if demand < 0.25:
            return "low"
        elif demand < 0.5:
            return "moderate"
        elif demand < 0.75:
            return "high"
        return "critical"

    def compare_policies(self, demand: float, hour: int, minute: int = 0) -> dict:
        """Compare pricing across different policies."""
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
        """Simple threshold-based reactive pricing."""
        if demand <= 0.5:
            return 1.0
        return 1.0 + (demand - 0.5) * 4.0  # max 3.0x at demand=1.0

    def _predictive_surge(self, demand: float, hour: int, minute: int = 0) -> float:
        """Anticipatory pricing that accounts for upcoming demand patterns."""
        base = self._reactive_surge(demand)
        # Anticipate upcoming rush: boost slightly in the hour before rush
        if hour in [6, 16] or (hour in [7, 17] and minute == 0):
            base += 0.15
        return min(base, config.SURGE_MAX)


pricing_service = PricingService()
