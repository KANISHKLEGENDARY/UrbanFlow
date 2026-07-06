"""
UrbanFlow -- API Data Models (Pydantic Schemas)

Defines the exact shape of every API request and response.
This is the "contract" between the frontend and backend.
Pydantic validates data automatically -- if the frontend sends
zone_id = "abc" instead of an integer, the API returns a clear error.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ─────────────────────────────────────────────
# Demand Prediction
# ─────────────────────────────────────────────

class DemandRequest(BaseModel):
    """Request body for single zone demand prediction."""
    zone_id: int = Field(..., ge=1, le=263, description="NYC taxi zone ID (1-263)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    minute: int = Field(0, ge=0, le=45, description="Minute of hour (0, 15, 30, 45)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    day_of_month: int = Field(15, ge=1, le=31, description="Day of month")
    month: int = Field(3, ge=1, le=12, description="Month (1-12)")
    is_weekend: Optional[int] = Field(None, description="Auto-computed if not provided")

    @field_validator("minute")
    @classmethod
    def minute_must_be_quarter_hour(cls, value: int) -> int:
        if value not in {0, 15, 30, 45}:
            raise ValueError("minute must be one of 0, 15, 30, or 45")
        return value


class DemandResponse(BaseModel):
    """Response for a single zone demand prediction."""
    zone_id: int
    zone_name: str
    borough: str
    predicted_demand: float = Field(..., description="Normalized demand [0-1]")
    demand_raw: float = Field(..., description="Estimated pickup count")
    surge_multiplier: float = Field(..., description="Recommended surge price multiplier")
    demand_level: str = Field(..., description="low / moderate / high / critical")


class HeatmapRequest(BaseModel):
    """Request for demand heatmap across all zones."""
    hour: int = Field(..., ge=0, le=23)
    minute: int = Field(0, ge=0, le=45)
    day_of_week: int = Field(..., ge=0, le=6)
    day_of_month: int = Field(15, ge=1, le=31)
    month: int = Field(3, ge=1, le=12)

    @field_validator("minute")
    @classmethod
    def minute_must_be_quarter_hour(cls, value: int) -> int:
        if value not in {0, 15, 30, 45}:
            raise ValueError("minute must be one of 0, 15, 30, or 45")
        return value


class ZoneDemand(BaseModel):
    """Single zone entry in the heatmap response."""
    zone_id: int
    zone_name: str
    borough: str
    lat: float
    lng: float
    predicted_demand: float
    demand_level: str
    surge_multiplier: float


class HeatmapResponse(BaseModel):
    """Response containing demand for all zones (for the map)."""
    hour: int
    minute: int
    day_of_week: int
    total_zones: int
    avg_demand: float
    zones: list[ZoneDemand]


# ─────────────────────────────────────────────
# Surge Pricing
# ─────────────────────────────────────────────

class PricingRequest(BaseModel):
    """Request for surge pricing calculation."""
    zone_id: int = Field(..., ge=1, le=263)
    hour: int = Field(..., ge=0, le=23)
    minute: int = Field(0, ge=0, le=45)
    day_of_week: int = Field(..., ge=0, le=6)
    day_of_month: int = Field(15, ge=1, le=31)
    month: int = Field(3, ge=1, le=12)
    base_fare: float = Field(10.0, ge=0, description="Base fare in dollars")

    @field_validator("minute")
    @classmethod
    def minute_must_be_quarter_hour(cls, value: int) -> int:
        if value not in {0, 15, 30, 45}:
            raise ValueError("minute must be one of 0, 15, 30, or 45")
        return value


class PricingFactor(BaseModel):
    """Individual factor contributing to surge pricing."""
    factor: str
    impact: str
    description: str


class PricingResponse(BaseModel):
    """Response with surge pricing breakdown."""
    zone_id: int
    zone_name: str
    surge_multiplier: float
    adjusted_fare: float
    demand_level: str
    factors: list[PricingFactor]


# ─────────────────────────────────────────────
# ETA Prediction
# ─────────────────────────────────────────────

class ETARequest(BaseModel):
    """Request for pickup-zone to dropoff-zone ETA estimation."""
    pickup_zone: int = Field(..., ge=1, le=263)
    dropoff_zone: int = Field(..., ge=1, le=263)
    hour: int = Field(..., ge=0, le=23)
    minute: int = Field(0, ge=0, le=45)
    day_of_week: int = Field(..., ge=0, le=6)

    @field_validator("minute")
    @classmethod
    def minute_must_be_quarter_hour(cls, value: int) -> int:
        if value not in {0, 15, 30, 45}:
            raise ValueError("minute must be one of 0, 15, 30, or 45")
        return value


class ETAResponse(BaseModel):
    """Response for an ETA estimate."""
    pickup_zone: int
    pickup_zone_name: str
    dropoff_zone: int
    dropoff_zone_name: str
    hour: int
    minute: int
    day_of_week: int
    base_eta_minutes: float
    adjusted_eta_minutes: float
    traffic_multiplier: float
    distance_miles: float
    route_summary: str
    graph_available: bool

# ─────────────────────────────────────────────
# Explainability (SHAP)
# ─────────────────────────────────────────────

class FeatureContribution(BaseModel):
    """Single feature's contribution to a prediction."""
    feature: str
    value: float
    impact: float
    direction: str = Field(..., description="positive or negative")


class ExplainResponse(BaseModel):
    """SHAP explanation for a specific prediction."""
    zone_id: int
    zone_name: str
    hour: int
    minute: int = 0
    predicted_demand: float
    base_value: float = Field(..., description="Model's average prediction")
    top_factors: list[FeatureContribution]


class FeatureImportanceItem(BaseModel):
    """Single feature in the global importance ranking."""
    rank: int
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    """Global feature importance ranking."""
    model_name: str
    total_features: int
    features: list[FeatureImportanceItem]


# ─────────────────────────────────────────────
# Health / Meta
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    zones_available: int
    eta_loaded: bool = False
    database_available: bool = False
    auth_enabled: bool = False
    version: str = "2.0.0"


# ─────────────────────────────────────────────
# Analytics / Database
# ─────────────────────────────────────────────

class PredictionStatsResponse(BaseModel):
    """Aggregated prediction analytics from the database."""
    total_predictions: int
    predictions_by_type: dict
    top_zones: list[dict]
    avg_demand: float
    predictions_by_hour: dict
    database_available: bool


class PricingHistoryItem(BaseModel):
    """Single pricing history record."""
    zone_id: int
    hour: int
    minute: int
    day_of_week: int
    base_fare: float
    surge_multiplier: float
    adjusted_fare: float
    demand_level: str
    created_at: str


class ETAStatsResponse(BaseModel):
    """Aggregated ETA usage statistics from the database."""
    total_queries: int
    popular_routes: list[dict]
    avg_eta_minutes: float
    graph_usage_ratio: float
    database_available: bool


# ─────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────

class AuthRegisterRequest(BaseModel):
    """Request body for user registration."""
    username: str = Field(..., min_length=3, max_length=80, description="Unique username")
    email: str = Field(..., min_length=5, max_length=255, description="Email address")
    password: str = Field(..., min_length=6, max_length=128, description="Password (min 6 chars)")


class AuthLoginRequest(BaseModel):
    """Request body for user login."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class AuthRefreshRequest(BaseModel):
    """Request body for token refresh."""
    refresh_token: str = Field(..., description="Valid refresh token")


class AuthTokenResponse(BaseModel):
    """Response with JWT access + refresh tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    username: str
    email: str


class AuthUserResponse(BaseModel):
    """Response with current user info."""
    id: int
    username: str
    email: str
    is_admin: bool = False
