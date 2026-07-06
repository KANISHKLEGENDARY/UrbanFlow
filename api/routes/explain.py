"""
UrbanFlow -- Explainability API Routes

Endpoints:
    GET /api/v1/explain/{zone_id}        -- SHAP explanation for a zone prediction
    GET /api/v1/explain/feature-importance -- Global feature importance ranking

This is the innovation that makes the project stand out:
    Most ride-hailing platforms show a surge price but never explain WHY.
    These endpoints provide transparent, SHAP-powered explanations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, Depends, HTTPException, Query
from api.auth.security import get_current_user
from api.schemas import (
    ExplainResponse, FeatureContribution,
    FeatureImportanceResponse, FeatureImportanceItem,
)
from api.services.demand_service import demand_service
import config

router = APIRouter(prefix="/api/v1/explain", tags=["Explainability"])


@router.get("/zone/{zone_id}", response_model=ExplainResponse)
async def explain_prediction(
    zone_id: int,
    hour: int = Query(..., ge=0, le=23),
    minute: int = Query(0, ge=0, le=45),
    day_of_week: int = Query(2, ge=0, le=6),
    day_of_month: int = Query(15, ge=1, le=31),
    month: int = Query(3, ge=1, le=12),
    _user: dict = Depends(get_current_user),
):
    """
    Explain WHY the model predicts a specific demand for this zone.

    Returns a SHAP-based breakdown of which features push the demand
    up or down relative to the model's average prediction.

    Example response:
        "Zone 161 at 5:15 PM -> predicted demand 0.85"
        - zone_slot_demand = 0.78  -> pushed demand UP by +0.25
        - time_slot = 69           -> pushed demand UP by +0.12
        - is_peak = 1              -> pushed demand UP by +0.08
    """
    if not demand_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    if demand_service.shap_explainer is None:
        raise HTTPException(status_code=404, detail="SHAP explainer not available")

    if minute not in {0, 15, 30, 45}:
        raise HTTPException(status_code=422, detail="minute must be one of 0, 15, 30, or 45")

    # Build features and get SHAP explanation
    features = demand_service._build_features(
        zone_id=zone_id,
        hour=hour,
        day_of_week=day_of_week,
        day_of_month=day_of_month,
        minute=minute,
        month=month,
    )
    explanation = demand_service.shap_explainer.explain_single(features)

    # Get zone metadata
    meta = demand_service.zone_stats["zone_meta"].get(zone_id, {})

    top_factors = []
    for item in explanation["top_factors"][:10]:
        top_factors.append(FeatureContribution(
            feature=item["feature"],
            value=item["value"],
            impact=item["impact"],
            direction="positive" if item["impact"] > 0 else "negative",
        ))

    return ExplainResponse(
        zone_id=zone_id,
        zone_name=meta.get("zone_name", f"Zone {zone_id}"),
        hour=hour,
        minute=minute,
        predicted_demand=explanation["prediction"],
        base_value=explanation["base_value"],
        top_factors=top_factors,
    )


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance():
    """
    Global feature importance ranking across all predictions.

    Shows which features the model relies on most -- useful for
    understanding the model and for dashboard visualizations.
    """
    if not demand_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Get importance from XGBoost model
    importance = demand_service.xgb_model.get_feature_importance(config.ALL_FEATURES)

    items = []
    for rank, (feature, score) in enumerate(importance.items(), 1):
        items.append(FeatureImportanceItem(
            rank=rank,
            feature=feature,
            importance=round(float(score), 4),
        ))

    return FeatureImportanceResponse(
        model_name="XGBoost",
        total_features=len(items),
        features=items[:20],  # top 20
    )
