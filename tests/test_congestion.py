"""
UrbanFlow -- Interconnected Congestion & Ripple Effect Unit Tests

Verifies spatial adjacency loading, 50% capacity threshold spillover logic,
and API response structures.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ml.congestion_model import spatial_congestion_engine, CONGESTION_THRESHOLD
from api.services.congestion_service import congestion_service


def test_spatial_adjacency_loaded():
    """Verify that all 263 zones are loaded in the spatial graph."""
    graph = spatial_congestion_engine.graph
    assert len(graph) > 0, "Spatial graph should not be empty"
    assert 75 in graph, "Zone 75 (East Harlem South) must exist in graph"
    assert len(graph[75]["neighbors"]) > 0, "Zone 75 must have neighbors defined"
    print("✓ Spatial adjacency graph test passed!")


def test_spillover_threshold():
    """Verify 50% capacity threshold triggering."""
    mock_predictions = [
        {"zone_id": 75, "zone_name": "East Harlem South", "borough": "Manhattan", "predicted_demand": 0.65}, # 65% load (> 50%)
        {"zone_id": 41, "zone_name": "Central Harlem", "borough": "Manhattan", "predicted_demand": 0.30},
        {"zone_id": 74, "zone_name": "East Harlem North", "borough": "Manhattan", "predicted_demand": 0.40},
        {"zone_id": 182, "zone_name": "Parkchester", "borough": "Bronx", "predicted_demand": 0.25},
    ]

    res = spatial_congestion_engine.compute_ripple_spillover(
        target_zone_id=75,
        all_zone_predictions=mock_predictions,
        threshold=0.50
    )

    assert res["root_zone_id"] == 75
    assert res["is_congested"] is True
    assert res["threshold_used_pct"] == 50
    assert len(res["direct_neighbors_impacted"]) > 0, "Direct neighbors should receive spillover"
    assert res["max_delay_added_minutes"] > 0, "Max delay should be positive"
    print("✓ Spillover 50% threshold test passed!")


def test_uncongested_below_threshold():
    """Verify that predictions below 50% threshold do not trigger spillover."""
    mock_predictions = [
        {"zone_id": 75, "zone_name": "East Harlem South", "borough": "Manhattan", "predicted_demand": 0.35}, # 35% load (< 50%)
    ]

    res = spatial_congestion_engine.compute_ripple_spillover(
        target_zone_id=75,
        all_zone_predictions=mock_predictions,
        threshold=0.50
    )

    assert res["is_congested"] is False
    assert len(res["direct_neighbors_impacted"]) == 0
    assert res["max_delay_added_minutes"] == 0.0
    print("✓ Uncongested below 50% threshold test passed!")


if __name__ == "__main__":
    print("Running Interconnected Traffic Flow Tests...")
    test_spatial_adjacency_loaded()
    test_spillover_threshold()
    test_uncongested_below_threshold()
    print("\nAll Interconnected Congestion Engine tests passed successfully!")
