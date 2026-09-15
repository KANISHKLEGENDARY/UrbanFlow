"""Quick end-to-end API test."""
import requests
import json

base = "http://127.0.0.1:8000"

# 1. Heatmap
r = requests.post(f"{base}/api/v1/demand/heatmap", json={"hour": 17, "day_of_week": 2, "day_of_month": 15})
data = r.json()
top3 = sorted(data["zones"], key=lambda x: -x["predicted_demand"])[:3]
print(f"Heatmap: {data['total_zones']} zones, avg_demand={data['avg_demand']:.3f}")
for z in top3:
    print(f"  {z['zone_name']}: {z['predicted_demand']:.3f} ({z['demand_level']})")

# 2. Feature importance
r = requests.get(f"{base}/api/v1/explain/feature-importance")
fi = r.json()
print(f"\nFeatures: {fi['total_features']} total, top = {fi['features'][0]['feature']}")

# 3. Explain zone 161
r = requests.get(f"{base}/api/v1/explain/zone/161?hour=17&day_of_week=2")
ex = r.json()
print(f"\nExplain zone 161: prediction={ex['predicted_demand']:.4f}")
print(f"  Top factor: {ex['top_factors'][0]['feature']} (impact={ex['top_factors'][0]['impact']:.4f})")

print("\n[OK] ALL ENDPOINTS PASSING")
