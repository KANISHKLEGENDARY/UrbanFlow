"""
UrbanFlow -- Zone Adjacency Graph Generator

Builds a spatial adjacency lookup (data/raw/zone_adjacency.json) connecting
each of the 263 NYC Taxi Zones to its nearest spatial neighbors (1-hop)
and major cross-borough transit corridors (bridges & tunnels).
"""

import json
import math
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ZONE_LOOKUP_PATH = PROJECT_ROOT / "data" / "raw" / "taxi_zone_lookup.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "zone_adjacency.json"

# Approximate centroids for NYC boroughs/zones if not provided, plus known key bridge/tunnel corridors
# Cross-borough bridges/tunnels:
# Manhattan (Yellow) <-> Brooklyn (Boro Zone): Williamsburg, Manhattan, Brooklyn Bridges
# Manhattan <-> Queens: Queensboro Bridge, Queens-Midtown Tunnel
# Manhattan <-> Bronx: Third Ave, Willis Ave, Triborough/RFK Bridges
# Staten Island <-> Brooklyn: Verrazzano-Narrows Bridge
CORRIDOR_PAIRS = [
    # Manhattan <-> Queens
    (161, 7),   # Midtown East -> Astoria
    (162, 138), # Midtown East -> LIC/Hunters Point
    (237, 7),   # Upper East Side South -> Astoria
    # Manhattan <-> Brooklyn
    (13, 181),  # Battery Park City -> Park Slope
    (12, 25),   # Battery Park -> Brooklyn Heights
    (148, 255), # Lower East Side -> Williamsburg
    (144, 255), # Little Italy -> Williamsburg
    # Manhattan <-> Bronx
    (75, 182),  # East Harlem South -> Mott Haven
    (74, 182),  # East Harlem North -> Mott Haven
    (127, 182), # Inwood -> Mott Haven
    # Staten Island <-> Brooklyn
    (14, 6),    # Bay Ridge -> Arrochar
]

def main():
    if not ZONE_LOOKUP_PATH.exists():
        print(f"Error: {ZONE_LOOKUP_PATH} not found.")
        return

    df = pd.read_csv(ZONE_LOOKUP_PATH)
    zones = {}
    for idx, row in df.iterrows():
        zid = int(row["LocationID"])
        zones[zid] = {
            "zone_id": zid,
            "borough": row["Borough"],
            "zone_name": row["Zone"],
        }

    # Generate spatial distance matrix heuristic based on borough & location
    # Groups by borough and finds nearest 3-6 neighbors for each zone
    adjacency = {}
    
    # We load demand service zone stats if available for exact centroids
    processed_train = PROJECT_ROOT / "data" / "processed" / "train.parquet"
    centroids = {}
    if processed_train.exists():
        try:
            train_df = pd.read_parquet(processed_train)
            if "lat" in train_df.columns and "lng" in train_df.columns:
                grouped = train_df.groupby("zone_id")[["lat", "lng"]].mean()
                for zid, row in grouped.iterrows():
                    centroids[int(zid)] = (float(row["lat"]), float(row["lng"]))
        except Exception as e:
            print(f"Centroid load warning: {e}")

    # Build neighbor graph
    for zid, zinfo in zones.items():
        neighbors = []
        borough = zinfo["borough"]
        
        # 1. Borough-based spatial neighbors
        same_borough_ids = [z for z, info in zones.items() if info["borough"] == borough and z != zid]
        
        if zid in centroids:
            lat1, lng1 = centroids[zid]
            # Distance-sorted neighbors in same borough
            dists = []
            for n_id in same_borough_ids:
                if n_id in centroids:
                    lat2, lng2 = centroids[n_id]
                    d = math.sqrt((lat2 - lat1)**2 + (lng2 - lng1)**2)
                    dists.append((d, n_id))
            dists.sort(key=lambda x: x[0])
            neighbors.extend([n_id for d, n_id in dists[:5]])
        else:
            # Fallback index offset neighbors
            neighbors.extend(same_borough_ids[:4])

        # 2. Add cross-borough corridor links
        for u, v in CORRIDOR_PAIRS:
            if zid == u and v not in neighbors:
                neighbors.append(v)
            elif zid == v and u not in neighbors:
                neighbors.append(u)

        adjacency[str(zid)] = {
            "zone_id": zid,
            "zone_name": zinfo["zone_name"],
            "borough": borough,
            "neighbors": list(set(neighbors))
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(adjacency, f, indent=2)

    print(f"[ZoneAdjacency] Created {OUTPUT_PATH} with {len(adjacency)} zone entries.")

if __name__ == "__main__":
    main()
