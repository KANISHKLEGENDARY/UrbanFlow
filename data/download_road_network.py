# This file is responsible for downloading the online map of newyork city using python's osmnx library and saving it to a graphml file. The map is downloaded in smaller chunks to reduce load on the RAM and an update mechanism is being used such that the user gets an update about ho much data has been downloaded and what is the total size of the data. If the system already contians the data then code skips the download and informs the user about the same.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def download_nyc_graph():
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError("Install osmnx before downloading the road network.") from exc

    config.ROAD_GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)

    if config.ROAD_GRAPH_PATH.exists():
        print(f"[OK] Road graph already exists: {config.ROAD_GRAPH_PATH}")
        return ox.load_graphml(config.ROAD_GRAPH_PATH)

    ox.settings.log_console = True
    ox.settings.timeout = 600 
    ox.settings.overpass_rate_limit = False

    NYC_NORTH = 40.9176
    NYC_SOUTH = 40.4774
    NYC_EAST = -73.7004
    NYC_WEST = -74.2591

    print("Downloading NYC drivable road network from OpenStreetMap...")
    print("(This may take 3-10 minutes for the full NYC area)")
    graph = ox.graph_from_bbox(
        bbox=(NYC_NORTH, NYC_SOUTH, NYC_EAST, NYC_WEST),
        network_type="drive",
    )
    print(f"Downloaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print("Adding edge speeds and travel times...")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    ox.save_graphml(graph, config.ROAD_GRAPH_PATH)
    print(f"[OK] Saved road graph: {config.ROAD_GRAPH_PATH}")
    return graph


if __name__ == "__main__":
    download_nyc_graph()
