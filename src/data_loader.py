"""
data_loader.py - Data Engine & Parsers for PathSnap Map-Matching Pipeline.

Member 1 Responsibility:
- Build parsers for .track, .nodes, .arcs, and .route files.
- Coordinate conversions (WGS84 -> local metric coordinates / UTM / Haversine).
- Build directed road graph using networkx.DiGraph.
- Chronological ground-truth route alignment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

import networkx as nx
import numpy as np
import pandas as pd

# Earth radius in meters (WGS84 spherical approximation)
EARTH_RADIUS_METERS: float = 6371000.0


def haversine_distance(
    lon1: Union[float, np.ndarray],
    lat1: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute geodesic distance between two points (or arrays of points) in meters
    using the Haversine formula.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def compute_bearing(
    lon1: Union[float, np.ndarray],
    lat1: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute initial bearing (forward azimuth) in degrees from point 1 to point 2.
    Range: [0, 360) degrees.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360.0) % 360.0
    return bearing_deg


def wgs84_to_local_metric(
    lons: Union[float, np.ndarray, pd.Series],
    lats: Union[float, np.ndarray, pd.Series],
    origin_lon: float,
    origin_lat: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert angular WGS84 coordinates (longitude, latitude) in degrees
    to local Euclidean metric coordinates (x, y) in meters relative to an origin point.
    Uses equirectangular projection centered on the origin.
    """
    lons_arr = np.asarray(lons, dtype=np.float64)
    lats_arr = np.asarray(lats, dtype=np.float64)

    cos_lat0 = np.cos(np.radians(origin_lat))
    x = np.radians(lons_arr - origin_lon) * EARTH_RADIUS_METERS * cos_lat0
    y = np.radians(lats_arr - origin_lat) * EARTH_RADIUS_METERS
    return x, y


def local_metric_to_wgs84(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    origin_lon: float,
    origin_lat: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert local Euclidean coordinates (x, y) in meters back to WGS84 (lon, lat).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    cos_lat0 = np.cos(np.radians(origin_lat))
    lon = origin_lon + np.degrees(x_arr / (EARTH_RADIUS_METERS * cos_lat0))
    lat = origin_lat + np.degrees(y_arr / EARTH_RADIUS_METERS)
    return lon, lat


def load_track(track_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load .track file containing GPS observations for a trajectory.
    Format: 3 whitespace-separated columns without headers:
            [longitude, latitude, timestamp_seconds].
    """
    track_path = Path(track_path)
    if not track_path.exists():
        raise FileNotFoundError(f"Track file not found: {track_path}")

    df = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
        dtype={"lon": np.float64, "lat": np.float64, "timestamp": np.float64},
    )

    # Derive sequence index and basic trajectory stats
    df["step_idx"] = np.arange(len(df), dtype=int)
    df["dt"] = df["timestamp"].diff().fillna(1.0)
    return df


def load_nodes(nodes_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load .nodes file containing geographical road network nodes.
    Format: 2 whitespace-separated columns without headers:
            [longitude, latitude] where 0-based row index is node_id.
    """
    nodes_path = Path(nodes_path)
    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")

    df = pd.read_csv(
        nodes_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": np.float64, "lat": np.float64},
    )
    df["node_id"] = df.index
    return df


def load_arcs(arcs_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load .arcs file containing directed road network segments.
    Format: 2 integer columns without headers:
            [from_node_id, to_node_id] where 0-based row index is arc_id.
    """
    arcs_path = Path(arcs_path)
    if not arcs_path.exists():
        raise FileNotFoundError(f"Arcs file not found: {arcs_path}")

    df = pd.read_csv(
        arcs_path,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype={"from_node": np.int64, "to_node": np.int64},
    )
    df["arc_id"] = df.index
    return df


def load_route(route_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load .route file containing chronological sequence of ground-truth arc IDs.
    Format: 1 integer column without headers: arc_id.
    """
    route_path = Path(route_path)
    if not route_path.exists():
        raise FileNotFoundError(f"Route file not found: {route_path}")

    df = pd.read_csv(
        route_path,
        header=None,
        names=["arc_id"],
        dtype={"arc_id": np.int64},
    )
    df["seq_order"] = np.arange(len(df), dtype=int)
    return df


DEFAULT_SEARCH_PATHS = [
    "data/Student folder/train",
    "data/Student folder/student_test",
    "data/Student folder/evaluator_test",
    "data/train",
    "data/student_test",
    "data",
    ".",
    "hack/data",
]


def resolve_trip_directory(trip_id: str, base_dirs: Optional[list[str]] = None) -> Path:
    """
    Resolve trip directory across possible locations:
    - data/Student folder/train/{trip_id}
    - data/Student folder/student_test/{trip_id}
    - data/{trip_id}
    """
    if base_dirs is None:
        base_dirs = DEFAULT_SEARCH_PATHS

    for b in base_dirs:
        candidate = Path(b) / trip_id
        if candidate.is_dir():
            return candidate

    # Recursive fallback in data/
    for root_dir in ["data", "."]:
        root_path = Path(root_dir)
        if root_path.is_dir():
            matches = [p for p in root_path.glob(f"**/{trip_id}") if p.is_dir()]
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"Trip directory '{trip_id}' not found in candidates: {base_dirs}"
    )


def list_trips(partition: str = "all", base_data_dir: str = "data") -> list[str]:
    """
    List available trip IDs for a partition ('train', 'student_test', 'all').
    Returns sorted list of trip IDs.
    """
    base = Path(base_data_dir)
    trips: list[str] = []

    if partition in ("train", "all"):
        train_dir = base / "Student folder" / "train"
        if train_dir.is_dir():
            trips.extend([d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    if partition in ("student_test", "test", "all"):
        test_dir = base / "Student folder" / "student_test"
        if test_dir.is_dir():
            trips.extend([d.name for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    # Fallback to direct numeric folders in base if Student folder is absent
    if not trips and base.is_dir():
        trips.extend([d.name for d in base.iterdir() if d.is_dir() and d.name.isdigit()])

    return sorted(list(set(trips)))


def load_trip(
    trip_id: str,
    base_dirs: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Load all 4 files for a trip:
    - {trip_id}.track
    - {trip_id}.nodes
    - {trip_id}.arcs
    - {trip_id}.route (if present)
    """
    trip_dir = resolve_trip_directory(trip_id, base_dirs)

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"

    track = load_track(track_file)
    nodes = load_nodes(nodes_file)
    arcs = load_arcs(arcs_file)

    route = None
    if route_file.exists():
        route = load_route(route_file)

    return {
        "trip_id": trip_id,
        "trip_dir": trip_dir,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }


def build_road_graph(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    origin_lon: Optional[float] = None,
    origin_lat: Optional[float] = None,
) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """
    Construct directed road graph (networkx.DiGraph) from nodes and arcs.
    Computes local metric coordinates, arc lengths (meters), and bearings.

    Returns:
        (G, augmented_arcs_df):
            - G: networkx.DiGraph with nodes having (x, y, lon, lat)
                 and edges having (arc_id, weight, length, bearing).
            - augmented_arcs_df: arcs DataFrame enriched with
              (u_lon, u_lat, v_lon, v_lat, u_x, u_y, v_x, v_y, arc_length, bearing).
    """
    if origin_lon is None:
        origin_lon = float(nodes_df["lon"].mean())
    if origin_lat is None:
        origin_lat = float(nodes_df["lat"].mean())

    # Metric coordinates for all nodes
    node_x, node_y = wgs84_to_local_metric(
        nodes_df["lon"].values,
        nodes_df["lat"].values,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
    )

    nodes_enriched = nodes_df.copy()
    nodes_enriched["x"] = node_x
    nodes_enriched["y"] = node_y

    u_idx = arcs_df["from_node"].values
    v_idx = arcs_df["to_node"].values

    u_lon = nodes_enriched["lon"].values[u_idx]
    u_lat = nodes_enriched["lat"].values[u_idx]
    v_lon = nodes_enriched["lon"].values[v_idx]
    v_lat = nodes_enriched["lat"].values[v_idx]

    u_x = node_x[u_idx]
    u_y = node_y[u_idx]
    v_x = node_x[v_idx]
    v_y = node_y[v_idx]

    dx = v_x - u_x
    dy = v_y - u_y
    arc_lengths = np.hypot(dx, dy)
    bearings = compute_bearing(u_lon, u_lat, v_lon, v_lat)

    augmented_arcs = arcs_df.copy()
    augmented_arcs["u_lon"] = u_lon
    augmented_arcs["u_lat"] = u_lat
    augmented_arcs["v_lon"] = v_lon
    augmented_arcs["v_lat"] = v_lat
    augmented_arcs["u_x"] = u_x
    augmented_arcs["u_y"] = u_y
    augmented_arcs["v_x"] = v_x
    augmented_arcs["v_y"] = v_y
    augmented_arcs["arc_length"] = arc_lengths
    augmented_arcs["bearing"] = bearings

    # Build NetworkX DiGraph
    G = nx.DiGraph()

    # Add node attributes
    for nid, row in nodes_enriched.iterrows():
        G.add_node(
            int(nid),
            lon=float(row["lon"]),
            lat=float(row["lat"]),
            x=float(row["x"]),
            y=float(row["y"]),
        )

    # Add edge attributes
    edge_tuples = []
    for row in augmented_arcs.itertuples():
        edge_tuples.append(
            (
                int(row.from_node),
                int(row.to_node),
                {
                    "arc_id": int(row.arc_id),
                    "weight": float(row.arc_length),
                    "length": float(row.arc_length),
                    "bearing": float(row.bearing),
                },
            )
        )
    G.add_edges_from(edge_tuples)

    return G, augmented_arcs
