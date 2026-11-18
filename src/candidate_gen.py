"""
candidate_gen.py - Spatial Indexing & Candidate Generation for PathSnap.

Member 1 Responsibility:
- Spatial index using shapely.STRtree and scipy.spatial.KDTree.
- Query top-K candidate arcs (e.g. K=5-10 within 50-100m) for each GPS observation.
- Tag ground-truth candidate arcs matching .route (Contract 1 compliant).
- Robust handling of GPS dropouts and isolated observations.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import scipy.spatial
import shapely

from src.data_loader import wgs84_to_local_metric, load_trip, build_road_graph


def align_trajectory_to_route(
    track_df: pd.DataFrame,
    route_df: pd.DataFrame,
    augmented_arcs_df: pd.DataFrame,
) -> np.ndarray:
    """
    Chronologically align each GPS observation to the active ground-truth arc in the route.
    Returns an array of arc_ids of length len(track_df), where element t is the ground-truth
    arc traversed by the vehicle at time t.
    """
    route_arc_ids = route_df["arc_id"].values
    route_arcs = augmented_arcs_df.loc[route_arc_ids]

    # Pre-extract segment lines for route arcs
    u_x = route_arcs["u_x"].values
    u_y = route_arcs["u_y"].values
    v_x = route_arcs["v_x"].values
    v_y = route_arcs["v_y"].values

    coords_u = np.column_stack([u_x, u_y])
    coords_v = np.column_stack([v_x, v_y])
    route_lines = shapely.linestrings(np.stack([coords_u, coords_v], axis=1))

    # GPS points
    gps_x = track_df["x"].values
    gps_y = track_df["y"].values
    gps_pts = shapely.points(gps_x, gps_y)

    n_pts = len(gps_pts)
    n_route = len(route_lines)

    aligned_arcs = np.zeros(n_pts, dtype=np.int64)
    route_idx = 0

    # Sequential forward-tracking window
    for t in range(n_pts):
        pt = gps_pts[t]
        w_start = max(0, route_idx - 2)
        w_end = min(n_route, route_idx + 15)

        sub_lines = route_lines[w_start:w_end]
        sub_dists = shapely.distance(pt, sub_lines)

        best_sub_idx = int(np.argmin(sub_dists))
        route_idx = w_start + best_sub_idx
        aligned_arcs[t] = route_arc_ids[route_idx]

    return aligned_arcs


class SpatialCandidateGenerator:
    """
    Spatial Candidate Generator for road networks.
    Supports both shapely.STRtree (exact segment R-tree) and scipy.spatial.KDTree.
    """

    def __init__(
        self,
        augmented_arcs_df: pd.DataFrame,
        origin_lon: float,
        origin_lat: float,
        index_type: str = "strtree",
    ):
        self.arcs_df = augmented_arcs_df.copy()
        self.origin_lon = origin_lon
        self.origin_lat = origin_lat
        self.index_type = index_type.lower()

        self.u_x = self.arcs_df["u_x"].values
        self.u_y = self.arcs_df["u_y"].values
        self.v_x = self.arcs_df["v_x"].values
        self.v_y = self.arcs_df["v_y"].values
        self.arc_ids = self.arcs_df["arc_id"].values

        # Build geometry lines
        coords_u = np.column_stack([self.u_x, self.u_y])
        coords_v = np.column_stack([self.v_x, self.v_y])
        self.lines = shapely.linestrings(np.stack([coords_u, coords_v], axis=1))

        # Midpoints for KDTree
        self.mid_x = (self.u_x + self.v_x) / 2.0
        self.mid_y = (self.u_y + self.v_y) / 2.0
        self.half_lens = self.arcs_df["arc_length"].values / 2.0

        # Construct spatial indexes
        self.strtree = shapely.STRtree(self.lines)
        self.kdtree = scipy.spatial.KDTree(np.column_stack([self.mid_x, self.mid_y]))

    def query_point(
        self,
        px: float,
        py: float,
        k: int = 10,
        max_distance: float = 100.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query candidate arcs for a single point (px, py) in local metric coords.
        Returns: (candidate_arc_ids, distances)
        """
        pt = shapely.Point(px, py)

        if self.index_type == "kdtree":
            # Search radius expanded by max segment half-length in local area (or 100m buffer)
            search_radius = max_distance + 150.0
            idx_list = self.kdtree.query_ball_point([px, py], r=search_radius)
            if len(idx_list) == 0:
                # Global fallback to nearest single arc
                _, nearest_idx = self.kdtree.query([px, py], k=1)
                idx_list = [nearest_idx]
            cand_indices = np.array(idx_list, dtype=int)
            cand_lines = self.lines[cand_indices]
            dists = shapely.distance(pt, cand_lines)
            valid = dists <= max_distance
            if np.any(valid):
                cand_indices = cand_indices[valid]
                dists = dists[valid]
            sort_order = np.argsort(dists)[:k]
            return self.arc_ids[cand_indices[sort_order]], dists[sort_order]
        else:
            # STRtree
            buf = shapely.buffer(pt, max_distance)
            cand_indices = self.strtree.query(buf, predicate="intersects")
            if len(cand_indices) == 0:
                # Nearest fallback
                nearest_idx = self.strtree.nearest(pt)
                cand_indices = np.array([nearest_idx], dtype=int)
            cand_lines = self.lines[cand_indices]
            dists = shapely.distance(pt, cand_lines)
            sort_order = np.argsort(dists)[:k]
            return self.arc_ids[cand_indices[sort_order]], dists[sort_order]

    def generate_candidates(
        self,
        trip_id: str,
        track_df: pd.DataFrame,
        route_df: Optional[pd.DataFrame] = None,
        k: int = 10,
        max_distance: float = 100.0,
    ) -> pd.DataFrame:
        """
        Generate candidate road pairs for all observations in track_df.

        Returns DataFrame matching Contract 1 schema:
        {
            "trip_id": str,
            "step_idx": int,
            "timestamp": float,
            "gps_lon": float,
            "gps_lat": float,
            "arc_id": int,
            "from_node": int,
            "to_node": int,
            "arc_length": float,
            "label": int,             # 1 if arc is the ground truth arc, 0 otherwise
            "dist_to_arc": float,     # perpendicular distance in meters
            "in_route": int           # 1 if arc_id in entire trip route set, 0 otherwise
        }
        """
        # Ensure metric coordinates exist in track_df
        if "x" not in track_df.columns or "y" not in track_df.columns:
            track_x, track_y = wgs84_to_local_metric(
                track_df["lon"].values,
                track_df["lat"].values,
                origin_lon=self.origin_lon,
                origin_lat=self.origin_lat,
            )
            track_df = track_df.copy()
            track_df["x"] = track_x
            track_df["y"] = track_y

        gps_x = track_df["x"].values
        gps_y = track_df["y"].values
        gps_pts = shapely.points(gps_x, gps_y)
        n_pts = len(gps_pts)

        # Ground truth route alignment if route_df is provided
        aligned_gt_arcs = None
        route_arcs_set = set()
        if route_df is not None:
            aligned_gt_arcs = align_trajectory_to_route(track_df, route_df, self.arcs_df)
            route_arcs_set = set(route_df["arc_id"].values)

        # Batch query using STRtree for ultra-fast vectorized retrieval
        buffers = shapely.buffer(gps_pts, max_distance)
        pt_idx, arc_idx = self.strtree.query(buffers, predicate="intersects")

        # Compute point-to-segment distances
        dists = shapely.distance(gps_pts[pt_idx], self.lines[arc_idx])

        # Assemble candidate pairs
        raw_candidates = pd.DataFrame({
            "step_idx": pt_idx,
            "arc_idx": arc_idx,
            "dist_to_arc": dists,
        })

        # Identify any points that had zero candidates within max_distance (fallback to nearest)
        queried_pts = set(pt_idx)
        all_pts = set(range(n_pts))
        missing_pts = sorted(list(all_pts - queried_pts))

        if missing_pts:
            fallback_records = []
            for m_idx in missing_pts:
                m_pt = gps_pts[m_idx]
                nearest_arc = int(self.strtree.nearest(m_pt))
                d = float(shapely.distance(m_pt, self.lines[nearest_arc]))
                fallback_records.append({
                    "step_idx": m_idx,
                    "arc_idx": nearest_arc,
                    "dist_to_arc": d,
                })
            fallback_df = pd.DataFrame(fallback_records)
            raw_candidates = pd.concat([raw_candidates, fallback_df], ignore_index=True)

        # Filter to top-k candidates per step_idx sorted by distance
        top_candidates = (
            raw_candidates.sort_values(["step_idx", "dist_to_arc"])
            .groupby("step_idx")
            .head(k)
            .reset_index(drop=True)
        )

        # Map candidate arc_idx to arc_id and metadata
        cand_arc_ids = self.arc_ids[top_candidates["arc_idx"].values]
        arc_meta = self.arcs_df.loc[cand_arc_ids].reset_index(drop=True)

        step_indices = top_candidates["step_idx"].values
        track_sub = track_df.iloc[step_indices].reset_index(drop=True)

        # Compute labels
        if aligned_gt_arcs is not None:
            active_gt = aligned_gt_arcs[step_indices]
            labels = (cand_arc_ids == active_gt).astype(int)
            in_route = np.array([1 if a in route_arcs_set else 0 for a in cand_arc_ids], dtype=int)
        else:
            labels = np.zeros(len(cand_arc_ids), dtype=int)
            in_route = np.zeros(len(cand_arc_ids), dtype=int)

        # Construct output DataFrame matching Contract 1
        candidates_df = pd.DataFrame({
            "trip_id": str(trip_id),
            "step_idx": step_indices,
            "timestamp": track_sub["timestamp"].values,
            "gps_lon": track_sub["lon"].values,
            "gps_lat": track_sub["lat"].values,
            "arc_id": cand_arc_ids,
            "from_node": arc_meta["from_node"].values,
            "to_node": arc_meta["to_node"].values,
            "arc_length": arc_meta["arc_length"].values,
            "label": labels,
            "dist_to_arc": top_candidates["dist_to_arc"].values,
            "in_route": in_route,
        })

        return candidates_df
