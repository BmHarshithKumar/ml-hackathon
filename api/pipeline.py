"""End-to-end map-matching pipeline orchestrator & Contract 1-3 adapter (Member 4).

Integrates directly with merged teammate modules while maintaining clean fallbacks:
  - Member 1 (`src.data_loader`, `src.candidate_gen`):
      Uses `load_trip`, `build_road_graph`, `SpatialCandidateGenerator` (STRtree/KDTree),
      and `align_trajectory_to_route` directly on `data/Student folder/student_test/`.
  - Member 2 (`src.features`, `artifacts/best_model.pkl`):
      Automatically delegates emission scoring when merged.
  - Member 3 (`src.hmm_viterbi.reconstruct_route`):
      Automatically delegates sequential decoding via Contract 3 when merged;
      otherwise runs a bounded-Dijkstra log-space HMM Viterbi decoder.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 coordinates."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward compass bearing in degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference in degrees [0, 180]."""
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _collapse_consecutive(seq: Sequence[int]) -> list[int]:
    """Remove consecutive duplicate arc IDs while preserving traversal order."""
    collapsed: list[int] = []
    for item in seq:
        val = int(item)
        if not collapsed or collapsed[-1] != val:
            collapsed.append(val)
    return collapsed


def _safe_import(module_name: str) -> Any | None:
    """Import an optional teammate module without raising if absent or missing deps."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


@dataclass
class ComponentStatus:
    """Tracks which teammate modules, dataset files, and model artifacts are actively wired."""

    data_loader: bool = False
    candidate_gen: bool = False
    features: bool = False
    model_artifact: bool = False
    hmm_viterbi: bool = False
    real_dataset_files: bool = False

    @property
    def mode(self) -> str:
        if all(
            [
                self.real_dataset_files,
                self.data_loader,
                self.candidate_gen,
                self.features,
                self.model_artifact,
                self.hmm_viterbi,
            ]
        ):
            return "full_real"
        if self.real_dataset_files:
            return "real_data_pipeline"
        if any(
            [
                self.data_loader,
                self.candidate_gen,
                self.features,
                self.model_artifact,
                self.hmm_viterbi,
            ]
        ):
            return "partial_real"
        return "mock_demo"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "p1_data_loader": self.data_loader,
            "p1_candidate_gen": self.candidate_gen,
            "p2_features": self.features,
            "p2_model_artifact": self.model_artifact,
            "p3_hmm_viterbi": self.hmm_viterbi,
            "real_dataset_files": self.real_dataset_files,
        }


class _SpatialArcIndex:
    """Fallback pure-NumPy spatial grid index used when Shapely/SciPy aren't installed."""

    def __init__(
        self,
        u_lats: np.ndarray,
        u_lons: np.ndarray,
        v_lats: np.ndarray,
        v_lons: np.ndarray,
        from_nodes: np.ndarray,
        to_nodes: np.ndarray,
        lengths_m: np.ndarray,
        bearings_deg: np.ndarray,
        cell_deg: float = 0.0020,
    ) -> None:
        self.u_lats = u_lats
        self.u_lons = u_lons
        self.v_lats = v_lats
        self.v_lons = v_lons
        self.from_nodes = from_nodes
        self.to_nodes = to_nodes
        self.lengths_m = lengths_m
        self.bearings_deg = bearings_deg
        self.cell_deg = cell_deg

        buckets: dict[tuple[int, int], list[int]] = {}
        min_r = np.floor(np.minimum(u_lats, v_lats) / cell_deg).astype(np.int64)
        max_r = np.floor(np.maximum(u_lats, v_lats) / cell_deg).astype(np.int64)
        min_c = np.floor(np.minimum(u_lons, v_lons) / cell_deg).astype(np.int64)
        max_c = np.floor(np.maximum(u_lons, v_lons) / cell_deg).astype(np.int64)

        for aid in range(len(u_lats)):
            r0, r1 = int(min_r[aid]), int(max_r[aid])
            c0, c1 = int(min_c[aid]), int(max_c[aid])
            if (r1 - r0) * (c1 - c0) > 400:
                buckets.setdefault(((r0 + r1) // 2, (c0 + c1) // 2), []).append(aid)
                continue
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    buckets.setdefault((r, c), []).append(aid)

        self.buckets: dict[tuple[int, int], np.ndarray] = {
            k: np.array(v, dtype=np.int32) for k, v in buckets.items()
        }

    def query_point(
        self,
        lat: float,
        lon: float,
        gps_heading: float,
        k: int = 8,
        max_radius_m: float = 120.0,
    ) -> list[dict[str, Any]]:
        cr = int(math.floor(lat / self.cell_deg))
        cc = int(math.floor(lon / self.cell_deg))

        cand_arrays: list[np.ndarray] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                arr = self.buckets.get((cr + dr, cc + dc))
                if arr is not None:
                    cand_arrays.append(arr)

        if not cand_arrays:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    arr = self.buckets.get((cr + dr, cc + dc))
                    if arr is not None:
                        cand_arrays.append(arr)

        cand_ids = (
            np.unique(np.concatenate(cand_arrays))
            if cand_arrays
            else np.arange(len(self.u_lats), dtype=np.int32)
        )
        dists, fracs = self._vectorized_point_to_segments(lat, lon, cand_ids)

        within = np.where(dists <= max_radius_m)[0]
        if len(within) >= min(k, 2):
            sub_ids, sub_dists, sub_fracs = cand_ids[within], dists[within], fracs[within]
        else:
            sub_ids, sub_dists, sub_fracs = cand_ids, dists, fracs

        if len(sub_ids) > k:
            top_idx = np.argpartition(sub_dists, k)[:k]
            top_idx = top_idx[np.argsort(sub_dists[top_idx])]
        else:
            top_idx = np.argsort(sub_dists)

        results: list[dict[str, Any]] = []
        for idx in top_idx:
            aid = int(sub_ids[idx])
            results.append(
                {
                    "arc_id": aid,
                    "from_node": int(self.from_nodes[aid]),
                    "to_node": int(self.to_nodes[aid]),
                    "length": float(self.lengths_m[aid]),
                    "distance_m": float(sub_dists[idx]),
                    "proj_frac": float(sub_fracs[idx]),
                    "heading_delta_deg": _angular_diff_deg(gps_heading, float(self.bearings_deg[aid])),
                }
            )
        return results

    def query_corridor_arc_ids(
        self,
        gps_points: Sequence[dict[str, Any]],
        max_arcs: int = 1200,
    ) -> list[int]:
        seen: set[int] = set()
        stride = max(1, len(gps_points) // 250)
        for pt in gps_points[::stride]:
            cr = int(math.floor(float(pt["lat"]) / self.cell_deg))
            cc = int(math.floor(float(pt["lon"]) / self.cell_deg))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    arr = self.buckets.get((cr + dr, cc + dc))
                    if arr is not None:
                        for aid in arr:
                            seen.add(int(aid))
                            if len(seen) >= max_arcs:
                                return sorted(seen)
        return sorted(seen)

    def _vectorized_point_to_segments(
        self, lat: float, lon: float, arc_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        cos_lat = math.cos(math.radians(lat))
        m_lat = 111_132.0
        m_lon = 111_320.0 * cos_lat

        u_lat = self.u_lats[arc_ids]
        u_lon = self.u_lons[arc_ids]
        v_lat = self.v_lats[arc_ids]
        v_lon = self.v_lons[arc_ids]

        px = (lon - u_lon) * m_lon
        py = (lat - u_lat) * m_lat
        dx = (v_lon - u_lon) * m_lon
        dy = (v_lat - u_lat) * m_lat

        seg_len_sq = dx * dx + dy * dy
        safe_denom = np.where(seg_len_sq > 1e-9, seg_len_sq, 1.0)
        t = np.where(seg_len_sq > 1e-9, np.clip((px * dx + py * dy) / safe_denom, 0.0, 1.0), 0.0)
        return np.hypot(px - t * dx, py - t * dy), t


@dataclass
class TripBundle:
    """Holds one evaluation trip's road graph, spatial index, GPS track, and server-side GT."""

    trip_id: str
    description: str
    graph: nx.DiGraph
    arc_lookup: dict[int, dict[str, Any]]
    spatial_index: _SpatialArcIndex
    gps_points: list[dict[str, Any]]
    ground_truth_route: list[int]
    # Optional Member 1 native objects when src.data_loader & src.candidate_gen are active
    track_df: pd.DataFrame | None = None
    nodes_df: pd.DataFrame | None = None
    arcs_df: pd.DataFrame | None = None
    aug_arcs_df: pd.DataFrame | None = None
    route_df: pd.DataFrame | None = None
    p1_candidate_gen: Any | None = None


class MapMatchingPipeline:
    """Server-side orchestrator for GPS trajectory map-matching and evaluation."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.status = ComponentStatus()

        self.model: Any | None = None
        self._p1_loader: Any | None = None
        self._p1_cand: Any | None = None
        self._p2_features: Any | None = None
        self._p3_hmm: Any | None = None

        self.eval_trip_dirs: dict[str, Path] = {}
        self.eval_trip_meta: dict[str, dict[str, Any]] = {}
        self._bundle_cache: dict[str, TripBundle] = {}

        self._initialize()

    # ------------------------------------------------------------------
    # Startup discovery
    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        self._p1_loader = _safe_import("src.data_loader")
        self._p1_cand = _safe_import("src.candidate_gen")
        self._p2_features = _safe_import("src.features")
        self._p3_hmm = _safe_import("src.hmm_viterbi")

        self.status.data_loader = bool(
            self._p1_loader is not None
            and hasattr(self._p1_loader, "load_trip")
            and hasattr(self._p1_loader, "build_road_graph")
        )
        self.status.candidate_gen = bool(
            self._p1_cand is not None
            and hasattr(self._p1_cand, "SpatialCandidateGenerator")
        )
        self.status.features = self._p2_features is not None
        self.status.hmm_viterbi = bool(
            self._p3_hmm is not None and hasattr(self._p3_hmm, "reconstruct_route")
        )

        if self.model_path.exists():
            self.model = self._load_model_artifact(self.model_path)
            self.status.model_artifact = self.model is not None

        if not self._discover_real_eval_trips():
            self._build_synthetic_fallback()

    @staticmethod
    def _load_model_artifact(path: Path) -> Any | None:
        try:
            import joblib

            return joblib.load(path)
        except Exception:
            try:
                with path.open("rb") as fh:
                    return pickle.load(fh)
            except Exception:
                return None

    def _find_student_test_dir(self) -> Path | None:
        candidates = [
            self.data_dir / "Student folder" / "student_test",
            self.data_dir / "student_test",
            self.data_dir / "test",
        ]
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                return cand
        if self.data_dir.exists():
            for match in self.data_dir.rglob("student_test"):
                if match.is_dir():
                    return match
        return None

    def _discover_real_eval_trips(self) -> bool:
        """Discover the 10 real evaluation trips in `data/Student folder/student_test/`."""
        test_dir = self._find_student_test_dir()
        if test_dir is None:
            return False

        trip_dirs: dict[str, Path] = {}
        trip_meta: dict[str, dict[str, Any]] = {}

        for sub in sorted(test_dir.iterdir()):
            if not sub.is_dir():
                continue
            tid = sub.name
            nodes_f = sub / f"{tid}.nodes"
            arcs_f = sub / f"{tid}.arcs"
            track_f = sub / f"{tid}.track"
            route_f = sub / f"{tid}.route"
            if not (nodes_f.exists() and arcs_f.exists() and track_f.exists()):
                continue

            track_lines = [
                l for l in track_f.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()
            ]
            num_nodes = sum(1 for l in nodes_f.open("r", encoding="utf-8", errors="ignore") if l.strip())
            num_arcs = sum(1 for l in arcs_f.open("r", encoding="utf-8", errors="ignore") if l.strip())

            duration_s = 0.0
            start_coord = ""
            if track_lines:
                first_parts = track_lines[0].split()
                last_parts = track_lines[-1].split()
                if len(first_parts) >= 3 and len(last_parts) >= 3:
                    try:
                        duration_s = round(float(last_parts[2]) - float(first_parts[2]), 1)
                    except ValueError:
                        duration_s = float(len(track_lines))
                if len(first_parts) >= 2:
                    try:
                        lon0, lat0 = float(first_parts[0]), float(first_parts[1])
                        start_coord = f"({lat0:.3f}, {lon0:.3f})"
                    except ValueError:
                        pass

            trip_dirs[tid] = sub
            trip_meta[tid] = {
                "trip_id": tid,
                "num_points": len(track_lines),
                "duration_s": duration_s,
                "num_nodes": num_nodes,
                "num_arcs": num_arcs,
                "has_ground_truth": route_f.exists(),
                "description": f"Real trip {tid} @ {start_coord} · {num_arcs:,} road arcs",
            }

        if not trip_dirs:
            return False

        self.eval_trip_dirs = trip_dirs
        self.eval_trip_meta = trip_meta
        self.status.real_dataset_files = True
        return True

    def _load_real_trip_bundle(self, trip_id: str) -> TripBundle:
        """Load and cache a real per-trip road graph using Member 1's `src.data_loader`."""
        if trip_id in self._bundle_cache:
            return self._bundle_cache[trip_id]

        trip_dir = self.eval_trip_dirs[trip_id]

        # Prefer Member 1's official `src.data_loader.load_trip` & `build_road_graph`
        if self.status.data_loader and self._p1_loader is not None:
            trip_data = self._p1_loader.load_trip(trip_id, base_dirs=[str(trip_dir.parent)])
            track_df: pd.DataFrame = trip_data["track"]
            nodes_df: pd.DataFrame = trip_data["nodes"]
            arcs_df: pd.DataFrame = trip_data["arcs"]
            route_df: pd.DataFrame | None = trip_data["route"]

            origin_lon = float(nodes_df["lon"].mean())
            origin_lat = float(nodes_df["lat"].mean())
            graph, aug_arcs = self._p1_loader.build_road_graph(
                nodes_df, arcs_df, origin_lon=origin_lon, origin_lat=origin_lat
            )

            u_lats = aug_arcs["u_lat"].values
            u_lons = aug_arcs["u_lon"].values
            v_lats = aug_arcs["v_lat"].values
            v_lons = aug_arcs["v_lon"].values
            from_nodes = aug_arcs["from_node"].values
            to_nodes = aug_arcs["to_node"].values
            lengths_m = aug_arcs["arc_length"].values
            bearings_deg = aug_arcs["bearing"].values

            arc_lookup: dict[int, dict[str, Any]] = {}
            for aid in range(len(aug_arcs)):
                arc_lookup[aid] = {
                    "arc_id": aid,
                    "from_node": int(from_nodes[aid]),
                    "to_node": int(to_nodes[aid]),
                    "length": round(float(lengths_m[aid]), 2),
                    "coords": [
                        [float(u_lats[aid]), float(u_lons[aid])],
                        [float(v_lats[aid]), float(v_lons[aid])],
                    ],
                }

            spatial_index = _SpatialArcIndex(
                u_lats=u_lats,
                u_lons=u_lons,
                v_lats=v_lats,
                v_lons=v_lons,
                from_nodes=from_nodes,
                to_nodes=to_nodes,
                lengths_m=lengths_m,
                bearings_deg=bearings_deg,
            )

            p1_gen = None
            if self.status.candidate_gen and self._p1_cand is not None:
                try:
                    p1_gen = self._p1_cand.SpatialCandidateGenerator(
                        aug_arcs,
                        origin_lon=origin_lon,
                        origin_lat=origin_lat,
                        index_type="strtree",
                    )
                except Exception as exc:
                    logging.getLogger("map_matching_api").warning(
                        "SpatialCandidateGenerator init failed (%s); using NumPy spatial index.", exc
                    )

            gps_points = [
                {
                    "step_idx": int(row.step_idx),
                    "timestamp": round(float(row.timestamp), 2),
                    "lat": float(row.lat),
                    "lon": float(row.lon),
                }
                for row in track_df.itertuples()
            ]
            gt_route = (
                [int(a) for a in route_df["arc_id"].values]
                if route_df is not None
                else []
            )

            bundle = TripBundle(
                trip_id=trip_id,
                description=self.eval_trip_meta[trip_id]["description"],
                graph=graph,
                arc_lookup=arc_lookup,
                spatial_index=spatial_index,
                gps_points=gps_points,
                ground_truth_route=gt_route,
                track_df=track_df,
                nodes_df=nodes_df,
                arcs_df=arcs_df,
                aug_arcs_df=aug_arcs,
                route_df=route_df,
                p1_candidate_gen=p1_gen,
            )
            self._bundle_cache[trip_id] = bundle
            return bundle

        # Fallback loader if src.data_loader is not available
        nodes_arr = np.loadtxt(str(trip_dir / f"{trip_id}.nodes"), dtype=np.float64)
        if nodes_arr.ndim == 1:
            nodes_arr = nodes_arr.reshape(1, -1)
        node_lons, node_lats = nodes_arr[:, 0], nodes_arr[:, 1]

        arcs_arr = np.loadtxt(str(trip_dir / f"{trip_id}.arcs"), dtype=np.int64)
        if arcs_arr.ndim == 1:
            arcs_arr = arcs_arr.reshape(1, -1)
        from_nodes, to_nodes = arcs_arr[:, 0], arcs_arr[:, 1]

        u_lats, u_lons = node_lats[from_nodes], node_lons[from_nodes]
        v_lats, v_lons = node_lats[to_nodes], node_lons[to_nodes]

        phi1, phi2 = np.radians(u_lats), np.radians(v_lats)
        dphi, dlam = np.radians(v_lats - u_lats), np.radians(v_lons - u_lons)
        a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
        lengths_m = 2.0 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
        by = np.sin(dlam) * np.cos(phi2)
        bx = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
        bearings_deg = (np.degrees(np.arctan2(by, bx)) + 360.0) % 360.0

        graph = nx.DiGraph()
        graph.add_nodes_from(
            (i, {"lat": float(node_lats[i]), "lon": float(node_lons[i])})
            for i in range(len(node_lats))
        )
        arc_lookup = {}
        edges_to_add = []
        for aid in range(len(from_nodes)):
            u, v = int(from_nodes[aid]), int(to_nodes[aid])
            length = float(lengths_m[aid])
            coords = [[float(u_lats[aid]), float(u_lons[aid])], [float(v_lats[aid]), float(v_lons[aid])]]
            edges_to_add.append((u, v, {"arc_id": aid, "length": length, "weight": length, "coords": coords}))
            arc_lookup[aid] = {"arc_id": aid, "from_node": u, "to_node": v, "length": round(length, 2), "coords": coords}
        graph.add_edges_from(edges_to_add)

        spatial_index = _SpatialArcIndex(
            u_lats=u_lats, u_lons=u_lons, v_lats=v_lats, v_lons=v_lons,
            from_nodes=from_nodes, to_nodes=to_nodes, lengths_m=lengths_m, bearings_deg=bearings_deg,
        )

        track_arr = np.loadtxt(str(trip_dir / f"{trip_id}.track"), dtype=np.float64)
        if track_arr.ndim == 1:
            track_arr = track_arr.reshape(1, -1)
        gps_points = [
            {
                "step_idx": idx,
                "timestamp": round(float(track_arr[idx, 2]) if track_arr.shape[1] >= 3 else float(idx), 2),
                "lat": float(track_arr[idx, 1]),
                "lon": float(track_arr[idx, 0]),
            }
            for idx in range(len(track_arr))
        ]
        route_f = trip_dir / f"{trip_id}.route"
        gt_route = (
            [int(line.strip().split()[0]) for line in route_f.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
            if route_f.exists()
            else []
        )

        bundle = TripBundle(
            trip_id=trip_id,
            description=self.eval_trip_meta[trip_id]["description"],
            graph=graph,
            arc_lookup=arc_lookup,
            spatial_index=spatial_index,
            gps_points=gps_points,
            ground_truth_route=gt_route,
        )
        self._bundle_cache[trip_id] = bundle
        return bundle

    # ------------------------------------------------------------------
    # Minimal synthetic fallback (only if `data/` is absent)
    # ------------------------------------------------------------------
    def _build_synthetic_fallback(self) -> None:
        rng = np.random.default_rng(seed=42)
        rows, cols = 10, 12
        base_lat, base_lon = 12.9640, 77.5840
        step_lat, step_lon = 0.00115, 0.00125

        node_lats = np.zeros(rows * cols, dtype=np.float64)
        node_lons = np.zeros(rows * cols, dtype=np.float64)
        graph = nx.DiGraph()
        for r in range(rows):
            for c in range(cols):
                nid = r * cols + c
                lat = base_lat + r * step_lat + float(rng.normal(0.0, 0.00008))
                lon = base_lon + c * step_lon + float(rng.normal(0.0, 0.00008))
                node_lats[nid], node_lons[nid] = lat, lon
                graph.add_node(nid, lat=lat, lon=lon)

        from_list, to_list = [], []
        for r in range(rows):
            for c in range(cols):
                u = r * cols + c
                if c + 1 < cols:
                    v = r * cols + c + 1
                    from_list.extend([u, v])
                    to_list.extend([v, u])
                if r + 1 < rows:
                    v = (r + 1) * cols + c
                    from_list.extend([u, v])
                    to_list.extend([v, u])

        from_nodes = np.array(from_list, dtype=np.int64)
        to_nodes = np.array(to_list, dtype=np.int64)
        u_lats, u_lons = node_lats[from_nodes], node_lons[from_nodes]
        v_lats, v_lons = node_lats[to_nodes], node_lons[to_nodes]
        lengths_m = np.array(
            [haversine_m(u_lats[i], u_lons[i], v_lats[i], v_lons[i]) for i in range(len(from_nodes))],
            dtype=np.float64,
        )
        bearings_deg = np.array(
            [_bearing_deg(u_lats[i], u_lons[i], v_lats[i], v_lons[i]) for i in range(len(from_nodes))],
            dtype=np.float64,
        )

        arc_lookup: dict[int, dict[str, Any]] = {}
        for aid in range(len(from_nodes)):
            u, v = int(from_nodes[aid]), int(to_nodes[aid])
            coords = [[float(u_lats[aid]), float(u_lons[aid])], [float(v_lats[aid]), float(v_lons[aid])]]
            graph.add_edge(u, v, arc_id=aid, length=float(lengths_m[aid]), weight=float(lengths_m[aid]), coords=coords)
            arc_lookup[aid] = {"arc_id": aid, "from_node": u, "to_node": v, "length": round(float(lengths_m[aid]), 2), "coords": coords}

        spatial_index = _SpatialArcIndex(
            u_lats=u_lats, u_lons=u_lons, v_lats=v_lats, v_lons=v_lons,
            from_nodes=from_nodes, to_nodes=to_nodes, lengths_m=lengths_m, bearings_deg=bearings_deg,
        )

        for idx in range(1, 11):
            tid = f"eval_trip_{idx:02d}"
            trng = np.random.default_rng(100 + idx)
            curr, prev = idx % len(node_lats), -1
            route: list[int] = []
            gps_points: list[dict[str, Any]] = []
            ts = 0.0
            for _ in range(14):
                nbrs = [n for n in graph.successors(curr) if n != prev] or list(graph.successors(curr))
                nxt = int(trng.choice(nbrs))
                route.append(int(graph[curr][nxt]["arc_id"]))
                for frac in (0.2, 0.55, 0.85):
                    lat = node_lats[curr] + frac * (node_lats[nxt] - node_lats[curr]) + float(trng.normal(0, 7.0)) / 111_132.0
                    lon = node_lons[curr] + frac * (node_lons[nxt] - node_lons[curr]) + float(trng.normal(0, 7.0)) / (111_320.0 * math.cos(math.radians(node_lats[curr])))
                    gps_points.append({"step_idx": len(gps_points), "timestamp": round(ts, 1), "lat": lat, "lon": lon})
                    ts += 3.0
                prev, curr = curr, nxt

            self._bundle_cache[tid] = TripBundle(
                trip_id=tid,
                description=f"Synthetic fallback trip {tid}",
                graph=graph,
                arc_lookup=arc_lookup,
                spatial_index=spatial_index,
                gps_points=gps_points,
                ground_truth_route=route,
            )
            self.eval_trip_meta[tid] = {
                "trip_id": tid,
                "num_points": len(gps_points),
                "duration_s": round(ts, 1),
                "num_nodes": graph.number_of_nodes(),
                "num_arcs": graph.number_of_edges(),
                "has_ground_truth": True,
                "description": f"Synthetic fallback trip {tid}",
            }

    def get_trip_bundle(self, trip_id: str) -> TripBundle:
        if trip_id in self._bundle_cache:
            return self._bundle_cache[trip_id]
        if trip_id in self.eval_trip_dirs:
            return self._load_real_trip_bundle(trip_id)
        raise KeyError(f"Unknown trip_id '{trip_id}'. Available: {sorted(self.eval_trip_meta)}")

    # ------------------------------------------------------------------
    # Public inspection helpers
    # ------------------------------------------------------------------
    def list_eval_trips(self) -> list[dict[str, Any]]:
        return [
            {
                "trip_id": tid,
                "num_points": meta["num_points"],
                "duration_s": meta["duration_s"],
                "num_nodes": meta["num_nodes"],
                "num_arcs": meta["num_arcs"],
                "description": meta["description"],
            }
            for tid, meta in sorted(self.eval_trip_meta.items())
        ]

    def health_summary(self) -> dict[str, Any]:
        total_nodes = sum(m["num_nodes"] for m in self.eval_trip_meta.values())
        total_arcs = sum(m["num_arcs"] for m in self.eval_trip_meta.values())
        return {
            "status": "ok",
            "pipeline_mode": self.status.mode,
            "components": self.status.to_dict(),
            "graph_stats": {
                "num_nodes": total_nodes,
                "num_arcs": total_arcs,
                "per_trip_graphs": self.status.real_dataset_files,
            },
            "num_eval_trips": len(self.eval_trip_meta),
            "eval_trip_ids": sorted(self.eval_trip_meta.keys()),
            "model_path": str(self.model_path),
            "model_loaded": self.model is not None,
        }

    # ------------------------------------------------------------------
    # Stage 2: Candidate Generation (uses Member 1's SpatialCandidateGenerator!)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_smoothed_headings(gps_points: Sequence[dict[str, Any]]) -> list[float]:
        n = len(gps_points)
        headings: list[float] = []
        for idx in range(n):
            i0 = max(0, idx - 1)
            i1 = min(n - 1, idx + 1)
            if i0 == i1:
                headings.append(0.0)
                continue
            lat0, lon0 = float(gps_points[i0]["lat"]), float(gps_points[i0]["lon"])
            lat1, lon1 = float(gps_points[i1]["lat"]), float(gps_points[i1]["lon"])
            if haversine_m(lat0, lon0, lat1, lon1) < 1.5 and headings:
                headings.append(headings[-1])
            else:
                headings.append(_bearing_deg(lat0, lon0, lat1, lon1))
        return headings

    def _generate_candidates(
        self,
        bundle: TripBundle,
        k: int = 8,
        max_radius_m: float = 120.0,
    ) -> tuple[list[list[dict[str, Any]]], pd.DataFrame | None]:
        """Generate candidate arcs using Member 1's STRtree `SpatialCandidateGenerator` if active."""
        gps_points = bundle.gps_points
        headings = self._compute_smoothed_headings(gps_points)

        # Native Member 1 Contract 1 pipeline
        if (
            bundle.p1_candidate_gen is not None
            and bundle.track_df is not None
            and bundle.aug_arcs_df is not None
        ):
            cands_df: pd.DataFrame = bundle.p1_candidate_gen.generate_candidates(
                trip_id=bundle.trip_id,
                track_df=bundle.track_df,
                route_df=None,  # Keep GT strictly isolated during inference!
                k=k,
                max_distance=max_radius_m,
            )

            # Vectorized projection fraction & heading delta enrichment for HMM
            cand_aids = cands_df["arc_id"].values
            step_ids = cands_df["step_idx"].values
            aug = bundle.aug_arcs_df
            u_x = aug["u_x"].values[cand_aids]
            u_y = aug["u_y"].values[cand_aids]
            v_x = aug["v_x"].values[cand_aids]
            v_y = aug["v_y"].values[cand_aids]
            arc_bearings = aug["bearing"].values[cand_aids]

            # Project GPS points into the same local metric coordinates
            gps_x, gps_y = self._p1_loader.wgs84_to_local_metric(
                cands_df["gps_lon"].values,
                cands_df["gps_lat"].values,
                origin_lon=bundle.p1_candidate_gen.origin_lon,
                origin_lat=bundle.p1_candidate_gen.origin_lat,
            )
            dx = v_x - u_x
            dy = v_y - u_y
            seg_len_sq = dx * dx + dy * dy
            safe_denom = np.where(seg_len_sq > 1e-9, seg_len_sq, 1.0)
            proj_fracs = np.where(
                seg_len_sq > 1e-9,
                np.clip(((gps_x - u_x) * dx + (gps_y - u_y) * dy) / safe_denom, 0.0, 1.0),
                0.0,
            )

            headings_arr = np.array(headings, dtype=np.float64)[step_ids]
            raw_hd = np.abs(headings_arr - arc_bearings) % 360.0
            hd_deg = np.minimum(raw_hd, 360.0 - raw_hd)

            from_nodes = cands_df["from_node"].values
            to_nodes = cands_df["to_node"].values
            lengths = cands_df["arc_length"].values
            dists = cands_df["dist_to_arc"].values

            candidates_per_step: list[list[dict[str, Any]]] = [[] for _ in range(len(gps_points))]
            for idx in range(len(cands_df)):
                s_idx = int(step_ids[idx])
                candidates_per_step[s_idx].append(
                    {
                        "arc_id": int(cand_aids[idx]),
                        "from_node": int(from_nodes[idx]),
                        "to_node": int(to_nodes[idx]),
                        "length": float(lengths[idx]),
                        "distance_m": float(dists[idx]),
                        "proj_frac": float(proj_fracs[idx]),
                        "heading_delta_deg": float(hd_deg[idx]),
                    }
                )
            return candidates_per_step, cands_df

        # Fallback NumPy spatial index
        candidates_per_step = []
        for idx, pt in enumerate(gps_points):
            cands = bundle.spatial_index.query_point(
                lat=float(pt["lat"]),
                lon=float(pt["lon"]),
                gps_heading=headings[idx],
                k=k,
                max_radius_m=max_radius_m,
            )
            candidates_per_step.append(cands)
        return candidates_per_step, None

    # ------------------------------------------------------------------
    # Stage 3: Classifier Emission Scoring (Contract 2)
    # ------------------------------------------------------------------
    def _score_candidates(
        self,
        bundle: TripBundle,
        candidates_per_step: list[list[dict[str, Any]]],
        candidates_df: pd.DataFrame | None = None,
    ) -> tuple[list[list[float]], list[int]]:
        """Score candidate arcs using Member 2's model if present, else Gaussian/Heading emissions."""
        if self.model is not None and self._p2_features is not None:
            for fn_name in ("extract_features", "compute_features", "build_features"):
                feat_fn = getattr(self._p2_features, fn_name, None)
                if callable(feat_fn):
                    try:
                        if candidates_df is not None:
                            X = feat_fn(candidates_df, bundle.graph)
                            probs_flat = self.model.predict_proba(X)[:, 1]
                            all_probs: list[list[float]] = []
                            greedy_ids: list[int] = []
                            cursor = 0
                            for step_cands in candidates_per_step:
                                m = len(step_cands)
                                step_p = [float(p) for p in probs_flat[cursor : cursor + m]]
                                cursor += m
                                all_probs.append(step_p)
                                greedy_ids.append(int(step_cands[int(np.argmax(step_p))]["arc_id"]))
                            return all_probs, greedy_ids
                    except Exception as exc:
                        logging.getLogger("map_matching_api").warning(
                            "Member 2 feature/model scoring raised %s; using Gaussian emissions.", exc
                        )
                        break

        sigma_dist = 12.0
        sigma_head = 50.0
        all_probs = []
        greedy_ids = []

        for step_cands in candidates_per_step:
            raw: list[float] = []
            for cand in step_cands:
                d = float(cand["distance_m"])
                hd = float(cand["heading_delta_deg"])
                score = math.exp(-0.5 * (d / sigma_dist) ** 2) * math.exp(
                    -0.45 * (hd / sigma_head) ** 2
                )
                raw.append(max(score, 1e-12))
            total = sum(raw) or 1.0
            probs = [s / total for s in raw]
            all_probs.append(probs)
            greedy_ids.append(int(step_cands[int(np.argmax(probs))]["arc_id"]))

        return all_probs, greedy_ids

    # ------------------------------------------------------------------
    # Stage 4: HMM Viterbi Decoding (Contract 3)
    # ------------------------------------------------------------------
    def _run_hmm_viterbi(
        self,
        bundle: TripBundle,
        candidates_per_step: list[list[dict[str, Any]]],
        emission_probs: list[list[float]],
    ) -> list[int]:
        """Delegate to Member 3's Contract 3 `reconstruct_route`, else run bounded-Dijkstra Viterbi."""
        if self._p3_hmm is not None and hasattr(self._p3_hmm, "reconstruct_route"):
            fn = self._p3_hmm.reconstruct_route
            sig = inspect.signature(fn)
            kwargs: dict[str, Any] = {}
            if "model" in sig.parameters and self.model is not None:
                kwargs["model"] = self.model
            if "candidates" in sig.parameters:
                kwargs["candidates"] = candidates_per_step
            if "emission_probs" in sig.parameters:
                kwargs["emission_probs"] = emission_probs
            try:
                result = fn(bundle.trip_id, bundle.gps_points, bundle.graph, **kwargs)
                if isinstance(result, (list, tuple)) and len(result) > 0:
                    return [int(a) for a in result]
            except Exception as exc:
                logging.getLogger("map_matching_api").warning(
                    "Member 3 reconstruct_route raised %s; falling back to built-in Viterbi.", exc
                )

        gps_points = bundle.gps_points
        n_steps = len(gps_points)
        if n_steps == 0:
            return []

        beta = 25.0
        graph = bundle.graph
        sssp_cache: dict[int, dict[int, float]] = {}

        def bounded_net_dist(src: int, dst: int, cutoff: float) -> float:
            if src == dst:
                return 0.0
            cached = sssp_cache.get(src)
            if cached is None:
                cached = nx.single_source_dijkstra_path_length(
                    graph, source=src, cutoff=max(cutoff, 450.0), weight="length"
                )
                sssp_cache[src] = cached
            return float(cached.get(dst, math.inf))

        dp: list[list[float]] = []
        backptr: list[list[int]] = []

        dp.append([math.log(max(p, 1e-12)) for p in emission_probs[0]])
        backptr.append([-1] * len(candidates_per_step[0]))

        for t in range(1, n_steps):
            gps_disp = haversine_m(
                float(gps_points[t - 1]["lat"]),
                float(gps_points[t - 1]["lon"]),
                float(gps_points[t]["lat"]),
                float(gps_points[t]["lon"]),
            )
            cutoff = max(450.0, gps_disp * 2.5 + 150.0)
            prev_cands = candidates_per_step[t - 1]
            curr_cands = candidates_per_step[t]

            curr_dp: list[float] = []
            curr_bp: list[int] = []

            for j, c_curr in enumerate(curr_cands):
                log_emit = math.log(max(emission_probs[t][j], 1e-12))
                f_curr = float(c_curr["proj_frac"])
                len_curr = float(c_curr["length"])
                best_score = -math.inf
                best_prev = 0

                for i, c_prev in enumerate(prev_cands):
                    f_prev = float(c_prev["proj_frac"])
                    len_prev = float(c_prev["length"])

                    if c_prev["arc_id"] == c_curr["arc_id"]:
                        along = (f_curr - f_prev) * len_prev
                        net_dist = abs(along)
                        back_pen = -3.0 if along < -20.0 else 0.0
                        trans_pen = -abs(net_dist - gps_disp) / beta + back_pen
                    elif c_prev["to_node"] == c_curr["from_node"]:
                        net_dist = (1.0 - f_prev) * len_prev + f_curr * len_curr
                        trans_pen = -abs(net_dist - gps_disp) / beta
                    elif (
                        c_prev["from_node"] == c_curr["to_node"]
                        and c_prev["to_node"] == c_curr["from_node"]
                    ):
                        trans_pen = -14.0
                    else:
                        sp = bounded_net_dist(c_prev["to_node"], c_curr["from_node"], cutoff)
                        if math.isfinite(sp):
                            net_dist = (1.0 - f_prev) * len_prev + sp + f_curr * len_curr
                            trans_pen = -abs(net_dist - gps_disp) / beta - 0.35
                        else:
                            trans_pen = -24.0

                    score = dp[t - 1][i] + trans_pen + log_emit
                    if score > best_score:
                        best_score = score
                        best_prev = i

                curr_dp.append(best_score)
                curr_bp.append(best_prev)

            if max(curr_dp) < -1e6:
                curr_dp = [math.log(max(p, 1e-12)) for p in emission_probs[t]]

            dp.append(curr_dp)
            backptr.append(curr_bp)

        best_last = int(np.argmax(dp[-1]))
        seq_indices = [best_last]
        for t in range(n_steps - 1, 0, -1):
            seq_indices.append(backptr[t][seq_indices[-1]])
        seq_indices.reverse()

        return [int(candidates_per_step[t][idx]["arc_id"]) for t, idx in enumerate(seq_indices)]

    # ------------------------------------------------------------------
    # Stage 5: Server-side GT Alignment & Metrics
    # ------------------------------------------------------------------
    def _align_gt_per_step(self, bundle: TripBundle) -> list[int | None]:
        """Chronologically align each GPS observation to the ground-truth `.route`."""
        # Prefer Member 1's chronological forward-tracking alignment when available
        if (
            self._p1_cand is not None
            and hasattr(self._p1_cand, "align_trajectory_to_route")
            and bundle.track_df is not None
            and bundle.route_df is not None
            and bundle.aug_arcs_df is not None
            and bundle.p1_candidate_gen is not None
        ):
            try:
                tdf = bundle.track_df
                if "x" not in tdf.columns or "y" not in tdf.columns:
                    tx, ty = self._p1_loader.wgs84_to_local_metric(
                        tdf["lon"].values,
                        tdf["lat"].values,
                        origin_lon=bundle.p1_candidate_gen.origin_lon,
                        origin_lat=bundle.p1_candidate_gen.origin_lat,
                    )
                    tdf = tdf.copy()
                    tdf["x"] = tx
                    tdf["y"] = ty
                aligned = self._p1_cand.align_trajectory_to_route(
                    tdf, bundle.route_df, bundle.aug_arcs_df
                )
                return [int(a) for a in aligned]
            except Exception:
                pass

        gt_route = bundle.ground_truth_route
        if not gt_route:
            return [None] * len(bundle.gps_points)

        valid_gt = np.array([a for a in gt_route if a in bundle.arc_lookup], dtype=np.int32)
        if len(valid_gt) == 0:
            return [None] * len(bundle.gps_points)

        aligned_list: list[int | None] = []
        for pt in bundle.gps_points:
            dists, _ = bundle.spatial_index._vectorized_point_to_segments(
                float(pt["lat"]), float(pt["lon"]), valid_gt
            )
            aligned_list.append(int(valid_gt[int(np.argmin(dists))]))
        return aligned_list

    def _compute_continuity(self, bundle: TripBundle, arc_sequence: Sequence[int]) -> dict[str, Any]:
        collapsed = _collapse_consecutive(arc_sequence)
        if len(collapsed) <= 1:
            return {
                "route_continuity": 1.0,
                "direct_continuity": 1.0,
                "transitions": 0,
                "valid_transitions": 0,
                "discontinuities": 0,
            }

        valid_transitions = 0
        direct_transitions = 0
        graph = bundle.graph

        for a_prev, a_curr in zip(collapsed[:-1], collapsed[1:]):
            prev_rec = bundle.arc_lookup.get(a_prev)
            curr_rec = bundle.arc_lookup.get(a_curr)
            if not prev_rec or not curr_rec:
                continue

            if prev_rec["to_node"] == curr_rec["from_node"]:
                direct_transitions += 1
                valid_transitions += 1
            else:
                try:
                    sp = nx.single_source_dijkstra_path_length(
                        graph, source=prev_rec["to_node"], cutoff=600.0, weight="length"
                    )
                    if curr_rec["from_node"] in sp:
                        valid_transitions += 1
                except Exception:
                    pass

        total = len(collapsed) - 1
        return {
            "route_continuity": round(valid_transitions / total, 4),
            "direct_continuity": round(direct_transitions / total, 4),
            "transitions": total,
            "valid_transitions": valid_transitions,
            "discontinuities": total - valid_transitions,
        }

    def _expand_route_with_bridges(
        self, bundle: TripBundle, raw_arc_seq: Sequence[int]
    ) -> list[int]:
        collapsed = _collapse_consecutive(raw_arc_seq)
        if len(collapsed) <= 1:
            return collapsed

        graph = bundle.graph
        reconstructed: list[int] = [collapsed[0]]
        for a_curr in collapsed[1:]:
            a_prev = reconstructed[-1]
            if a_prev == a_curr:
                continue
            prev_rec = bundle.arc_lookup.get(a_prev)
            curr_rec = bundle.arc_lookup.get(a_curr)
            if prev_rec and curr_rec and prev_rec["to_node"] != curr_rec["from_node"]:
                try:
                    path_nodes = nx.shortest_path(
                        graph,
                        source=prev_rec["to_node"],
                        target=curr_rec["from_node"],
                        weight="length",
                    )
                    bridge_len = sum(
                        float(graph[u][v].get("length", 0.0))
                        for u, v in zip(path_nodes[:-1], path_nodes[1:])
                    )
                    if bridge_len <= 600.0:
                        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                            reconstructed.append(int(graph[u][v]["arc_id"]))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
            reconstructed.append(a_curr)

        return _collapse_consecutive(reconstructed)

    def _compute_metrics(
        self,
        bundle: TripBundle,
        gt_step_arcs: Sequence[int | None],
        pred_step_arcs: Sequence[int],
        greedy_step_arcs: Sequence[int],
        reconstructed_route: Sequence[int],
    ) -> dict[str, Any]:
        pred_cont = self._compute_continuity(bundle, reconstructed_route)
        greedy_cont = self._compute_continuity(bundle, greedy_step_arcs)

        gt_route = _collapse_consecutive(bundle.ground_truth_route)
        gt_set = set(gt_route)
        pred_set = set(reconstructed_route)

        if len(pred_step_arcs) == len(gt_step_arcs) and len(pred_step_arcs) > 0:
            top1_acc = sum(
                1 for g, p in zip(gt_step_arcs, pred_step_arcs) if (p == g or p in gt_set)
            ) / len(pred_step_arcs)
            greedy_top1 = sum(
                1 for g, p in zip(gt_step_arcs, greedy_step_arcs) if (p == g or p in gt_set)
            ) / len(greedy_step_arcs)
        else:
            top1_acc = sum(1 for p in reconstructed_route if p in gt_set) / max(
                1, len(reconstructed_route)
            )
            greedy_top1 = sum(1 for p in greedy_step_arcs if p in gt_set) / max(
                1, len(greedy_step_arcs)
            )

        tp = len(gt_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gt_set) if gt_set else 0.0
        route_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        jaccard = tp / len(gt_set | pred_set) if (gt_set | pred_set) else 0.0

        len_tp = sum(bundle.arc_lookup[a]["length"] for a in (gt_set & pred_set) if a in bundle.arc_lookup)
        len_pred = sum(bundle.arc_lookup[a]["length"] for a in pred_set if a in bundle.arc_lookup)
        len_gt = sum(bundle.arc_lookup[a]["length"] for a in gt_set if a in bundle.arc_lookup)
        lw_prec = len_tp / len_pred if len_pred > 0 else 0.0
        lw_rec = len_tp / len_gt if len_gt > 0 else 0.0
        lw_f1 = (2 * lw_prec * lw_rec / (lw_prec + lw_rec)) if (lw_prec + lw_rec) > 0 else 0.0

        return {
            "top1_accuracy": round(top1_acc, 4),
            "route_continuity": pred_cont["route_continuity"],
            "direct_continuity": pred_cont["direct_continuity"],
            "discontinuities": pred_cont["discontinuities"],
            "route_precision": round(precision, 4),
            "route_recall": round(recall, 4),
            "route_f1": round(route_f1, 4),
            "length_weighted_f1": round(lw_f1, 4),
            "sequence_jaccard": round(jaccard, 4),
            "classifier_only_top1_accuracy": round(greedy_top1, 4),
            "classifier_only_continuity": greedy_cont["route_continuity"],
            "classifier_only_discontinuities": greedy_cont["discontinuities"],
            "hmm_accuracy_gain": round(top1_acc - greedy_top1, 4),
            "hmm_continuity_gain": round(
                pred_cont["route_continuity"] - greedy_cont["route_continuity"], 4
            ),
            "num_gt_unique_arcs": len(gt_route),
            "num_pred_unique_arcs": len(reconstructed_route),
        }

    # ------------------------------------------------------------------
    # Main Orchestration Entrypoint
    # ------------------------------------------------------------------
    def predict_trip(self, trip_id: str) -> dict[str, Any]:
        """Execute full map-matching pipeline for `trip_id` and return routes, geometry & metrics."""
        t_start = time.perf_counter()
        bundle = self.get_trip_bundle(trip_id)
        gps_points = bundle.gps_points

        # Stage 2: Candidate generation (Member 1 STRtree SpatialCandidateGenerator)
        t0 = time.perf_counter()
        candidates_per_step, candidates_df = self._generate_candidates(
            bundle, k=8, max_radius_m=120.0
        )
        candidate_ms = (time.perf_counter() - t0) * 1000.0
        total_candidates = sum(len(c) for c in candidates_per_step)

        # Stage 3: Classifier emission scoring (Contract 2)
        t1 = time.perf_counter()
        emission_probs, greedy_step_arcs = self._score_candidates(
            bundle, candidates_per_step, candidates_df=candidates_df
        )
        inference_ms = (time.perf_counter() - t1) * 1000.0

        # Stage 4: HMM Viterbi sequential decoding (Contract 3)
        t2 = time.perf_counter()
        raw_pred_arcs = self._run_hmm_viterbi(bundle, candidates_per_step, emission_probs)
        if len(raw_pred_arcs) == len(gps_points):
            pred_step_arcs = raw_pred_arcs
            reconstructed_route = self._expand_route_with_bridges(bundle, raw_pred_arcs)
        else:
            pred_step_arcs = raw_pred_arcs
            reconstructed_route = _collapse_consecutive(raw_pred_arcs)
        viterbi_ms = (time.perf_counter() - t2) * 1000.0

        # Stage 5: Server-side ground-truth evaluation & geometry packing
        t3 = time.perf_counter()
        gt_step_arcs = self._align_gt_per_step(bundle)
        metrics = self._compute_metrics(
            bundle=bundle,
            gt_step_arcs=gt_step_arcs,
            pred_step_arcs=pred_step_arcs,
            greedy_step_arcs=greedy_step_arcs,
            reconstructed_route=reconstructed_route,
        )

        gt_unique = _collapse_consecutive(bundle.ground_truth_route)
        gt_set = set(gt_unique)
        gt_arcs = [bundle.arc_lookup[a] for a in gt_unique if a in bundle.arc_lookup]
        pred_arcs = [bundle.arc_lookup[a] for a in reconstructed_route if a in bundle.arc_lookup]
        corridor_ids = bundle.spatial_index.query_corridor_arc_ids(gps_points, max_arcs=1100)
        local_road_arcs = [bundle.arc_lookup[a] for a in corridor_ids if a in bundle.arc_lookup]

        step_details: list[dict[str, Any]] = []
        aligned_pred = len(pred_step_arcs) == len(gps_points)
        for idx, pt in enumerate(gps_points):
            gt_a = gt_step_arcs[idx] if idx < len(gt_step_arcs) else None
            pred_a = pred_step_arcs[idx] if aligned_pred else None
            greedy_a = greedy_step_arcs[idx] if idx < len(greedy_step_arcs) else None
            is_match = (
                (pred_a == gt_a or pred_a in gt_set)
                if (pred_a is not None and gt_a is not None)
                else None
            )
            step_details.append(
                {
                    "step_idx": int(pt["step_idx"]),
                    "timestamp": float(pt["timestamp"]),
                    "lat": float(pt["lat"]),
                    "lon": float(pt["lon"]),
                    "ground_truth_arc_id": gt_a,
                    "classifier_only_arc_id": greedy_a,
                    "predicted_arc_id": pred_a,
                    "is_match": is_match,
                }
            )

        metrics_ms = (time.perf_counter() - t3) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0
        metrics["processing_time_ms"] = round(total_ms, 2)

        stage_timings_ms = {
            "candidate_gen_ms": round(candidate_ms, 2),
            "model_inference_ms": round(inference_ms, 2),
            "hmm_viterbi_ms": round(viterbi_ms, 2),
            "metrics_and_pack_ms": round(metrics_ms, 2),
            "total_ms": round(total_ms, 2),
        }

        return {
            "trip_id": trip_id,
            "description": bundle.description,
            "pipeline_mode": self.status.mode,
            "num_gps_points": len(gps_points),
            "num_candidate_pairs": total_candidates,
            "graph_stats": {
                "num_nodes": bundle.graph.number_of_nodes(),
                "num_arcs": bundle.graph.number_of_edges(),
            },
            "metrics": metrics,
            "stage_timings_ms": stage_timings_ms,
            "gps_points": gps_points,
            "road_network_arcs": local_road_arcs,
            "ground_truth_arc_ids": gt_unique,
            "predicted_arc_ids": reconstructed_route,
            "classifier_only_arc_ids": _collapse_consecutive(greedy_step_arcs),
            "ground_truth_arcs": gt_arcs,
            "predicted_arcs": pred_arcs,
            "step_details": step_details,
        }
