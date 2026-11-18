"""Unified interactive Folium map visualizer for PathSnap (Members 1 & 4).

Exposes both interfaces cleanly with zero breakage:
  1. Member 1 EDA / DataFrame interface:
       - `create_folium_map(track_df, nodes_df, arcs_df, route_df=..., ...)`
       - `save_folium_map(m, output_path)`
  2. Member 4 FastAPI / Streamlit Laptop 2 interface:
       - `create_trip_map_from_payload(payload, tiles=..., show_layers=...)`
       - `create_trip_map_from_graph(trip_id, gps_points, graph, ...)`
       - `map_to_html_str(fmap)` / `export_map_html(fmap, output_path)`

Renders all 4 core layers plus optional diagnostics:
  - Layer 1: Raw GPS observations (blue circle markers)
  - Layer 2: Road network arcs (light gray lines / GeoJSON)
  - Layer 3: Ground-truth reference route (green polylines)
  - Layer 4: Model predicted reconstructed route (purple dashed polylines)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Set, Union

import folium
import networkx as nx
import numpy as np
import pandas as pd


LAYER_COLORS = {
    "gps": "#2563EB",          # Blue circle markers
    "road_network": "#94A3B8", # Light gray road network lines
    "ground_truth": "#16A34A", # Green ground-truth route
    "predicted": "#9333EA",    # Purple predicted route (high contrast over green)
    "mismatch": "#DC2626",     # Red highlights for mismatched GPS points
}

SUPPORTED_TILES = {
    "OpenStreetMap": ("OpenStreetMap", None),
    "Light Minimal (Esri Gray)": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
    ),
    "Street Detail (Esri Street)": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "Tiles &copy; Esri",
    ),
}


# ======================================================================
# Member 1 DataFrame Interface (preserved 100% for notebooks & test_member1)
# ======================================================================
def create_folium_map(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route_df: Optional[pd.DataFrame] = None,
    candidate_arc_ids: Optional[Union[List[int], Set[int], np.ndarray]] = None,
    predicted_arcs: Optional[List[int]] = None,
    title: str = "PathSnap Trajectory Visualization",
    gps_sample_step: int = 1,
) -> folium.Map:
    """Generate an interactive multi-layer Folium map from Member 1 DataFrames."""
    center_lat = float(track_df["lat"].mean())
    center_lon = float(track_df["lon"].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Layer 2: Road Network Corridor (Light Gray)
    corridor_arcs: Set[int] = set()
    if candidate_arc_ids is not None:
        corridor_arcs.update(int(a) for a in candidate_arc_ids)
    if route_df is not None:
        corridor_arcs.update(int(a) for a in route_df["arc_id"].values)

    if corridor_arcs:
        network_group = folium.FeatureGroup(name="Road Network Corridor (Light Gray)", show=True)
        selected_arc_ids = list(corridor_arcs)[:8000]
        subset_arcs = arcs_df.loc[selected_arc_ids]

        for row in subset_arcs.itertuples():
            u_node = nodes_df.iloc[row.from_node]
            v_node = nodes_df.iloc[row.to_node]
            folium.PolyLine(
                locations=[[u_node["lat"], u_node["lon"]], [v_node["lat"], v_node["lon"]]],
                color="#7f8c8d",
                weight=2.0,
                opacity=0.6,
                tooltip=f"Arc ID: {row.arc_id} ({row.from_node} -> {row.to_node})",
            ).add_to(network_group)
        network_group.add_to(m)

    # Layer 3: Ground-Truth Reference Route (Green)
    if route_df is not None:
        gt_group = folium.FeatureGroup(name="Ground-Truth Route (Green)", show=True)
        route_arc_ids = route_df["arc_id"].values
        for aid in route_arc_ids:
            if aid < len(arcs_df):
                arc_row = arcs_df.iloc[aid]
                u_node = nodes_df.iloc[arc_row["from_node"]]
                v_node = nodes_df.iloc[arc_row["to_node"]]
                folium.PolyLine(
                    locations=[[u_node["lat"], u_node["lon"]], [v_node["lat"], v_node["lon"]]],
                    color="#27ae60",
                    weight=4.5,
                    opacity=0.9,
                ).add_to(gt_group)
        gt_group.add_to(m)

    # Layer 4: Model Predicted Route (Purple)
    if predicted_arcs is not None:
        pred_group = folium.FeatureGroup(name="Predicted Route (Purple)", show=True)
        for aid in predicted_arcs:
            if aid < len(arcs_df):
                arc_row = arcs_df.iloc[aid]
                u_node = nodes_df.iloc[arc_row["from_node"]]
                v_node = nodes_df.iloc[arc_row["to_node"]]
                folium.PolyLine(
                    locations=[[u_node["lat"], u_node["lon"]], [v_node["lat"], v_node["lon"]]],
                    color="#8e44ad",
                    weight=3.5,
                    opacity=0.85,
                ).add_to(pred_group)
        pred_group.add_to(m)

    # Layer 1: Raw GPS Observations (Blue Circle Markers & Trace)
    gps_group = folium.FeatureGroup(name="Raw GPS Observations (Blue)", show=True)
    gps_coords = track_df[["lat", "lon"]].values.tolist()
    folium.PolyLine(
        locations=gps_coords,
        color="#2980b9",
        weight=1.5,
        dash_array="4, 6",
        opacity=0.6,
    ).add_to(gps_group)

    step = max(1, gps_sample_step)
    if len(track_df) > 1000 and gps_sample_step == 1:
        step = max(1, len(track_df) // 800)

    for idx, row in track_df.iloc[::step].iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3.5,
            color="#2980b9",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.8,
            popup=(
                f"<b>Observation</b>: {row.get('step_idx', idx)}<br>"
                f"<b>Time</b>: {row['timestamp']:.1f} s<br>"
                f"<b>Lat</b>: {row['lat']:.6f}<br>"
                f"<b>Lon</b>: {row['lon']:.6f}"
            ),
        ).add_to(gps_group)

    start_pt = track_df.iloc[0]
    end_pt = track_df.iloc[-1]
    folium.Marker(
        location=[start_pt["lat"], start_pt["lon"]],
        popup=f"<b>Trip Start</b>: t={start_pt['timestamp']:.1f}s",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(gps_group)
    folium.Marker(
        location=[end_pt["lat"], end_pt["lon"]],
        popup=f"<b>Trip End</b>: t={end_pt['timestamp']:.1f}s",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(gps_group)

    gps_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def save_folium_map(
    m: folium.Map,
    output_path: Union[str, Path],
) -> Path:
    """Save folium map to HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    return output_path


# ======================================================================
# Member 4 High-Performance API / Streamlit Laptop 2 Interface
# ======================================================================
def _merge_contiguous_polylines(arcs: Sequence[Mapping[str, Any]]) -> list[list[list[float]]]:
    """Merge chained road arcs into smooth continuous polylines for fast Leaflet rendering."""
    chains: list[list[list[float]]] = []
    current_chain: list[list[float]] = []

    for arc in arcs:
        coords = arc.get("coords", [])
        if len(coords) < 2:
            continue
        p0 = [float(coords[0][0]), float(coords[0][1])]
        p1 = [float(coords[-1][0]), float(coords[-1][1])]

        if not current_chain:
            current_chain = [p0, p1]
        else:
            last = current_chain[-1]
            if abs(last[0] - p0[0]) < 1e-6 and abs(last[1] - p0[1]) < 1e-6:
                current_chain.append(p1)
            else:
                chains.append(current_chain)
                current_chain = [p0, p1]

    if current_chain:
        chains.append(current_chain)
    return chains


def _build_legend_html(
    trip_id: str,
    metrics: Mapping[str, Any] | None = None,
    gps_note: str = "",
) -> str:
    """Create a clean floating HTML legend box for the Folium map."""
    metrics_block = ""
    if metrics:
        top1 = metrics.get("top1_accuracy")
        cont = metrics.get("route_continuity")
        ms = metrics.get("processing_time_ms")
        parts: list[str] = []
        if top1 is not None:
            parts.append(f"Top-1: <b>{float(top1) * 100:.1f}%</b>")
        if cont is not None:
            parts.append(f"Continuity: <b>{float(cont) * 100:.1f}%</b>")
        if ms is not None:
            parts.append(f"Time: <b>{float(ms):.1f} ms</b>")
        if parts:
            metrics_block = (
                "<div style='margin-top:6px;padding-top:6px;border-top:1px solid #E2E8F0;"
                "font-size:11px;color:#334155;'>"
                + " &middot; ".join(parts)
                + "</div>"
            )

    note_block = (
        f"<div style='margin-top:4px;font-size:10.5px;color:#64748B;'>{gps_note}</div>"
        if gps_note
        else ""
    )

    return f"""
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.14);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        color: #0F172A;
        min-width: 235px;
    ">
      <div style="font-weight: 700; font-size: 13px; margin-bottom: 6px;">
        Trip: {trip_id}
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                     background:{LAYER_COLORS['gps']};border:1.5px solid #1E3A8A;"></span>
        <span>Layer 1: Raw GPS Observations</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <span style="display:inline-block;width:18px;height:3px;
                     background:{LAYER_COLORS['road_network']};"></span>
        <span>Layer 2: Road Network Arcs</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <span style="display:inline-block;width:18px;height:4px;
                     background:{LAYER_COLORS['ground_truth']};border-radius:2px;"></span>
        <span>Layer 3: Ground-Truth Route</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <span style="display:inline-block;width:18px;height:0;
                     border-top:3.5px dashed {LAYER_COLORS['predicted']};"></span>
        <span>Layer 4: Predicted HMM Route</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                     background:{LAYER_COLORS['mismatch']};border:1.5px solid #7F1D1D;"></span>
        <span>Mismatched Observations</span>
      </div>
      {metrics_block}
      {note_block}
    </div>
    """


def _compute_center_and_bounds(
    gps_points: Sequence[Mapping[str, Any]],
    arcs: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[list[float]] | None]:
    lats: list[float] = []
    lons: list[float] = []

    for pt in gps_points:
        lat = pt.get("lat", pt.get("gps_lat"))
        lon = pt.get("lon", pt.get("gps_lon"))
        if lat is not None and lon is not None:
            lats.append(float(lat))
            lons.append(float(lon))

    if not lats:
        for arc in arcs:
            for coord in arc.get("coords", []):
                if len(coord) >= 2:
                    lats.append(float(coord[0]))
                    lons.append(float(coord[1]))

    if not lats:
        return [12.9716, 77.5946], None

    center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    return center, bounds


def create_trip_map_from_payload(
    payload: Mapping[str, Any],
    tiles: str = "OpenStreetMap",
    show_layers: Mapping[str, bool] | None = None,
    max_gps_markers: int = 380,
) -> folium.Map:
    """Build an interactive 4-layer Folium map from a `/predict_trip` JSON payload."""
    trip_id = str(payload.get("trip_id", "unknown_trip"))
    gps_points: Sequence[Mapping[str, Any]] = payload.get("gps_points", [])
    road_arcs: Sequence[Mapping[str, Any]] = payload.get("road_network_arcs", [])
    gt_arcs: Sequence[Mapping[str, Any]] = payload.get("ground_truth_arcs", [])
    pred_arcs: Sequence[Mapping[str, Any]] = payload.get("predicted_arcs", [])
    step_details: Sequence[Mapping[str, Any]] = payload.get("step_details", [])
    metrics: Mapping[str, Any] | None = payload.get("metrics")

    visibility = {
        "gps": True,
        "road_network": True,
        "ground_truth": True,
        "predicted": True,
        "mismatches": True,
    }
    if show_layers:
        visibility.update(show_layers)

    tile_spec, tile_attr = SUPPORTED_TILES.get(tiles, ("OpenStreetMap", None))
    center, bounds = _compute_center_and_bounds(gps_points, gt_arcs or pred_arcs or road_arcs)

    map_kwargs: dict[str, Any] = {
        "location": center,
        "zoom_start": 14,
        "tiles": tile_spec,
        "control_scale": True,
        "prefer_canvas": True,
    }
    if tile_attr:
        map_kwargs["attr"] = tile_attr
    fmap = folium.Map(**map_kwargs)

    # Layer 2: Road Network Arcs via fast GeoJSON FeatureCollection
    road_layer = folium.FeatureGroup(
        name="Layer 2: Road Network Arcs",
        show=visibility["road_network"],
    )
    if road_arcs:
        features = []
        for arc in road_arcs:
            coords = arc.get("coords", [])
            if len(coords) < 2:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[float(c[1]), float(c[0])] for c in coords],
                    },
                    "properties": {
                        "arc_id": arc.get("arc_id", -1),
                        "length": arc.get("length", 0.0),
                    },
                }
            )
        if features:
            folium.GeoJson(
                {"type": "FeatureCollection", "features": features},
                style_function=lambda _feat: {
                    "color": LAYER_COLORS["road_network"],
                    "weight": 1.8,
                    "opacity": 0.55,
                },
            ).add_to(road_layer)
    road_layer.add_to(fmap)

    # Layer 3: Ground-Truth Reference Route (Green merged polylines)
    gt_layer = folium.FeatureGroup(
        name="Layer 3: Ground-Truth Route (Green)",
        show=visibility["ground_truth"],
    )
    for chain_idx, chain in enumerate(_merge_contiguous_polylines(gt_arcs), start=1):
        folium.PolyLine(
            locations=chain,
            color=LAYER_COLORS["ground_truth"],
            weight=6.5,
            opacity=0.82,
            tooltip=f"Ground-Truth Route Segment #{chain_idx} ({len(gt_arcs)} total arcs)",
        ).add_to(gt_layer)
    gt_layer.add_to(fmap)

    # Layer 4: Model Predicted Reconstructed Route (Purple dashed merged polylines)
    pred_layer = folium.FeatureGroup(
        name="Layer 4: Predicted HMM Route (Purple)",
        show=visibility["predicted"],
    )
    for chain_idx, chain in enumerate(_merge_contiguous_polylines(pred_arcs), start=1):
        folium.PolyLine(
            locations=chain,
            color=LAYER_COLORS["predicted"],
            weight=3.8,
            opacity=0.95,
            dash_array="8, 6",
            tooltip=f"Predicted HMM Route Segment #{chain_idx} ({len(pred_arcs)} total arcs)",
        ).add_to(pred_layer)
    pred_layer.add_to(fmap)

    detail_by_step: dict[int, Mapping[str, Any]] = {
        int(item["step_idx"]): item for item in step_details if "step_idx" in item
    }

    # Layer 1: Raw GPS Observations (Blue circle markers)
    gps_layer = folium.FeatureGroup(
        name="Layer 1: Raw GPS Observations (Blue)",
        show=visibility["gps"],
    )
    mismatch_layer = folium.FeatureGroup(
        name="Diagnostics: Mismatches & Endpoints",
        show=visibility["mismatches"],
    )

    n_gps = len(gps_points)
    stride = max(1, n_gps // max_gps_markers) if n_gps > max_gps_markers else 1
    sampled_indices = set(range(0, n_gps, stride))
    if n_gps > 0:
        sampled_indices.add(0)
        sampled_indices.add(n_gps - 1)

    mismatch_count = 0
    for idx, pt in enumerate(gps_points):
        step_idx = int(pt.get("step_idx", idx))
        detail = detail_by_step.get(step_idx, {})
        is_match = detail.get("is_match")

        show_blue = idx in sampled_indices
        show_red = is_match is False and mismatch_count < 160
        if not show_blue and not show_red:
            continue

        lat = float(pt.get("lat", pt.get("gps_lat")))
        lon = float(pt.get("lon", pt.get("gps_lon")))
        ts = pt.get("timestamp", idx)
        gt_arc = detail.get("ground_truth_arc_id", "N/A")
        pred_arc = detail.get("predicted_arc_id", "N/A")
        status_label = "MATCH" if is_match is True else ("MISMATCH" if is_match is False else "N/A")

        popup_html = (
            f"<div style='font-family:sans-serif;font-size:12px;min-width:170px;'>"
            f"<b>GPS Observation #{step_idx}</b><br>"
            f"Timestamp: {ts}s<br>"
            f"Coords: ({lat:.5f}, {lon:.5f})<br>"
            f"GT Arc: <b>{gt_arc}</b><br>"
            f"Predicted Arc: <b>{pred_arc}</b><br>"
            f"Status: <b>{status_label}</b>"
            f"</div>"
        )

        if show_blue:
            folium.CircleMarker(
                location=[lat, lon],
                radius=4.0,
                color="#1E3A8A",
                weight=1.0,
                fill=True,
                fill_color=LAYER_COLORS["gps"],
                fill_opacity=0.85,
                tooltip=f"GPS #{step_idx} | GT={gt_arc} | Pred={pred_arc}",
                popup=folium.Popup(popup_html, max_width=240),
            ).add_to(gps_layer)

        if show_red:
            mismatch_count += 1
            folium.CircleMarker(
                location=[lat, lon],
                radius=7.0,
                color=LAYER_COLORS["mismatch"],
                weight=2.0,
                fill=True,
                fill_color=LAYER_COLORS["mismatch"],
                fill_opacity=0.35,
                tooltip=f"MISMATCH at GPS #{step_idx}: GT={gt_arc} vs Pred={pred_arc}",
                popup=folium.Popup(popup_html, max_width=240),
            ).add_to(mismatch_layer)

    gps_layer.add_to(fmap)

    if gps_points:
        first_pt, last_pt = gps_points[0], gps_points[-1]
        folium.Marker(
            location=[
                float(first_pt.get("lat", first_pt.get("gps_lat"))),
                float(first_pt.get("lon", first_pt.get("gps_lon"))),
            ],
            tooltip=f"Trip Start ({trip_id})",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(mismatch_layer)
        folium.Marker(
            location=[
                float(last_pt.get("lat", last_pt.get("gps_lat"))),
                float(last_pt.get("lon", last_pt.get("gps_lon"))),
            ],
            tooltip=f"Trip End ({trip_id})",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        ).add_to(mismatch_layer)

    mismatch_layer.add_to(fmap)

    if bounds is not None:
        fmap.fit_bounds(bounds, padding=(28, 28))

    gps_note = (
        f"Displaying {len(sampled_indices)} / {n_gps:,} GPS markers (1-in-{stride} stride)"
        if stride > 1
        else f"Displaying all {n_gps:,} GPS observations"
    )
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(_build_legend_html(trip_id, metrics, gps_note)))
    return fmap


def _extract_arc_record_from_graph(
    graph: nx.DiGraph,
    u: Any,
    v: Any,
    edge_data: Mapping[str, Any],
) -> dict[str, Any] | None:
    coords = edge_data.get("coords")
    if not coords:
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        u_lat = u_data.get("lat", u_data.get("y"))
        u_lon = u_data.get("lon", u_data.get("x"))
        v_lat = v_data.get("lat", v_data.get("y"))
        v_lon = v_data.get("lon", v_data.get("x"))
        if None in (u_lat, u_lon, v_lat, v_lon):
            return None
        coords = [[float(u_lat), float(u_lon)], [float(v_lat), float(v_lon)]]

    return {
        "arc_id": int(edge_data.get("arc_id", -1)),
        "from_node": int(u) if isinstance(u, int) else u,
        "to_node": int(v) if isinstance(v, int) else v,
        "length": float(edge_data.get("length", edge_data.get("weight", 0.0))),
        "coords": coords,
    }


def create_trip_map_from_graph(
    trip_id: str,
    gps_points: Sequence[Mapping[str, Any]],
    graph: nx.DiGraph,
    ground_truth_arc_ids: Sequence[int],
    predicted_arc_ids: Sequence[int],
    metrics: Mapping[str, Any] | None = None,
    tiles: str = "OpenStreetMap",
    bbox_pad_deg: float = 0.0025,
) -> folium.Map:
    """Build an interactive 4-layer Folium map directly from a NetworkX DiGraph."""
    arc_lookup: dict[int, dict[str, Any]] = {}
    all_arcs: list[dict[str, Any]] = []

    for u, v, data in graph.edges(data=True):
        rec = _extract_arc_record_from_graph(graph, u, v, data)
        if rec is None:
            continue
        all_arcs.append(rec)
        if rec["arc_id"] != -1:
            arc_lookup[rec["arc_id"]] = rec

    lats = [float(p.get("lat", p.get("gps_lat"))) for p in gps_points]
    lons = [float(p.get("lon", p.get("gps_lon"))) for p in gps_points]
    if lats and lons:
        min_lat, max_lat = min(lats) - bbox_pad_deg, max(lats) + bbox_pad_deg
        min_lon, max_lon = min(lons) - bbox_pad_deg, max(lons) + bbox_pad_deg
        local_road_arcs = [
            arc
            for arc in all_arcs
            if any(min_lat <= c[0] <= max_lat and min_lon <= c[1] <= max_lon for c in arc["coords"])
        ][:1100]
    else:
        local_road_arcs = all_arcs[:1100]

    gt_arcs = [arc_lookup[aid] for aid in ground_truth_arc_ids if aid in arc_lookup]
    pred_arcs = [arc_lookup[aid] for aid in predicted_arc_ids if aid in arc_lookup]

    payload = {
        "trip_id": trip_id,
        "gps_points": list(gps_points),
        "road_network_arcs": local_road_arcs,
        "ground_truth_arcs": gt_arcs,
        "predicted_arcs": pred_arcs,
        "step_details": [],
        "metrics": dict(metrics) if metrics else None,
    }
    return create_trip_map_from_payload(payload=payload, tiles=tiles)


def map_to_html_str(fmap: folium.Map) -> str:
    """Serialize a Folium map to a standalone HTML string."""
    return fmap.get_root().render()


def export_map_html(fmap: folium.Map, output_path: str | Path) -> Path:
    """Save an interactive Folium map to an HTML file on disk."""
    return save_folium_map(fmap, output_path)
