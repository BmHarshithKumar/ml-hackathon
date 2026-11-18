
import numpy as np
import pandas as pd

from pyproj import CRS, Transformer


MAX_SPEED_GAP_SECONDS = 5.0


def create_transformer(longitudes, latitudes):
    """Create a local UTM transformer based on trip coordinates."""
    lon = float(np.mean(longitudes))
    lat = float(np.mean(latitudes))

    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone

    return Transformer.from_crs(
        "EPSG:4326",
        CRS.from_epsg(epsg),
        always_xy=True
    )


def calculate_bearing(dx, dy):
    """Calculate bearing clockwise from north, in degrees."""
    return (np.degrees(np.arctan2(dx, dy)) + 360) % 360


def calculate_heading_difference(a, b):
    """Smallest absolute angular difference between two bearings."""
    return np.abs((a - b + 180) % 360 - 180)


def calculate_gps_motion(track, gps_x, gps_y):
    """Calculate GPS speed and heading using actual timestamps."""
    n = len(track)

    speed = np.full(n, np.nan)
    heading = np.full(n, np.nan)
    dt_values = np.full(n, np.nan)

    timestamps = track[:, 2]

    for i in range(1, n):
        dt = timestamps[i] - timestamps[i - 1]
        dt_values[i] = dt

        if not (0 < dt <= MAX_SPEED_GAP_SECONDS):
            continue

        dx = gps_x[i] - gps_x[i - 1]
        dy = gps_y[i] - gps_y[i - 1]

        distance = np.hypot(dx, dy)

        speed[i] = distance / dt

        if distance > 0.5:
            heading[i] = calculate_bearing(dx, dy)

    return speed, heading, dt_values


def generate_candidate_features(
    track,
    nodes,
    arcs,
    candidate_arc_rows,
    trip_id=None
):
    """
    Generate one feature row per GPS observation and candidate arc.

    Coordinates:
        track[:, 0] = longitude
        track[:, 1] = latitude
        track[:, 2] = timestamp in seconds

    arcs contain directed node-index pairs.

    Candidate arc rows are row indices into arcs.
    """

    track = np.asarray(track, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    arcs = np.asarray(arcs, dtype=int)

    if len(track) != len(candidate_arc_rows):
        raise ValueError(
            "Number of candidate lists must match GPS observations."
        )

    if track.shape[1] < 3:
        raise ValueError("Track must contain longitude, latitude and time.")

    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError("Arc endpoint references an invalid node.")

    transformer = create_transformer(
        track[:, 0],
        track[:, 1]
    )

    gps_x, gps_y = transformer.transform(
        track[:, 0],
        track[:, 1]
    )

    node_x, node_y = transformer.transform(
        nodes[:, 0],
        nodes[:, 1]
    )

    gps_x = np.asarray(gps_x)
    gps_y = np.asarray(gps_y)
    node_x = np.asarray(node_x)
    node_y = np.asarray(node_y)

    speed, gps_heading, dt = calculate_gps_motion(
        track,
        gps_x,
        gps_y
    )

    rows = []

    for gps_index, candidates in enumerate(candidate_arc_rows):

        if len(candidates) == 0:
            continue

        px = gps_x[gps_index]
        py = gps_y[gps_index]

        for arc_row in candidates:

            arc_row = int(arc_row)

            if not 0 <= arc_row < len(arcs):
                raise ValueError(
                    f"Invalid candidate arc row: {arc_row}"
                )

            start_node, end_node = arcs[arc_row]

            x1 = node_x[start_node]
            y1 = node_y[start_node]

            x2 = node_x[end_node]
            y2 = node_y[end_node]

            dx = x2 - x1
            dy = y2 - y1

            segment_length = np.hypot(dx, dy)

            if segment_length == 0:
                projection_fraction = 0.0
                distance = np.hypot(px - x1, py - y1)
                along_track = 0.0
                road_heading = np.nan

            else:
                projection_fraction = np.clip(
                    ((px - x1) * dx + (py - y1) * dy)
                    / (segment_length ** 2),
                    0.0,
                    1.0
                )

                closest_x = x1 + projection_fraction * dx
                closest_y = y1 + projection_fraction * dy

                distance = np.hypot(
                    px - closest_x,
                    py - closest_y
                )

                along_track = (
                    projection_fraction * segment_length
                )

                road_heading = calculate_bearing(dx, dy)

            if (
                np.isfinite(gps_heading[gps_index])
                and np.isfinite(road_heading)
            ):
                heading_difference = calculate_heading_difference(
                    gps_heading[gps_index],
                    road_heading
                )
            else:
                heading_difference = np.nan

            rows.append({
                "trip_id": trip_id,
                "gps_index": gps_index,
                "timestamp": track[gps_index, 2],
                "candidate_arc_row": arc_row,
                "start_node": int(start_node),
                "end_node": int(end_node),
                "distance_m": distance,
                "gps_speed_mps": speed[gps_index],
                "gps_heading_deg": gps_heading[gps_index],
                "road_heading_deg": road_heading,
                "heading_difference_deg": heading_difference,
                "segment_length_m": segment_length,
                "projection_fraction": projection_fraction,
                "along_track_m": along_track,
                "time_delta_s": dt[gps_index]
            })

    columns = [
        "trip_id",
        "gps_index",
        "timestamp",
        "candidate_arc_row",
        "start_node",
        "end_node",
        "distance_m",
        "gps_speed_mps",
        "gps_heading_deg",
        "road_heading_deg",
        "heading_difference_deg",
        "segment_length_m",
        "projection_fraction",
        "along_track_m",
        "time_delta_s"
    ]

    return pd.DataFrame(rows, columns=columns)
