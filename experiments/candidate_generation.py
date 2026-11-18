
import subprocess
from io import StringIO

import numpy as np
import shapely
from shapely import STRtree, LineString, Point
from pyproj import Transformer


TRIP_ID = "00000086"
ROOT = "data/Student folder/train"

SEARCH_RADIUS_M = 50
MAX_CANDIDATES = 16


def load_file(extension):
    path = f"{ROOT}/{TRIP_ID}/{TRIP_ID}.{extension}"

    raw = subprocess.check_output(
        ["git", "show", f"upstream/master:{path}"],
        text=True
    )

    return np.loadtxt(StringIO(raw), ndmin=2)


def main():
    nodes = load_file("nodes")
    arcs = load_file("arcs").astype(int)
    track = load_file("track")

    gps = track[:, :2]

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32640",
        always_xy=True
    )

    node_x, node_y = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )

    gps_x, gps_y = transformer.transform(
        gps[:, 0], gps[:, 1]
    )

    projected_nodes = np.column_stack((node_x, node_y))
    projected_gps = np.column_stack((gps_x, gps_y))

    # Build actual directed road-segment geometries.
    start_points = projected_nodes[arcs[:, 0]]
    end_points = projected_nodes[arcs[:, 1]]

    segments = np.array([
        LineString([start, end])
        for start, end in zip(start_points, end_points)
    ], dtype=object)

    tree = STRtree(segments)

    candidate_counts = []
    nearest_distances = []
    candidate_arc_rows = []

    for point_xy in projected_gps:
        point = Point(point_xy)

        indices = tree.query(
            point,
            predicate="dwithin",
            distance=SEARCH_RADIUS_M
        )

        if len(indices) == 0:
            candidate_counts.append(0)
            nearest_distances.append(float("inf"))
            candidate_arc_rows.append([])
            continue

        distances = shapely.distance(
            segments[indices], point
        )

        order = np.argsort(distances)[:MAX_CANDIDATES]

        selected_indices = indices[order]
        selected_distances = distances[order]

        candidate_counts.append(len(selected_indices))
        nearest_distances.append(float(selected_distances[0]))
        candidate_arc_rows.append(selected_indices.tolist())

    candidate_counts = np.asarray(candidate_counts)
    nearest_distances = np.asarray(nearest_distances)

    covered = np.isfinite(nearest_distances)

    print("\n========== STRtree CANDIDATE GENERATION ==========")
    print("Shapely version:", shapely.__version__)
    print("GPS observations:", len(gps))
    print("Road segments:", len(segments))
    print("Search radius:", SEARCH_RADIUS_M, "metres")
    print("Maximum candidates:", MAX_CANDIDATES)

    print("\nCandidate availability:")
    print(
        "Observations with candidates:",
        f"{covered.sum()}/{len(gps)}"
    )
    print("Coverage:", f"{covered.mean() * 100:.2f}%")

    print("\nCandidate count:")
    print("Mean:", round(float(candidate_counts.mean()), 2))
    print("Median:", float(np.median(candidate_counts)))
    print("Maximum:", int(candidate_counts.max()))

    if covered.any():
        print("\nNearest segment distance:")
        print(
            "Median (m):",
            round(float(np.median(nearest_distances[covered])), 2)
        )
        print(
            "95th percentile (m):",
            round(float(np.percentile(nearest_distances[covered], 95)), 2)
        )
        print(
            "Maximum (m):",
            round(float(nearest_distances[covered].max()), 2)
        )

    # Save candidate row indices for later alignment experiments.
    np.savez_compressed(
        f"experiments/{TRIP_ID}_candidates.npz",
        candidate_arc_rows=np.array(
            candidate_arc_rows, dtype=object
        ),
        candidate_counts=candidate_counts,
        nearest_distances=nearest_distances
    )

    print("\nSaved candidate results.")


if __name__ == "__main__":
    main()
