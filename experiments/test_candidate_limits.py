
import subprocess
from io import StringIO

import numpy as np
from pyproj import Transformer
from shapely import STRtree, LineString, Point
import shapely


TRIP_ID = "00000086"
ROOT = "data/Student folder/train"

SEARCH_RADIUS_M = 50
CANDIDATE_LIMITS = [8, 16, 32, 64]


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
    route = load_file("route").ravel().astype(int)
    track = load_file("track")

    gps = track[:, :2]
    reference_arcs = set(route.tolist())

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

    start_points = projected_nodes[arcs[:, 0]]
    end_points = projected_nodes[arcs[:, 1]]

    segments = np.array([
        LineString([start, end])
        for start, end in zip(start_points, end_points)
    ], dtype=object)

    tree = STRtree(segments)

    missed_observations = [3840, 3841]

    print("\n========== CANDIDATE LIMIT EXPERIMENT ==========")
    print("Shapely:", shapely.__version__)
    print("Search radius:", SEARCH_RADIUS_M, "metres")

    for limit in CANDIDATE_LIMITS:
        covered = 0
        total = len(gps)
        missed = []

        for i, point_xy in enumerate(projected_gps):
            point = Point(point_xy)

            indices = tree.query(
                point,
                predicate="dwithin",
                distance=SEARCH_RADIUS_M
            )

            if len(indices) == 0:
                missed.append(i)
                continue

            distances = shapely.distance(
                segments[indices], point
            )

            order = np.argsort(distances)[:limit]
            selected = indices[order]

            if reference_arcs.intersection(set(selected.tolist())):
                covered += 1
            else:
                missed.append(i)

        print(f"\nTop {limit} candidates:")
        print("Reference-route coverage:",
              f"{covered}/{total} ({covered / total * 100:.2f}%)")
        print("Missed observations:", len(missed))

        if limit == CANDIDATE_LIMITS[0]:
            for observation in missed_observations:
                point = Point(projected_gps[observation])

                indices = tree.query(
                    point,
                    predicate="dwithin",
                    distance=SEARCH_RADIUS_M
                )

                distances = shapely.distance(
                    segments[indices], point
                )

                order = np.argsort(distances)
                ranked_indices = indices[order]
                ranked_distances = distances[order]

                print(
                    f"\nDetailed ranking for GPS observation "
                    f"{observation}:"
                )

                for rank, (arc_row, distance) in enumerate(
                    zip(ranked_indices[:40], ranked_distances[:40]),
                    start=1
                ):
                    is_reference = int(arc_row) in reference_arcs

                    print(
                        f"Rank {rank:2d} | "
                        f"Arc row {arc_row:6d} | "
                        f"Distance {distance:7.2f} m | "
                        f"Reference route: {is_reference}"
                    )


if __name__ == "__main__":
    main()
