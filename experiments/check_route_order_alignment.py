
import subprocess
from io import StringIO

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point


TRIP_ID = "00000086"
ROOT = "data/Student folder/train"
CANDIDATE_FILE = "experiments/00000086_candidates.npz"

REVERSALS = [0, 3354, 3744, 3772]
WINDOW = 2


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

    saved = np.load(CANDIDATE_FILE, allow_pickle=True)
    candidate_rows = saved["candidate_arc_rows"]

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32640", always_xy=True
    )

    node_x, node_y = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )
    gps_x, gps_y = transformer.transform(
        track[:, 0], track[:, 1]
    )

    route_positions = {}
    for position, arc_row in enumerate(route):
        route_positions.setdefault(int(arc_row), []).append(position)

    # Cache geometries for candidate road segments.
    geometry_cache = {}

    def get_geometry(arc_row):
        if arc_row not in geometry_cache:
            start_node, end_node = arcs[arc_row]
            geometry_cache[arc_row] = LineString([
                (node_x[start_node], node_y[start_node]),
                (node_x[end_node], node_y[end_node])
            ])
        return geometry_cache[arc_row]

    print("\n========== ROUTE REVERSAL INSPECTION ==========")

    for reversal in REVERSALS:
        print(f"\n\n--- GPS transition {reversal} -> {reversal + 1} ---")

        start = max(0, reversal - WINDOW)
        end = min(len(track), reversal + WINDOW + 2)

        for gps_index in range(start, end):
            point = Point(gps_x[gps_index], gps_y[gps_index])
            matches = []

            for arc_row in candidate_rows[gps_index]:
                arc_row = int(arc_row)
                positions = route_positions.get(arc_row)

                if not positions:
                    continue

                distance = point.distance(get_geometry(arc_row))

                for position in positions:
                    matches.append((distance, position, arc_row))

            matches.sort()

            print(
                f"\nGPS {gps_index} | "
                f"time={track[gps_index, 2]:.1f} | "
                f"lon={track[gps_index, 0]:.6f} | "
                f"lat={track[gps_index, 1]:.6f}"
            )

            for distance, position, arc_row in matches[:8]:
                start_node, end_node = arcs[arc_row]
                print(
                    f"  route_pos={position:4d} | "
                    f"arc={arc_row:6d} | "
                    f"distance={distance:7.2f} m | "
                    f"edge={start_node}->{end_node}"
                )


if __name__ == "__main__":
    main()
