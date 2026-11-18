
import subprocess
from io import StringIO
from collections import defaultdict

import numpy as np


TRIP_ID = "00000086"
ROOT = "data/Student folder/train"


def load_file(extension):
    path = f"{ROOT}/{TRIP_ID}/{TRIP_ID}.{extension}"

    raw = subprocess.check_output(
        ["git", "show", f"upstream/master:{path}"],
        text=True
    )

    return np.loadtxt(StringIO(raw), ndmin=2)


def main():
    arcs = load_file("arcs").astype(int)
    route = load_file("route").ravel().astype(int)

    route_edges = arcs[route]

    positions_by_arc = defaultdict(list)

    for position, arc_row in enumerate(route):
        positions_by_arc[int(arc_row)].append(position)

    print("\n========== ROUTE ANOMALY INSPECTION ==========")

    print("\nRepeated arc rows:")

    repeated = {
        arc: positions
        for arc, positions in positions_by_arc.items()
        if len(positions) > 1
    }

    if not repeated:
        print("No repeated arcs.")
    else:
        for arc, positions in repeated.items():
            print(
                f"Arc row {arc}: positions {positions}, "
                f"edge {arcs[arc, 0]} -> {arcs[arc, 1]}"
            )

    print("\nDisconnected directed transitions:")

    disconnected = []

    for i in range(len(route_edges) - 1):
        current_end = route_edges[i, 1]
        next_start = route_edges[i + 1, 0]

        if current_end != next_start:
            disconnected.append(i)

            print(
                f"Position {i} -> {i + 1}: "
                f"arc {route[i]} "
                f"({route_edges[i, 0]} -> {current_end}) "
                f"then arc {route[i + 1]} "
                f"({next_start} -> {route_edges[i + 1, 1]})"
            )

    print("\nSummary:")
    print("Repeated arc rows:", len(repeated))
    print("Disconnected transitions:", len(disconnected))


if __name__ == "__main__":
    main()
