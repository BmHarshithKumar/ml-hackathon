
import subprocess
from io import StringIO

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

    unique_count = len(np.unique(route))
    repeated_count = len(route) - unique_count

    connected = (
        route_edges[:-1, 1] == route_edges[1:, 0]
    )

    print("\n========== ROUTE PROGRESSION ==========")

    print("Route entries:", len(route))
    print("Unique route arcs:", unique_count)
    print("Repeated route entries:", repeated_count)

    print(
        "Directed connected transitions:",
        f"{connected.sum()}/{len(connected)}"
    )

    print(
        "Directed connectivity:",
        f"{connected.mean() * 100:.2f}%"
    )

    print("\nFirst 20 route transitions:")

    for i in range(min(20, len(route) - 1)):
        print(
            f"Position {i:3d}: "
            f"arc row {route[i]:6d} "
            f"({route_edges[i, 0]} -> {route_edges[i, 1]})"
            f" | next arc row {route[i + 1]:6d} "
            f"({route_edges[i + 1, 0]} -> {route_edges[i + 1, 1]})"
        )


if __name__ == "__main__":
    main()
