
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
    nodes = load_file("nodes")
    arcs = load_file("arcs").astype(int)
    route = load_file("route").ravel().astype(int)
    track = load_file("track")

    gps = track[:, :2]
    route_nodes = np.unique(arcs[route].ravel())
    route_coords = nodes[route_nodes]

    print("\n========== COORDINATE INSPECTION ==========")

    print("\nNode coordinate ranges:")
    print("Column 1:", nodes[:, 0].min(), nodes[:, 0].max())
    print("Column 2:", nodes[:, 1].min(), nodes[:, 1].max())

    print("\nGPS coordinate ranges:")
    print("Longitude:", gps[:, 0].min(), gps[:, 0].max())
    print("Latitude:", gps[:, 1].min(), gps[:, 1].max())

    print("\nReference route coordinate ranges:")
    print("Column 1:", route_coords[:, 0].min(),
          route_coords[:, 0].max())
    print("Column 2:", route_coords[:, 1].min(),
          route_coords[:, 1].max())

    print("\nGPS observations:", len(gps))
    print("Unique route nodes:", len(route_nodes))

    # Check whether all GPS points lie within the overall
    # bounding box of the reference route.
    inside = (
        (gps[:, 0] >= route_coords[:, 0].min())
        & (gps[:, 0] <= route_coords[:, 0].max())
        & (gps[:, 1] >= route_coords[:, 1].min())
        & (gps[:, 1] <= route_coords[:, 1].max())
    )

    print("\nGPS points inside route bounding box:",
          f"{inside.sum()}/{len(gps)}",
          f"({inside.mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()

