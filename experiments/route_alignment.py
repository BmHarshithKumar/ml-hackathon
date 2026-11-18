
import subprocess
from io import StringIO

import numpy as np
from scipy.spatial.distance import cdist


TRIP_ID = "00000086"
ROOT = "data/Student folder/train"


def load_trip_file(extension):
    path = f"{ROOT}/{TRIP_ID}/{TRIP_ID}.{extension}"

    raw = subprocess.check_output(
        ["git", "show", f"upstream/master:{path}"],
        text=True
    )

    return np.loadtxt(StringIO(raw), ndmin=2)


def haversine(lon1, lat1, lon2, lat2):
    radius = 6371000

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


def main():
    nodes = load_trip_file("nodes")
    arcs = load_trip_file("arcs").astype(int)
    route = load_trip_file("route").ravel().astype(int)
    track = load_trip_file("track")

    # Reconstruct the reference route's node coordinates.
    route_arcs = arcs[route]

    route_start = nodes[route_arcs[:, 0]]
    route_end = nodes[route_arcs[:, 1]]

    # GPS coordinates: longitude, latitude.
    gps = track[:, :2]
    timestamps = track[:, 2]

    # Calculate distances between consecutive GPS observations.
    gps_distances = haversine(
        gps[:-1, 0],
        gps[:-1, 1],
        gps[1:, 0],
        gps[1:, 1]
    )

    time_intervals = np.diff(timestamps)

    valid_time = time_intervals > 0

    gps_speeds = np.zeros(len(time_intervals))

    gps_speeds[valid_time] = (
        gps_distances[valid_time]
        / time_intervals[valid_time]
    )

    print("\n========== ROUTE ALIGNMENT DIAGNOSTICS ==========")

    print("GPS observations:", len(gps))
    print("Reference route arcs:", len(route))

    print("\nGPS trajectory:")
    print("Start:", gps[0])
    print("End:", gps[-1])

    print("\nReference route:")
    print("Start node:", route_start[0])
    print("End node:", route_end[-1])

    print("\nGPS movement:")
    print("Total GPS distance (km):",
          round(gps_distances.sum() / 1000, 3))

    print("Median GPS speed (m/s):",
          round(np.median(gps_speeds[valid_time]), 3))

    print("\nRoute geometry:")
    print("First 5 route arc IDs:", route[:5])
    print("Last 5 route arc IDs:", route[-5:])

    # Approximate geographic endpoint agreement.
    start_distance = haversine(
        gps[0, 0], gps[0, 1],
        route_start[0, 0], route_start[0, 1]
    )

    end_distance = haversine(
        gps[-1, 0], gps[-1, 1],
        route_end[-1, 0], route_end[-1, 1]
    )

    print("\nEndpoint agreement:")
    print("Start distance (m):", round(float(start_distance), 2))
    print("End distance (m):", round(float(end_distance), 2))


if __name__ == "__main__":
    main()

