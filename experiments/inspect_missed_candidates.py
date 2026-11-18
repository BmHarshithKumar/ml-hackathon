
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
    track = load_file("track")
    route = load_file("route").ravel().astype(int)

    saved = np.load(
        f"experiments/{TRIP_ID}_candidates.npz",
        allow_pickle=True
    )

    candidate_rows = saved["candidate_arc_rows"]
    nearest_distances = saved["nearest_distances"]

    reference_arcs = set(route.tolist())

    missed = []

    for i, candidates in enumerate(candidate_rows):
        if not reference_arcs.intersection(set(candidates)):
            missed.append(i)

    print("\n========== MISSED REFERENCE CANDIDATES ==========")
    print("Total missed observations:", len(missed))

    for i in missed:
        print(f"\nGPS observation: {i}")
        print("Coordinates (lon, lat):", track[i, :2])
        print("Timestamp:", track[i, 2])
        print("Nearest road distance (m):", nearest_distances[i])
        print("Candidate arc rows:", candidate_rows[i])

        if i > 0:
            print("Previous GPS coordinates:", track[i - 1, :2])

        if i < len(track) - 1:
            print("Next GPS coordinates:", track[i + 1, :2])


if __name__ == "__main__":
    main()
