
import subprocess
from io import StringIO

import numpy as np

from dynamic_route_alignment import (
    load_file,
    build_candidate_states,
    align_segment,
)

TRIP_ID = "00000086"
CANDIDATE_FILE = "experiments/00000086_candidates.npz"


def main():
    nodes = load_file("nodes")
    arcs = load_file("arcs").astype(int)
    route = load_file("route").ravel().astype(int)
    track = load_file("track")

    saved = np.load(CANDIDATE_FILE, allow_pickle=True)
    candidate_rows = saved["candidate_arc_rows"]

    states = build_candidate_states(
        track, nodes, arcs, route, candidate_rows
    )

    aligned = [None] * len(track)
    segments = []
    start = None

    for i, gps_states in enumerate(states):
        if gps_states and start is None:
            start = i

        if start is not None and (
            not gps_states or i == len(states) - 1
        ):
            end = i if not gps_states else i + 1
            segments.append((start, end))
            start = None

    for start, end in segments:
        selected = align_segment(states[start:end])

        for offset, state in enumerate(selected):
            aligned[start + offset] = state

    valid = [
        (i, state)
        for i, state in enumerate(aligned)
        if state is not None
    ]

    positions = np.array([
        state["position"] for _, state in valid
    ])

    distances = np.array([
        state["distance"] for _, state in valid
    ])

    increments = np.diff(positions)

    used_positions = np.unique(positions)

    print("\n========== DP ALIGNMENT AUDIT ==========")
    print("GPS observations:", len(track))
    print("Aligned observations:", len(valid))
    print("Unaligned observations:", len(track) - len(valid))

    print("\nRoute progression:")
    print("First assigned position:", positions[0])
    print("Last assigned position:", positions[-1])
    print("Minimum assigned position:", positions.min())
    print("Maximum assigned position:", positions.max())
    print("Unique route positions used:", len(used_positions))
    print("Reference route positions:", len(route))

    print("\nProgression increments:")
    print("Zero increments:", np.sum(increments == 0))
    print("One-position increments:", np.sum(increments == 1))
    print("Multi-position jumps:", np.sum(increments > 1))
    print("Maximum forward jump:", increments.max())

    print("\nGeometric fit:")
    print("Mean distance:", f"{distances.mean():.2f} m")
    print("Median distance:", f"{np.median(distances):.2f} m")
    print("95th percentile:", f"{np.percentile(distances, 95):.2f} m")
    print("Maximum distance:", f"{distances.max():.2f} m")

    print("\nFirst 10 selected route positions:")
    print(positions[:10].tolist())

    print("\nLast 10 selected route positions:")
    print(positions[-10:].tolist())

    print("\nFirst and last GPS timestamps:")
    print(track[valid[0][0], 2], track[valid[-1][0], 2])


if __name__ == "__main__":
    main()
