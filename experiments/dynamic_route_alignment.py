
import subprocess
from io import StringIO

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point


TRIP_ID = "00000086"
ROOT = "data/Student folder/train"
CANDIDATE_FILE = "experiments/00000086_candidates.npz"

GPS_SIGMA_METERS = 10.0
SKIP_PENALTY = 0.05


def load_file(extension):
    path = f"{ROOT}/{TRIP_ID}/{TRIP_ID}.{extension}"

    raw = subprocess.check_output(
        ["git", "show", f"upstream/master:{path}"],
        text=True
    )

    return np.loadtxt(StringIO(raw), ndmin=2)


def build_candidate_states(track, nodes, arcs, route, candidate_rows):
    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32640",
        always_xy=True
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

    geometry_cache = {}

    def get_geometry(arc_row):
        if arc_row not in geometry_cache:
            start_node, end_node = arcs[arc_row]

            geometry_cache[arc_row] = LineString([
                (node_x[start_node], node_y[start_node]),
                (node_x[end_node], node_y[end_node])
            ])

        return geometry_cache[arc_row]

    all_states = []

    for gps_index, candidates in enumerate(candidate_rows):
        point = Point(gps_x[gps_index], gps_y[gps_index])

        states = {}

        for arc_row in candidates:
            arc_row = int(arc_row)
            positions = route_positions.get(arc_row, [])

            if not positions:
                continue

            distance = point.distance(get_geometry(arc_row))

            for position in positions:
                # Each route occurrence is a separate state.
                states[position] = {
                    "position": position,
                    "arc_row": arc_row,
                    "distance": distance
                }

        all_states.append(
            sorted(states.values(), key=lambda state: state["position"])
        )

    return all_states


def align_segment(segment_states):
    """
    Find the minimum-cost non-decreasing route-position sequence.
    """

    costs = []
    backpointers = []

    first_costs = np.array([
        (state["distance"] / GPS_SIGMA_METERS) ** 2 / 2
        for state in segment_states[0]
    ])

    costs.append(first_costs)
    backpointers.append(
        np.full(len(first_costs), -1, dtype=int)
    )

    for t in range(1, len(segment_states)):
        previous_states = segment_states[t - 1]
        current_states = segment_states[t]

        current_costs = np.full(len(current_states), np.inf)
        current_backpointers = np.full(
            len(current_states), -1, dtype=int
        )

        for j, current in enumerate(current_states):
            emission_cost = (
                current["distance"] / GPS_SIGMA_METERS
            ) ** 2 / 2

            for i, previous in enumerate(previous_states):
                delta = (
                    current["position"] - previous["position"]
                )

                # Do not allow route-position reversals.
                if delta < 0:
                    continue

                # Small penalty for skipping route positions.
                transition_cost = SKIP_PENALTY * max(
                    0, delta - 1
                )

                total_cost = (
                    costs[t - 1][i]
                    + transition_cost
                    + emission_cost
                )

                if total_cost < current_costs[j]:
                    current_costs[j] = total_cost
                    current_backpointers[j] = i

        costs.append(current_costs)
        backpointers.append(current_backpointers)

    if not np.isfinite(costs[-1]).any():
        raise RuntimeError(
            "No valid monotonic alignment found for a segment."
        )

    best_index = int(np.argmin(costs[-1]))
    selected = [None] * len(segment_states)

    for t in range(len(segment_states) - 1, -1, -1):
        selected[t] = segment_states[t][best_index]

        if t > 0:
            best_index = backpointers[t][best_index]

    return selected


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

    # Independent nearest-candidate baseline.
    greedy = []

    for gps_states in states:
        if gps_states:
            greedy.append(
                min(gps_states, key=lambda state: state["distance"])
            )
        else:
            greedy.append(None)

    # Align each contiguous block with available route candidates.
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

    greedy_valid = [
        state for state in greedy if state is not None
    ]

    aligned_valid = [
        state for state in aligned if state is not None
    ]

    greedy_positions = [
        state["position"] for state in greedy_valid
    ]

    aligned_positions = [
        state["position"] for state in aligned_valid
    ]

    greedy_decreases = sum(
        current < previous
        for previous, current in zip(
            greedy_positions[:-1], greedy_positions[1:]
        )
    )

    aligned_decreases = sum(
        current < previous
        for previous, current in zip(
            aligned_positions[:-1], aligned_positions[1:]
        )
    )

    greedy_distances = [
        state["distance"] for state in greedy_valid
    ]

    aligned_distances = [
        state["distance"] for state in aligned_valid
    ]

    print("\n========== DYNAMIC ROUTE ALIGNMENT ==========")
    print("GPS observations:", len(track))
    print("Reference route entries:", len(route))
    print("Observations with route candidates:", len(aligned_valid))
    print("Observations without route candidates:",
          len(track) - len(aligned_valid))
    print("Contiguous alignment segments:", len(segments))

    print("\n--- Greedy nearest-candidate baseline ---")
    print("Route-position decreases:", greedy_decreases)
    print(
        "Median geometric distance:",
        f"{np.median(greedy_distances):.2f} m"
    )
    print(
        "95th percentile distance:",
        f"{np.percentile(greedy_distances, 95):.2f} m"
    )

    print("\n--- Dynamic-programming alignment ---")
    print("Route-position decreases:", aligned_decreases)
    print(
        "Median geometric distance:",
        f"{np.median(aligned_distances):.2f} m"
    )
    print(
        "95th percentile distance:",
        f"{np.percentile(aligned_distances, 95):.2f} m"
    )

    print("\nFirst 20 aligned observations:")

    for i in range(min(20, len(track))):
        state = aligned[i]

        if state is None:
            print(f"GPS {i:4d} | no route candidate")
        else:
            print(
                f"GPS {i:4d} | time={track[i, 2]:8.1f} | "
                f"route_pos={state['position']:4d} | "
                f"arc={state['arc_row']:6d} | "
                f"distance={state['distance']:7.2f} m"
            )

    print("\nAlignment around the four earlier reversals:")

    for center in [0, 3354, 3744, 3772]:
        print(f"\nGPS window around {center}:")

        for i in range(max(0, center - 2),
                       min(len(track), center + 4)):
            state = aligned[i]

            if state is None:
                print(f"GPS {i}: no route candidate")
            else:
                print(
                    f"GPS {i:4d} | "
                    f"route_pos={state['position']:4d} | "
                    f"arc={state['arc_row']:6d} | "
                    f"distance={state['distance']:7.2f} m"
                )


if __name__ == "__main__":
    main()
