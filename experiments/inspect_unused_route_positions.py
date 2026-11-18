
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString

from dynamic_route_alignment import (
    load_file,
    build_candidate_states,
    align_segment,
    CANDIDATE_FILE,
)


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
    start = None

    for i, gps_states in enumerate(states):
        if gps_states and start is None:
            start = i

        if start is not None and (
            not gps_states or i == len(states) - 1
        ):
            end = i if not gps_states else i + 1
            selected = align_segment(states[start:end])

            for offset, state in enumerate(selected):
                aligned[start + offset] = state

            start = None

    used_positions = {
        state["position"]
        for state in aligned
        if state is not None
    }

    all_positions = set(range(len(route)))
    unused_positions = sorted(all_positions - used_positions)

    increments = []

    for previous, current in zip(aligned[:-1], aligned[1:]):
        if previous is not None and current is not None:
            increments.append(
                current["position"] - previous["position"]
            )

    skipped_between = []

    for i, delta in enumerate(increments):
        if delta > 1:
            previous_position = aligned[i]["position"]

            for position in range(
                previous_position + 1,
                previous_position + delta
            ):
                skipped_between.append(position)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32640",
        always_xy=True
    )

    node_x, node_y = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )

    print("\n========== UNUSED ROUTE POSITION AUDIT ==========")

    print("Reference route positions:", len(route))
    print("Used route positions:", len(used_positions))
    print("Unused route positions:", len(unused_positions))

    print("\nFirst selected position:",
          min(used_positions) if used_positions else None)

    print("Last selected position:",
          max(used_positions) if used_positions else None)

    print("\nInitial unused positions:")
    initial = [
        p for p in unused_positions
        if p < min(used_positions)
    ]
    print(initial)

    print("\nPositions skipped between selected states:")
    print(sorted(skipped_between))

    print("\n========== UNUSED ARC DETAILS ==========")

    for position in unused_positions:
        arc_row = int(route[position])
        start_node, end_node = arcs[arc_row]

        x1, y1 = node_x[start_node], node_y[start_node]
        x2, y2 = node_x[end_node], node_y[end_node]

        length = LineString([
            (x1, y1), (x2, y2)
        ]).length

        if position < min(used_positions):
            reason = "INITIAL POSITION NOT SELECTED"
        elif position > max(used_positions):
            reason = "AFTER LAST SELECTED POSITION"
        else:
            reason = "SKIPPED DURING FORWARD TRANSITION"

        print(
            f"Route position: {position:4d} | "
            f"Arc row: {arc_row:6d} | "
            f"From: {start_node:6d} | "
            f"To: {end_node:6d} | "
            f"Length: {length:8.2f} m | "
            f"{reason}"
        )

    print("\n========== AUDIT SUMMARY ==========")

    print("Initial unused:", len(initial))
    print("Skipped internally:", len(skipped_between))
    print("Unused after final position:",
          sum(p > max(used_positions) for p in unused_positions))

    print("\nAudit completed.")


if __name__ == "__main__":
    main()
