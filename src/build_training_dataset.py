import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from src.features import generate_candidate_features


UPSTREAM = "upstream/master"
DATA_ROOT = "data/Student folder/train"
OUTPUT = Path("artifacts/training_dataset.csv")

SEARCH_RADIUS_M = 50.0
MAX_CANDIDATES = 16
GPS_SIGMA_M = 10.0
SKIP_PENALTY = 0.05
MAX_ALIGNMENT_GAP_S = 300.0


def run_git(args):
    return subprocess.run(
        args, check=True, capture_output=True, text=True
    ).stdout


def load_trip_file(trip_id, extension):
    path = f"{DATA_ROOT}/{trip_id}/{trip_id}.{extension}"
    content = run_git(["git", "show", f"{UPSTREAM}:{path}"])
    return np.loadtxt(content.splitlines())


def get_training_trip_ids():
    result = run_git([
        "git", "ls-tree", "-d", "--name-only",
        f"{UPSTREAM}:{DATA_ROOT}",
    ])
    return sorted(x.strip() for x in result.splitlines() if x.strip())


def build_candidates(track, nodes, arcs):
    lon = float(np.mean(track[:, 0]))
    lat = float(np.mean(track[:, 1]))
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone

    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )

    node_x, node_y = transformer.transform(nodes[:, 0], nodes[:, 1])
    gps_x, gps_y = transformer.transform(track[:, 0], track[:, 1])

    segments = [
        LineString([
            (node_x[int(start)], node_y[int(start)]),
            (node_x[int(end)], node_y[int(end)]),
        ])
        for start, end in arcs
    ]

    tree = STRtree(segments)
    candidates = []

    for x, y in zip(gps_x, gps_y):
        point = Point(x, y)
        indices = tree.query(
            point, predicate="dwithin", distance=SEARCH_RADIUS_M
        )

        ranked = sorted(
            (
                (int(i), float(point.distance(segments[int(i)])))
                for i in indices
            ),
            key=lambda item: item[1],
        )
        candidates.append([row for row, _ in ranked[:MAX_CANDIDATES]])

    return candidates


def solve_alignment_chunk(states):
    """Find the minimum-cost non-decreasing route-position path."""
    costs = []
    backpointers = []

    for index, observation_states in enumerate(states):
        current_cost = {}
        current_back = {}

        for position, distance in observation_states:
            emission = (distance / GPS_SIGMA_M) ** 2 / 2.0

            if index == 0:
                current_cost[position] = emission
                current_back[position] = None
                continue

            valid_previous = [
                (previous_position, previous_cost)
                for previous_position, previous_cost in costs[-1].items()
                if previous_position <= position
            ]

            if not valid_previous:
                continue

            previous_position, previous_cost = min(
                valid_previous,
                key=lambda item: (
                    item[1]
                    + SKIP_PENALTY
                    * max(0, position - item[0] - 1)
                ),
            )

            skipped = max(0, position - previous_position - 1)
            current_cost[position] = (
                previous_cost + SKIP_PENALTY * skipped + emission
            )
            current_back[position] = previous_position

        costs.append(current_cost)
        backpointers.append(current_back)

    if not costs or not costs[-1]:
        return None

    position = min(costs[-1], key=costs[-1].get)
    result = [-1] * len(states)

    for index in range(len(states) - 1, -1, -1):
        result[index] = position
        position = backpointers[index][position]

    return result


def align_route_labels(features, track, route):
    occurrences = defaultdict(list)

    for position, arc_row in enumerate(route.astype(int)):
        occurrences[int(arc_row)].append(position)

    states_by_observation = [[] for _ in range(len(track))]

    for gps_index, group in features.groupby("gps_index", sort=True):
        states = {}

        for row in group.itertuples():
            arc_row = int(row.candidate_arc_row)
            distance = float(row.distance_m)

            for position in occurrences.get(arc_row, []):
                states[position] = distance

        states_by_observation[int(gps_index)] = sorted(states.items())

    aligned = np.full(len(track), -1, dtype=int)

    # Split into runs of consecutive observations with route candidates.
    i = 0
    while i < len(track):
        if not states_by_observation[i]:
            i += 1
            continue

        run_start = i
        run_states = [states_by_observation[i]]
        i += 1

        while i < len(track):
            dt = track[i, 2] - track[i - 1, 2]

            if (
                dt <= 0
                or dt > MAX_ALIGNMENT_GAP_S
                or not states_by_observation[i]
            ):
                break

            run_states.append(states_by_observation[i])
            i += 1

        # Split whenever globally reachable route positions
        # cannot continue into the next observation.
        chunk_start = 0
        reachable = {position for position, _ in run_states[0]}

        for j in range(1, len(run_states)):
            current_positions = {
                position for position, _ in run_states[j]
            }

            next_reachable = {
                current
                for current in current_positions
                if any(previous <= current for previous in reachable)
            }

            if not next_reachable:
                chunk = run_states[chunk_start:j]
                result = solve_alignment_chunk(chunk)

                if result is not None:
                    aligned[
                        run_start + chunk_start:
                        run_start + j
                    ] = result

                chunk_start = j
                reachable = current_positions
            else:
                reachable = next_reachable

        chunk = run_states[chunk_start:]
        result = solve_alignment_chunk(chunk)

        if result is not None:
            aligned[
                run_start + chunk_start:
                run_start + len(run_states)
            ] = result

    return aligned


def process_trip(trip_id):
    nodes = np.atleast_2d(load_trip_file(trip_id, "nodes"))
    arcs = np.atleast_2d(load_trip_file(trip_id, "arcs")).astype(int)
    route = np.atleast_1d(load_trip_file(trip_id, "route")).astype(int)
    track = np.atleast_2d(load_trip_file(trip_id, "track"))

    candidates = build_candidates(track, nodes, arcs)

    features = generate_candidate_features(
        track, nodes, arcs, candidates, trip_id=trip_id
    )

    aligned_positions = align_route_labels(features, track, route)
    aligned_mask = aligned_positions >= 0

    aligned_arc_rows = np.full(len(track), -1, dtype=int)
    aligned_arc_rows[aligned_mask] = route[aligned_positions[aligned_mask]]

    gps_indices = features["gps_index"].astype(int).to_numpy()

    features["aligned_route_position"] = aligned_positions[gps_indices]
    features["is_match"] = (
        features["candidate_arc_row"].to_numpy()
        == aligned_arc_rows[gps_indices]
    ).astype(int)

    features = features[
        features["aligned_route_position"] >= 0
    ].copy()

    # Preserve trip IDs as zero-padded strings in the output.
    features["trip_id"] = f"{int(trip_id):08d}"

    candidate_observations = sum(bool(x) for x in candidates)
    route_arcs = set(map(int, route))

    route_candidate_observations = sum(
        any(arc in route_arcs for arc in observation)
        for observation in candidates
    )

    return (
        features,
        int(aligned_mask.sum()),
        len(track),
        candidate_observations,
        route_candidate_observations,
    )


def validate_dataset(path):
    data = pd.read_csv(path, dtype={"trip_id": str})

    required = {"trip_id", "gps_index", "candidate_arc_row", "is_match"}
    missing = required - set(data.columns)

    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    counts = data.groupby(["trip_id", "gps_index"])["is_match"].sum()

    invalid = counts[counts != 1]

    if len(invalid):
        raise RuntimeError(
            f"{len(invalid)} observations do not have exactly one positive label."
        )

    print("\n========== LABEL VALIDATION ==========")
    print(f"Dataset rows: {len(data):,}")
    print(f"Represented trips: {data['trip_id'].nunique()}")
    print(f"Labelled observations: {len(counts):,}")
    print(f"Positive labels: {int(data['is_match'].sum()):,}")
    print("Every retained observation has exactly one positive candidate.")


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    trip_ids = get_training_trip_ids()
    print(f"Training trips found: {len(trip_ids)}")

    if len(trip_ids) != 80:
        raise RuntimeError(f"Expected 80 trips, found {len(trip_ids)}")

    totals = {
        "gps": 0,
        "candidate": 0,
        "route_candidate": 0,
        "aligned": 0,
        "rows": 0,
    }

    first_write = True
    trip_report = []

    for index, trip_id in enumerate(trip_ids, start=1):
        (
            features,
            aligned,
            observations,
            candidate_observations,
            route_candidate_observations,
        ) = process_trip(trip_id)

        features.to_csv(
            OUTPUT,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False

        totals["gps"] += observations
        totals["candidate"] += candidate_observations
        totals["route_candidate"] += route_candidate_observations
        totals["aligned"] += aligned
        totals["rows"] += len(features)

        coverage = aligned / observations if observations else 0.0

        trip_report.append({
            "trip_id": f"{int(trip_id):08d}",
            "gps_observations": observations,
            "candidate_observations": candidate_observations,
            "route_candidate_observations": route_candidate_observations,
            "aligned_observations": aligned,
            "alignment_coverage": coverage,
            "feature_rows": len(features),
        })

        print(
            f"[{index:02d}/80] {trip_id}: "
            f"GPS={observations}, "
            f"candidates={candidate_observations}, "
            f"route_candidates={route_candidate_observations}, "
            f"aligned={aligned}, "
            f"coverage={coverage:.2%}, "
            f"rows={len(features)}"
        )

    report_path = Path("artifacts/trip_alignment_report.csv")
    pd.DataFrame(trip_report).to_csv(report_path, index=False)

    print("\n========== DATASET BUILD SUMMARY ==========")
    print(f"Trips processed: {len(trip_ids)}")
    print(f"GPS observations: {totals['gps']:,}")
    print(f"Observations with candidates: {totals['candidate']:,}")
    print(f"Observations with route candidates: {totals['route_candidate']:,}")
    print(f"Aligned observations: {totals['aligned']:,}")
    print(f"Alignment coverage: {totals['aligned'] / totals['gps']:.2%}")
    print(f"Candidate feature rows: {totals['rows']:,}")
    print(f"Saved dataset: {OUTPUT}")
    print(f"Saved trip report: {report_path}")

    if totals["rows"] == 0:
        raise RuntimeError("Dataset is empty.")

    validate_dataset(OUTPUT)


if __name__ == "__main__":
    main()
