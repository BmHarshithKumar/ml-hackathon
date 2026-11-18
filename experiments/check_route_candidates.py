import subprocess
from io import StringIO

import numpy as np

TRIP_ID = "00000086"
ROOT = "data/Student folder/train"


def load_route():
    path = f"{ROOT}/{TRIP_ID}/{TRIP_ID}.route"

    raw = subprocess.check_output(
        ["git", "show", f"upstream/master:{path}"],
        text=True
    )

    return np.loadtxt(
        StringIO(raw),
        dtype=int
    ).ravel()


def main():
    route = load_route()

    saved = np.load(
        f"experiments/{TRIP_ID}_candidates.npz",
        allow_pickle=True
    )

    candidate_rows = saved["candidate_arc_rows"]

    reference_arc_rows = set(route.tolist())

    points_with_reference_candidate = 0
    total_reference_candidate_matches = 0

    for candidates in candidate_rows:

        # Candidate entries are Python lists.
        candidate_set = set(candidates)

        matches = candidate_set.intersection(
            reference_arc_rows
        )

        if matches:
            points_with_reference_candidate += 1

        total_reference_candidate_matches += len(matches)

    coverage = (
        points_with_reference_candidate / len(candidate_rows)
    ) * 100

    print("\n========== REFERENCE ROUTE CANDIDATE CHECK ==========")

    print("GPS observations:", len(candidate_rows))
    print("Reference route entries:", len(route))
    print("Unique reference route arcs:", len(reference_arc_rows))

    print(
        "GPS points with at least one reference-route candidate:",
        points_with_reference_candidate
    )

    print(
        "Reference-route candidate coverage:",
        f"{coverage:.2f}%"
    )

    print(
        "Total reference-route candidate matches:",
        total_reference_candidate_matches
    )


if __name__ == "__main__":
    main()