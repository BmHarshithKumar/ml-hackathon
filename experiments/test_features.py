
import sys
from pathlib import Path

import numpy as np

# Add the project root to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import generate_candidate_features
from experiments.dynamic_route_alignment import load_file


TRIP_ID = "00000086"


def main():
    nodes = load_file("nodes")
    arcs = load_file("arcs").astype(int)
    track = load_file("track")

    saved = np.load(
        f"experiments/{TRIP_ID}_candidates.npz",
        allow_pickle=True
    )

    candidates = saved["candidate_arc_rows"]

    features = generate_candidate_features(
        track=track,
        nodes=nodes,
        arcs=arcs,
        candidate_arc_rows=candidates,
        trip_id=TRIP_ID
    )

    print("\n========== FEATURE ENGINEERING TEST ==========")
    print("Trip:", TRIP_ID)
    print("GPS observations:", len(track))
    print("Candidate feature rows:", len(features))
    print("Feature columns:", len(features.columns))

    print("\nFeature names:")
    print(features.columns.tolist())

    print("\nMissing values:")
    print(features.isna().sum())

    print("\nFeature statistics:")
    print(
        features[
            [
                "distance_m",
                "gps_speed_mps",
                "heading_difference_deg",
                "segment_length_m",
                "projection_fraction"
            ]
        ].describe().round(3)
    )

    output_path = f"experiments/{TRIP_ID}_features.csv"
    features.to_csv(output_path, index=False)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
