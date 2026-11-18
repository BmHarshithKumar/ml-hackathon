"""
test_member1.py - Automated validation suite for Member 1 deliverables.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_trip, build_road_graph
from src.candidate_gen import SpatialCandidateGenerator
from src.visualizer import create_folium_map, save_folium_map


def test_data_loader():
    trip = load_trip("00000086")
    assert trip["trip_id"] == "00000086"
    assert len(trip["track"]) == 4143
    assert len(trip["nodes"]) == 114868
    assert len(trip["arcs"]) == 237656
    assert len(trip["route"]) == 671

    G, aug_arcs = build_road_graph(trip["nodes"], trip["arcs"])
    assert G.number_of_nodes() == 114868
    assert G.number_of_edges() == 237593
    assert len(aug_arcs) == 237656


def test_candidate_gen_and_contract1():
    trip = load_trip("00000086")
    G, aug_arcs = build_road_graph(trip["nodes"], trip["arcs"])
    gen = SpatialCandidateGenerator(
        aug_arcs,
        origin_lon=float(trip["nodes"]["lon"].mean()),
        origin_lat=float(trip["nodes"]["lat"].mean()),
    )
    cands = gen.generate_candidates(
        trip["trip_id"], trip["track"], trip["route"], k=10, max_distance=100.0
    )

    contract_cols = [
        "trip_id",
        "step_idx",
        "timestamp",
        "gps_lon",
        "gps_lat",
        "arc_id",
        "from_node",
        "to_node",
        "arc_length",
        "label",
    ]
    for col in contract_cols:
        assert col in cands.columns, f"Missing Contract 1 column: {col}"

    assert cands.isnull().sum().sum() == 0, "Null values found in candidates_df"

    # Verify 100% recall of ground-truth arcs within Top-10
    step_recall = (cands.groupby("step_idx")["label"].max() == 1).mean()
    assert step_recall == 1.0, f"Expected 100% recall, got {step_recall*100:.2f}%"


def test_visualizer_and_artifacts():
    map_path = Path("artifacts/maps/trip_00000086_map.html")
    assert map_path.exists(), "Folium map missing"
    assert map_path.stat().st_size > 500_000, "Map file empty or corrupted"

    csv_path = Path("artifacts/candidates_00000086.csv")
    assert csv_path.exists(), "Contract 1 candidates CSV missing"
    assert csv_path.stat().st_size > 1_000_000, "Candidates CSV empty or corrupted"


def test_notebook_and_report():
    nb_path = Path("notebooks/01_eda.ipynb")
    assert nb_path.exists(), "01_eda.ipynb missing"
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 10

    report_path = Path("report/section_2_2_eda.md")
    assert report_path.exists(), "section_2_2_eda.md missing"
    assert report_path.stat().st_size > 2000

    fig_names = [
        "eda_sampling_rate.png",
        "eda_speed_trajectory.png",
        "eda_road_network_topology.png",
        "eda_candidate_spatial_stats.png",
        "eda_trajectory_spatial_plot.png",
    ]
    for fn in fig_names:
        p = Path("report/figures") / fn
        assert p.exists() and p.stat().st_size > 10_000, f"Figure missing or corrupted: {fn}"


if __name__ == "__main__":
    print("Running Member 1 tests...")
    test_data_loader()
    print("test_data_loader passed!")
    test_candidate_gen_and_contract1()
    print("test_candidate_gen_and_contract1 passed!")
    test_visualizer_and_artifacts()
    print("test_visualizer_and_artifacts passed!")
    test_notebook_and_report()
    print("test_notebook_and_report passed!")
    print("\nALL MEMBER 1 DELIVERABLES FULLY VERIFIED!")
