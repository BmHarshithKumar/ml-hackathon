"""
generate_eda_plots.py - Generate all EDA figures and metrics for Section 2.2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set Matplotlib temp cache directory inside workspace or temp
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent.parent / "artifacts" / ".matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.candidate_gen import SpatialCandidateGenerator
from src.data_loader import build_road_graph, haversine_distance, load_trip

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

OUTPUT_DIRS = [Path("report/figures")]
for d in OUTPUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def save_plot(fig: plt.Figure, filename: str):
    for d in OUTPUT_DIRS:
        out_path = d / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {filename}")


def main():
    trip_id = "00000086"
    print(f"Loading trip {trip_id}...")
    trip = load_trip(trip_id)
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]

    print("Building road graph...")
    G, aug_arcs = build_road_graph(nodes, arcs)

    print("Running candidate generator...")
    gen = SpatialCandidateGenerator(
        aug_arcs,
        origin_lon=float(nodes["lon"].mean()),
        origin_lat=float(nodes["lat"].mean()),
    )
    cands = gen.generate_candidates(trip_id, track, route, k=10, max_distance=100.0)

    # -------------------------------------------------------------
    # 1. Sampling Rate & Gaps Distribution
    # -------------------------------------------------------------
    dt = track["timestamp"].diff().dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Bar chart of top sampling deltas
    top_dts = dt.value_counts().head(5)
    labels = [f"{v:.1f}s" for v in top_dts.index]
    percentages = (top_dts.values / len(dt)) * 100
    bars = ax1.bar(labels, percentages, color="#2980b9", edgecolor="black", alpha=0.85)
    ax1.set_title(f"GPS Sampling Interval Distribution (Nominal 1 Hz: {percentages[0]:.2f}%)")
    ax1.set_xlabel("Time Delta (dt in seconds)")
    ax1.set_ylabel("Percentage of Steps (%)")
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(0, 105)

    # Cumulative time and gap occurrences
    gap_steps = track[track["dt"] > 1.5]
    ax2.plot(track["step_idx"], track["dt"], color="#e74c3c", lw=1.2, label="Sampling Interval (dt)")
    ax2.axhline(1.0, color="#27ae60", ls="--", lw=1.5, label="Nominal 1.0s")
    if not gap_steps.empty:
        ax2.scatter(gap_steps["step_idx"], gap_steps["dt"], color="#c0392b", s=30, zorder=5, label=f"Gaps > 1.5s (n={len(gap_steps)})")
    ax2.set_title("Temporal Continuity & Observation Gaps Across Trajectory")
    ax2.set_xlabel("Step Index (Sequential Observation)")
    ax2.set_ylabel("dt (seconds)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    save_plot(fig, "eda_sampling_rate.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # 2. Velocity Profile & Trajectory Length
    # -------------------------------------------------------------
    lat1, lon1 = track["lat"].values[:-1], track["lon"].values[:-1]
    lat2, lon2 = track["lat"].values[1:], track["lon"].values[1:]
    step_dists = haversine_distance(lon1, lat1, lon2, lat2)
    speeds_mps = step_dists / np.maximum(dt.values, 1e-3)
    speeds_kmh = speeds_mps * 3.6
    cum_dist_km = np.cumsum(np.insert(step_dists, 0, 0.0)) / 1000.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    ax1.plot(track["step_idx"].values[1:], speeds_kmh, color="#34495e", lw=1.0, label="Instantaneous Speed (km/h)")
    # 30-step moving average
    ma_speed = pd.Series(speeds_kmh).rolling(window=30, min_periods=1).mean()
    ax1.plot(track["step_idx"].values[1:], ma_speed, color="#e67e22", lw=2.0, label="Smoothed Trend (30s MA)")
    ax1.set_title(f"Vehicle Speed Dynamics (Trip {trip_id}: Total Length = {cum_dist_km[-1]:.2f} km)")
    ax1.set_ylabel("Speed (km/h)")
    ax1.legend(loc="upper right")

    ax2.plot(track["step_idx"], cum_dist_km, color="#16a085", lw=2.0, label="Cumulative Distance (km)")
    ax2.set_title("Cumulative Trajectory Progress Over Time")
    ax2.set_xlabel("Step Index (Time in Seconds)")
    ax2.set_ylabel("Distance (km)")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    save_plot(fig, "eda_speed_trajectory.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # 3. Road Network Topology & Degree Distribution
    # -------------------------------------------------------------
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    arc_lengths = aug_arcs["arc_length"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.arange(0, 10) - 0.5
    ax1.hist([in_degrees, out_degrees], bins=bins, label=["In-degree", "Out-degree"], color=["#3498db", "#9b59b6"], rwidth=0.8, density=True)
    ax1.set_title(f"Road Network Node Degree Distribution (N = {G.number_of_nodes():,})")
    ax1.set_xlabel("Degree (Number of Connected Arcs)")
    ax1.set_ylabel("Density")
    ax1.set_xticks(range(0, 8))
    ax1.legend()

    ax2.hist(arc_lengths, bins=50, range=(0, 500), color="#2c3e50", edgecolor="white", density=True, alpha=0.85)
    median_len = np.median(arc_lengths)
    mean_len = np.mean(arc_lengths)
    ax2.axvline(median_len, color="#e74c3c", ls="--", lw=2, label=f"Median: {median_len:.1f} m")
    ax2.axvline(mean_len, color="#f39c12", ls=":", lw=2, label=f"Mean: {mean_len:.1f} m")
    ax2.set_title(f"Road Segment Length Distribution (Total Arcs = {len(arc_lengths):,})")
    ax2.set_xlabel("Arc Length (meters)")
    ax2.set_ylabel("Density")
    ax2.legend()

    plt.tight_layout()
    save_plot(fig, "eda_road_network_topology.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # 4. Spatial Candidate & Error Distribution
    # -------------------------------------------------------------
    cands_per_step = cands.groupby("step_idx").size()
    gt_cands = cands[cands["label"] == 1]
    non_gt_cands = cands[cands["label"] == 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.hist(cands_per_step, bins=np.arange(0.5, 12.5, 1.0), color="#1abc9c", edgecolor="black", rwidth=0.8, alpha=0.85)
    ax1.set_title(f"Candidates per GPS Observation (Mean = {cands_per_step.mean():.2f})")
    ax1.set_xlabel("Number of Candidate Arcs (Top K ≤ 10)")
    ax1.set_ylabel("Observation Count")
    ax1.set_xticks(range(1, 11))

    sns.kdeplot(gt_cands["dist_to_arc"], ax=ax2, color="#27ae60", fill=True, label=f"Ground-Truth Arcs (Mean: {gt_cands['dist_to_arc'].mean():.1f}m)")
    sns.kdeplot(non_gt_cands["dist_to_arc"], ax=ax2, color="#e74c3c", fill=True, label=f"Distractor Arcs (Mean: {non_gt_cands['dist_to_arc'].mean():.1f}m)")
    ax2.set_title("Perpendicular GPS-to-Arc Distance Separation")
    ax2.set_xlabel("Perpendicular Distance (meters)")
    ax2.set_ylabel("Kernel Density")
    ax2.set_xlim(0, 100)
    ax2.legend()

    plt.tight_layout()
    save_plot(fig, "eda_candidate_spatial_stats.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # 5. Static 2D Trajectory & Route Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    # Route lines
    route_arc_ids = route["arc_id"].values
    route_arcs = aug_arcs.loc[route_arc_ids]
    for row in route_arcs.itertuples():
        ax.plot([row.u_lon, row.v_lon], [row.u_lat, row.v_lat], color="#27ae60", lw=2.5, zorder=2)
    # Custom legend handle for route
    ax.plot([], [], color="#27ae60", lw=2.5, label="Ground-Truth Route (.route)")

    # GPS observations
    ax.scatter(track["lon"].values[::5], track["lat"].values[::5], s=8, color="#2980b9", alpha=0.6, zorder=3, label="GPS Observations (1-in-5 subsample)")
    ax.scatter(track["lon"].iloc[0], track["lat"].iloc[0], s=80, marker="o", color="#2ecc71", edgecolor="black", zorder=5, label="Trip Origin")
    ax.scatter(track["lon"].iloc[-1], track["lat"].iloc[-1], s=80, marker="X", color="#e74c3c", edgecolor="black", zorder=5, label="Trip Destination")

    ax.set_title(f"PathSnap 2D Trajectory & Road Route: Trip {trip_id}")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="best")
    plt.tight_layout()
    save_plot(fig, "eda_trajectory_spatial_plot.png")
    plt.close(fig)

    print("All EDA figures successfully generated!")


if __name__ == "__main__":
    main()
