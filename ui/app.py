"""Streamlit Evaluator Dashboard — Laptop 2 (Member 4).

Connects over LAN to Laptop 1's FastAPI inference server (`api/main.py`), lets the
evaluator select any of the 10 evaluation trips, displays returned metrics
(Top-1 Accuracy, Route Continuity, Route F1, Processing Time), and renders the
interactive 4-layer Folium map via `src/visualizer.py`.

Launch on Laptop 2:
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# Ensure repo root is on sys.path when invoked via `streamlit run ui/app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualizer import SUPPORTED_TILES, create_trip_map_from_payload, map_to_html_str


DEFAULT_EVAL_TRIPS = [
    "00000000",
    "00000007",
    "00000009",
    "00000040",
    "00000052",
    "00000056",
    "00000065",
    "00000076",
    "00000077",
    "00000091",
]


def _normalize_base_url(host_input: str, port: int) -> str:
    """Build a normalized HTTP base URL from an IP/hostname or full URL string."""
    raw = host_input.strip().rstrip("/")
    if not raw:
        raw = "127.0.0.1"
    if raw.startswith("http://") or raw.startswith("https://"):
        # Append port only if no explicit port is present after scheme
        after_scheme = raw.split("://", 1)[1]
        if ":" not in after_scheme and "/" not in after_scheme:
            return f"{raw}:{port}"
        return raw
    if ":" in raw:
        return f"http://{raw}"
    return f"http://{raw}:{port}"


def _fetch_health(base_url: str, timeout: float = 4.0) -> tuple[dict[str, Any] | None, str | None]:
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def _fetch_trips(base_url: str, timeout: float = 4.0) -> tuple[list[dict[str, Any]], str | None]:
    try:
        resp = requests.get(f"{base_url}/trips", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return list(data.get("trips", [])), None
    except requests.RequestException as exc:
        return [], str(exc)


def _predict_single_trip(
    base_url: str, trip_id: str, timeout: float = 30.0
) -> tuple[dict[str, Any] | None, float, str | None]:
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/predict_trip",
            json={"trip_id": trip_id},
            timeout=timeout,
        )
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code != 200:
            return None, rtt_ms, f"HTTP {resp.status_code}: {resp.text}"
        return resp.json(), round(rtt_ms, 2), None
    except requests.RequestException as exc:
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        return None, round(rtt_ms, 2), str(exc)


def _predict_batch(
    base_url: str, trip_ids: list[str] | None = None, timeout: float = 60.0
) -> tuple[dict[str, Any] | None, float, str | None]:
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/predict_batch",
            json={"trip_ids": trip_ids},
            timeout=timeout,
        )
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code != 200:
            return None, rtt_ms, f"HTTP {resp.status_code}: {resp.text}"
        return resp.json(), round(rtt_ms, 2), None
    except requests.RequestException as exc:
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        return None, round(rtt_ms, 2), str(exc)


def _render_batch_chart(df: pd.DataFrame):
    """Build a crisp Classifier-Only vs HMM Viterbi comparison chart across evaluation trips."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=130)
    x = range(len(df))
    width = 0.36
    labels = [t.replace("eval_", "") for t in df["trip_id"]]

    # Accuracy subplot
    axes[0].bar(
        [i - width / 2 for i in x],
        df["classifier_only_top1_accuracy"] * 100,
        width=width,
        label="Classifier Only",
        color="#94A3B8",
    )
    axes[0].bar(
        [i + width / 2 for i in x],
        df["top1_accuracy"] * 100,
        width=width,
        label="Classifier + HMM Viterbi",
        color="#2563EB",
    )
    axes[0].set_title("Top-1 Accuracy (%) by Trip", fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 108)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8)

    # Continuity subplot
    axes[1].bar(
        [i - width / 2 for i in x],
        df["classifier_only_continuity"] * 100,
        width=width,
        label="Classifier Only",
        color="#F59E0B",
    )
    axes[1].bar(
        [i + width / 2 for i in x],
        df["route_continuity"] * 100,
        width=width,
        label="Classifier + HMM Viterbi",
        color="#16A34A",
    )
    axes[1].set_title("Route Continuity (%) by Trip", fontweight="bold")
    axes[1].set_ylabel("Continuity (%)")
    axes[1].set_ylim(0, 108)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="GPS Map-Matching Evaluator (Laptop 2)",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
        div[data-testid="stMetric"] {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 12px 16px;
        }
        .status-pill-ok {
            display:inline-block; padding:4px 10px; border-radius:999px;
            background:#DCFCE7; color:#166534; font-weight:600; font-size:12px;
        }
        .status-pill-err {
            display:inline-block; padding:4px 10px; border-radius:999px;
            background:#FEE2E2; color:#991B1B; font-weight:600; font-size:12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Session state initialization
    st.session_state.setdefault("last_prediction", None)
    st.session_state.setdefault("last_rtt_ms", None)
    st.session_state.setdefault("last_batch", None)
    st.session_state.setdefault("last_batch_rtt_ms", None)

    # ------------------------------------------------------------------
    # Sidebar: Laptop 1 Connection, Trip Selector & Map Layer Controls
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Laptop 1 Server Config")
        host_ip = st.text_input(
            "Laptop 1 IP Address / URL",
            value="127.0.0.1",
            help="Enter Laptop 1's LAN IP (e.g. 192.168.1.15) or 127.0.0.1 for local testing.",
        )
        port = st.number_input("API Port", min_value=1, max_value=65535, value=8000, step=1)
        base_url = _normalize_base_url(host_ip, int(port))
        st.caption(f"Endpoint: `{base_url}`")

        ping_clicked = st.button("Test Connection / Refresh", use_container_width=True)
        health_data, health_err = _fetch_health(base_url)

        if health_data:
            mode = str(health_data.get("pipeline_mode", "unknown")).upper()
            st.markdown(
                f"<span class='status-pill-ok'>● ONLINE &middot; {mode}</span>",
                unsafe_allow_html=True,
            )
            g_stats = health_data.get("graph_stats", {})
            st.caption(
                f"Graph: {g_stats.get('num_nodes', 0)} nodes / "
                f"{g_stats.get('num_arcs', 0)} arcs | "
                f"Trips: {health_data.get('num_eval_trips', 0)}"
            )
        else:
            st.markdown(
                "<span class='status-pill-err'>● DISCONNECTED</span>",
                unsafe_allow_html=True,
            )
            if ping_clicked or health_err:
                st.error(f"Cannot reach Laptop 1: {health_err}")

        st.divider()
        st.header("Evaluation Trip")

        trips_meta, _ = _fetch_trips(base_url) if health_data else ([], None)
        if trips_meta:
            trip_options = [t["trip_id"] for t in trips_meta]
            trip_desc_map = {
                t["trip_id"]: f"{t['trip_id']} ({t['num_points']} pts, {t['duration_s']}s)"
                for t in trips_meta
            }
        else:
            trip_options = DEFAULT_EVAL_TRIPS
            trip_desc_map = {t: t for t in DEFAULT_EVAL_TRIPS}

        selected_trip = st.selectbox(
            "Select Evaluation Trip (10 Trips)",
            options=trip_options,
            format_func=lambda tid: trip_desc_map.get(tid, tid),
        )

        run_clicked = st.button(
            "Run Map Matching",
            type="primary",
            use_container_width=True,
            disabled=(health_data is None),
        )

        st.divider()
        st.header("Map Layers & Style")
        tile_style = st.selectbox("Basemap Theme", options=list(SUPPORTED_TILES.keys()), index=0)
        show_gps = st.checkbox("Layer 1: Raw GPS Observations (Blue)", value=True)
        show_roads = st.checkbox("Layer 2: Road Network Arcs (Gray)", value=True)
        show_gt = st.checkbox("Layer 3: Ground-Truth Route (Green)", value=True)
        show_pred = st.checkbox("Layer 4: Predicted HMM Route (Purple)", value=True)
        show_mismatch = st.checkbox("Diagnostics: Mismatches & Endpoints", value=True)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.title("GPS Map-Matching & HMM Route Reconstruction Evaluator")
    st.caption(
        "Two-Laptop Live Evaluation Dashboard (Laptop 2 Client -> Laptop 1 FastAPI Inference Engine). "
        "Ground-truth routes are isolated server-side."
    )

    if run_clicked:
        with st.spinner(f"Requesting POST {base_url}/predict_trip for {selected_trip}..."):
            pred_data, rtt_ms, pred_err = _predict_single_trip(base_url, selected_trip)
        if pred_err:
            st.error(f"Prediction request failed: {pred_err}")
        else:
            st.session_state["last_prediction"] = pred_data
            st.session_state["last_rtt_ms"] = rtt_ms

    tab_single, tab_batch, tab_diag = st.tabs(
        [
            "Trip Inspector & Interactive Map",
            "10-Trip Batch Evaluation",
            "System & LAN Diagnostics",
        ]
    )

    # ------------------------------------------------------------------
    # Tab 1: Single Trip Evaluation & 4-Layer Folium Map
    # ------------------------------------------------------------------
    with tab_single:
        pred = st.session_state.get("last_prediction")
        rtt_ms = st.session_state.get("last_rtt_ms")

        if pred is None:
            st.info(
                "Select one of the 10 evaluation trips in the sidebar and click "
                "**Run Map Matching** to query Laptop 1 and visualize the reconstructed route."
            )
        else:
            m = pred["metrics"]
            timings = pred["stage_timings_ms"]

            st.subheader(f"Results for `{pred['trip_id']}`")
            if pred.get("description"):
                st.caption(
                    f"{pred['description']} · {pred['num_gps_points']} GPS observations · "
                    f"{pred['num_candidate_pairs']} candidate pairs · Mode: `{pred['pipeline_mode']}`"
                )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Top-1 Accuracy (HMM)",
                f"{m['top1_accuracy'] * 100:.2f}%",
                delta=f"{m['hmm_accuracy_gain'] * 100:+.2f}% vs Classifier-Only",
            )
            c2.metric(
                "Route Continuity",
                f"{m['route_continuity'] * 100:.2f}%",
                delta=f"{m['hmm_continuity_gain'] * 100:+.2f}% vs Classifier-Only",
            )
            c3.metric(
                "Server Processing Time",
                f"{m['processing_time_ms']:.2f} ms",
                delta=f"LAN RTT: {rtt_ms:.1f} ms" if rtt_ms is not None else None,
                delta_color="off",
            )
            c4.metric(
                "Route F1 / Discontinuities",
                f"{m['route_f1'] * 100:.2f}%",
                delta=f"{m['discontinuities']} broken transitions",
                delta_color="inverse" if m["discontinuities"] > 0 else "off",
            )

            # Build and render the 4-layer Folium map
            layer_flags = {
                "gps": show_gps,
                "road_network": show_roads,
                "ground_truth": show_gt,
                "predicted": show_pred,
                "mismatches": show_mismatch,
            }
            fmap = create_trip_map_from_payload(
                pred,
                tiles=tile_style,
                show_layers=layer_flags,
            )
            html_str = map_to_html_str(fmap)

            st.markdown("#### Interactive 4-Layer Folium Route Map")
            components.html(html_str, height=560, scrolling=False)

            dl_col1, dl_col2, _ = st.columns([1.2, 1.2, 3])
            dl_col1.download_button(
                label="Download Interactive HTML Map",
                data=html_str.encode("utf-8"),
                file_name=f"{pred['trip_id']}_map.html",
                mime="text/html",
                use_container_width=True,
            )
            dl_col2.download_button(
                label="Download Raw API Response (JSON)",
                data=json.dumps(pred, indent=2).encode("utf-8"),
                file_name=f"{pred['trip_id']}_prediction.json",
                mime="application/json",
                use_container_width=True,
            )

            # Classifier-Only vs HMM Comparison + Stage Timings
            col_cmp, col_lat = st.columns(2)
            with col_cmp:
                st.markdown("#### Classifier-Only vs. Classifier + HMM Viterbi")
                cmp_df = pd.DataFrame(
                    [
                        {
                            "Pipeline": "Classifier Only (Greedy)",
                            "Top-1 Accuracy (%)": round(m["classifier_only_top1_accuracy"] * 100, 2),
                            "Route Continuity (%)": round(m["classifier_only_continuity"] * 100, 2),
                            "Discontinuities": m["classifier_only_discontinuities"],
                        },
                        {
                            "Pipeline": "Classifier + HMM Viterbi",
                            "Top-1 Accuracy (%)": round(m["top1_accuracy"] * 100, 2),
                            "Route Continuity (%)": round(m["route_continuity"] * 100, 2),
                            "Discontinuities": m["discontinuities"],
                        },
                    ]
                )
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            with col_lat:
                st.markdown("#### Server Pipeline Stage Breakdown")
                stage_df = pd.DataFrame(
                    [
                        {"Stage": "Candidate Generation (Spatial)", "Latency (ms)": timings["candidate_gen_ms"]},
                        {"Stage": "Classifier Emission Scoring", "Latency (ms)": timings["model_inference_ms"]},
                        {"Stage": "HMM Viterbi Decoding", "Latency (ms)": timings["hmm_viterbi_ms"]},
                        {"Stage": "Metrics & Geometry Packing", "Latency (ms)": timings["metrics_and_pack_ms"]},
                        {"Stage": "Total Server Time", "Latency (ms)": timings["total_ms"]},
                    ]
                )
                st.dataframe(stage_df, use_container_width=True, hide_index=True)

            with st.expander("Step-by-Step GPS Observation Audit Table", expanded=False):
                step_df = pd.DataFrame(pred.get("step_details", []))
                if not step_df.empty:
                    st.dataframe(step_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Tab 2: 10-Trip Batch Evaluator
    # ------------------------------------------------------------------
    with tab_batch:
        st.subheader("Full 10-Trip Evaluation Suite")
        st.write(
            "Evaluate all 10 held-out trips stored on Laptop 1 in a single batch call "
            "to inspect aggregate accuracy, continuity, and latency."
        )
        if st.button(
            "Evaluate All 10 Trips",
            type="primary",
            disabled=(health_data is None),
        ):
            with st.spinner(f"Running POST {base_url}/predict_batch across all evaluation trips..."):
                batch_data, batch_rtt, batch_err = _predict_batch(base_url)
            if batch_err:
                st.error(f"Batch evaluation failed: {batch_err}")
            else:
                st.session_state["last_batch"] = batch_data
                st.session_state["last_batch_rtt_ms"] = batch_rtt

        batch = st.session_state.get("last_batch")
        if batch:
            s = batch["summary"]
            b1, b2, b3, b4 = st.columns(4)
            b1.metric(
                "Mean Top-1 Accuracy",
                f"{s['mean_top1_accuracy'] * 100:.2f}%",
                delta=f"{(s['mean_top1_accuracy'] - s['mean_classifier_only_top1']) * 100:+.2f}% vs Greedy",
            )
            b2.metric(
                "Mean Route Continuity",
                f"{s['mean_route_continuity'] * 100:.2f}%",
                delta=f"{(s['mean_route_continuity'] - s['mean_classifier_only_continuity']) * 100:+.2f}% vs Greedy",
            )
            b3.metric("Mean Route F1", f"{s['mean_route_f1'] * 100:.2f}%")
            b4.metric(
                "Mean Server Latency",
                f"{s['mean_processing_time_ms']:.2f} ms",
                delta=f"Total Discontinuities: {s['total_discontinuities']}",
                delta_color="off",
            )

            df_batch = pd.DataFrame(batch["results"])
            fig = _render_batch_chart(df_batch)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            display_df = df_batch.copy()
            for col in (
                "top1_accuracy",
                "route_continuity",
                "route_f1",
                "sequence_jaccard",
                "classifier_only_top1_accuracy",
                "classifier_only_continuity",
            ):
                display_df[col] = (display_df[col] * 100).round(2)

            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.download_button(
                label="Download 10-Trip Benchmark CSV",
                data=df_batch.to_csv(index=False).encode("utf-8"),
                file_name="eval_10_trips_benchmark.csv",
                mime="text/csv",
            )

    # ------------------------------------------------------------------
    # Tab 3: System & LAN Diagnostics
    # ------------------------------------------------------------------
    with tab_diag:
        st.subheader("Two-Laptop Deployment & Contract Diagnostics")
        if health_data:
            comps = health_data.get("components", {})
            comp_rows = [
                {"Contract / Component": "Member 1: src/data_loader.py", "Active": comps.get("p1_data_loader", False)},
                {"Contract / Component": "Member 1: src/candidate_gen.py", "Active": comps.get("p1_candidate_gen", False)},
                {"Contract / Component": "Member 2: src/features.py", "Active": comps.get("p2_features", False)},
                {"Contract / Component": "Member 2: artifacts/best_model.pkl", "Active": comps.get("p2_model_artifact", False)},
                {"Contract / Component": "Member 3: src/hmm_viterbi.py (reconstruct_route)", "Active": comps.get("p3_hmm_viterbi", False)},
                {"Contract / Component": "Dataset files in data/", "Active": comps.get("real_dataset_files", False)},
            ]
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            st.markdown("#### Quick Viva Verification `curl` Command")
            st.code(
                f'curl -X POST {base_url}/predict_trip \\\n'
                f'  -H "Content-Type: application/json" \\\n'
                f'  -d \'{{"trip_id": "{selected_trip}"}}\'',
                language="bash",
            )

            with st.expander("Raw GET /health Payload", expanded=False):
                st.json(health_data)
        else:
            st.warning(
                "Laptop 1 is currently unreachable. Start the server on Laptop 1 with:\n\n"
                "`uvicorn api.main:app --host 0.0.0.0 --port 8000`"
            )


if __name__ == "__main__":
    main()
