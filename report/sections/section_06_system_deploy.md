# 6. System Architecture, Two-Laptop Deployment & Interactive Visualization

## 6.1 Client-Server LAN Architecture

To support live evaluator inspection without exposing ground-truth labels on the client machine, the system is architected as a decoupled two-laptop HTTP client-server deployment over a local area network (LAN):

- **Laptop 1 — Inference & Evaluation Engine (`api/main.py`, `api/pipeline.py`)**: Runs a Uvicorn-hosted FastAPI service bound to `0.0.0.0:8000`. At startup, the lifespan manager initializes the directed road network (`networkx.DiGraph`), loads the serialized classifier (`artifacts/best_model.pkl`), binds the HMM Viterbi decoder (`reconstruct_route`), and caches the 10 held-out evaluation trips alongside their internal ground-truth `.route` sequences strictly server-side.
- **Laptop 2 — Evaluator Dashboard (`ui/app.py`, `src/visualizer.py`)**: Runs a stateless Streamlit web application that communicates with Laptop 1 over HTTP using a configurable target IP address and port. Laptop 2 stores no model weights, no road graph files, and no ground-truth routes; it displays server-computed metrics and renders interactive Folium maps directly from the returned JSON geometries.

![Two-Laptop Deployment Architecture](figures/system_architecture.png)

## 6.2 API Contracts & Pipeline Orchestration

The FastAPI server exposes four structured endpoints with Cross-Origin Resource Sharing (CORS) enabled for seamless LAN interaction:

| Endpoint | Method | Request Schema | Purpose |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | — | Reports readiness, active mode (`full_real`, `real_data_pipeline`, or `mock_demo`), detected LAN IP, graph statistics, and loaded teammate modules. |
| `/trips` | `GET` | — | Lists metadata (trip ID, GPS observation count, duration) for the 10 evaluation trips without exposing ground truth. |
| `/predict_trip` | `POST` | `{"trip_id": str}` | Executes the full inference pipeline (Raw Track -> Candidate Generation -> Classifier Emissions -> HMM Viterbi -> Metrics & Geometry Packing). |
| `/predict_batch` | `POST` | `{"trip_ids": list or null}` | Evaluates all 10 trips sequentially and aggregates mean Top-1 Accuracy, Route Continuity, Route F1, and server latency. |

To eliminate team bottlenecks and Git merge conflicts, `api/pipeline.py` implements dynamic module discovery across Contracts 1–3. Whenever Member 1 (`src/data_loader.py`, `src/candidate_gen.py`), Member 2 (`src/features.py`, `artifacts/best_model.pkl`), or Member 3 (`src/hmm_viterbi.py` exposing `reconstruct_route(trip_id, gps_points, graph)`) are merged into the repository, the orchestrator automatically delegates to them. In their absence, it falls back cleanly to a deterministic 14x16 directed urban road network and 10 synthetic evaluation trips so LAN deployment and UI verification remain operational throughout the sprint.

## 6.3 Viva Observability & Latency Optimization

Two engineering decisions directly support responsive live demonstrations:

1. **Structured Terminal Logging**: Every `POST /predict_trip` invocation emits a color-coded, timestamped audit block in Laptop 1's terminal detailing the client IP, trip ID, observation count, candidate count, Classifier-Only vs. HMM Viterbi accuracy and continuity, per-stage latencies (candidate generation, model inference, Viterbi decoding, geometry packing), and total execution time.
2. **Bounding-Box Network Clipping**: Serializing an entire city-scale road network over Wi-Fi for every trip request would inflate payload sizes and UI render latency. Instead, Laptop 1 clips Layer 2 road arcs to the trip's padded WGS84 bounding box (approx. +/- 250m around the trajectory), keeping JSON payloads compact while preserving full local street context.

## 6.4 Interactive Four-Layer Folium Visualization

`src/visualizer.py` constructs interactive HTML maps using Folium (`OpenStreetMap`, `Esri Light Minimal`, or `Esri Street Detail` tiles — requiring no third-party API keys) with an interactive `LayerControl` panel and floating KPI legend covering all four required layers plus an optional error-diagnostics layer:

- **Layer 1 — Raw GPS Observations (Blue Circle Markers)**: Displays every noisy GPS coordinate (`#2563EB`) with click popups showing step index, timestamp, WGS84 coordinates, ground-truth arc ID, predicted arc ID, and match status.
- **Layer 2 — Road Network Arcs (Light Gray Lines)**: Renders local directed road segments (`#94A3B8`) with hover tooltips reporting arc ID, endpoint nodes, and metric length.
- **Layer 3 — Ground-Truth Reference Route (Green Line)**: Renders the true server-side route (`#16A34A`, 6.5px stroke) beneath the predicted track.
- **Layer 4 — Model Predicted Reconstructed Route (Purple Dashed Overlay)**: Overlays the HMM Viterbi output (`#9333EA`, 4.0px dashed stroke) so exact matches remain clearly verifiable alongside the green reference line while divergences stand out immediately.
- **Diagnostics Overlay**: Highlights trip start/end markers and any mismatched GPS observations in red (`#DC2626`) for rapid qualitative error analysis.

Evaluators can toggle any layer directly inside the Streamlit dashboard, download standalone `.html` maps for offline archival, or export the full 10-trip batch benchmark as CSV.
