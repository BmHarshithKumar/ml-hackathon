# PathSnap: Distributed Map-Matching & Route Reconstruction

PathSnap is an end-to-end machine learning and Hidden Markov Model (HMM) map-matching system that reconstructs physically valid vehicle routes from noisy GPS trajectories on a directed road network, deployed via a **two-laptop LAN architecture** (FastAPI inference server on Laptop 1 + Streamlit Evaluator Dashboard with interactive 4-layer Folium maps on Laptop 2).

---

## Repository Architecture

```text
ml-hackathon/
├── data/                         # Raw trip directories (.track, .nodes, .arcs, .route)
│   └── Student folder/
│       ├── train/                # 80 training trips
│       └── student_test/         # 10 held-out evaluation trips (00000000..00000091)
├── artifacts/                    # Processed candidates, best_model.pkl, interactive HTML maps
│   ├── candidates_00000086.csv
│   └── maps/
│       └── trip_00000086_map.html
├── src/                          # Core modular source code
│   ├── __init__.py
│   ├── data_loader.py            # [Member 1] Parsers, coordinate conversions & DiGraph builder
│   ├── candidate_gen.py          # [Member 1] STRtree/KDTree spatial candidate matcher
│   ├── generate_eda_plots.py     # [Member 1] EDA plot generator
│   ├── visualizer.py             # [Member 1 & 4] Unified multi-layer interactive Folium maps
│   ├── features.py               # [Member 2] Feature extraction pipeline
│   ├── train_models.py           # [Member 2] 5 distinct classification models
│   ├── bayesian_opt.py           # [Member 2] Optuna Bayesian hyperparameter search
│   └── hmm_viterbi.py            # [Member 3] HMM transition model & Viterbi sequential decoder
├── api/                          # [Member 4] FastAPI backend service (Laptop 1)
│   ├── __init__.py
│   ├── pipeline.py               # End-to-end orchestrator & Contract 1-3 adapter
│   └── main.py                   # FastAPI app & viva terminal logger
├── ui/                           # [Member 4] Evaluator Streamlit dashboard (Laptop 2)
│   └── app.py
├── notebooks/                    # Exploratory data analysis & experiment notebooks
│   └── 01_eda.ipynb              # [Member 1] Phase 1 exploratory data analysis
├── report/                       # Consolidated technical report documentation
│   ├── section_2_2_eda.md        # [Member 1] Dataset understanding and EDA report
│   ├── sections/                 # Report sections (Sec 1-7)
│   ├── figures/                  # High-resolution plots & architecture diagrams
│   ├── build_report.py           # [Member 4] Automated PDF compiler
│   └── Technical_Report.pdf      # Compiled technical report
├── tests/
│   ├── test_member1.py           # Member 1 automated validation suite
│   └── test_member4_integration.py  # Member 4 API/UI/Visualizer/PDF integration tests
├── requirements.txt
└── README.md
```

---

## Quick Setup (Both Laptops)

```bash
git clone https://github.com/AtharvaGore/ml-hackathon.git
cd ml-hackathon

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Member 1 Highlights — Data Engine, Spatial Indexing & EDA

- **Parsers & Road Graph (`src/data_loader.py`)**: Headerless parsers for `.track`, `.nodes`, `.arcs`, `.route`, WGS84 -> local metric equirectangular projection, Haversine distance & bearing utilities, and `networkx.DiGraph` builder.
- **Spatial Indexing (`src/candidate_gen.py`)**: Two-tier `shapely.STRtree` + `scipy.spatial.KDTree` spatial matcher achieving **100.00% candidate recall** at $K=10, R=100\text{ m}$ and exporting Contract 1 `candidates_df` (`artifacts/candidates_00000086.csv`).
- **EDA & Report (`notebooks/01_eda.ipynb`, `report/section_2_2_eda.md`)**: Sampling rate, speed dynamics, topology, distance separation plots in `report/figures/`, and standalone HTML map in `artifacts/maps/trip_00000086_map.html`.

---

## Two-Laptop LAN Deployment Guide — Member 4 (Live Evaluator Viva)

Ensure **Laptop 1** and **Laptop 2** are connected to the same Wi-Fi network or mobile hotspot.

### Laptop 1 — FastAPI Inference Server

1. Start the inference server bound to all network interfaces (`0.0.0.0`):
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
2. Keep Laptop 1's terminal visible during the viva. On startup it prints the detected LAN URL (e.g. `http://192.168.1.15:8000`) and logs every incoming prediction with stage latencies, Classifier-Only vs. HMM Viterbi metrics, and discontinuity counts.

### Laptop 2 — Streamlit Evaluator Dashboard

1. Launch the Evaluator UI:
   ```bash
   streamlit run ui/app.py
   ```
2. Enter **Laptop 1's IP Address** and port `8000` in the sidebar, click **Test Connection / Refresh**, select any of the **10 evaluation trips** (`00000000`–`00000091`), and click **Run Map Matching**.
3. Inspect the KPI cards, interactive **4-layer Folium map** (Layer 1 Raw GPS blue markers, Layer 2 Road Network gray lines, Layer 3 Ground-Truth green route, Layer 4 Predicted HMM purple overlay), and the **10-Trip Batch Evaluation** tab.

---

## API Contract Specification (Laptop 1)

| Method | Endpoint | Payload | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | — | Server status, pipeline mode (`full_real` / `real_data_pipeline` / `mock_demo`), graph stats, loaded modules |
| `GET` | `/trips` | — | Metadata for the 10 held-out evaluation trips in `data/Student folder/student_test/` (zero GT leakage) |
| `POST` | `/predict_trip` | `{"trip_id": "00000052"}` | Runs full pipeline (`Track -> Candidates -> Classifier -> HMM Viterbi -> Metrics`) |
| `POST` | `/predict_batch` | `{"trip_ids": null}` | Evaluates all 10 trips and returns per-trip + macro summary metrics |

Quick CLI verification:
```bash
curl -X POST http://<LAPTOP_1_IP>:8000/predict_trip \
  -H "Content-Type: application/json" \
  -d '{"trip_id": "00000052"}'
```

---

## Compiling the Final Technical Report PDF

`report/build_report.py` automatically merges Member 1's `report/section_2_2_eda.md` (and its 5 EDA figures), Member 2–3's sections, and Member 4's Sections 6 & 7 into `report/Technical_Report.pdf`:

```bash
python report/build_report.py
```

---

## Running Tests

```bash
pytest -v tests/
```
