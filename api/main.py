"""FastAPI Map-Matching Inference Backend — Laptop 1 (Member 4).

Exposes:
  - GET  /health         : Server health, detected LAN IP, loaded components & mode.
  - GET  /trips          : Lists the 10 evaluation trips (metadata only; no GT leakage).
  - POST /predict_trip   : Exact contract payload `{"trip_id": str}`. Runs the end-to-end
                           pipeline (Track -> Candidates -> Model -> HMM Viterbi -> Metrics)
                           with server-side ground truth and clean viva terminal logs.
  - POST /predict_batch  : Evaluates all (or selected) evaluation trips and returns macro stats.

Run on Laptop 1 for LAN evaluation:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.pipeline import MapMatchingPipeline


# ----------------------------------------------------------------------
# Clean Viva Terminal Logger
# ----------------------------------------------------------------------
class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    DIM = "\033[2m"


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("map_matching_api")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _configure_logger()


def _detect_lan_ip() -> str:
    """Best-effort detection of Laptop 1's local LAN IP address for viva display."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


# ----------------------------------------------------------------------
# Pydantic Contracts
# ----------------------------------------------------------------------
class PredictTripRequest(BaseModel):
    trip_id: str = Field(
        ...,
        description="Identifier of an evaluation trip stored server-side (e.g., 'eval_trip_01').",
        examples=["eval_trip_01"],
    )


class PredictBatchRequest(BaseModel):
    trip_ids: Optional[list[str]] = Field(
        default=None,
        description="Optional subset of trip IDs. Omit to evaluate all server-side trips.",
    )


# ----------------------------------------------------------------------
# Application State & Lifespan
# ----------------------------------------------------------------------
_STATE: dict[str, Any] = {
    "pipeline": None,
    "started_at": None,
    "lan_ip": "127.0.0.1",
    "requests_served": 0,
}


def _get_pipeline() -> MapMatchingPipeline:
    pipeline = _STATE.get("pipeline")
    if pipeline is None:
        pipeline = MapMatchingPipeline()
        _STATE["pipeline"] = pipeline
    return pipeline


def _log_startup_banner(pipeline: MapMatchingPipeline, lan_ip: str) -> None:
    health = pipeline.health_summary()
    mode = health["pipeline_mode"]
    mode_color = _Ansi.GREEN if mode in ("full_real", "real_data_pipeline") else _Ansi.YELLOW
    port = os.getenv("API_PORT", "8000")

    lines = [
        "",
        f"{_Ansi.CYAN}{'=' * 76}{_Ansi.RESET}",
        f"{_Ansi.BOLD}{_Ansi.CYAN}  GPS MAP-MATCHING & HMM VITERBI SERVER  (LAPTOP 1 — MEMBER 4){_Ansi.RESET}",
        f"{_Ansi.CYAN}{'=' * 76}{_Ansi.RESET}",
        f"  Status          : {_Ansi.GREEN}READY{_Ansi.RESET}",
        f"  Pipeline Mode   : {mode_color}{mode.upper()}{_Ansi.RESET}",
        f"  LAN Endpoint    : {_Ansi.BOLD}http://{lan_ip}:{port}{_Ansi.RESET}  (Enter this on Laptop 2 UI)",
        f"  Local Swagger   : http://127.0.0.1:{port}/docs",
        f"  Road Graph      : {health['graph_stats']['num_nodes']} nodes | {health['graph_stats']['num_arcs']} directed arcs",
        f"  Model Artifact  : {'LOADED' if health['model_loaded'] else 'FALLBACK (Gaussian/Heading Emission)'}",
        f"  Evaluation Trips: {health['num_eval_trips']} trips loaded server-side (ground truth isolated)",
        f"{_Ansi.CYAN}{'=' * 76}{_Ansi.RESET}",
        "",
    ]
    logger.info("\n".join(lines))


def _log_prediction_viva(client_ip: str, result: dict[str, Any]) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    m = result["metrics"]
    t = result["stage_timings_ms"]
    top1_pct = m["top1_accuracy"] * 100.0
    cont_pct = m["route_continuity"] * 100.0
    clf_top1_pct = m["classifier_only_top1_accuracy"] * 100.0
    clf_cont_pct = m["classifier_only_continuity"] * 100.0
    f1_pct = m["route_f1"] * 100.0

    block = [
        f"{_Ansi.DIM}{'-' * 76}{_Ansi.RESET}",
        (
            f"{_Ansi.BOLD}[{ts}] POST /predict_trip{_Ansi.RESET} | "
            f"client={_Ansi.CYAN}{client_ip}{_Ansi.RESET} | "
            f"trip_id={_Ansi.MAGENTA}{result['trip_id']}{_Ansi.RESET} | "
            f"mode={result['pipeline_mode']}"
        ),
        (
            f"  ├─ Stage 1 Track       : {result['num_gps_points']} GPS observations "
            f"({result.get('description', '')})"
        ),
        (
            f"  ├─ Stage 2 Candidates  : {result['num_candidate_pairs']} pairs "
            f"[{t['candidate_gen_ms']:6.2f} ms]"
        ),
        (
            f"  ├─ Stage 3 Classifier  : Top-1={clf_top1_pct:5.1f}% | "
            f"Continuity={clf_cont_pct:5.1f}% [{t['model_inference_ms']:6.2f} ms]"
        ),
        (
            f"  ├─ Stage 4 HMM Viterbi : "
            f"{_Ansi.GREEN}Top-1={top1_pct:5.1f}%{_Ansi.RESET} | "
            f"{_Ansi.GREEN}Continuity={cont_pct:5.1f}%{_Ansi.RESET} | "
            f"Route F1={f1_pct:5.1f}% [{t['hmm_viterbi_ms']:6.2f} ms]"
        ),
        (
            f"  └─ Stage 5 Completed   : Total={_Ansi.BOLD}{t['total_ms']:.2f} ms{_Ansi.RESET} | "
            f"Pred Arcs={m['num_pred_unique_arcs']} vs GT Arcs={m['num_gt_unique_arcs']} | "
            f"Discontinuities={m['discontinuities']}"
        ),
        f"{_Ansi.DIM}{'-' * 76}{_Ansi.RESET}",
    ]
    logger.info("\n".join(block))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pipeline = MapMatchingPipeline()
    lan_ip = _detect_lan_ip()
    _STATE["pipeline"] = pipeline
    _STATE["started_at"] = time.time()
    _STATE["lan_ip"] = lan_ip
    _log_startup_banner(pipeline, lan_ip)
    yield
    logger.info(f"{_Ansi.YELLOW}[SHUTDOWN] Map-Matching FastAPI Server stopped.{_Ansi.RESET}")


app = FastAPI(
    title="GPS Map-Matching & HMM Viterbi Inference API",
    description=(
        "Laptop 1 inference server for two-laptop live evaluation. "
        "Loads road graph, trained ML model, HMM Viterbi decoder, and server-side "
        "ground truth to evaluate GPS trajectories."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Laptop 2 Streamlit/browser clients over LAN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/", tags=["System"])
def root() -> dict[str, Any]:
    """Root overview endpoint."""
    return {
        "service": "GPS Map-Matching & HMM Viterbi API (Laptop 1)",
        "endpoints": {
            "health": "GET /health",
            "trips": "GET /trips",
            "predict_trip": "POST /predict_trip",
            "predict_batch": "POST /predict_batch",
            "docs": "GET /docs",
        },
    }


@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    """Return server readiness, loaded modules, graph size, and uptime."""
    pipeline = _get_pipeline()
    summary = pipeline.health_summary()
    started = _STATE.get("started_at")
    summary["lan_ip"] = _STATE.get("lan_ip", "127.0.0.1")
    summary["uptime_s"] = round(time.time() - started, 1) if started else 0.0
    summary["requests_served"] = _STATE.get("requests_served", 0)
    return summary


@app.get("/trips", tags=["Evaluation"])
def list_trips() -> dict[str, Any]:
    """Return metadata for the evaluation trips available on Laptop 1."""
    pipeline = _get_pipeline()
    trips = pipeline.list_eval_trips()
    return {
        "count": len(trips),
        "pipeline_mode": pipeline.status.mode,
        "trips": trips,
    }


@app.post("/predict_trip", tags=["Inference"])
def predict_trip(payload: PredictTripRequest, request: Request) -> dict[str, Any]:
    """Execute end-to-end map-matching & server-side evaluation for one trip.

    Exact contract payload: ``{"trip_id": str}``.
    """
    pipeline = _get_pipeline()
    client_ip = request.client.host if request.client else "unknown"

    try:
        result = pipeline.predict_trip(payload.trip_id)
    except KeyError as exc:
        logger.warning(
            f"{_Ansi.RED}[404] POST /predict_trip | client={client_ip} | "
            f"trip_id={payload.trip_id} not found{_Ansi.RESET}"
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"{_Ansi.RED}[500] Pipeline error for trip_id={payload.trip_id}{_Ansi.RESET}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    _STATE["requests_served"] = int(_STATE.get("requests_served", 0)) + 1
    _log_prediction_viva(client_ip, result)
    return result


@app.post("/predict_batch", tags=["Inference"])
def predict_batch(payload: PredictBatchRequest, request: Request) -> dict[str, Any]:
    """Run inference across all (or selected) evaluation trips and aggregate metrics."""
    pipeline = _get_pipeline()
    client_ip = request.client.host if request.client else "unknown"
    target_ids = payload.trip_ids or sorted(pipeline.eval_trip_meta.keys())

    rows: list[dict[str, Any]] = []
    for tid in target_ids:
        if tid not in pipeline.eval_trip_meta:
            raise HTTPException(status_code=404, detail=f"Unknown trip_id '{tid}'.")
        res = pipeline.predict_trip(tid)
        m = res["metrics"]
        rows.append(
            {
                "trip_id": tid,
                "description": res.get("description", ""),
                "num_gps_points": res["num_gps_points"],
                "top1_accuracy": m["top1_accuracy"],
                "route_continuity": m["route_continuity"],
                "route_f1": m["route_f1"],
                "length_weighted_f1": m.get("length_weighted_f1", m["route_f1"]),
                "sequence_jaccard": m["sequence_jaccard"],
                "discontinuities": m["discontinuities"],
                "classifier_only_top1_accuracy": m["classifier_only_top1_accuracy"],
                "classifier_only_continuity": m["classifier_only_continuity"],
                "hmm_accuracy_gain": m["hmm_accuracy_gain"],
                "hmm_continuity_gain": m["hmm_continuity_gain"],
                "processing_time_ms": m["processing_time_ms"],
            }
        )

    n = max(1, len(rows))
    summary = {
        "num_trips": len(rows),
        "mean_top1_accuracy": round(sum(r["top1_accuracy"] for r in rows) / n, 4),
        "mean_route_continuity": round(sum(r["route_continuity"] for r in rows) / n, 4),
        "mean_route_f1": round(sum(r["route_f1"] for r in rows) / n, 4),
        "mean_classifier_only_top1": round(
            sum(r["classifier_only_top1_accuracy"] for r in rows) / n, 4
        ),
        "mean_classifier_only_continuity": round(
            sum(r["classifier_only_continuity"] for r in rows) / n, 4
        ),
        "mean_processing_time_ms": round(sum(r["processing_time_ms"] for r in rows) / n, 2),
        "total_discontinuities": sum(r["discontinuities"] for r in rows),
    }

    _STATE["requests_served"] = int(_STATE.get("requests_served", 0)) + len(rows)
    ts = datetime.now().strftime("%H:%M:%S")
    logger.info(
        f"{_Ansi.BOLD}[{ts}] POST /predict_batch{_Ansi.RESET} | client={client_ip} | "
        f"trips={len(rows)} | Mean Top-1={summary['mean_top1_accuracy']*100:.1f}% | "
        f"Mean Continuity={summary['mean_route_continuity']*100:.1f}% | "
        f"Mean Latency={summary['mean_processing_time_ms']:.2f} ms"
    )

    return {
        "pipeline_mode": pipeline.status.mode,
        "summary": summary,
        "results": rows,
    }


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False, log_level="warning")
