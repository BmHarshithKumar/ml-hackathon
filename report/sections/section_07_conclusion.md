# 7. Limitations, Future Work & Conclusion

## 7.1 Operational & Modeling Limitations

While combining a supervised tabular emission classifier with an HMM Viterbi decoder resolves the vast majority of greedy matching jumps, several practical edge cases remain:

- **Dense Parallel Arterials & Multi-Level Overpasses**: In urban canyons with elevated highways above service roads, 2D horizontal GPS coordinates and heading deltas can remain indistinguishable over multiple consecutive observations until a diverging exit ramp provides topological disambiguation.
- **Severe GPS Dropouts & Cold-Start Ambiguity**: Extended signal loss (e.g., tunnels or underground parking > 60 seconds) widens the feasible shortest-path horizon, increasing transition computation cost and permitting multiple equal-length detours between anchor points.
- **Offline Global Decoding vs. Real-Time Streaming**: Standard Viterbi decoding backtracks over the complete trajectory after trip completion. Latency-sensitive live navigation requires bounded-window (sliding-window or fixed-lag) Viterbi smoothing to emit online matches within 2–5 observations.
- **Graph Scale & Map Freshness**: Shortest-path queries grow costly on metropolitan road graphs containing hundreds of thousands of edges without contraction hierarchies, landmark A* heuristics, or bounded Dijkstra cutoffs. Likewise, stale road-network maps (missing newly opened streets or temporary one-way diversions) constrain the HMM to invalid detours.

## 7.2 Future Work

1. **Online Fixed-Lag Viterbi & Streaming Ingestion**: Extend FastAPI with a WebSocket (`/ws/stream_match`) endpoint running fixed-lag Viterbi (lag = 3–5 observations) for live vehicle tracking.
2. **Precomputed Contraction Hierarchies & Spatial Caching**: Replace ad-hoc shortest-path calls with bounded bidirectional Dijkstra or precomputed distance caches for sub-millisecond transitions on city-scale OpenStreetMap extracts.
3. **Uncertainty Calibration & Anomaly Flagging**: Apply isotonic or Platt calibration to classifier emission probabilities and surface low-margin Viterbi transitions directly as yellow confidence warnings on the Folium map.
4. **Containerized Multi-Node Deployment**: Package Laptop 1 and Laptop 2 services into Docker Compose profiles with Prometheus latency histograms for production monitoring.

## 7.3 Conclusion

We designed, implemented, and deployed an end-to-end GPS map-matching pipeline that unites spatial candidate indexing, feature-engineered supervised classification with Optuna hyperparameter tuning, log-space HMM Viterbi sequence reconstruction, and a two-laptop LAN evaluation platform. Separating the FastAPI inference engine (Laptop 1) from the Streamlit + Folium evaluator dashboard (Laptop 2) guarantees strict ground-truth isolation, transparent viva observability, and clear qualitative verification of route reconstruction across raw GPS tracks, road network geometry, reference routes, and predicted sequences.
