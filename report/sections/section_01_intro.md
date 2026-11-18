# 1. Introduction and Problem Formulation

*(Team Overview — finalize after Phase 1 & 2 results are merged.)*

Urban GPS trajectories suffer from multipath reflections, atmospheric delay, and variable sampling intervals, causing raw coordinates to deviate by 5–30 meters from the true underlying road segments. Map-matching resolves this ambiguity by projecting noisy sequential GPS observations onto a directed road network graph $G = (V, E)$ to recover the physically valid sequence of traversed road arcs.

This project implements a four-stage production pipeline:
1. Spatial indexing and candidate road-arc generation on a metric-projected directed road graph (Member 1).
2. Tabular spatio-temporal feature engineering, 5-family supervised model benchmarking with GroupKFold cross-validation, and Bayesian hyperparameter optimization via Optuna (Member 2).
3. Hidden Markov Model (HMM) formulation and log-space Viterbi dynamic programming to enforce road-network connectivity (Member 3).
4. A two-laptop FastAPI + Streamlit deployment system with interactive 4-layer Folium trajectory visualization and live server-side evaluation (Member 4).
