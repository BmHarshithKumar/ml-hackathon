
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


DATASET = Path("artifacts/training_dataset.csv")
OUTPUT_DIR = Path("artifacts")
RESULTS_FILE = OUTPUT_DIR / "benchmark_results.csv"

FEATURE_COLUMNS = [
    "distance_m",
    "gps_speed_mps",
    "gps_heading_deg",
    "road_heading_deg",
    "heading_difference_deg",
    "segment_length_m",
    "projection_fraction",
    "along_track_m",
    "time_delta_s",
]


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )),
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )),
        ]),

        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=150,
                max_depth=25,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )),
        ]),

        "LightGBM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                n_jobs=-1,
                verbosity=-1,
                random_state=42,
            )),
        ]),

        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=(
                    1205342 / 170883
                ),
                n_jobs=-1,
                tree_method="hist",
                random_state=42,
            )),
        ]),
    }


def top1_accuracy(frame, probabilities):
    evaluation = frame[
        ["trip_id", "gps_index", "is_match"]
    ].copy()

    evaluation["probability"] = probabilities

    # Select the highest-probability candidate for each GPS observation.
    selected = evaluation.loc[
        evaluation.groupby(
            ["trip_id", "gps_index"]
        )["probability"].idxmax()
    ]

    return float(selected["is_match"].mean())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training dataset...")
    data = pd.read_csv(DATASET)

    X = data[FEATURE_COLUMNS]
    y = data["is_match"].astype(int)
    groups = data["trip_id"].astype(str)

    print(f"Rows: {len(data):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Trips: {groups.nunique()}")
    print(f"Positive rate: {y.mean():.4f}")

    splitter = GroupKFold(n_splits=5)
    models = build_models()
    results = []

    for model_name, model in models.items():
        print(f"\n========== {model_name} ==========")

        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y, groups), start=1
        ):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            test_frame = data.iloc[test_idx]

            fitted_model = clone(model)

            start = time.perf_counter()
            fitted_model.fit(X_train, y_train)
            train_seconds = time.perf_counter() - start

            start = time.perf_counter()
            probabilities = fitted_model.predict_proba(X_test)[:, 1]
            inference_seconds = time.perf_counter() - start

            predictions = (probabilities >= 0.5).astype(int)

            macro_f1 = f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )

            loss = log_loss(
                y_test,
                probabilities,
                labels=[0, 1],
            )

            top1 = top1_accuracy(test_frame, probabilities)

            fold_metrics.append({
                "fold": fold,
                "macro_f1": macro_f1,
                "log_loss": loss,
                "top1_accuracy": top1,
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
            })

            print(
                f"Fold {fold}: "
                f"Macro F1={macro_f1:.4f}, "
                f"Log Loss={loss:.4f}, "
                f"Top-1={top1:.4f}, "
                f"Train={train_seconds:.1f}s"
            )

        summary = {
            "model": model_name,
            "macro_f1_mean": np.mean([
                x["macro_f1"] for x in fold_metrics
            ]),
            "macro_f1_std": np.std([
                x["macro_f1"] for x in fold_metrics
            ]),
            "log_loss_mean": np.mean([
                x["log_loss"] for x in fold_metrics
            ]),
            "top1_accuracy_mean": np.mean([
                x["top1_accuracy"] for x in fold_metrics
            ]),
            "train_seconds_mean": np.mean([
                x["train_seconds"] for x in fold_metrics
            ]),
            "inference_seconds_mean": np.mean([
                x["inference_seconds"] for x in fold_metrics
            ]),
        }

        results.append(summary)

        print("Mean Macro F1:", round(summary["macro_f1_mean"], 4))
        print("Mean Top-1:", round(summary["top1_accuracy_mean"], 4))

    results_df = pd.DataFrame(results).sort_values(
        "top1_accuracy_mean",
        ascending=False,
    )

    results_df.to_csv(RESULTS_FILE, index=False)

    print("\n========== FINAL MODEL BENCHMARK ==========")
    print(results_df.to_string(index=False))
    print(f"\nSaved results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
