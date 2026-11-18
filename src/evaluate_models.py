import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

try:
    from train_models import (
        DATASET,
        OUTPUT_DIR,
        FEATURE_COLUMNS,
        build_models,
    )
except ImportError:
    from src.train_models import (
        DATASET,
        OUTPUT_DIR,
        FEATURE_COLUMNS,
        build_models,
    )

ALIGNMENT_FILE = OUTPUT_DIR / "trip_alignment_report.csv"
SUMMARY_FILE = OUTPUT_DIR / "evaluation_summary.csv"
FOLDS_FILE = OUTPUT_DIR / "evaluation_fold_metrics.csv"
CONFUSION_FILE = OUTPUT_DIR / "confusion_matrices.csv"
PLOT_FILE = OUTPUT_DIR / "evaluation_comparison.png"
REPORT_FILE = OUTPUT_DIR / "evaluation_report.md"

RANDOM_SEED = 42
N_SPLITS = 5


def top1_predictions(frame, probabilities):
    evaluation = frame[
        ["trip_id", "gps_index", "is_match"]
    ].copy()

    evaluation["probability"] = probabilities

    selected = evaluation.loc[
        evaluation.groupby(
            ["trip_id", "gps_index"]
        )["probability"].idxmax()
    ].copy()

    selected["correct"] = selected["is_match"].astype(int)
    return selected


def route_continuity(frame, probabilities):
    selected = frame[
        ["trip_id", "gps_index", "candidate_arc_row",
         "start_node", "end_node"]
    ].copy()

    selected["probability"] = probabilities

    selected = selected.loc[
        selected.groupby(
            ["trip_id", "gps_index"]
        )["probability"].idxmax()
    ].copy()

    selected = selected.sort_values(
        ["trip_id", "gps_index"]
    )

    selected["previous_trip"] = selected["trip_id"].shift()
    selected["previous_index"] = selected["gps_index"].shift()
    selected["previous_arc"] = selected["candidate_arc_row"].shift()
    selected["previous_end"] = selected["end_node"].shift()

    consecutive = (
        selected["trip_id"].eq(selected["previous_trip"])
        & selected["gps_index"].eq(
            selected["previous_index"] + 1
        )
    )

    same_arc = selected["candidate_arc_row"].eq(
        selected["previous_arc"]
    )

    connected = selected["start_node"].eq(
        selected["previous_end"]
    )

    valid_transitions = consecutive.sum()
    continuous_transitions = (
        consecutive & (same_arc | connected)
    ).sum()

    if valid_transitions == 0:
        return 0.0, 0

    return (
        float(continuous_transitions / valid_transitions),
        int(valid_transitions),
    )


def sequence_metrics(selected):
    trip_scores = selected.groupby("trip_id")["correct"].mean()

    return {
        "mean_trip_top1_accuracy": float(trip_scores.mean()),
        "exact_sequence_accuracy": float(
            (trip_scores == 1.0).mean()
        ),
        "evaluated_trips": int(len(trip_scores)),
    }


def candidate_miss_rate():
    alignment = pd.read_csv(ALIGNMENT_FILE)

    total_observations = alignment["gps_observations"].sum()
    route_candidate_observations = (
        alignment["route_candidate_observations"].sum()
    )

    if total_observations == 0:
        raise ValueError("No GPS observations in alignment report.")

    return float(
        1 - route_candidate_observations / total_observations
    )


def save_confusion_plot(matrix, model_name):
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], ["Non-match", "Match"])
    ax.set_yticks([0, 1], ["Non-match", "Match"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"confusion_matrix_{slug}.png",
        dpi=160
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    data = pd.read_csv(DATASET)

    X = data[FEATURE_COLUMNS]
    y = data["is_match"].astype(int)
    groups = data["trip_id"].astype(str)

    print(f"Candidate rows: {len(data):,}")
    print(f"GPS observations: {data[['trip_id', 'gps_index']].drop_duplicates().shape[0]:,}")
    print(f"Trips: {groups.nunique()}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    miss_rate = candidate_miss_rate()
    print(f"Candidate miss rate: {miss_rate:.4%}")

    from sklearn.model_selection import GroupKFold

    splitter = GroupKFold(n_splits=N_SPLITS)
    models = build_models()

    fold_results = []
    summary_results = []
    confusion_results = []

    for model_name, model in models.items():
        print(f"\n========== {model_name} ==========")

        oof_probabilities = np.full(len(data), np.nan)
        model_folds = []

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y, groups),
            start=1,
        ):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            fitted_model = clone(model)

            start = time.perf_counter()
            fitted_model.fit(X_train, y_train)
            train_seconds = time.perf_counter() - start

            start = time.perf_counter()
            probabilities = fitted_model.predict_proba(X_test)[:, 1]
            inference_seconds = time.perf_counter() - start

            predictions = (probabilities >= 0.5).astype(int)

            metrics = {
                "model": model_name,
                "fold": fold,
                "accuracy": accuracy_score(y_test, predictions),
                "precision_macro": precision_score(
                    y_test, predictions,
                    average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_test, predictions,
                    average="macro", zero_division=0
                ),
                "macro_f1": f1_score(
                    y_test, predictions,
                    average="macro", zero_division=0
                ),
                "log_loss": log_loss(
                    y_test, probabilities, labels=[0, 1]
                ),
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
            }

            fold_results.append(metrics)
            model_folds.append(metrics)
            oof_probabilities[test_idx] = probabilities

            print(
                f"Fold {fold}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"Macro F1={metrics['macro_f1']:.4f}, "
                f"Log Loss={metrics['log_loss']:.4f}"
            )

        if np.isnan(oof_probabilities).any():
            raise RuntimeError(
                f"Missing out-of-fold predictions for {model_name}"
            )

        oof_predictions = (
            oof_probabilities >= 0.5
        ).astype(int)

        matrix = confusion_matrix(
            y, oof_predictions, labels=[0, 1]
        )

        confusion_results.append({
            "model": model_name,
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        })

        save_confusion_plot(matrix, model_name)

        selected = top1_predictions(data, oof_probabilities)

        continuity, transition_count = route_continuity(
            data, oof_probabilities
        )

        sequences = sequence_metrics(selected)

        summary = {
            "model": model_name,
            "accuracy": accuracy_score(y, oof_predictions),
            "precision_macro": precision_score(
                y, oof_predictions,
                average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y, oof_predictions,
                average="macro", zero_division=0
            ),
            "macro_f1": f1_score(
                y, oof_predictions,
                average="macro", zero_division=0
            ),
            "log_loss": log_loss(
                y, oof_probabilities, labels=[0, 1]
            ),
            "top1_accuracy": selected["correct"].mean(),
            "candidate_miss_rate": miss_rate,
            "route_continuity": continuity,
            "route_transitions_evaluated": transition_count,
            "train_seconds_mean": np.mean([
                item["train_seconds"] for item in model_folds
            ]),
            "inference_seconds_mean": np.mean([
                item["inference_seconds"] for item in model_folds
            ]),
            **sequences,
        }

        summary_results.append(summary)

        print(
            f"OOF Top-1: {summary['top1_accuracy']:.4f} | "
            f"Route continuity: {continuity:.4f} | "
            f"Exact sequence accuracy: "
            f"{summary['exact_sequence_accuracy']:.4f}"
        )

    summary_df = pd.DataFrame(summary_results).sort_values(
        "top1_accuracy", ascending=False
    )

    folds_df = pd.DataFrame(fold_results)
    confusion_df = pd.DataFrame(confusion_results)

    summary_df.to_csv(SUMMARY_FILE, index=False)
    folds_df.to_csv(FOLDS_FILE, index=False)
    confusion_df.to_csv(CONFUSION_FILE, index=False)

    # Model comparison plot
    plot_metrics = [
        "accuracy",
        "macro_f1",
        "top1_accuracy",
        "route_continuity",
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    summary_df.set_index("model")[plot_metrics].plot(
        kind="bar", ax=ax
    )

    ax.set_title("Five-Model Evaluation Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=180)
    plt.close(fig)

    best = summary_df.iloc[0]

    report = f"""# Model Evaluation Report

## Evaluation methodology

- Dataset: `{DATASET}`
- Models evaluated: {len(models)}
- Cross-validation: {N_SPLITS}-fold GroupKFold
- Grouping variable: `trip_id`
- Random seed: {RANDOM_SEED}
- Classification threshold: 0.5
- Predictions: out-of-fold predictions for all retained candidate rows.

## Candidate generation

Candidate miss rate: **{miss_rate:.4%}**

Calculated from the alignment report as:

1 - (total route-candidate observations / total GPS observations).

## Sequence evaluation

- Top-1 accuracy selects the highest-probability candidate for each GPS observation.
- Route continuity measures consecutive observations whose selected arcs are identical or connected end-to-start.
- Exact sequence accuracy is the fraction of trips for which every retained observation has the correct top-1 candidate.
- Sequence metrics apply to retained, aligned observations only.

## Best model by observation-level Top-1 accuracy

**{best['model']}**

- Top-1 accuracy: {best['top1_accuracy']:.4f}
- Macro F1: {best['macro_f1']:.4f}
- Log Loss: {best['log_loss']:.4f}
- Route continuity: {best['route_continuity']:.4f}
- Mean trip Top-1 accuracy: {best['mean_trip_top1_accuracy']:.4f}
- Exact sequence accuracy: {best['exact_sequence_accuracy']:.4f}

## Output files

- `evaluation_summary.csv`
- `evaluation_fold_metrics.csv`
- `confusion_matrices.csv`
- `evaluation_comparison.png`
- Individual model confusion matrix plots.

## Interpretation and limitations

These are cross-validation results on the training dataset, not evaluator-test results.
Candidate miss rate measures candidate-generation coverage, not classifier error.
Route continuity is a simple graph-transition proxy and does not replace HMM/Viterbi
sequence decoding. Exact sequence accuracy is sensitive to trip length.
"""

    REPORT_FILE.write_text(report)

    print("\n========== EVALUATION COMPLETE ==========")
    print(summary_df.to_string(index=False))
    print("\nSaved:")
    for path in [
        SUMMARY_FILE,
        FOLDS_FILE,
        CONFUSION_FILE,
        PLOT_FILE,
        REPORT_FILE,
    ]:
        print(path)


if __name__ == "__main__":
    main()
