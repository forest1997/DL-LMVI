"""Recompute all seven ablation metrics from the packaged held-out predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import trimodal_pipeline as pipeline


ROOT = Path(__file__).resolve().parent.parent
CLASS_ORDER = ["HeLa", "OA", "HepG2", "2DG"]
MODELS = [
    ("D-FFOCT", "ablation_predictions/DFFOCT_only/test_predictions.csv", 42),
    ("SRS Lipid/Protein", "ablation_predictions/SRS_only/test_predictions.csv", 43),
    ("2PEF FAD/(FAD+NADH)", "ablation_predictions/2PEF_only/test_predictions.csv", 44),
    ("D-FFOCT + SRS", "ablation_predictions/DFFOCT_SRS/test_predictions.csv", 45),
    ("D-FFOCT + 2PEF", "ablation_predictions/DFFOCT_2PEF/test_predictions.csv", 46),
    ("SRS + 2PEF", "ablation_predictions/SRS_2PEF/test_predictions.csv", 47),
    ("Trimodal", "trimodal/test_predictions_with_modality_weights.csv", 20260812),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "recomputed")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"class_order": CLASS_ORDER}
    metric_rows: list[dict[str, float | str]] = []
    confusion_rows: list[dict[str, float | int | str]] = []

    for model_name, relative_path, bootstrap_seed in MODELS:
        table = pd.read_csv(args.results_dir / relative_path)
        y_true = table["true_index"].to_numpy(dtype=int)
        probability = table[[f"prob_{label}" for label in CLASS_ORDER]].to_numpy(dtype=float)
        y_pred = probability.argmax(axis=1)
        metrics = pipeline.metric_dict(y_true, probability, len(CLASS_ORDER))
        intervals = pipeline.group_bootstrap_ci(
            table, cfg, repeats=args.bootstrap_repeats, seed=bootstrap_seed
        )
        metric_rows.append(
            {
                "input": model_name,
                "bootstrap_seed": bootstrap_seed,
                "accuracy": metrics["accuracy"],
                "accuracy_ci_low": intervals["accuracy"][0],
                "accuracy_ci_high": intervals["accuracy"][1],
                "macro_f1": metrics["f1_macro"],
                "macro_f1_ci_low": intervals["f1_macro"][0],
                "macro_f1_ci_high": intervals["f1_macro"][1],
                "macro_auc_ovr": metrics["auc_macro_ovr"],
                "macro_auc_ci_low": intervals["auc_macro_ovr"][0],
                "macro_auc_ci_high": intervals["auc_macro_ovr"][1],
            }
        )

        matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_ORDER)))
        row_sum = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sum, where=row_sum != 0)
        for true_index, true_label in enumerate(CLASS_ORDER):
            for pred_index, pred_label in enumerate(CLASS_ORDER):
                confusion_rows.append(
                    {
                        "input": model_name,
                        "true_class": true_label,
                        "predicted_class": pred_label,
                        "count": int(matrix[true_index, pred_index]),
                        "row_fraction": float(normalized[true_index, pred_index]),
                    }
                )

    metrics_table = pd.DataFrame(metric_rows)
    metrics_table.to_csv(args.output_dir / "metrics_recomputed.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(
        args.output_dir / "confusion_matrices_recomputed.csv", index=False
    )

    expected = pd.read_csv(args.results_dir / "ablation_metrics.csv")
    comparison = metrics_table.merge(expected, on="input", suffixes=("_new", "_stored"))
    checks = {
        "accuracy": float(np.max(np.abs(comparison.accuracy_new - comparison.accuracy_stored))),
        "macro_f1": float(np.max(np.abs(comparison.macro_f1_new - comparison.macro_f1_stored))),
        "macro_auc": float(
            np.max(np.abs(comparison.macro_auc_ovr - comparison.macro_auc))
        ),
        "accuracy_ci_low": float(
            np.max(np.abs(comparison.accuracy_ci_low_new - comparison.accuracy_ci_low_stored))
        ),
        "accuracy_ci_high": float(
            np.max(np.abs(comparison.accuracy_ci_high_new - comparison.accuracy_ci_high_stored))
        ),
    }
    tolerance = 5e-10
    report = {
        "class_order": CLASS_ORDER,
        "held_out_cells": int(len(pd.read_csv(args.results_dir / MODELS[-1][1]))),
        "bootstrap_unit": "source_sample, stratified by class",
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seeds": {name: seed for name, _, seed in MODELS},
        "maximum_absolute_differences": checks,
        "stored_results_verified": bool(max(checks.values()) <= tolerance),
        "tolerance": tolerance,
    }
    (args.output_dir / "verification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(metrics_table.to_string(index=False))
    print(json.dumps(report, indent=2))
    if not report["stored_results_verified"]:
        raise SystemExit("Stored results did not match recomputed values within tolerance.")


if __name__ == "__main__":
    main()
