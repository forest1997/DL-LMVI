"""Reproduce held-out predictions from the selected trimodal checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

import trimodal_pipeline as pipeline
from train_trimodal import MODALITIES, TrimodalClassifier, load_cfg, pipeline_cfg, predict


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "trimodal.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "trimodal_best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "reproduced_test")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    seed = int(checkpoint.get("seed", checkpoint.get("config", {}).get("seed", 42)))
    run_cfg = pipeline_cfg(cfg, seed)
    records = pipeline.discover_records(run_cfg)
    manifest = pd.read_csv(cfg["split_manifest"])
    loaders = pipeline.make_loaders(records, manifest, MODALITIES, run_cfg)

    model = TrimodalClassifier(
        len(cfg["class_order"]), int(cfg["projection_dim"]), int(cfg["gate_hidden_dim"]),
        float(cfg["dropout"]), float(cfg["modality_dropout"]),
        str(cfg["fusion_mode"]), float(cfg["gate_min_weight"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    y_true, probability, embedding, weights, record_indices = predict(
        model, loaders["test"], device
    )
    prediction_index = probability.argmax(axis=1)
    metrics = pipeline.metric_dict(y_true, probability, len(cfg["class_order"]))
    cm = confusion_matrix(y_true, prediction_index, labels=np.arange(len(cfg["class_order"])))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table = manifest.set_index("record_index").loc[record_indices].reset_index()
    table["true_index"] = y_true
    table["pred_index"] = prediction_index
    table["pred_class"] = [cfg["class_order"][index] for index in prediction_index]
    for index, label in enumerate(cfg["class_order"]):
        table[f"prob_{label}"] = probability[:, index]
    for index, label in enumerate(["dffoct", "srs", "2pef_orr"]):
        table[f"weight_{label}"] = weights[:, index]
    table.to_csv(output_dir / "test_predictions_with_modality_weights.csv", index=False)
    np.save(output_dir / "test_embeddings.npy", embedding)
    np.save(output_dir / "confusion_matrix_counts.npy", cm)
    np.save(output_dir / "confusion_matrix_normalized.npy", cm / cm.sum(axis=1, keepdims=True))
    try:
        checkpoint_label = str(args.checkpoint.resolve().relative_to(ROOT))
    except ValueError:
        checkpoint_label = args.checkpoint.name
    result = {
        "model": "trimodal",
        "checkpoint": checkpoint_label,
        "device": str(device),
        "selected_seed": seed,
        **metrics,
        "confusion_matrix": cm.tolist(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
