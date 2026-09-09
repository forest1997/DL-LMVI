"""Train the three-encoder trimodal classifier used in the final analysis."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, f1_score

import trimodal_pipeline as pipeline


MODALITIES = ["oct", "srs_ratio", "tpef_orr_minimal"]
MODALITY_LABELS = ["D-FFOCT", "SRS", "2PEF ORR"]
MODALITY_COLORS = ["#4C78A8", "#72B7B2", "#F2A65A"]


class TrimodalClassifier(nn.Module):
    def __init__(self, num_classes: int, projection_dim: int, gate_hidden: int,
                 dropout: float, modality_dropout: float, fusion_mode: str = "weighted_sum",
                 gate_min_weight: float = 0.0) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({
            modality: pipeline.ResNet18Encoder(pipeline.MODALITY_CHANNELS[modality])
            for modality in MODALITIES
        })
        self.projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(512, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ) for modality in MODALITIES
        })
        self.gate = nn.Sequential(
            nn.Linear(projection_dim * len(MODALITIES), gate_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, len(MODALITIES)),
        )
        if fusion_mode not in {"weighted_sum", "gated_concatenation"}:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.gate_min_weight = gate_min_weight
        if fusion_mode == "gated_concatenation":
            self.fusion_head = nn.Sequential(
                nn.Linear(projection_dim * len(MODALITIES), projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.fusion_head = nn.Identity()
        self.classifier = nn.Linear(projection_dim, num_classes)
        self.auxiliary_heads = nn.ModuleDict({
            modality: nn.Linear(projection_dim, num_classes) for modality in MODALITIES
        })
        self.modality_dropout = modality_dropout

    def forward(self, inputs: dict[str, torch.Tensor], apply_modality_dropout: bool = False):
        features = {m: self.encoders[m](inputs[m]) for m in MODALITIES}
        projected = {m: self.projections[m](features[m]) for m in MODALITIES}
        batch = next(iter(projected.values())).shape[0]
        keep = torch.ones(batch, len(MODALITIES), device=next(iter(projected.values())).device,
                          dtype=torch.bool)
        if self.training and apply_modality_dropout and self.modality_dropout > 0:
            keep = torch.rand_like(keep, dtype=torch.float32) >= self.modality_dropout
            missing_all = ~keep.any(dim=1)
            if missing_all.any():
                missing_indices = missing_all.nonzero(as_tuple=False).squeeze(1)
                replacement = torch.randint(0, len(MODALITIES), (int(missing_all.sum()),),
                                            device=keep.device)
                keep[missing_all] = False
                keep[missing_indices, replacement] = True
        gated_inputs = [projected[m] * keep[:, i:i + 1] for i, m in enumerate(MODALITIES)]
        gate_logits = self.gate(torch.cat(gated_inputs, dim=1))
        gate_logits = gate_logits.masked_fill(~keep, -1e4)
        gates = torch.softmax(gate_logits, dim=1)
        if self.gate_min_weight > 0:
            keep_float = keep.float()
            kept_count = keep_float.sum(dim=1, keepdim=True)
            available = (1 - self.gate_min_weight * kept_count).clamp_min(0)
            gates = gates * available + self.gate_min_weight * keep_float
            gates = gates / gates.sum(dim=1, keepdim=True)
        if self.fusion_mode == "gated_concatenation":
            # Multiplication by the number of modalities preserves feature scale
            # when gates are close to uniform while retaining cell-adaptive emphasis.
            concatenated = torch.cat([
                len(MODALITIES) * gates[:, i:i + 1] * projected[m]
                for i, m in enumerate(MODALITIES)
            ], dim=1)
            fused = self.fusion_head(concatenated)
        else:
            fused = self.fusion_head(sum(
                gates[:, i:i + 1] * projected[m] for i, m in enumerate(MODALITIES)
            ))
        logits = self.classifier(fused)
        auxiliary = {m: self.auxiliary_heads[m](projected[m]) for m in MODALITIES}
        return logits, fused, gates, auxiliary


def load_cfg(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent
    for key in ("data_root", "output_dir", "base_results_dir", "teacher_checkpoints_dir",
                "split_manifest"):
        value = Path(cfg[key])
        if not value.is_absolute():
            value = root / value
        cfg[key] = str(value.resolve())
    return cfg


def pipeline_cfg(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        **cfg,
        "seed": seed,
        "embedding_dim": int(cfg["projection_dim"]),
        "experiments": {"trimodal": MODALITIES},
    }


def initialize_encoders(model: TrimodalClassifier, base_dir: Path) -> None:
    checkpoint_names = {
        "oct": "DFFOCT_only",
        "srs_ratio": "SRS_only",
        "tpef_orr_minimal": "2PEF_ORR_only",
    }
    for modality, checkpoint_name in checkpoint_names.items():
        checkpoint = torch.load(base_dir / checkpoint_name / "best_model.pt",
                                map_location="cpu", weights_only=False)
        prefix = f"encoders.{modality}."
        encoder_state = {
            key[len(prefix):]: value for key, value in checkpoint["state_dict"].items()
            if key.startswith(prefix)
        }
        model.encoders[modality].load_state_dict(encoder_state, strict=True)


def load_teachers(cfg: dict[str, Any], device: torch.device):
    teacher_root = Path(cfg["teacher_checkpoints_dir"])
    definitions = [
        ("DFFOCT_SRS", ["oct", "srs_ratio"]),
        ("SRS_2PEF", ["srs_ratio", "tpef_orr_minimal"]),
    ]
    teachers = []
    for checkpoint_name, modalities in definitions:
        checkpoint = torch.load(teacher_root / checkpoint_name / "best_model.pt",
                                map_location=device, weights_only=False)
        model = pipeline.MultimodalClassifier(
            modalities, len(cfg["class_order"]), int(checkpoint["config"]["embedding_dim"]),
            float(checkpoint["config"]["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        teachers.append(model)
    return teachers


@torch.no_grad()
def predict(model: TrimodalClassifier, loader, device: torch.device):
    model.eval()
    ys, probabilities, embeddings, gates, indices = [], [], [], [], []
    for inputs, labels, record_indices in loader:
        inputs = pipeline.move_inputs(inputs, device)
        logits, embedding, weights, _ = model(inputs)
        ys.append(labels.numpy())
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        embeddings.append(embedding.cpu().numpy())
        gates.append(weights.cpu().numpy())
        indices.append(record_indices.numpy())
    return tuple(np.concatenate(part) for part in (ys, probabilities, embeddings, gates, indices))


def train_seed(cfg: dict[str, Any], seed: int, records, manifest: pd.DataFrame,
               device: torch.device, output_dir: Path) -> dict[str, Any]:
    run_cfg = pipeline_cfg(cfg, seed)
    pipeline.set_seed(seed)
    loaders = pipeline.make_loaders(records, manifest, MODALITIES, run_cfg)
    model = TrimodalClassifier(
        len(cfg["class_order"]), int(cfg["projection_dim"]), int(cfg["gate_hidden_dim"]),
        float(cfg["dropout"]), float(cfg["modality_dropout"]),
        str(cfg.get("fusion_mode", "weighted_sum")), float(cfg.get("gate_min_weight", 0.0)),
    ).to(device)
    initialize_encoders(model, Path(cfg["base_results_dir"]))
    freeze_encoders = bool(cfg.get("freeze_encoders", False))
    if freeze_encoders:
        for encoder in model.encoders.values():
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
    distillation_weight = float(cfg.get("distillation_weight", 0.0))
    teachers = load_teachers(cfg, device) if distillation_weight > 0 else []
    label_to_index = {label: index for index, label in enumerate(cfg["class_order"])}
    train_y = manifest.loc[manifest.split == "train", "raw_class"].map(label_to_index).to_numpy()
    criterion = nn.CrossEntropyLoss(
        weight=pipeline.class_weights(train_y, len(cfg["class_order"])).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_f1, best_epoch, best_state, patience = -math.inf, 0, None, 0
    history = []
    start = time.time()
    uniform = torch.full((len(MODALITIES),), 1 / len(MODALITIES), device=device)
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        if freeze_encoders:
            for encoder in model.encoders.values():
                encoder.eval()
        total_loss = 0.0
        seen = 0
        for inputs, labels, _ in loaders["train"]:
            inputs = pipeline.move_inputs(inputs, device)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, _, gates, auxiliary = model(inputs, apply_modality_dropout=True)
                main_loss = criterion(logits, labels)
                aux_loss = torch.stack([criterion(auxiliary[m], labels) for m in MODALITIES]).mean()
                mean_gate = gates.mean(dim=0).clamp_min(1e-6)
                balance_loss = torch.sum(mean_gate * torch.log(mean_gate / uniform))
                if teachers:
                    temperature = float(cfg["distillation_temperature"])
                    with torch.no_grad():
                        teacher_probabilities = []
                        for teacher in teachers:
                            teacher_logits, _ = teacher(inputs)
                            teacher_probabilities.append(
                                torch.softmax(teacher_logits / temperature, dim=1)
                            )
                        teacher_probability = torch.stack(teacher_probabilities).mean(dim=0)
                    distillation_loss = nn.functional.kl_div(
                        torch.log_softmax(logits / temperature, dim=1),
                        teacher_probability,
                        reduction="batchmean",
                    ) * temperature ** 2
                else:
                    distillation_loss = logits.new_zeros(())
                loss = ((1 - distillation_weight) * main_loss
                        + distillation_weight * distillation_loss
                        + float(cfg["auxiliary_loss_weight"]) * aux_loss
                        + float(cfg["gate_balance_weight"]) * balance_loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * labels.size(0)
            seen += labels.size(0)
        scheduler.step()
        val_y, val_prob, _, val_gate, _ = predict(model, loaders["val"], device)
        metrics = pipeline.metric_dict(val_y, val_prob, len(cfg["class_order"]))
        history.append({
            "epoch": epoch, "train_loss": total_loss / seen,
            "val_accuracy": metrics["accuracy"], "val_f1_macro": metrics["f1_macro"],
            "val_auc_macro_ovr": metrics["auc_macro_ovr"],
            **{f"mean_gate_{label}": float(val_gate[:, i].mean())
               for i, label in enumerate(MODALITY_LABELS)},
        })
        print(f"[seed {seed}] epoch {epoch:02d} loss={total_loss/seen:.4f} "
              f"val_acc={metrics['accuracy']:.3f} val_f1={metrics['f1_macro']:.3f}", flush=True)
        if metrics["f1_macro"] > best_f1 + 1e-4:
            best_f1, best_epoch = metrics["f1_macro"], epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(cfg["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("No trimodal checkpoint selected")
    seed_dir = output_dir / "candidate_seeds" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"seed": seed, "best_epoch": best_epoch, "best_val_f1": best_f1,
                "state_dict": best_state, "config": pipeline.portable_config(cfg)},
               seed_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(seed_dir / "training_history.csv", index=False)
    return {"seed": seed, "best_epoch": best_epoch, "best_val_f1": best_f1,
            "training_seconds": time.time() - start,
            "checkpoint": str(Path("candidate_seeds") / f"seed_{seed}" / "best_model.pt")}


def plot_results(output_dir: Path, prediction: pd.DataFrame, cm: np.ndarray,
                 metrics: dict[str, float], cfg: dict[str, Any]) -> None:
    pipeline.setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    gate_columns = ["weight_dffoct", "weight_srs", "weight_2pef_orr"]
    means = prediction.groupby("raw_class", sort=False)[gate_columns].mean().loc[cfg["class_order"]]
    x = np.arange(len(means))
    bottom = np.zeros(len(means))
    for column, label, color in zip(gate_columns, MODALITY_LABELS, MODALITY_COLORS):
        axes[0].bar(x, means[column], bottom=bottom, label=label, color=color, width=0.72)
        bottom += means[column].to_numpy()
    axes[0].set_xticks(x, cfg["class_order"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Mean learned modality weight")
    axes[0].set_title("Condition-level modality use", fontweight="bold")
    axes[0].legend(fontsize=5.5, loc="upper center", ncol=3)

    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    sns.heatmap(normalized, vmin=0, vmax=1, cmap="Blues", annot=True, fmt=".2f",
                cbar=False, square=True, xticklabels=cfg["class_order"],
                yticklabels=cfg["class_order"], annot_kws={"size": 6}, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Trimodal", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].tick_params(axis="y", rotation=0)

    sorted_prediction = prediction.sort_values(["raw_class", "cell_id"],
                                               key=lambda s: pd.Categorical(s, cfg["class_order"]) if s.name == "raw_class" else s)
    matrix = sorted_prediction[gate_columns].to_numpy().T
    sns.heatmap(matrix, vmin=0, vmax=1, cmap="viridis", cbar_kws={"label": "Gate weight"},
                xticklabels=False, yticklabels=MODALITY_LABELS, ax=axes[2])
    axes[2].set_xlabel("Held-out cells grouped by condition")
    axes[2].set_title("Cell-specific modality weights", fontweight="bold")
    fig.suptitle(f"Trimodal: accuracy {metrics['accuracy']:.3f}, "
                 f"macro F1 {metrics['f1_macro']:.3f}, AUC {metrics['auc_macro_ovr']:.3f}",
                 fontsize=8, fontweight="bold")
    pipeline.save_publication_figure(fig, output_dir / "figure_trimodal_analysis")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/trimodal.yaml"))
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    discovery_cfg = pipeline_cfg(cfg, int(cfg["seeds"][0]))
    records = pipeline.discover_records(discovery_cfg)
    manifest = pd.read_csv(Path(cfg["split_manifest"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = [train_seed(cfg, int(seed), records, manifest, device, output_dir)
                  for seed in cfg["seeds"]]
    candidate_table = pd.DataFrame(candidates).sort_values(
        ["best_val_f1", "seed"], ascending=[False, True]
    )
    candidate_table.to_csv(output_dir / "validation_seed_selection.csv", index=False)
    selected = candidate_table.iloc[0]
    selected_checkpoint = output_dir / str(selected.checkpoint)
    checkpoint = torch.load(selected_checkpoint, map_location=device, weights_only=False)
    model = TrimodalClassifier(
        len(cfg["class_order"]), int(cfg["projection_dim"]), int(cfg["gate_hidden_dim"]),
        float(cfg["dropout"]), float(cfg["modality_dropout"]),
        str(cfg.get("fusion_mode", "weighted_sum")), float(cfg.get("gate_min_weight", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loaders = pipeline.make_loaders(records, manifest, MODALITIES,
                                    pipeline_cfg(cfg, int(selected.seed)))
    test_y, test_prob, embeddings, gates, record_indices = predict(model, loaders["test"], device)
    metrics = pipeline.metric_dict(test_y, test_prob, len(cfg["class_order"]))
    pred = test_prob.argmax(axis=1)
    prediction = manifest.set_index("record_index").loc[record_indices].reset_index()
    prediction["true_index"] = test_y
    prediction["pred_index"] = pred
    prediction["pred_class"] = [cfg["class_order"][i] for i in pred]
    for index, label in enumerate(cfg["class_order"]):
        prediction[f"prob_{label}"] = test_prob[:, index]
    for index, label in enumerate(["dffoct", "srs", "2pef_orr"]):
        prediction[f"weight_{label}"] = gates[:, index]
    prediction.to_csv(output_dir / "test_predictions_with_modality_weights.csv", index=False)
    np.save(output_dir / "test_embeddings.npy", embeddings)
    cm = confusion_matrix(test_y, pred, labels=np.arange(len(cfg["class_order"])))
    np.save(output_dir / "confusion_matrix_counts.npy", cm)
    np.save(output_dir / "confusion_matrix_normalized.npy",
            cm / np.maximum(cm.sum(axis=1, keepdims=True), 1))
    final_checkpoint = {**checkpoint, "selected_by": "validation_macro_f1_across_seeds",
                        "candidate_seeds": list(map(int, cfg["seeds"]))}
    torch.save(final_checkpoint, output_dir / "best_model.pt")
    result = {"model": "trimodal", "modalities": MODALITIES,
              "fusion_mode": str(cfg.get("fusion_mode", "weighted_sum")),
              "gate_min_weight": float(cfg.get("gate_min_weight", 0.0)),
              "distillation_weight": float(cfg.get("distillation_weight", 0.0)),
              "freeze_encoders": bool(cfg.get("freeze_encoders", False)),
              "selected_seed": int(selected.seed), "best_epoch": int(selected.best_epoch),
              "best_val_f1": float(selected.best_val_f1), **metrics,
              "mean_gate_weights": {label: float(gates[:, i].mean())
                                    for i, label in enumerate(MODALITY_LABELS)}}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_results(output_dir, prediction, cm, metrics, cfg)
    print(json.dumps(result, indent=2), flush=True)
    print("confusion_matrix", cm.tolist(), flush=True)


if __name__ == "__main__":
    main()
