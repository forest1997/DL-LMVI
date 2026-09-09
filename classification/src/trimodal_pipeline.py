from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18


MODALITY_CHANNELS = {
    "oct": 3,
    "srs_ratio": 1,
    "tpef_ratio": 1,
    "tpef_orr_snr": 1,
    "tpef_orr_equalized": 1,
    "tpef_orr_pattern": 1,
    "tpef_orr_minimal": 1,
}
MODALITY_TITLES = {
    "oct": "D-FFOCT",
    "srs_ratio": "SRS lipid/protein",
    "tpef_ratio": "2PEF FAD/NADH",
    "tpef_orr_snr": "2PEF SNR-corrected ORR",
    "tpef_orr_equalized": "2PEF corrected ORR (SNR/dispersion/shape)",
    "tpef_orr_pattern": "2PEF relative ORR spatial pattern only",
    "tpef_orr_minimal": "2PEF minimally processed ORR",
}
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def portable_config(cfg: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(cfg)
    for key in ("data_root", "output_dir", "split_manifest"):
        if key not in result:
            continue
        path = Path(result[key]).resolve()
        try:
            result[key] = str(path.relative_to(PACKAGE_ROOT))
        except ValueError:
            result[key] = path.name
    return result
@dataclass(frozen=True)
class CellRecord:
    cell_dir: str
    cell_id: str
    raw_class: str
    source_sample: str
    oct_path: str
    protein_path: str
    lipid_path: str
    nadh_path: str
    fad_path: str
    mask_path: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    config_dir = path.resolve().parent
    workspace = config_dir.parent
    output = Path(cfg["output_dir"])
    if not output.is_absolute():
        output = workspace / output
    cfg["output_dir"] = str(output.resolve())
    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        data_root = workspace / data_root
    cfg["data_root"] = str(data_root.resolve())
    return cfg


def find_named_tif(cell_dir: Path, suffix: str) -> Path:
    matches = [p for p in cell_dir.glob("*.tif") if p.stem.lower().endswith("_" + suffix.lower())]
    if len(matches) != 1:
        raise ValueError(f"Expected one *_{suffix}.tif in {cell_dir}, found {len(matches)}")
    return matches[0]


def discover_records(cfg: dict[str, Any]) -> list[CellRecord]:
    root = Path(cfg["data_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    records: list[CellRecord] = []
    for raw_class in cfg["class_order"]:
        class_dir = root / raw_class
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        for cell_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            meta_path = cell_dir / "metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing metadata.json: {cell_dir}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("cell_type") != raw_class:
                raise ValueError(f"Class mismatch in {meta_path}: {meta.get('cell_type')} != {raw_class}")
            records.append(
                CellRecord(
                    cell_dir=str(cell_dir),
                    cell_id=cell_dir.name,
                    raw_class=raw_class,
                    source_sample=str(meta["source_sample"]),
                    oct_path=str(find_named_tif(cell_dir, "OCT")),
                    protein_path=str(find_named_tif(cell_dir, "protein")),
                    lipid_path=str(find_named_tif(cell_dir, "lipid")),
                    nadh_path=str(find_named_tif(cell_dir, "nadh")),
                    fad_path=str(find_named_tif(cell_dir, "fad")),
                    mask_path=str(find_named_tif(cell_dir, "mask")),
                )
            )
    if not records:
        raise ValueError("No cells discovered")
    return records


def audit_records(records: Sequence[CellRecord], cfg: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected = ["oct_path", "protein_path", "lipid_path", "nadh_path", "fad_path", "mask_path"]
    for r in records:
        arrays = {}
        for field in expected:
            path = getattr(r, field)
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError(f"OpenCV could not read {path}")
            arrays[field] = arr
        spatial = {a.shape[:2] for a in arrays.values()}
        if len(spatial) != 1:
            raise ValueError(f"Spatial shape mismatch in {r.cell_dir}: {spatial}")
        if arrays["oct_path"].ndim != 3 or arrays["oct_path"].shape[2] != 3:
            raise ValueError(f"OCT must be 3-channel: {r.oct_path}")
        for field in expected[1:]:
            if arrays[field].ndim != 2:
                raise ValueError(f"{field} must be single-channel in {r.cell_dir}")
        if any(a.dtype != np.uint8 for a in arrays.values()):
            raise ValueError(f"Expected uint8 TIFFs in {r.cell_dir}")
        mask = arrays["mask_path"] > 0
        if not mask.any():
            raise ValueError(f"Empty segmentation mask: {r.mask_path}")
        rows.append(
            {
                "cell_id": r.cell_id,
                "raw_class": r.raw_class,
                "source_sample": r.source_sample,
                "height": arrays["mask_path"].shape[0],
                "width": arrays["mask_path"].shape[1],
                "mask_fraction": float(mask.mean()),
                "oct_max": int(arrays["oct_path"].max()),
                "protein_max": int(arrays["protein_path"].max()),
                "lipid_max": int(arrays["lipid_path"].max()),
                "nadh_max": int(arrays["nadh_path"].max()),
                "fad_max": int(arrays["fad_path"].max()),
            }
        )
    audit = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "data_audit.csv", index=False)
    counts = (
        audit.groupby("raw_class", sort=False)
        .agg(cells=("cell_id", "size"), source_samples=("source_sample", "nunique"))
        .reindex(cfg["class_order"])
        .reset_index()
    )
    counts.to_csv(out_dir / "class_counts.csv", index=False)
    return audit


def build_group_split(records: Sequence[CellRecord], cfg: dict[str, Any]) -> pd.DataFrame:
    labels = np.array([r.raw_class for r in records])
    groups = np.array([r.source_sample for r in records])
    indices = np.arange(len(records))
    split = np.full(len(records), "", dtype=object)
    # Classes and source groups are nested, which can defeat a generic
    # StratifiedGroupKFold assignment. Split groups independently within each
    # class so every partition contains every class by construction.
    rng = np.random.default_rng(int(cfg["seed"]))
    test_fraction = 1.0 / int(cfg["outer_folds"])
    val_fraction = 1.0 / int(cfg["outer_folds"])
    for raw_class in cfg["class_order"]:
        class_groups = np.unique(groups[labels == raw_class])
        if len(class_groups) < 5:
            raise RuntimeError(
                f"Class {raw_class} has only {len(class_groups)} source groups; "
                "at least five are required for a 60/20/20 group split"
            )
        shuffled = rng.permutation(class_groups)
        n_test = max(1, int(round(len(shuffled) * test_fraction)))
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        test_groups = set(shuffled[:n_test])
        val_groups = set(shuffled[n_test : n_test + n_val])
        class_mask = labels == raw_class
        split[class_mask] = "train"
        split[class_mask & np.isin(groups, list(val_groups))] = "val"
        split[class_mask & np.isin(groups, list(test_groups))] = "test"
    if (split == "").any():
        raise RuntimeError("Unassigned split records")
    manifest = pd.DataFrame(
        {
            "record_index": indices,
            "cell_id": [r.cell_id for r in records],
            "raw_class": labels,
            "source_sample": groups,
            "split": split,
            "cell_dir": [r.cell_dir for r in records],
        }
    )
    group_splits = manifest.groupby("source_sample")["split"].nunique()
    if int(group_splits.max()) != 1:
        raise RuntimeError("source_sample leakage detected across splits")
    for s in ("train", "val", "test"):
        present = set(manifest.loc[manifest.split == s, "raw_class"])
        if present != set(cfg["class_order"]):
            raise RuntimeError(f"Split {s} does not contain all classes: {present}")
    return manifest


def write_split_summary(manifest: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]) -> None:
    order = pd.MultiIndex.from_product(
        [["train", "val", "test"], cfg["class_order"]], names=["split", "raw_class"]
    )
    summary = (
        manifest.groupby(["split", "raw_class"])
        .agg(cells=("cell_id", "size"), source_samples=("source_sample", "nunique"))
        .reindex(order, fill_value=0)
        .reset_index()
    )
    summary.to_csv(out_dir / "split_summary.csv", index=False)


def _read(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Could not read {path}")
    return arr


def _letterbox(arr: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interpolation = cv2.INTER_NEAREST if is_mask else (cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    resized = cv2.resize(arr, (nw, nh), interpolation=interpolation)
    if arr.ndim == 2:
        canvas = np.zeros((size, size), dtype=resized.dtype)
    else:
        canvas = np.zeros((size, size, arr.shape[2]), dtype=resized.dtype)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _normalized_gaussian(
    arr: np.ndarray, mask: np.ndarray, sigma: float = 1.0
) -> np.ndarray:
    """Smooth foreground without bleeding the zero-filled exterior into the cell."""
    mask_f = mask.astype(np.float32)
    weight = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    signal = cv2.GaussianBlur(
        arr.astype(np.float32) * mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma
    )
    return signal / np.maximum(weight, 1e-6)


def _canonical_cell_map(
    arr: np.ndarray, mask: np.ndarray, size: int, fill_value: float
) -> np.ndarray:
    """Remove absolute cell size/aspect while making the exterior non-informative."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise ValueError("Cannot canonicalize an empty cell mask")
    cropped = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()
    cropped_mask = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    cropped[~cropped_mask] = fill_value
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def _corrected_tpef_orr(
    fad: np.ndarray,
    nadh: np.ndarray,
    mask: np.ndarray,
    signal_floor: float,
    size: int,
    target_iqr: float | None,
    pattern_only: bool = False,
) -> np.ndarray:
    """Create a bounded, denoised ORR map with training-derived SNR correction."""
    fad_s = _normalized_gaussian(fad, mask)
    nadh_s = _normalized_gaussian(nadh, mask)
    total = fad_s + nadh_s
    orr = fad_s / np.maximum(total, 1e-6)
    reliable = mask & (total >= signal_floor)
    if not reliable.any():
        reliable = mask
    median = float(np.median(orr[reliable]))
    corrected = orr.copy()
    corrected[mask & ~reliable] = median
    q25, q75 = np.quantile(corrected[mask], [0.25, 0.75])
    current_iqr = max(float(q75 - q25), 1e-4)
    if pattern_only:
        corrected[mask] = np.clip(
            (corrected[mask] - median) / (3.0 * current_iqr), -1.0, 1.0
        )
        corrected = _canonical_cell_map(corrected, mask, size, 0.0)
        return corrected.astype(np.float32)
    if target_iqr is not None:
        corrected[mask] = median + (corrected[mask] - median) * target_iqr / current_iqr
    corrected = np.clip(corrected, 0.0, 1.0)
    corrected = _canonical_cell_map(corrected, mask, size, median)
    return corrected.astype(np.float32) * 2.0 - 1.0


def _minimal_tpef_orr(
    fad: np.ndarray,
    nadh: np.ndarray,
    mask: np.ndarray,
    signal_floor: float,
    size: int,
) -> np.ndarray:
    """Denoise and bound ORR while preserving the original multimodal geometry."""
    fad_s = _normalized_gaussian(fad, mask)
    nadh_s = _normalized_gaussian(nadh, mask)
    total = fad_s + nadh_s
    orr = fad_s / np.maximum(total, 1e-6)
    reliable = mask & (total >= signal_floor)
    if not reliable.any():
        reliable = mask
    median = float(np.median(orr[reliable]))
    orr[mask & ~reliable] = median
    # Center 0.5 ORR at zero. The exterior and letterbox padding are also zero,
    # matching the spatial convention used by OCT and SRS.
    scaled = np.zeros_like(orr, dtype=np.float32)
    scaled[mask] = np.clip(orr[mask] * 2.0 - 1.0, -1.0, 1.0)
    return _letterbox(scaled, size).astype(np.float32)


def estimate_tpef_correction_stats(
    records: Sequence[CellRecord], train_indices: Sequence[int], signal_quantile: float
) -> tuple[float, float]:
    """Estimate the SNR floor and target ORR IQR from training cells only."""
    sampled_totals: list[np.ndarray] = []
    for idx in train_indices:
        r = records[int(idx)]
        mask = _read(r.mask_path) > 0
        total = _read(r.fad_path).astype(np.float32) + _read(r.nadh_path).astype(np.float32)
        values = total[mask]
        stride = max(1, len(values) // 512)
        sampled_totals.append(values[::stride])
    signal_floor = float(np.quantile(np.concatenate(sampled_totals), signal_quantile))
    iqrs = []
    for idx in train_indices:
        r = records[int(idx)]
        mask = _read(r.mask_path) > 0
        fad_s = _normalized_gaussian(_read(r.fad_path), mask)
        nadh_s = _normalized_gaussian(_read(r.nadh_path), mask)
        total = fad_s + nadh_s
        orr = fad_s / np.maximum(total, 1e-6)
        reliable = mask & (total >= signal_floor)
        if not reliable.any():
            reliable = mask
        median = float(np.median(orr[reliable]))
        values = orr[mask].copy()
        values[total[mask] < signal_floor] = median
        q25, q75 = np.quantile(values, [0.25, 0.75])
        iqrs.append(float(q75 - q25))
    return signal_floor, float(np.median(iqrs))


class CellDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CellRecord],
        indices: Sequence[int],
        modalities: Sequence[str],
        label_to_idx: dict[str, int],
        image_size: int,
        ratio_epsilon: float,
        ratio_log_clip: float,
        augment: bool,
        tpef_signal_floor: float = 0.0,
        tpef_target_iqr: float = 0.1,
    ) -> None:
        self.records = records
        self.indices = list(map(int, indices))
        self.modalities = tuple(modalities)
        self.label_to_idx = label_to_idx
        self.image_size = image_size
        self.ratio_epsilon = ratio_epsilon
        self.ratio_log_clip = ratio_log_clip
        self.augment = augment
        self.tpef_signal_floor = tpef_signal_floor
        self.tpef_target_iqr = tpef_target_iqr
        self.cache: dict[int, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def _load_modalities(self, record_idx: int) -> dict[str, torch.Tensor]:
        if record_idx in self.cache:
            return {k: v.clone() for k, v in self.cache[record_idx].items()}
        r = self.records[record_idx]
        mask = _read(r.mask_path) > 0
        resized_mask = _letterbox(mask.astype(np.uint8), self.image_size, is_mask=True) > 0
        tensors: dict[str, torch.Tensor] = {}
        if "oct" in self.modalities:
            oct_img = cv2.cvtColor(_read(r.oct_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            oct_img[~mask] = 0.0
            oct_img = _letterbox(oct_img, self.image_size)
            oct_img[~resized_mask] = 0.0
            tensors["oct"] = torch.from_numpy(np.moveaxis(oct_img, -1, 0).copy())
        if "srs_ratio" in self.modalities:
            numerator = _read(r.lipid_path).astype(np.float32)
            denominator = _read(r.protein_path).astype(np.float32)
            ratio = np.log((numerator + self.ratio_epsilon) / (denominator + self.ratio_epsilon))
            ratio = np.clip(ratio, -self.ratio_log_clip, self.ratio_log_clip) / self.ratio_log_clip
            ratio[~mask] = 0.0
            ratio = _letterbox(ratio, self.image_size)
            ratio[~resized_mask] = 0.0
            tensors["srs_ratio"] = torch.from_numpy(ratio[None].copy())
        if "tpef_ratio" in self.modalities:
            numerator = _read(r.fad_path).astype(np.float32)
            denominator = _read(r.nadh_path).astype(np.float32)
            ratio = np.log((numerator + self.ratio_epsilon) / (denominator + self.ratio_epsilon))
            ratio = np.clip(ratio, -self.ratio_log_clip, self.ratio_log_clip) / self.ratio_log_clip
            ratio[~mask] = 0.0
            ratio = _letterbox(ratio, self.image_size)
            ratio[~resized_mask] = 0.0
            tensors["tpef_ratio"] = torch.from_numpy(ratio[None].copy())
        corrected_keys = {
            "tpef_orr_snr",
            "tpef_orr_equalized",
            "tpef_orr_pattern",
            "tpef_orr_minimal",
        }.intersection(self.modalities)
        if corrected_keys:
            fad = _read(r.fad_path).astype(np.float32)
            nadh = _read(r.nadh_path).astype(np.float32)
            for key in sorted(corrected_keys):
                if key == "tpef_orr_minimal":
                    corrected = _minimal_tpef_orr(
                        fad,
                        nadh,
                        mask,
                        self.tpef_signal_floor,
                        self.image_size,
                    )
                    tensors[key] = torch.from_numpy(corrected[None].copy())
                    continue
                target_iqr = self.tpef_target_iqr if key == "tpef_orr_equalized" else None
                corrected = _corrected_tpef_orr(
                    fad,
                    nadh,
                    mask,
                    self.tpef_signal_floor,
                    self.image_size,
                    target_iqr,
                    pattern_only=key == "tpef_orr_pattern",
                )
                tensors[key] = torch.from_numpy(corrected[None].copy())
        self.cache[record_idx] = {k: v.clone() for k, v in tensors.items()}
        return tensors

    @staticmethod
    def _shared_augment(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if random.random() < 0.5:
            inputs = {k: torch.flip(v, dims=(-1,)) for k, v in inputs.items()}
        if random.random() < 0.5:
            inputs = {k: torch.flip(v, dims=(-2,)) for k, v in inputs.items()}
        k = random.randrange(4)
        if k:
            inputs = {name: torch.rot90(value, k, dims=(-2, -1)) for name, value in inputs.items()}
        return inputs

    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], int, int]:
        record_idx = self.indices[item]
        r = self.records[record_idx]
        inputs = self._load_modalities(record_idx)
        if self.augment:
            inputs = self._shared_augment(inputs)
        return inputs, self.label_to_idx[r.raw_class], record_idx


class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalClassifier(nn.Module):
    def __init__(
        self, modalities: Sequence[str], num_classes: int, embedding_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.encoders = nn.ModuleDict(
            {m: ResNet18Encoder(MODALITY_CHANNELS[m]) for m in self.modalities}
        )
        fused_dim = 512 * len(self.modalities)
        self.embedding = nn.Sequential(
            nn.Linear(fused_dim, embedding_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        features = [self.encoders[m](inputs[m]) for m in self.modalities]
        fused = torch.cat(features, dim=1)
        embedding = self.embedding(fused)
        return self.classifier(embedding), embedding


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_loaders(
    records: Sequence[CellRecord], manifest: pd.DataFrame, modalities: Sequence[str], cfg: dict[str, Any]
) -> dict[str, DataLoader]:
    label_to_idx = {name: i for i, name in enumerate(cfg["class_order"])}
    loaders = {}
    train_indices = manifest.loc[manifest.split == "train", "record_index"].to_numpy()
    corrected_requested = any(m.startswith("tpef_orr_") for m in modalities)
    if corrected_requested:
        signal_floor, target_iqr = estimate_tpef_correction_stats(
            records, train_indices, float(cfg.get("tpef_signal_quantile", 0.10))
        )
        message = f"Corrected 2PEF training-only signal_floor={signal_floor:.3f}"
        if "tpef_orr_equalized" in modalities:
            message += f", target_iqr={target_iqr:.5f}"
        print(message, flush=True)
    else:
        signal_floor, target_iqr = 0.0, 0.1
    for split in ("train", "val", "test"):
        indices = manifest.loc[manifest.split == split, "record_index"].to_numpy()
        ds = CellDataset(
            records,
            indices,
            modalities,
            label_to_idx,
            int(cfg["image_size"]),
            float(cfg["ratio_epsilon"]),
            float(cfg["ratio_log_clip"]),
            augment=split == "train",
            tpef_signal_floor=signal_floor,
            tpef_target_iqr=target_iqr,
        )
        generator = torch.Generator().manual_seed(int(cfg["seed"]))
        loaders[split] = DataLoader(
            ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=split == "train",
            num_workers=int(cfg["num_workers"]),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=int(cfg["num_workers"]) > 0,
            generator=generator,
        )
    return loaders


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_y, all_prob, all_embedding, all_record_idx = [], [], [], []
    for inputs, labels, record_idx in loader:
        inputs = move_inputs(inputs, device)
        logits, embedding = model(inputs)
        all_y.append(labels.numpy())
        all_prob.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_embedding.append(embedding.cpu().numpy())
        all_record_idx.append(record_idx.numpy())
    return (
        np.concatenate(all_y),
        np.concatenate(all_prob).reshape(-1, num_classes),
        np.concatenate(all_embedding),
        np.concatenate(all_record_idx),
    )


def metric_dict(y: np.ndarray, prob: np.ndarray, num_classes: int) -> dict[str, float]:
    pred = prob.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    try:
        result["auc_macro_ovr"] = float(
            roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=np.arange(num_classes))
        )
    except ValueError:
        result["auc_macro_ovr"] = float("nan")
    return result


def train_one_experiment(
    model_name: str,
    modalities: Sequence[str],
    records: Sequence[CellRecord],
    manifest: pd.DataFrame,
    cfg: dict[str, Any],
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    exp_dir = Path(cfg["output_dir"]) / "models" / model_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.json"
    pred_path = exp_dir / "test_predictions.csv"
    embedding_path = exp_dir / "test_embeddings.npy"
    if metrics_path.exists() and pred_path.exists() and embedding_path.exists() and not force:
        print(f"[{model_name}] Complete outputs found; skipping (use --force to retrain).", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    set_seed(int(cfg["seed"]))
    loaders = make_loaders(records, manifest, modalities, cfg)
    label_to_idx = {name: i for i, name in enumerate(cfg["class_order"])}
    train_labels = manifest.loc[manifest.split == "train", "raw_class"].map(label_to_idx).to_numpy()
    weights = class_weights(train_labels, len(label_to_idx)).to(device)
    model = MultimodalClassifier(
        modalities, len(label_to_idx), int(cfg["embedding_dim"]), float(cfg["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    criterion = nn.CrossEntropyLoss(weight=weights)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, float]] = []
    best_f1 = -math.inf
    best_epoch = 0
    best_state = None
    patience = 0
    start = time.time()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, labels, _ in loaders["train"]:
            inputs = move_inputs(inputs, device)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits, _ = model(inputs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * labels.size(0)
            seen += labels.size(0)
        scheduler.step()
        val_y, val_prob, _, _ = predict(model, loaders["val"], device, len(label_to_idx))
        val_metrics = metric_dict(val_y, val_prob, len(label_to_idx))
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"[{model_name}] epoch {epoch:02d} loss={row['train_loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} val_f1={val_metrics['f1_macro']:.3f}",
            flush=True,
        )
        if val_metrics["f1_macro"] > best_f1 + 1e-4:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(cfg["early_stopping_patience"]):
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {model_name}")
    model.load_state_dict(best_state)
    checkpoint = {
        "model_name": model_name,
        "modalities": list(modalities),
        "class_order": cfg["class_order"],
        "display_names": cfg["display_names"],
        "config": portable_config(cfg),
        "best_epoch": best_epoch,
        "state_dict": best_state,
    }
    torch.save(checkpoint, exp_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(exp_dir / "training_history.csv", index=False)

    test_y, test_prob, embedding, record_indices = predict(
        model, loaders["test"], device, len(label_to_idx)
    )
    test_metrics = metric_dict(test_y, test_prob, len(label_to_idx))
    pred_labels = test_prob.argmax(axis=1)
    pred_df = manifest.set_index("record_index").loc[record_indices].reset_index()
    pred_df["true_index"] = test_y
    pred_df["pred_index"] = pred_labels
    pred_df["pred_class"] = [cfg["class_order"][i] for i in pred_labels]
    for i, cls in enumerate(cfg["class_order"]):
        pred_df[f"prob_{cls}"] = test_prob[:, i]
    pred_df.to_csv(pred_path, index=False)
    np.save(embedding_path, embedding)
    cm = confusion_matrix(test_y, pred_labels, labels=np.arange(len(label_to_idx)))
    np.save(exp_dir / "confusion_matrix_counts.npy", cm)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    np.save(exp_dir / "confusion_matrix_normalized.npy", cm_norm)
    metrics = {
        "model": model_name,
        "modalities": list(modalities),
        "best_epoch": best_epoch,
        "training_seconds": time.time() - start,
        **test_metrics,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    del model, optimizer, loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _prediction_arrays(pred_df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    label_to_idx = {name: i for i, name in enumerate(cfg["class_order"])}
    y = pred_df["raw_class"].map(label_to_idx).to_numpy()
    prob = pred_df[[f"prob_{c}" for c in cfg["class_order"]]].to_numpy()
    # CSV round-tripping can move row sums a few ulps away from one.  Renormalize
    # before AUC/log-loss evaluation so bootstrap resampling remains warning-free.
    prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)
    return y, prob


def group_bootstrap_ci(
    pred_df: pd.DataFrame, cfg: dict[str, Any], repeats: int, seed: int
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    metric_samples: dict[str, list[float]] = defaultdict(list)
    groups_by_class = {
        cls: pred_df.loc[pred_df.raw_class == cls, "source_sample"].unique()
        for cls in cfg["class_order"]
    }
    for _ in range(repeats):
        sampled_parts = []
        for cls, groups in groups_by_class.items():
            chosen = rng.choice(groups, size=len(groups), replace=True)
            for replicate_id, group in enumerate(chosen):
                part = pred_df[(pred_df.raw_class == cls) & (pred_df.source_sample == group)].copy()
                part["bootstrap_group"] = f"{cls}-{replicate_id}"
                sampled_parts.append(part)
        sampled = pd.concat(sampled_parts, ignore_index=True)
        y, prob = _prediction_arrays(sampled, cfg)
        for key, value in metric_dict(y, prob, len(cfg["class_order"])).items():
            if np.isfinite(value):
                metric_samples[key].append(value)
    return {
        key: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for key, values in metric_samples.items()
    }


def aggregate_results(cfg: dict[str, Any], model_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for offset, model_name in enumerate(model_names):
        exp_dir = Path(cfg["output_dir"]) / "models" / model_name
        metrics = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
        pred_df = pd.read_csv(exp_dir / "test_predictions.csv")
        cis = group_bootstrap_ci(
            pred_df, cfg, int(cfg["bootstrap_repeats"]), int(cfg["seed"]) + offset
        )
        row = dict(metrics)
        for metric, (low, high) in cis.items():
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(Path(cfg["output_dir"]) / "metrics_summary.csv", index=False)
    return result


def save_publication_figure(fig: plt.Figure, base: Path, dpi: int = 600) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def setup_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def write_run_metadata(cfg: dict[str, Any], out_dir: Path, device: torch.device) -> None:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    metadata = {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
        "device": str(device),
        "config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Subset such as DFFOCT_only SRS_only; default is all configured models",
    )
    parser.add_argument("--force", action="store_true", help="Retrain even when complete outputs exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_run_metadata(cfg, out_dir, device)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    records = discover_records(cfg)
    audit_records(records, cfg, out_dir)
    frozen_manifest = cfg.get("split_manifest")
    manifest_path = Path(frozen_manifest) if frozen_manifest else out_dir / "split_manifest.csv"
    if frozen_manifest:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Frozen split manifest not found: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        if len(manifest) != len(records):
            raise RuntimeError("Frozen split manifest length differs from current dataset")
    elif manifest_path.exists() and not args.force:
        manifest = pd.read_csv(manifest_path)
        if len(manifest) != len(records):
            raise RuntimeError("Existing split manifest length differs from current dataset; use --force")
    else:
        manifest = build_group_split(records, cfg)
        manifest.to_csv(manifest_path, index=False)
    write_split_summary(manifest, out_dir, cfg)
    print(pd.read_csv(out_dir / "split_summary.csv").to_string(index=False), flush=True)
    if args.audit_only:
        return

    configured = cfg["experiments"]
    model_names = args.models or list(configured)
    unknown = set(model_names) - set(configured)
    if unknown:
        raise ValueError(f"Unknown models requested: {sorted(unknown)}")
    metrics = []
    for model_name in model_names:
        metrics.append(
            train_one_experiment(
                model_name,
                configured[model_name],
                records,
                manifest,
                cfg,
                device,
                args.force,
            )
        )
    result = aggregate_results(cfg, model_names)
    print("\nCompleted model metrics:", flush=True)
    print(
        result[["model", "accuracy", "f1_macro", "auc_macro_ovr"]].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    cv2.setNumThreads(0)
    main()
