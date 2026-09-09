# Package for Four-Class Multimodal Cell Classification

## Purpose

This folder contains only the files required to reproduce the supervised classification task and its evaluation. Presentation-only documents and figures are not included.

The four labels are experimental cell conditions:

- `HeLa`: untreated/control HeLa cells
- `OA`: OA-treated HeLa cells
- `HepG2`: untreated/control HepG2 cells
- `2DG`: 2DG-treated HepG2 cells

`HeLa` is not relabeled as a generic "Normal" class.

## Inputs and model variants

Each segmented single-cell folder contains five registered images and a mask:

- D-FFOCT: `OCT.tif`
- SRS: `protein.tif` and `lipid.tif`
- 2PEF: `nadh.tif` and `fad.tif`
- segmentation mask: `mask.tif`

The classifier uses three biologically interpretable inputs:

- D-FFOCT image
- SRS lipid/protein ratio
- 2PEF optical redox ratio, `FAD / (FAD + NADH)`

The 2PEF branch uses minimal mask-aware processing. No per-cell histogram equalization, histogram matching, or geometric warping is applied.

Seven held-out comparisons are included: three single modalities, three pairwise combinations, and one trimodal model. The trimodal network uses three independent ResNet-18 encoders, feature projections, an adaptive gate, gated feature concatenation, and an MLP classification head.

## Dataset and split

- Total cells: 502
- Source samples: 58
- Training: 338 cells from 36 source samples
- Validation: 85 cells from 11 source samples
- Held-out test: 79 cells from 11 source samples

The split is performed at the `source_sample` level to prevent cells from the same acquisition/source sample from leaking across partitions. The exact split is frozen in `data/manifests/split_manifest.csv` and is used by both ablation and trimodal training.

## Directory layout

```text
configs/                  Training configurations
data/raw/res/             Complete 502-cell dataset
data/manifests/           Frozen split and raw-file manifests
models/                   Final checkpoint and required training dependencies
results/                  Stored held-out predictions and reference metrics
scripts/                  Windows command entry points
src/                      Training, inference, environment, and evaluation code
requirements.txt          Python dependencies
PACKAGE_MANIFEST_SHA256.csv  File-level integrity manifest
```

The initialization and teacher checkpoints under `models/` are necessary dependencies of the selected trimodal training procedure. They are not additional test-time inputs; final inference uses only `models/trimodal_best_model.pt`.

## Tested environment

- Windows
- Python 3.10.18
- PyTorch 2.9.1 with CUDA 12.6
- NVIDIA GeForce RTX 4090

The command files use the `python` executable available in the active environment. Pass another Python executable as the first argument if needed. For example:

```bat
scripts\00_check_gpu_environment.cmd D:\path\to\python.exe
```

To create a new environment, install a CUDA-compatible PyTorch build first and then run:

```bat
python -m pip install -r requirements.txt
```

## Fast exact verification

Run these commands from the package root:

```bat
scripts\00_check_gpu_environment.cmd
scripts\03_test_trimodal.cmd
scripts\04_recompute_metrics.cmd
```

`03_test_trimodal.cmd` loads the packaged final checkpoint, performs inference on the frozen 79-cell test set, and writes `results/reproduced_test/`.

`04_recompute_metrics.cmd` independently recalculates accuracy, macro F1, macro one-vs-rest AUC, source-sample bootstrap 95% confidence intervals, and all seven confusion matrices from the stored held-out prediction tables. It writes `results/recomputed/verification_report.json` and exits with an error if the recomputed values do not match `results/ablation_metrics.csv` within tolerance.

## Training

Train the six single- and dual-modality ablations:

```bat
scripts\01_train_ablation_models.cmd
```

Outputs are written to `results/ablation_retraining/`. Existing packaged reference results are not overwritten.

Retrain the trimodal model using the packaged unimodal initialization and training-only teacher checkpoints:

```bat
scripts\02_train_trimodal.cmd
```

Outputs are written to `results/trimodal_retraining/`. Five seeds are evaluated on the validation set; the test set is evaluated only after seed selection.

## Reference held-out results

| Input               | Accuracy | Macro F1 | Macro OvR AUC |
| ------------------- | -------: | -------: | ------------: |
| D-FFOCT             |   0.7215 |   0.7254 |        0.9224 |
| SRS Lipid/Protein   |   0.6076 |   0.5852 |        0.8630 |
| 2PEF FAD/(FAD+NADH) |   0.8354 |   0.8396 |        0.9758 |
| D-FFOCT + SRS       |   0.8861 |   0.8830 |        0.9800 |
| D-FFOCT + 2PEF      |   0.8987 |   0.9015 |        0.9868 |
| SRS + 2PEF          |   0.9114 |   0.9180 |        0.9884 |
| Trimodal            |   0.9494 |   0.9509 |        0.9911 |

The trimodal confusion matrix uses class order `[HeLa, OA, HepG2, 2DG]`:

```text
[[17, 0,  0,  1],
 [ 0, 14, 0,  0],
 [ 1, 0, 20,  0],
 [ 0, 1,  1, 24]]
```

## Metric definitions

- Accuracy: fraction of correctly classified held-out cells.
- Macro F1: unweighted mean of the four class-wise F1 scores.
- Macro OvR AUC: unweighted mean of one-vs-rest ROC AUC across the four classes.
- Confidence intervals: 2.5th and 97.5th percentiles from 1,000 source-sample bootstrap replicates, stratified by class. Resampling source samples rather than individual cells respects within-sample dependence.

## Reproducibility notes

- Always run commands from the package root or use the supplied command files.
- Do not change `data/manifests/split_manifest.csv` when comparing against the stored metrics.
- GPU training can still show small platform-dependent variation despite fixed seeds. Exact numerical verification should use the packaged checkpoint and prediction tables.
- The raw-file manifest and `PACKAGE_MANIFEST_SHA256.csv` can be used to audit package integrity.
