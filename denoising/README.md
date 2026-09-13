# D-FFOCT quantitative denoising minimal source release v2

This release updates the v1 package to the final frame-count-normalized,
log-V, intensity-preserving model used for the revised evaluation. It includes
the trained checkpoint, portable MATLAB source, one matched test pair, and
expected quantitative and RGB outputs. No linear-V model is included.

## Quick start

Unzip the entire folder. In MATLAB, change the current folder to `source` and run:

```matlab
run_DFFOCT_denoising
```

This uses a supported GPU when available. To force CPU inference:

```matlab
run_DFFOCT_denoising(false)
```

No training or raw-data preprocessing is required for the included example.
Results are written to `output/denoised/quantitative_maps` and
`output/denoised/rgb_preview`. Running inference overwrites only same-named
output files; the supplied checkpoint and input maps are not modified.
The matched long reference is not loaded during inference.

## Requirements

- MATLAB R2025b, used for release verification. Earlier releases are not tested.
- Deep Learning Toolbox and Image Processing Toolbox.
- Parallel Computing Toolbox and a supported GPU only for GPU execution.
- Full retraining requires the complete paired dataset, not just this example.

## Layout

- `source/computeDFFOCTQuantMaps.m`: temporal acquisition to quantitative H/S/V.
- `source/generate_DFFOCT_quantMaps_fromRaw.m`: batch preprocessing.
- `source/train_DFFOCT_48HVto512HV_quant_customLoop.m`: first-stage training,
  complete residual U-Net, fixed split, normalization, loss and validation.
- `source/train_DFFOCT_logV_intensityPreserving_finetune.m`: second-stage
  intensity-preserving fine-tuning.
- `source/run_DFFOCT_denoising.m`: reference-free inference and RGB export.
- `source/validateDFFOCTInputMaps.m`: input units and acquisition checks.
- `source/quantMapsToRgbPreview.m`: original HSV-style display mapping.
- `model/`: final trained checkpoint, not the preliminary model.
- `config/`: the 35 reserved image names and complete 91-pair split manifest.
- `data/quant_maps/`: one normalized short/long acquisition pair.
- `output/denoised/`: expected outputs for that example.
- `tests/verify_packaged_example.m`: preprocessing, guard and inference tests.
- `provenance/`: original training logs, source hashes and release verification.
- `MANIFEST_SHA256.txt`: checksums for the packaged files.

## Quantitative definitions and compatibility

The source data are sampled at 100 Hz. The short acquisition has 48 frames;
the reference has 512. Each raw frame is divided by its spatial mean before FFT.
The MATLAB FFT is unnormalized; DC is excluded.

```text
V_power = sum(abs(FFT).^2 / N^2), for positive-frequency bins in [1, 50] Hz
H_Hz = sum(P(f)*f) / sum(P(f)), over all non-DC positive-frequency bins
```

H and S use all positive bins through the Nyquist bin, as in the original
implementation; the configurable band applies specifically to V. The code uses
one-sided power without doubling the interior positive bins. These are the
implemented arbitrary units, not calibrated radiometric intensity or a
two-sided Parseval total. Frame-mean normalization also means V is relative
to the spatially normalized acquisition rather than absolute detector power.

`maps.V_power` is ALREADY normalized by N squared. The model must not divide it
again. Legacy v1 maps, including the old example, must not be placed directly
in the v2 input folder. The new example is a replacement, not a relabeled copy.
Inputs must record `powerMode="amplitudeSquared"`, `frameCount=48`,
`samplingRateHz=100`, `bandHz=[1 50]`, and `frameNormalize=true`.
Metadata checks cannot detect falsely labeled numerical values.

The two network channels are fixed-range normalized H (0-50 Hz mapped to
[-1,1]) and standardized log10(V), using the checkpoint's training-only
statistics. No per-image intensity normalization is performed before the
network. The inverse transform restores linear V after prediction; the
released numerical inverse retains the original `max(10.^logV - vEps, 0)`
convention. Predictions are blended from 256 x 256 tiles with 96-pixel overlap
using a Hann window floored at 0.001.

After inference, V is multiplied by the ratio of total input V to total
predicted V over valid positive pixels at or above the 20th percentile of input
log-V. The ratio is clamped to [0.5, 2.0]. Finite positive predictions are also
required for a pixel to enter this mask. The 512-frame reference is never used
to calculate this factor. H is unaffected. This anchoring is part of the
released pipeline and is essential to reproducing its reported results.

RGB is display-only. The original HSV-style mapping uses linear V, not log-V,
with predicted saturation fixed to 0.85 because S is not predicted. Display
contrast adaptation is not a quantitative intensity correction. The optional
paper plotting units `V_plot = 512^2 * V_power` are not applied to saved maps.

## Training and evaluation provenance

The cleaned dataset contains 91 matched image pairs: 35 reserved for evaluation,
47 for training and 9 for validation. The split is deterministic (seed 0 after
excluding the 35 reserved names). Validation and test pairs do not enter gradient
updates; the test pairs are not used for checkpoint selection or early stopping.
This establishes image-pair separation, not culture, specimen or acquisition-
batch independence. Refinement followed earlier benchmark diagnostics; these
35 images are not a fresh prospective external validation cohort.

The network has four encoder/decoder levels (32/64/128/256 channels), a
512-channel bottleneck, group normalization, ReLU, concatenated skip connections
and a two-channel residual output. Training uses 256 x 256 patches, batch size
6, Adam and matched flips/90-degree rotations.

First-stage training starts at learning rate 2e-4, with H/log-V Charbonnier
weights 1/1 and gradient weight 0.05. It stopped at epoch 26; the best checkpoint
was from epoch 18. Fine-tuning starts from that checkpoint at learning rate
3e-5, with weights H=1, log-V=1, gradient=0.03, symmetric relative linear-V=0.25,
and patch-mean V conservation=0.80. It stopped at epoch 7; the final selected
checkpoint was from epoch 1. The original CSV histories are included.
Both stages select checkpoints using validation readouts, not test-set scores;
the exact score expressions and stopping settings remain in the source.

For full retraining, populate both map folders with all 91 named pairs, then run:

```matlab
train_DFFOCT_48HVto512HV_quant_customLoop
train_DFFOCT_logV_intensityPreserving_finetune
```

Training outputs go to `output/training`, leaving the supplied checkpoint
unchanged. To use retrained weights, explicitly update `modelFile` in the
inference entry point. The single included test example is insufficient for
training, and training intentionally stops if reserved data are missing.
Hardware and MATLAB versions can affect exact numerical reproducibility.

The included OA-cell field is
`F_OAFOV3_OAcellsLFOV3_images_20251113_220325`, also used in v1. It belongs to
the reserved 35-pair set and is not a HepG2 or excluded-outlier acquisition.
Its expected outputs are regenerated with v2, not copied from v1.
The complete 35-image assessment and its plots are a separate evaluation
release; one example does not reproduce cohort-level statistics.

## Raw preprocessing and interpretation

Place headerless little-endian uint16 `.raw` files in `data/raw`. The default
frame layout is 800 x 550, reshaped to 550 x 800. The script uses the first 48
and first 512 frames of each file; these windows are nested, not independently
acquired realizations. Check the camera layout and acquisition settings before
using new data. Generating maps alone does not validate a new imaging regime.

N-squared normalization removes explicit FFT frame-count scaling, but cannot
restore the different observation windows or frequency-bin spacing (100/N Hz).
The network estimates long-acquisition-equivalent maps; it does not recover the
original 512-frame time series or prove unbiasedness across all conditions.
Further validation is required at new imaging depths or noise levels.

## Verify the package

From the package root in MATLAB:

```matlab
addpath('source', 'tests')
verify_packaged_example(false)  % CPU; use true to test GPU
```

This compares a fresh prediction with the supplied expected maps, checks the
input-only anchor, confirms the example's split, and tests FFT normalization
and rejection of legacy input. Temporary test outputs are separate from the
supplied expected outputs. See `provenance/release_verification.json` for the
release-time tests actually performed.

CPU and GPU calculations are not necessarily bit-identical. The packaged
cross-device checks require maximum H difference <0.05 Hz, mean absolute H
difference <0.005 Hz, and maximum V difference <0.001 of the expected V peak.
These are software regression tolerances, not claims of biological accuracy.
Release verification separately compares GPU inference with the original
evaluation pipeline using tighter tolerances.
