# D-FFOCT quantitative denoising: minimal source release

This package contains the minimal MATLAB source required to reproduce the
quantitative denoising workflow used in the study. The network maps a short
48-frame acquisition to the quantitative-map equivalent of a matched
512-frame acquisition.

The two model channels are:

- `H_Hz`: dynamic center frequency in Hz.
- `V_power`: dynamic intensity, defined as band-integrated FFT power.

`V_power` is transformed to `log10(V_power)` only for stable network training.
Predictions are inverse-transformed and saved in linear `V_power` units. RGB
images are display-only products generated with the original HSV mapping.

## Package contents

- `source/computeDFFOCTQuantMaps.m`: raw time series to H, S, and V maps.
- `source/generate_DFFOCT_quantMaps_fromRaw.m`: batch preprocessing entry point.
- `source/train_DFFOCT_48HVto512HV_quant_customLoop.m`: complete residual U-Net
  definition, data split, normalization, loss, training, validation, and model
  checkpointing.
- `source/run_DFFOCT_denoising.m`: inference-only entry point.
- `source/quantMapsToRgbPreview.m`: original HSV-style RGB visualization.
- `model/DFFOCT_resUNet_H_logV_checkpoint.mat`: trained checkpoint (included in
  the full release, omitted from the source-only archive).
- `data/quant_maps/short_acquisition/`: one ready-to-run short-acquisition test
  map is included.
- `data/quant_maps/long_reference/`: the matched long-acquisition reference for
  the test map is included.
- `output/denoised/`: expected quantitative and RGB outputs for the included
  test map.

## Requirements

- MATLAB R2023b or newer; the release was checked with R2025b.
- Deep Learning Toolbox.
- Image Processing Toolbox.
- A CUDA-capable GPU is optional. CPU inference is supported, while training is
  considerably faster on a GPU.

## Input data

Place raw `uint16` time-series files in `data/raw/`. The provided preprocessing
script assumes each frame is 800 by 550 pixels, stored with the same orientation
as the study data, sampled at 100 Hz. Edit the clearly marked configuration
block when using another camera format or sampling rate.

Generated MAT files contain:

```matlab
maps.H_Hz       % dynamic center frequency, Hz
maps.S_invHz    % inverse spectral width, used only for HSV display
maps.V_power    % band-integrated FFT power, linear units
maps.validMask  % finite pixels with non-zero spectral power
```

The paired filenames use `_Range48_quant.mat` and `_Range512_quant.mat` only as
machine-readable acquisition tags. The scientific variables and manuscript
terminology are short acquisition and long-acquisition reference.

## Run order

From MATLAB, set the current folder to `source/`, then run:

```matlab
generate_DFFOCT_quantMaps_fromRaw
train_DFFOCT_48HVto512HV_quant_customLoop
run_DFFOCT_denoising
```

For inference with the supplied checkpoint, only the last command is needed
after placing quantitative short-acquisition MAT files in
`data/quant_maps/short_acquisition/`.

The inference script writes quantitative MAT files and RGB previews to
`output/denoised/`.

## Included test example

The package includes the representative OA-cell field
`F_OAFOV3_OAcellsLFOV3_images_20251113_220325`. It was also used as a visual
example in the manuscript figure preparation. It is not a HepG2 sample or a
quarantined acquisition outlier.

With the full release, open MATLAB in `source/` and run:

```matlab
run_DFFOCT_denoising
```

The generated MAT file should contain a `maps` struct with `H_Hz`, linear
`V_power`, `validMask`, and provenance fields. The expected output supplied in
the package allows the file structure and RGB appearance to be checked without
retraining the network.

## Reproducibility details

- Training/validation split: deterministic random split with seed 0; 85% for
  training and 15% for validation. For the study dataset this yielded 81
  training pairs and 15 independent validation pairs.
- Patch size: 256 by 256 pixels; batch size: 6.
- Optimizer: Adam, initial learning rate `2e-4`, halved every 20 epochs.
- Network: four-level residual U-Net with 32, 64, 128, and 256 channels and a
  512-channel bottleneck; group normalization and skip concatenation are used.
- Loss: foreground-masked Charbonnier losses for normalized H and log-V plus a
  gradient-consistency term with weight 0.05.
- Tiled inference: 256 by 256 patches with 96-pixel overlap and Hann blending.

## Important interpretation

The output is a learned long-acquisition-equivalent reconstruction. It should
not be described as a direct measurement of the full 5.12 s time series from
the 0.48 s acquisition. Quantitative validation against matched held-out
long-acquisition references is therefore required when applying the model to a
new acquisition regime.
