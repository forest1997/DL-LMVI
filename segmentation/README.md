# Automatic segmentation

Automatic cell segmentation with U-Net++ / ResNet34 and Cellpose, including model weights and validation data.

- [Original package documentation (中文)](README_zh.md)
- [Validation data documentation (中文)](Reproducibility/README_zh.md)
- [Environment dependency list](Automatic_Segmentation/environment.yml)

## Repository contents

The 237 project files from `Segmentation_Automatic_With_Validation_Data_20260909.zip` are preserved unchanged, including inference code, three model files, 31 input images, 186 ground-truth/predicted masks, and existing validation records.

The archive's approximately 5.35 GB `Environment/` directory (Python, installed packages, and CUDA libraries) is omitted from Git. Model weights are stored with Git LFS; run `git lfs pull` from the repository root after cloning.

## Running

Run from this `segmentation/` directory with an environment matching the supplied dependency list. The original package documents Python 3.10.20, PyTorch 2.4.1+cu124, Cellpose 3.1.1.1, and segmentation-models-pytorch 0.3.3.

To recreate the environment on Windows with Conda, the CUDA-specific PyTorch wheels need their package index:

```powershell
$env:PIP_EXTRA_INDEX_URL = 'https://download.pytorch.org/whl/cu124'
conda env create -f Automatic_Segmentation/environment.yml
conda activate unetpp-resnet34-cellseg
python run_automatic.py --input 'D:\Images' --output 'D:\Results'
```

Environment recreation and inference have not been rerun as part of this repository upload. The validation records are supplied results from the original archive.

Alternatively, copy `Environment/` from the original archive into this directory. The original `RUN_AUTOMATIC.bat` uses `Environment/python.exe` and requires that folder; it does not use the activated Conda environment.
