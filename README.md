# DL-LMVI

Deep-learning workflows for LMVI.

## Projects

- [denoising](denoising/README.md): MATLAB D-FFOCT quantitative denoising, trained checkpoint, example input maps, and expected outputs. [中文说明](denoising/README_CN.md)
- [classification](classification/README.md): trimodal cell classification using D-FFOCT, SRS, and 2PEF, with source code, configurations, the packaged 502-cell dataset, model checkpoints, and reference results.

- [segmentation](segmentation/README.md): automatic U-Net++ / ResNet34 and Cellpose segmentation, model weights, and validation inputs, masks, and records.

- [ld-tracking-plots](ld-tracking-plots/README.md): lipid droplet trajectory data and MATLAB plotting code for Fig. 4o.

Run each workflow from its own project directory as described in its README.

## Downloading model files

Model checkpoints (classification, denoising, and segmentation weights) are stored with Git LFS. Install Git LFS, then clone this repository:

```sh
git lfs install
git clone https://github.com/forest1997/DL-LMVI.git
cd DL-LMVI
git lfs pull
```

Access to this private repository requires an authorized GitHub account.


