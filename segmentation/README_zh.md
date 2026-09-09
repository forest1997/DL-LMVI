# 自动细胞分割模型（含核验数据）

本包包含已核验的自动分割代码、模型权重、运行环境、启动文件，以及 Reproducibility 文件夹中的核验数据。不含手工标注 GUI 或核验脚本。

## 使用

完整解压后，双击 `RUN_AUTOMATIC.bat`，输入图片文件夹和独立的输出文件夹。

- 输入为 D-FFOCT RGB 图像，支持原脚本中的 PNG/JPG/TIF/TIFF/BMP。
- 处理所选文件夹中的图片，不递归处理子文件夹。
- 输出包含 `FinalMasks/Cytoplasm_binary`、`Nucleus_binary`、`Nucleolus_binary`，以及原程序的其他 mask 和叠加预览。
- 使用包内 `Environment/python.exe`，不依赖 E 盘或系统 Python。手工 GUI 和核验入口均不包含在本包中。

也可在 PowerShell 中运行：

```powershell
& .\Environment\python.exe .\run_automatic.py --input 'D:\Images' --output 'D:\Results'
```

## 已确认的模型配置

- 自动推理入口：`Automatic_Segmentation/UNet++-ResNet34_cellpose_outline_improved.py`，2026-07-30 改进版（原脚本没有单独的语义版本号）。
- 细胞核和核仁：U-Net++ / ResNet34，权重 `trained_models/best_unetpp_resnet34.pth`，512 × 512 letterbox 输入，阈值 0.5。
- 整细胞：Cellpose 3.1.1.1 的 legacy `cyto`，权重 `cellpose_models/cytotorch_0`，绿色通道 `[2, 0]`，直径 80 px，flow threshold 0.4，cell probability threshold 0.0。
- 胞质 = 整细胞减最终细胞核；核限制在整细胞内，核仁限制在核内。
- `cyto3` 是原程序保留的备用模型，并非本图使用的主模型。正常日志应显示 `model_type='cyto'`；如出现 fallback 提示，该次运行不再等同于已验证配置。

## 环境及核验结论

包内为已测试的 Windows 运行环境：Python 3.10.20、PyTorch 2.4.1+cu124、Cellpose 3.1.1.1、segmentation-models-pytorch 0.3.3。完整依赖清单见 `Automatic_Segmentation/environment.yml`。NVIDIA 驱动由目标电脑提供；原程序在无可用 GPU 时会使用 CPU。

打包前已在本机 RTX 4050、包内复制环境中重跑全部 31 帧：93 张最终预测 mask 与评估时的结果逐像素一致。核验工作已完成；现附 31 张输入图、186 张 GT/预测 mask、逐帧 Dice、配对表及既有核验结果，但不附核验脚本。数据说明见 Reproducibility/README_zh.md。不同硬件/驱动的浮点细节可能不同，不承诺跨平台逐像素一致。

推理代码和模型权重保持原样；启动器只负责输入输出路径传递。
