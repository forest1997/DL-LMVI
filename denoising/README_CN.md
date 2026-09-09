# D-FFOCT 定量去噪源码包

本文件夹包含复现本文定量去噪流程所需的MATLAB 源码。网络输入为
48 帧短采集得到的定量图，目标为匹配的 512 帧长采集定量图。

网络处理两个通道：动态中心频率 `H_Hz` 和动态强度 `V_power`。其中
`V_power` 仅在网络内部使用 `log10` 变换及训练集全局统计量归一化；推理
结果会逆变换并以线性 `V_power` 保存。最终 RGB 图只是使用原始 HSV 映射
生成的显示结果，不参与训练和定量测量。

## 运行环境

- MATLAB R2023b 或更新版本；本包使用 R2025b 检查。
- Deep Learning Toolbox。
- Image Processing Toolbox。
- GPU 可选；CPU 可以推理，但训练建议使用 GPU。

## 使用方法

1. 将 `uint16` 原始时间序列放入 `data/raw/`。默认图像尺寸为 800 x 550，
   采样率为 100 Hz；如数据格式不同，请修改预处理脚本顶部配置。
2. 在 MATLAB 中进入 `source/`，依次运行：

```matlab
generate_DFFOCT_quantMaps_fromRaw
train_DFFOCT_48HVto512HV_quant_customLoop
run_DFFOCT_denoising
```

若直接使用随包提供的模型，只需把短采集定量 MAT 文件放入
`data/quant_maps/short_acquisition/`，然后运行 `run_DFFOCT_denoising.m`。
输出保存在 `output/denoised/`，同时包含定量 H/V 图和 RGB 预览图。

包内已经放入一套可直接运行的 OA 细胞测试数据：短采集定量图位于
`data/quant_maps/short_acquisition/`，匹配的长采集参考位于
`data/quant_maps/long_reference/`。该样本不是 HepG2，也不是此前剔除的
异常采集。含模型版本解压后进入 `source/`，直接运行：

```matlab
run_DFFOCT_denoising
```

即可重新生成 `output/denoised/` 中随包提供的预期定量结果和 RGB 图，无需
先执行训练。

## 数据划分和训练参数

训练和验证采用固定随机种子 0 的 85%/15% 非重叠划分。在本文数据中为
81 对训练图和 15 对验证图。网络为四层残差 U-Net，编码通道数为
32/64/128/256，瓶颈层为 512。训练 patch 为 256 x 256，batch size 为 6，
优化器为 Adam，初始学习率 `2e-4`，每 20 个 epoch 减半。损失函数由掩膜
Charbonnier H 损失、log-V 损失和权重 0.05 的梯度一致性损失组成。

## 科学解释边界

网络结果应描述为“长采集等效重建”，而不是从 0.48 s 序列直接测得完整
5.12 s 时间序列的频谱统计。将模型用于新的成像深度或噪声条件时，仍应
使用匹配的长采集数据进行独立定量验证。
