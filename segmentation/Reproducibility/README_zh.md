# 核验数据

本文件夹为已完成核验所使用的数据，不包含核验脚本。

- Inputs：31 张 D-FFOCT RGB 原始输入图，按 2DG、4T1、HeLa、HepG2、OA 分组。
- Masks：每帧包含胞质、细胞核和核仁各一张 GT、一张预测 mask，共 186 张。
- frame_manifest.json：输入图、GT、预测 mask 和预期 Dice 的逐帧配对表；路径相对于解压后的包根目录。
- Figure_Frame_Dice.csv：图中使用的 93 条 frame-level Dice 和像素计数；其中原始来源路径仅作溯源，本包实际路径见 frame_manifest.json。
- verification_results.csv / .json：31 帧重新推理的既有核验结果，93 张预测 mask 与归档结果的差异像素数均为 0。
- Original_Run_Logs / source_audit.json：原始运行日志及来源核对记录。

组别帧数：2DG 6、4T1 6、HeLa 7、HepG2 5、OA 7。分析单位是图像/帧，不是单细胞。读取 GT 和预测 mask 时，像素值大于 0 视为前景；Dice = 2 × 交集像素数 /（GT 前景像素数 + 预测前景像素数）。原表 Dice 保留 10 位小数。
