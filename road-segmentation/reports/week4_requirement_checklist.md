# 第四周要求逐条核对

## PDF 学习与实现指标

| 序号 | PDF 要求 | 完成证据 | 状态 |
|---|---|---|---|
| 1 | 学习 U-Net 语义分割算法原理 | `docs/reference_analysis.md`、`segmentation/model.py`；编码器/解码器、跳跃连接和三分类输出均已实现 | 通过 |
| 2 | 标注并制作车道线和可行驶区域分割数据集 | CARLA 同步 RGB/语义相机；主数据 800/250/175，另有 75 张锁定审计集；`reports/dataset_validation.json` | 通过 |
| 3 | 训练车道线和可行驶区域分割模型 | `segmentation/train_unet.py`、`runs/unet_night_adapt/best.pt`、`weights/unet_week4_best.pt` | 通过 |
| 4 | 实现车道线拟合和可行驶区域提取 | `segmentation/postprocess.py`、`segmentation/inference.py`；连通域多边形和自车左右标线跟踪/二次拟合 | 通过 |

## PDF 交付物

| 交付物 | 文件 | 检查结果 |
|---|---|---|
| 语义分割训练代码和权重 | `segmentation/`、`weights/unet_week4_best.pt` | 精简权重 31,115,187 字节，可重新加载并完成推理 |
| 车道线拟合和可行驶区域提取代码 | `segmentation/postprocess.py`、`segmentation/inference.py` | 单元自检全部通过，审计样例人工复核通过 |
| 同时展示目标检测和分割的 1 分钟视频 | `demo/week4_integrated_perception_1min.mp4` | 900 帧、15 FPS、60.000 秒、640×360、H.264/yuv420p |
| 中文实验报告 | `reports/第四周实验报告.pdf` | 沿用前三周版式，14 页 A4，已逐页渲染复查 |

## 指标与边界

- 混合验证集最佳道路/车道 mIoU：0.820071。
- 固定测试集 175 张：道路 IoU 0.749066，车道 IoU 0.602862，mIoU 0.675964。
- 锁定审计集 75 张：道路 IoU 0.877213，车道 IoU 0.714785，mIoU 0.795999。
- 夜间适配显著改善 Town10 夜间，但 Town05 暴雨 mIoU 下降 0.022717；该权衡已保留在 `reports/night_adaptation_comparison.json`，没有只展示有利场景。
- 视频中的目标检测沿用第三周两个专用 YOLOv8 checkpoint；第四周新增内容是 U-Net 分割、道路区域提取、车道拟合及集成。
