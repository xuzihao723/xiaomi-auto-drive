# Xiaomi Auto Drive：第四周语义分割

本目录严格对应项目 PDF 的第四周任务：制作车道线/可行驶区域分割数据集，训练 U-Net，提取可行驶区域并拟合车道线，最后与第三周目标检测合成 1 分钟演示视频。

## 已完成内容

- CARLA 0.9.15 同步 RGB 与语义相机采集。
- 1225 组主数据按完整场景固定划分为 train/val/test = 800/250/175；另有模型固定后一次性评估的 75 组 audit，共 1300 组 640×360 图像/掩码。
- Town05、Town10HD_Opt 两张地图，包含晴天、阴天、湿地、雨天、黄昏和夜间。
- 三分类标签：`Background=0`、`DrivableArea=1`、`LaneMarking=2`。
- 轻量 U-Net、加权交叉熵＋Dice 损失、逐类 IoU/Dice 与混淆矩阵。
- 默认保留最佳 checkpoint，并在验证指标连续 5 轮未提升时早停，避免提交过拟合的最后一轮。
- 可行驶区域最大连通域/多边形提取，左右车道线二次曲线拟合。
- 第三周双权重 YOLOv8 与第四周 U-Net 的视频融合脚本。

> 重要：本机 CARLA 0.9.15 Python API 的实际枚举为 `Roads=1`、`RoadLines=24`。采集器在运行时读取 `carla.CityObjectLabel`，不硬编码网页旧顺序值。

## 目录说明

- `configs/`：地图、天气、随机种子和固定数据划分。
- `segmentation/`：采集、U-Net、训练、评估、推理与后处理代码。
- `integration/`：目标检测＋分割的一分钟融合演示。
- `tools/`：数据集验证、预览和训练曲线工具。
- `data/carla_segmentation/`：RGB 图像、三分类掩码、场景清单。
- `runs/unet_final/`：基础训练日志/配置；`runs/unet_night_adapt/`：夜间适配日志/配置。提交包省略带优化器状态的 93 MB 训练 checkpoint，统一使用 `weights/unet_week4_best.pt` 推理与评估。
- `weights/`：去除优化器状态后的提交/推理权重。
- `reports/`：数据校验、独立测试指标、可视化和视频摘要。
- `demo/`：第四周集成演示视频。

> GitHub 与 Release 提交包不重复上传 1300 张原始训练图像；包内保留完整采集代码、场景配置、`manifest.json`、验证摘要和预览证据。最终权重、融合视频与第四周实验报告均包含在提交包中。

## 1. 数据采集

Windows 启动 CARLA Server 后，在 WSL2 中运行：

```bash
export WIN_HOST=$(ip route | awk '/default/ {print $3; exit}')

python segmentation/collect_carla_segmentation.py \
  --host "$WIN_HOST" --port 2000 \
  --config configs/segmentation_scenarios.json \
  --output data/carla_segmentation
```

## 2. 完整性检查

```bash
python tools/validate_segmentation_dataset.py \
  --data data/carla_segmentation \
  --output reports/dataset_validation.json
```

检查项目包括图像/掩码一一对应、尺寸、合法类别、各类像素统计，以及跨 train/val/test 的 SHA-256 重复检查。

## 3. 训练与断点恢复

```bash
python segmentation/train_unet.py \
  --data data/carla_segmentation \
  --output runs/unet_final \
  --epochs 30 --batch-size 4 --workers 2 \
  --device cuda --base-channels 32
```

如果进程中断，可从 `last.pt` 继续；checkpoint 会保存模型、优化器、学习率调度器和 GradScaler 状态：

```bash
python segmentation/train_unet.py \
  --data data/carla_segmentation \
  --output runs/unet_final \
  --epochs 30 --batch-size 4 --workers 2 \
  --device cuda --base-channels 32 \
  --resume runs/unet_final/last.pt
```

若中断来自 AMP/CUDA 稳定性问题，应关闭 `--amp`，从已验证的最佳权重恢复并重置优化器：

```bash
python segmentation/train_unet.py \
  --data data/carla_segmentation \
  --output runs/unet_final \
  --epochs 30 --batch-size 4 --workers 2 \
  --device cuda --base-channels 32 \
  --learning-rate 5e-5 \
  --resume runs/unet_final/best.pt --reset-optimizer
```

当训练/验证场景发生变化（例如新增夜间场景）时，应另建输出目录并增加 `--reset-best`，避免用旧验证分布的分数阻止新 checkpoint 保存。

## 4. 独立测试集评估

```bash
python segmentation/evaluate.py \
  --data data/carla_segmentation \
  --weights weights/unet_week4_best.pt \
  --output reports/test_evaluation_night_adapt \
  --device cuda

python tools/plot_training.py \
  --csv runs/unet_final/results.csv \
  --output reports/training_curves.png
```

## 5. 单图分割、区域提取和车道拟合

```bash
python segmentation/inference.py \
  --weights weights/unet_week4_best.pt \
  --source data/carla_segmentation/images/test/town05_hard_rain_test_00000.png \
  --output reports/example_overlay.png \
  --device cuda
```

除叠加图外，还会输出同名 JSON，记录可行驶区域多边形和车道曲线系数。

## 6. 一分钟集成视频

```bash
python integration/integrated_demo.py \
  --source data/carla_segmentation/images \
  --segmentation-weights runs/unet_final/best.pt \
  --road-user-weights ../xiaomi_week3/weights/road_user_best.pt \
  --traffic-control-weights ../xiaomi_week3/weights/traffic_control_best.pt \
  --output demo/week4_integrated_perception_1min.mp4 \
  --summary reports/integrated_video_summary.json \
  --fps 15 --seconds 60 --device cuda
```

资料依据、文档链接与路线取舍见 `docs/reference_analysis.md`。

## 最终复核结果

- 最佳混合验证集：道路/车道 mIoU 0.820。
- 固定测试集 175 张：道路 IoU 0.749、车道 IoU 0.603、mIoU 0.676。
- 模型固定后新增审计集 75 张：道路 IoU 0.877、车道 IoU 0.715、mIoU 0.796。
- 夜间适配前后：Town10 夜间 mIoU 从 0.004 提升至 0.778；Town05 暴雨 mIoU 从 0.659 小幅降至 0.637。
- 正式视频：900 帧、15 FPS、60.000 秒、640×360、H.264/yuv420p。

## GitHub 与下载

- 累计项目仓库：<https://github.com/xuzihao723/xiaomi-auto-drive>
- 第四周代码目录：<https://github.com/xuzihao723/xiaomi-auto-drive/tree/main/week4-segmentation>
- 第四周 Release：<https://github.com/xuzihao723/xiaomi-auto-drive/releases/tag/week4-submission>
- 第四周压缩包：<https://github.com/xuzihao723/xiaomi-auto-drive/releases/download/week4-submission/xiaomi_week4.zip>
- 中文实验报告：`reports/第四周实验报告.pdf`
