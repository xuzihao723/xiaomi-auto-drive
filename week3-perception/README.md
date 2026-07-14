# Xiaomi AD Week 3 - YOLOv8 多目标检测

> 本目录保存可直接审阅的第三周代码、配置、评估图表和报告。模型权重、完整数据及演示视频位于 [Week 3 Release](https://github.com/xuzihao723/xiaomi-auto-drive/releases/tag/week3-submission) 的 `xiaomi_week3.zip` 中。

本工程对应项目第三周“感知模块开发”，完成车辆、行人、交通灯和交通标志检测，包含数据准备、YOLOv8 微调、独立测试集评估、FPS 测试与一分钟演示视频。

## 1. 第三周交付物

- `detection/`：数据转换、训练、评估、推理、FPS 和视频生成代码
- `configs/data.yaml`：可移植的相对路径数据配置
- `weights/best.pt`、`weights/last.pt`：四类别 YOLOv8 微调权重
- `evaluation/`：混淆矩阵、PR 曲线及测试集预测示例
- `reports/第三周目标检测性能评估报告.pdf`：中文性能评估报告
- `demo/yolov8_detection_demo_1min.mp4`：约一分钟多场景检测视频

## 2. 检测类别

| ID | 类别 | 标签来源 |
| ---: | --- | --- |
| 0 | Car | 第二周 CARLA/KITTI 真值 |
| 1 | Pedestrian | 第二周 CARLA/KITTI 真值 |
| 2 | TrafficLight | COCO YOLOv8 伪标注并抽查 |
| 3 | TrafficSign | COCO YOLOv8 伪标注并抽查 |

交通控制目标使用伪标注是因为第二周数据只提供了车辆和行人标签。该限制已在性能报告中明确说明。

## 2.1 YOLOv8 原理简述

YOLOv8 属于单阶段目标检测算法，一次前向传播即可同时输出目标类别、置信度和边界框，适合自动驾驶实时感知：

- **Backbone**：使用卷积和 C2f 模块提取多尺度图像特征，SPPF 模块扩大感受野。
- **Neck**：通过 FPN/PAN 式特征融合，把高层语义信息与低层位置细节结合起来，增强不同尺度目标的表达。
- **Detection Head**：采用解耦、Anchor-Free 检测头，分别预测分类与边界框，不依赖预设锚框。
- **训练损失**：分类分支与边界框回归分支联合优化，框回归使用 IoU 类损失和 Distribution Focal Loss。
- **推理后处理**：根据置信度筛选候选框，再使用 NMS 去除高度重叠的重复检测框。

本项目选择轻量级 `yolov8n`，在检测精度、模型体积和实时性之间取得平衡。

## 3. 环境

- Windows 11 / WSL2 Ubuntu 22.04 均可
- Python 3.10+
- PyTorch 2.0+
- Ultralytics YOLOv8
- OpenCV、Matplotlib、ReportLab

安装依赖：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# WSL2: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. 准备第二周数据

将第二周压缩包解压至：

```text
data/raw/xiaomi_week2/dataset/image_2
data/raw/xiaomi_week2/dataset/label_2
```

检查 KITTI 数据：

```bash
python tools/check_kitti_dataset.py \
  --dataset-root data/raw/xiaomi_week2/dataset
```

## 5. 生成四类 YOLO 数据集

首次运行会使用官方 `yolov8n.pt` 生成交通灯和交通标志伪标注：

```bash
python detection/kitti_to_yolo.py \
  --dataset-root data/raw/xiaomi_week2/dataset \
  --output-root data/yolo_carla \
  --data-yaml configs/data.yaml \
  --split-mode random \
  --seed 42 \
  --pseudo-model yolov8n.pt \
  --pseudo-imgsz 960 \
  --traffic-light-conf 0.20 \
  --traffic-sign-conf 0.06 \
  --visualize 40 \
  --summary-json reports/dataset_summary.json
```

## 6. 训练

稳定训练命令：

```bash
python detection/train_yolov8.py \
  --data configs/data.yaml \
  --model yolov8n.pt \
  --imgsz 768 \
  --epochs 40 \
  --batch 8 \
  --device 0 \
  --workers 0 \
  --project runs/detect \
  --name train_complete \
  --patience 12 \
  --seed 42 \
  --no-amp \
  --no-cache
```

没有 NVIDIA GPU 时将 `--device 0` 改为 `--device cpu`。

## 7. 独立测试集评估

```bash
python detection/evaluate.py \
  --weights weights/best.pt \
  --data configs/data.yaml \
  --split test \
  --imgsz 768 \
  --batch 8 \
  --device 0 \
  --project runs/detect \
  --name test_complete \
  --output-json reports/evaluation_metrics.json
```

评估重点包括 Precision、Recall、mAP50、mAP50-95、混淆矩阵和 PR 曲线。

## 8. FPS 测试

```bash
python detection/benchmark_fps.py \
  --weights weights/best.pt \
  --source data/yolo_carla/images/test \
  --imgsz 768 \
  --device cpu \
  --max-images 150 \
  --output-json reports/fps_benchmark.json
```

## 9. 推理与演示视频

```bash
python detection/detect_images.py \
  --weights weights/best.pt \
  --source data/raw/xiaomi_week2/dataset/image_2 \
  --output runs/detect/full_sequence_complete \
  --imgsz 768 \
  --conf 0.15 \
  --device 0 \
  --line-width 1 \
  --compact-labels \
  --save-txt \
  --save-conf

python detection/make_demo_video.py \
  --source runs/detect/full_sequence_complete \
  --output demo/yolov8_detection_demo_1min.mp4 \
  --fps 15 \
  --max-frames 1000 \
  --summary-json reports/video_summary.json
```

## 10. 生成 PDF 性能报告

```bash
python tools/build_week3_report.py \
  --dataset-summary reports/dataset_summary.json \
  --evaluation-metrics reports/evaluation_metrics.json \
  --fps-summary reports/fps_benchmark.json \
  --video-summary reports/video_summary.json \
  --train-results training/results.csv \
  --evaluation-dir evaluation \
  --output reports/第三周目标检测性能评估报告.pdf
```

## 11. 说明

- ZIP 不重复包含第二周的 1000 帧原始数据，以控制提交体积。
- `configs/data.yaml` 使用相对路径，不包含个人电脑盘符或用户名。
- 提交包内报告和图表均可直接打开；演示视频已进行完整解码检查。
