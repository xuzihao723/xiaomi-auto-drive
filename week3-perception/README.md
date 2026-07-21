# Xiaomi Auto Drive：第三周目标检测完整成果

本目录是第三周感知模块的完整最终版本，不是单独的“问题修复包”。内容覆盖数据采集与标签生成、YOLOv8 四类别训练、严格分组评估、PR 曲线与混淆矩阵、类别专长双模型融合、推理测速和一分钟演示视频。后续发现的问题已经直接融合进最终数据、代码、权重和评估结果中，修复记录仅作为质量说明保留。

## 第三周文档要求与完成情况

| 文档任务 | 完整成果 | 状态 |
| --- | --- | --- |
| 学习并实现 YOLOv8 目标检测 | 完成数据加载、训练、推理、评估和视频生成代码 | 完成 |
| 制作城市道路四类别检测数据集 | 完成 `Car`、`Pedestrian`、`TrafficLight`、`TrafficSign` 标签与场景级划分 | 完成 |
| 训练车辆、行人及交通控制目标检测模型 | 稳定完成 40 轮训练，最终采用道路参与者模型与交通控制模型融合 | 完成 |
| 计算 Precision、Recall、mAP 和 FPS | 输出总体/逐类指标及 CPU、GPU、ONNX、TensorRT 同口径测速 | 完成 |
| 输出 PR 曲线和混淆矩阵 | `evaluation/` 中包含 P、R、F1、PR 曲线及原始/归一化混淆矩阵 | 完成 |
| 生成一分钟目标检测演示视频 | H.264、1280×720、15 FPS、1000 帧、66.67 秒，完整解码通过 | 完成 |
| 提交训练代码和权重 | `detection/`、`tools/`、训练记录及最终权重均已纳入提交材料 | 完成 |

## 完整技术流程

```text
CARLA 多地图/多天气场景
        ↓
同步图像采集与四类别标签生成
        ↓
按场景和连续帧块划分 train / val / test
        ↓
YOLOv8 稳定训练与最佳 checkpoint 选择
        ↓
道路参与者模型 + 交通控制模型类别专长融合
        ↓
严格 test 评估、PR 曲线、混淆矩阵与人工复核
        ↓
CPU / GPU / ONNX / TensorRT 测速与演示视频
```

## 1. 数据集与标签体系

数据来自 CARLA 0.9.15 的 Town05 和 Town10HD_Opt，覆盖晴天、阴天、湿地、日落、软雨、暴雨、夜间和不同随机种子。检测类别统一为：

- `Car`
- `Pedestrian`
- `TrafficLight`
- `TrafficSign`

最终分组数据共 1600 张图像，划分结果如下：

| split | 图像 | Car | Pedestrian | TrafficLight | TrafficSign |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 960 | 2776 | 189 | 4223 | 571 |
| val | 340 | 404 | 138 | 3291 | 320 |
| test | 300 | 602 | 125 | 2480 | 265 |

划分不是把连续帧随机打散，而是按场景和连续帧块固定分组。Town10 原连续序列以 100 帧为一组，组间保留 200 帧缓冲；跨 split 场景组重叠和图像 SHA-256 重叠均为 0。完整统计见：

- `reports/dataset_summary_grouped.json`
- `reports/grouped_validation.json`
- `configs/data_grouped.yaml`
- `configs/scenarios.json`

## 2. YOLOv8 模型训练

训练环境为 Windows 11 + WSL2 Ubuntu 22.04、Python 3.10.12、PyTorch 2.10.0+cu126 和 Ultralytics 8.4.90。最终稳定配置完成全部 40 轮训练，并保存可复核的配置、指标和曲线：

- 训练入口：`detection/train_yolov8.py`
- 最终训练配置：`training/args.yaml`
- 40 轮指标：`training/results.csv`
- 训练曲线：`training/results.png`
- 标签分布：`training/labels.jpg`

最终系统采用类别专长双模型融合：

| 模型 | 负责类别 | 提交权重 |
| --- | --- | --- |
| 道路参与者模型 | Car、Pedestrian | `weights/road_user_best.pt` |
| 交通控制模型 | TrafficLight、TrafficSign | `weights/traffic_control_best.pt` |

融合脚本 `detection/detect_class_fusion.py` 和 `detection/evaluate_class_fusion.py` 分别读取两个模型，只保留其负责类别，再统一执行结果合并与评估。该方案没有覆盖原模型文件，单模型与融合模型结果均可追溯。

## 3. 严格独立测试结果

最终评估使用固定的 300 张严格分组 test，共 3472 个目标。跨 split 没有场景组或完全相同图像重叠。

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Car | 0.952 | 0.829 | 0.857 | 0.732 |
| Pedestrian | 0.885 | 0.804 | 0.882 | 0.615 |
| TrafficLight | 0.839 | 0.322 | 0.522 | 0.196 |
| TrafficSign | 0.843 | 0.404 | 0.554 | 0.071 |
| **总体** | **0.880** | **0.590** | **0.704** | **0.403** |

在相同严格 test 和相同融合评估器口径下：

| 方案 | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| 旧单模型承担四类 | 0.473 | 0.361 |
| 最终类别专长融合 | 0.704 | 0.403 |
| 绝对提升 | +0.231 | +0.042 |

逐类指标和原始混淆矩阵数据位于 `reports/evaluation_metrics_fusion_test.json`；完整 P、R、F1、PR 曲线及混淆矩阵位于 `evaluation/`。

## 4. 评估图表与演示视频

`evaluation/` 包含：

- `BoxP_curve.png`
- `BoxR_curve.png`
- `BoxF1_curve.png`
- `BoxPR_curve.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`

演示视频由 `detection/make_demo_video.py` 根据模型逐帧推理生成，不是人工绘制：

| 项目 | 结果 |
| --- | --- |
| 文件 | `demo/yolov8_detection_demo_1min.mp4` |
| 方法 | 道路参与者 + 交通控制双模型融合 |
| 编码 | H.264 |
| 分辨率 | 1280×720 |
| 帧率 | 15 FPS |
| 总帧数 | 1000 |
| 时长 | 66.67 秒 |
| 完整性 | 1000/1000 帧解码通过 |

GitHub 主分支不重复保存大权重和视频，完整文件位于第三周 Release 压缩包中；在线目录通过 `weights/README.md` 和 `demo/README.md` 提供下载及校验说明。

## 5. 已融合到最终版本的质量完善

以下内容不是独立于第三周主体的另一套结果，而是已经合并进上述最终数据、模型和评估流程：

| 完善项 | 在完整项目中的最终处理 |
| --- | --- |
| CUDA 训练稳定性 | 使用干净的 WSL2/PyTorch 环境、关闭不稳定 AMP 路线、采用稳定 batch，并完成全部 40 轮 |
| 交通控制伪标签可信度 | 对独立旧 test 的 150 张图像、414 个现有控制目标及 97 个漏标候选进行人工复核 |
| TrafficSign 样本不足 | 多场景采集后唯一实例总数增至 1156，严格 test 含 265 个 TrafficSign 目标 |
| 连续帧随机划分 | 改为场景/连续帧块分组，并设置缓冲帧，跨 split 重叠为 0 |
| 单一地图 | 扩展为 Town05 + Town10HD_Opt，并加入多天气、光照和随机种子 |
| 推理速度口径 | 在同一批预加载图像上完成 CPU、GPU、PyTorch、ONNX、TensorRT 和双模型融合测速 |

### 独立测试集人工复核

人工检查固定的 150 张旧独立测试图像，原始标签保持不覆盖，删除、重分类和新增操作均可追溯：

| 项目 | 复核前 | 复核操作 | 复核后 |
| --- | ---: | --- | ---: |
| TrafficLight | 403 | 删除误标、重分类、补漏标 | 338 |
| TrafficSign | 11 | 重分类、补漏标 | 20 |
| 交通控制总数 | 414 | 删除 92、重分类 6、新增 36 | 358 |

| 同口径指标 | 复核前 | 复核后 |
| --- | ---: | ---: |
| 总体 Precision | 0.808 | 0.736 |
| 总体 Recall | 0.536 | 0.611 |
| 总体 mAP50 | 0.640 | 0.645 |
| 总体 mAP50-95 | 0.469 | 0.475 |

TrafficSign 目标由 11 个增加到 20 个后，其 mAP50 从 0.338 变为 0.320，说明旧指标受到漏标影响而偏乐观。复核工作提高的是评估可信度，不是选择性保证所有指标上升。完整证据保存在 `evaluation/traffic_review/`。

### 多后端推理速度

测试统一使用 150 张预加载图像、`batch=1`、`imgsz=640`、20 次预热；计时包含模型预处理、推理和 NMS/后处理，不包含磁盘解码。

| 后端 | 平均延迟 | P95 | FPS |
| --- | ---: | ---: | ---: |
| PyTorch CPU | 18.97 ms | 22.30 ms | 52.71 |
| ONNX CPU | 23.94 ms | 25.72 ms | 41.77 |
| PyTorch GPU | 5.18 ms | 5.83 ms | 192.98 |
| ONNX GPU | 5.65 ms | 7.47 ms | 176.84 |
| TensorRT GPU FP16 | 3.12 ms | 3.72 ms | 320.63 |
| 双模型融合 CPU | 34.35 ms | 38.35 ms | 29.11 |
| 双模型融合 GPU | 9.65 ms | 10.35 ms | 103.60 |

测速代码为 `detection/benchmark_fps.py`，逐后端 JSON 与汇总图位于 `reports/inference_benchmarks/` 和 `reports/inference_benchmark_comparison.png`。

## 6. 目录说明

```text
week3-perception/
├── configs/       # 数据集、场景、划分和人工复核配置
├── detection/     # 采集、转换、训练、推理、融合、评估和测速代码
├── evaluation/    # PR 曲线、混淆矩阵及人工复核证据
├── reports/       # 最终指标、数据校验、测速摘要和第三周实验报告
├── training/      # 最终 40 轮训练配置、CSV、曲线和样例
├── tools/         # 数据验证、人工复核、导出、预览和报告生成工具
├── weights/       # 最终 PyTorch/ONNX/TensorRT 权重（Release 中提供大文件）
└── demo/          # 目标检测演示视频（Release 中提供大文件）
```

## 7. 环境与复现命令

```bash
python -m pip install -r requirements.txt

python detection/train_yolov8.py \
  --data configs/data_grouped.yaml \
  --epochs 40 --batch 8 --device 0

python detection/evaluate_class_fusion.py \
  --road-user-weights weights/road_user_best.pt \
  --traffic-control-weights weights/traffic_control_best.pt \
  --data configs/data_grouped.yaml --split test \
  --imgsz 640 --batch 8 --device 0 \
  --output-dir evaluation \
  --output-json reports/evaluation_metrics_fusion_test.json

python detection/make_demo_video.py \
  --source data/yolo_grouped/images/test \
  --road-user-weights weights/road_user_best.pt \
  --traffic-control-weights weights/traffic_control_best.pt \
  --output demo/yolov8_detection_demo_1min.mp4
```

## 8. 提交材料与下载

- 第三周代码目录：<https://github.com/xuzihao723/xiaomi-auto-drive/tree/main/week3-perception>
- 第三周 Release：<https://github.com/xuzihao723/xiaomi-auto-drive/releases/tag/week3-submission>
- 完整压缩包：<https://github.com/xuzihao723/xiaomi-auto-drive/releases/download/week3-submission/xiaomi_week3.zip>
- 中文实验报告：`reports/第三周实验报告.pdf`

完整压缩包包含代码、配置、训练记录、最终权重、ONNX/TensorRT 导出文件、评估图表、人工复核证据、演示视频和实验报告。GitHub 主目录保留代码及可在线审阅的小型结果，大二进制文件统一由 Release 提供。

## 当前边界

- 完整训练集仍主要使用 CARLA 自动生成标签；人工复核范围是固定的 150 张独立测试图像。
- 交通灯和交通标志属于远距离小目标，定位精度仍低于车辆和行人。
- 推理速度来自笔记本软件管线，尚未完成真实汽车计算平台的摄像头到控制输出端到端延迟测试。

