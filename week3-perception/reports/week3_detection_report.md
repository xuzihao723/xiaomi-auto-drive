# 第三周实验报告：基于 YOLOv8 的目标检测模块

## 一、实验目的

本周目标是在第二周 CARLA 仿真数据集基础上完成城市道路目标检测模块开发。实验内容包括 KITTI 标签到 YOLO 标签的转换、YOLOv8 模型微调、车辆与行人目标检测、模型性能评估和演示视频生成。

## 二、实验环境

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Windows + WSL2 数据环境 |
| Python | 3.12.8 |
| 深度学习框架 | PyTorch 2.10.0+cu126 |
| 训练框架 | Ultralytics 8.4.90 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| 推理评估设备 | CPU，避免 CUDA 训练中断影响复现 |

## 三、数据集说明

- 数据来源：第二周 CARLA Town10HD_Opt 仿真采集数据
- 图像数量：1000 张
- 图像尺寸：1280 x 720
- 原始标签格式：KITTI 2D detection
- 训练标签格式：YOLO normalized xywh
- 检测类别：Car、Pedestrian
- 类别数量：Car 3534 个，Pedestrian 597 个
- 空标签帧：64 张

第二周数据尚未包含交通标志、交通灯等类别，因此本周主线先完成车辆和行人两类检测闭环。

## 四、实验流程

1. 检查第二周 KITTI 数据集完整性。
2. 将 KITTI 二维框转换为 YOLO 格式。
3. 按连续帧段划分训练集、验证集和测试集，比例为 70% / 15% / 15%。
4. 使用 `yolov8n.pt` 作为预训练权重训练车辆和行人检测模型。
5. 在 test 集上评估 Precision、Recall、mAP 和混淆矩阵。
6. 对 1000 帧完整序列执行推理并生成 1 分钟演示视频。

## 五、训练结果

训练命令：

```powershell
python .\detection\train_yolov8.py `
  --data .\configs\data.yaml `
  --model yolov8n.pt `
  --imgsz 640 `
  --epochs 50 `
  --batch 8 `
  --device 0 `
  --workers 2 `
  --project .\runs\detect `
  --name train `
  --patience 15
```

训练在第 17 轮遇到 CUDA `illegal instruction` 中断。由于前 16 轮已经正常保存 `best.pt` 和 `last.pt`，本次实验使用已保存的 `best.pt` 继续完成评估、推理和演示。

训练阶段最佳记录：

| 指标 | 最佳 epoch | 数值 |
| --- | ---: | ---: |
| Precision | 14 | 0.95101 |
| Recall | 13 | 0.26747 |
| mAP50 | 13 | 0.29232 |
| mAP50-95 | 12 | 0.20892 |

## 六、测试集评估

测试集包含 150 张图像、1027 个目标，其中 Car 852 个、Pedestrian 175 个。

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| all | 0.960 | 0.187 | 0.202 | 0.159 |
| Car | 0.921 | 0.374 | 0.404 | 0.318 |
| Pedestrian | 1.000 | 0.000 | 0.000 | 0.000 |

评估输出：

- `runs/detect/test_eval/confusion_matrix.png`
- `runs/detect/test_eval/confusion_matrix_normalized.png`
- `runs/detect/test_eval/BoxPR_curve.png`
- `runs/detect/test_eval/BoxF1_curve.png`
- `runs/detect/test_eval/val_batch*_pred.jpg`

## 七、速度测试

CPU 逐张图片推理 benchmark：

| 指标 | 数值 |
| --- | ---: |
| 测试图片数 | 150 |
| 输入尺寸 | 640 |
| 平均延迟 | 67.34 ms |
| 中位延迟 | 66.33 ms |
| FPS | 14.85 |

Ultralytics test 验证阶段统计的模型推理耗时约为 17.7 ms/帧；脚本级 benchmark 包含逐张图片调用、读图和 Python 调度开销，因此 FPS 更低。

## 八、演示结果

- 完整序列推理结果：`runs/detect/full_sequence`
- 演示视频：`demo/yolov8_detection_demo_1min.mp4`
- 视频帧数：1000 帧
- 视频帧率：15 FPS
- 视频时长：约 66 秒

## 九、问题分析

本周模型已经能够检测车辆，但行人检测效果较差。主要原因包括：

1. Pedestrian 样本量只有 597 个，明显少于 Car 的 3534 个。
2. 行人在 1280 x 720 前视图中通常尺度较小，远距离行人更难学习。
3. 当前数据来自单一地图和连续帧，场景多样性有限。
4. 第 17 轮 CUDA 中断导致训练未完成原计划 50 轮。

后续改进建议：

1. 补采行人密集场景，提高行人样本数量和尺度多样性。
2. 增加路口、人行横道、公交站附近场景。
3. 尝试提高输入尺寸到 960，增强小目标学习能力。
4. 使用更稳定的 CUDA/PyTorch 组合重新训练完整 50-80 轮。
5. 扩展交通灯、交通标志类别，满足项目 PDF 中更完整的目标检测要求。

## 十、实验总结

本周完成了目标检测模块从数据检查、格式转换、模型训练、测试评估到演示视频生成的完整闭环。当前模型可作为第四周感知模块集成的车辆检测基线，但行人检测仍需通过补采数据、调整训练策略和稳定训练环境继续提升。
