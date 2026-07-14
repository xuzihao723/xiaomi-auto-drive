# 第三周执行记录

## 1. 对照项目要求

项目 PDF 第三周要求：学习 YOLOv8、使用自制仿真数据微调、实现车辆/行人/交通标志等目标实时检测，并使用 mAP、FPS、混淆矩阵和 PR 曲线评估，提交训练代码与权重、性能报告和一分钟演示视频。

## 2. 数据准备

- 来源：第二周 CARLA Town10HD_Opt 自制数据
- 原始图像：1000 张，1280 x 720
- KITTI 真值：Car 3534、Pedestrian 597
- 交通控制伪标注：TrafficLight 2466、TrafficSign 51
- 固定随机种子：42
- 划分：train 700、val 150、test 150

车辆和行人使用第二周 KITTI 真值；交通灯和交通标志使用官方 COCO YOLOv8 伪标注，并经过置信度、尺寸过滤和可视化抽查。

## 3. 正式训练

- 模型：YOLOv8n
- 输入尺寸：768
- Batch：8
- 设备：NVIDIA GeForce RTX 4070 Laptop GPU
- 稳定性设置：workers=0、AMP=false、cache=false
- 最佳 checkpoint：第 27 轮

训练在第 28 轮遇到 CUDA illegal instruction，断点恢复后训练至第 30 轮；第 31 轮又出现驱动级 illegal memory access。因此最终固定使用第 27 轮最佳权重。该权重已正常加载并通过独立 CPU 评估。

## 4. 独立 test 集评估

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Car | 0.904 | 0.866 | 0.926 | 0.801 |
| Pedestrian | 0.860 | 0.606 | 0.637 | 0.395 |
| TrafficLight | 0.784 | 0.816 | 0.886 | 0.647 |
| TrafficSign | 0.320 | 0.545 | 0.379 | 0.308 |
| Overall | - | - | 0.707 | 0.538 |

评估目录包含混淆矩阵、归一化混淆矩阵、PR/P/R/F1 曲线和测试批次预测示例。

## 5. FPS 测试

- 设备：CPU（13th Gen Intel Core i9-13900HX）
- 测试图像：150
- 输入尺寸：768
- 平均延迟：65.50 ms
- 中位延迟：64.77 ms
- FPS：15.27

## 6. 完整序列推理与视频

1000 帧、阈值 0.15 的检测统计：

- Car：4355 个检测，覆盖 934 帧
- Pedestrian：398 个检测，覆盖 270 帧
- TrafficLight：3473 个检测，覆盖 798 帧
- TrafficSign：111 个检测，覆盖 95 帧

演示视频为 1280 x 720、15 FPS、1000 帧、66.67 秒，覆盖城区直道、路口、交通灯与交通标志等不同道路场景。

## 7. 完成状态

- [x] 目标检测训练代码
- [x] 四类别微调权重
- [x] mAP 与 FPS 指标
- [x] 混淆矩阵与 PR 曲线
- [x] 中文 PDF 性能评估报告
- [x] 约一分钟多场景目标检测视频
