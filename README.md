# Xiaomi Auto Drive

基于视觉的城市道路端到端自动驾驶仿真系统。本仓库按周整理 CARLA 仿真、数据采集、目标检测、语义分割以及后续规划控制成果。

> 本项目用于课程学习与仿真实验，不代表小米汽车官方产品。

## 项目进度

| 周次 | 阶段 | 核心成果 | 完整提交包 |
| --- | --- | --- | --- |
| Week 1 | 环境搭建 | WSL2、Ubuntu 22.04、CARLA 0.9.15 环境与基础仿真验证 | [xiaomi_week1.zip](https://github.com/xuzihao723/xiaomi-auto-drive/releases/download/week1-submission/xiaomi_week1.zip) |
| Week 2 | 数据管线 | RGB + LiDAR 同步采集、1000 帧 KITTI 数据和自动验收 | [xiaomi_week2.zip](https://github.com/xuzihao723/xiaomi-auto-drive/releases/download/week2-submission/xiaomi_week2.zip) |
| Week 3 | 目标检测 | YOLOv8 四类别检测、严格分组评估、人工复核和多后端测速 | [xiaomi_week3.zip](https://github.com/xuzihao723/xiaomi-auto-drive/releases/download/week3-submission/xiaomi_week3.zip) |
| Week 4 | 语义分割 | U-Net 车道线/可行驶区域分割、几何后处理与检测融合 | [xiaomi_week4.zip](https://github.com/xuzihao723/xiaomi-auto-drive/releases/download/week4-submission/xiaomi_week4.zip) |

## 第四周最终结果

第四周完成 1300 组 CARLA 同步 RGB/语义标签数据、U-Net 三分类训练、夜间域适配、可行驶区域提取、左右车道线拟合以及目标检测与分割融合视频。

| 评估范围 | 图像 | Pixel Accuracy | Road IoU | Lane IoU | Road/Lane mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 严格 test | 175 | 0.901 | 0.749 | 0.603 | 0.676 |
| 锁定 audit | 75 | 0.960 | 0.877 | 0.715 | 0.796 |

- Town10 夜间 test mIoU：`0.004 → 0.778`。
- Town05 暴雨 test mIoU：`0.659 → 0.637`，保留了适配后的轻微天气权衡。
- 融合视频：H.264/yuv420p、640×360、15 FPS、900 帧、60.0 秒。
- 详细代码、指标证据和中文实验报告见 [`week4-segmentation/`](week4-segmentation/README.md)。

## 第三周最终结果

第三周最终方案采用类别专长双模型融合：`road_user_best.pt` 负责车辆和行人，`traffic_control_best.pt` 负责交通灯和交通标志。

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Car | 0.952 | 0.829 | 0.857 | 0.732 |
| Pedestrian | 0.885 | 0.804 | 0.882 | 0.615 |
| TrafficLight | 0.839 | 0.322 | 0.522 | 0.196 |
| TrafficSign | 0.843 | 0.404 | 0.554 | 0.071 |
| 总体 | 0.880 | 0.590 | 0.704 | 0.403 |

第三周还完成了独立测试集人工复核，以及 PyTorch CPU/GPU、ONNX 和 TensorRT 的同口径推理测速。测速结果来自笔记本软件推理，不等于真实车载端到端延迟。

## 仓库结构

```text
xiaomi-auto-drive/
├── docs/                         # 系统架构与阶段进度
├── week1-environment/            # 环境搭建文档与截图
├── week2-data-pipeline/          # CARLA 数据采集与 KITTI 转换
├── week3-perception/             # YOLOv8 训练、评估、人工复核与部署基准
└── week4-segmentation/           # U-Net 分割、几何后处理与感知融合
```

Git 主分支保存代码、配置、文档和可审阅结果。完整数据、最终权重及演示视频放在对应 GitHub Release 附件中。

## 快速开始

```bash
cd week4-segmentation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

第四周完整复现命令见 [Week 4 README](week4-segmentation/README.md)，前三周内容仍保留在各自目录与 Release 中。

## 当前边界

- 训练和测试数据主要来自 CARLA，尚不能代表真实道路相机域。
- 当前语义分割覆盖 Town05 与 Town10HD_Opt；夜间适配后暴雨能力存在小幅权衡。
- 车道线拟合仍是像素结果上的几何启发式处理，强遮挡和极端曲率下需要时序约束。
- TensorRT 等速度数据来自笔记本软件推理，尚未完成真实汽车计算平台端到端延迟测试。

## 技术栈

- CARLA 0.9.15
- Windows 11 + WSL2 Ubuntu 22.04
- Python 3.10+
- PyTorch / Ultralytics YOLOv8 / U-Net
- OpenCV / NumPy / Matplotlib / ReportLab
