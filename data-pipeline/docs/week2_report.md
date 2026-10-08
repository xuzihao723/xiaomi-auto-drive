# 第二周实验报告

## 1. 本周目标

本周主要学习 CARLA Python API、虚拟传感器配置、仿真数据采集、KITTI 数据格式和 Git 版本控制。

## 2. 完成内容

- 完成 CARLA Python Client 与 Windows CARLA Server 连接。
- 完成 Ego Vehicle 生成和自动驾驶控制。
- 完成 RGB 相机和激光雷达配置。
- 完成40辆 NPC 车辆和20名行人的交通场景生成。
- 使用同步模式和固定时间步采集数据。
- 完成1000帧图像、点云和元数据采集。
- 完成 CARLA 原始数据到 KITTI 格式转换。
- 完成数据集自动检查和包围框可视化。
- 完成 GitHub 项目仓库初始化。

## 3. 采集配置

- 地图：Town10HD_Opt
- 仿真帧率：20 FPS
- 图像分辨率：1280×720
- 相机水平视场角：90度
- 激光雷达线数：32
- 激光雷达范围：50米
- 采集数量：1000帧

## 4. 数据集结构

```text
training/
├── image_2/
├── velodyne/
├── label_2/
└── calib/
