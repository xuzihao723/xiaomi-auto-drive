# 阶段进度说明

## Week 1：环境搭建与仿真验证

- 安装并验证 WSL2 Ubuntu 22.04。
- 启动 CARLA 0.9.15 Windows Server。
- 验证 Python Client、交通流生成和人工驾驶。
- 保存运行截图、环境说明与 30 秒演示视频。

详细材料见 [`week1-environment/`](../week1-environment/) 和 [Week 1 Release](https://github.com/xuzihao723/xiaomi-auto-drive/releases/tag/week1-submission)。

## Week 2：数据采集与 KITTI 转换

- 配置 RGB 相机与 LiDAR。
- 使用同步模式采集图像、点云和 Actor 真值。
- 生成车辆与行人交通场景。
- 将原始数据转换为 KITTI Object 格式。
- 完成 1000 帧数据自动验收。

详细材料见 [`week2-data-pipeline/`](../week2-data-pipeline/) 和 [Week 2 Release](https://github.com/xuzihao723/xiaomi-auto-drive/releases/tag/week2-submission)。

## Week 3：YOLOv8 四类别感知

- 将 KITTI 数据转换为 YOLO 格式并采用固定随机种子划分数据。
- 在车辆和行人真值基础上补充交通灯、交通标志伪标注。
- 微调 YOLOv8n，保存最佳和最后 checkpoint。
- 在独立 test 集计算 Precision、Recall、mAP 和混淆矩阵。
- 完成 CPU FPS 基准与 1000 帧连续推理演示。

最终测试结果为 mAP@0.5 `0.707`、mAP@0.5:0.95 `0.538`，CPU 推理速度 `15.27 FPS`。

详细材料见 [`week3-perception/`](../week3-perception/) 和 [Week 3 Release](https://github.com/xuzihao723/xiaomi-auto-drive/releases/tag/week3-submission)。

