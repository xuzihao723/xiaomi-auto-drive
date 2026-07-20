# 第四周资料与技术路线

## 文档要求

项目 PDF 第 2 页规定第四周必须完成：

1. 学习 U-Net 语义分割算法原理。
2. 标注并制作车道线和可行驶区域分割数据集。
3. 训练车道线和可行驶区域分割模型。
4. 实现车道线拟合和可行驶区域提取。

交付物为分割训练代码和权重、车道拟合/可行驶区域提取代码，以及同时展示目标检测与分割结果的 1 分钟视频。

## 文档内链接的作用

- CARLA 0.9.15 文档：https://carla.readthedocs.io/en/0.9.15/
  - 传感器参考：https://carla.readthedocs.io/en/0.9.15/ref_sensors/
  - 使用 `sensor.camera.rgb` 和 `sensor.camera.semantic_segmentation` 同步采集。
  - 原始标签位于语义图像红通道。在线 0.9.15 页面仍显示旧顺序表，但本机 0.9.15 Python API 的运行时枚举为 `Roads=1`、`RoadLines=24`，因此代码必须读取 `carla.CityObjectLabel`，不能硬编码网页旧值。
- PyTorch 文档：https://pytorch.org/docs/stable/index.html
  - Dataset/DataLoader：https://docs.pytorch.org/docs/stable/data.html
  - CrossEntropyLoss：https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
  - 使用 `Dataset`/`DataLoader`、`CrossEntropyLoss`、AdamW 和 `state_dict` checkpoint。
- OpenCDA：https://github.com/ucla-mobility/OpenCDA
  - 借鉴感知、规划、控制分层和模块化接口，不直接复制完整框架。
- UniAD：https://github.com/OpenDriveLab/UniAD
  - 用于理解感知结果如何服务后续规划，但第四周保持可解释的独立分割模块。

## 补充原始资料

- U-Net 原始论文：https://arxiv.org/abs/1505.04597

U-Net 使用收缩路径提取上下文、对称扩张路径恢复分辨率，并通过跳跃连接融合高分辨率定位信息。本项目采用三类像素输出：Background、DrivableArea、LaneMarking。

## 数据与评估设计

- 标签来源：CARLA 原生逐像素语义真值，不用颜色阈值或目标检测框伪造分割标签；标签 ID 由运行时 API 枚举确定。
- 划分原则：按地图/天气/场景固定 train、val、test，不随机打散连续帧。
- 指标：每类 IoU、Dice、道路/车道 mean IoU、像素准确率和混淆矩阵。
- 后处理：可行驶区域取最大连通域和轮廓多边形；车道线在下方 ROI 内按左右区域拟合二次曲线。
