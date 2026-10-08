# 第三周完善执行记录

## 数据与划分

- 新增 Town05 七个场景：晴天中午、湿地阴天、晴天日落、软雨日落、阴天中午、暴雨、晴天夜间。
- 使用 CARLA instance segmentation 生成可见像素框，类别标签核对为 Car=14/15/16、Pedestrian=12、TrafficLight=7、TrafficSign=8。
- 合并 Town10HD_Opt 和 Town05 后，按场景/连续帧块构建 960/340/300 的 train/val/test。
- 分组交集：false；跨 split 精确图像哈希交集：false；丢弃边界缓冲 200 帧。
- TrafficSign 唯一实例：571/320/265，总计 1156。

## CUDA 诊断与训练

- Windows 原环境存在 PyTorch c10.dll 初始化问题；改用 WSL2 Ubuntu 22.04 清洁环境。
- 稳定环境：Python 3.10.12、PyTorch 2.10.0+cu126、Ultralytics 8.4.90。
- 200 轮 CUDA 前向/反向矩阵压力测试通过。
- AMP 路线复现 CUDA 非法访问；AMP 关闭后原始分组数据完成 40 轮。
- 均衡数据 batch=16 在第 13 轮复现非法内存访问；重启 WSL、恢复 batch=8 后从第 12 轮检查点续训并完成全部 40 轮。
- 最终 40 轮 CSV：`training/results.csv`；最佳验证轮为 epoch 18，mAP50-95=0.2417。

## 独立测试与融合

- 单一旧模型在严格 grouped test：mAP50=0.4727、mAP50-95=0.3613（使用融合评估器自检口径）。
- 最终类别专长融合：旧模型负责 Car/Pedestrian，新模型负责 TrafficLight/TrafficSign。
- 最终 grouped test：Precision=0.880、Recall=0.590、mAP50=0.704、mAP50-95=0.403。
- 相同评估口径下绝对提升：mAP50 +0.231，mAP50-95 +0.042。
- 评估器同模型自检与 Ultralytics 原生结果差异：mAP50 约 0.005、mAP50-95 约 0.003。

## 性能与演示

- 人工复核独立测试集 150 张图像、414 个现有交通控制目标和 97 个未匹配候选；删除 92 个误标、重分类 6 个、新增 36 个漏标，原标签保持不覆盖。
- 复核前后总体 mAP50 为 0.640 -> 0.645，mAP50-95 为 0.469 -> 0.475；TrafficSign 目标由 11 增至 20 后，其 mAP50 由 0.338 降至 0.320，说明旧指标受漏标影响而偏乐观。
- 预加载 150 张、batch=1、640 像素同口径速度：PyTorch CPU/GPU 为 52.71/192.98 FPS，ONNX CPU/GPU 为 41.77/176.84 FPS，TensorRT GPU FP16 为 320.63 FPS。
- 双模型融合由 CPU 29.11 FPS 提升到 GPU 103.60 FPS。上述数据为笔记本软件推理，不等于真实车载摄像头到控制输出端到端延迟。
- 最终演示视频：H.264，1280×720，15 FPS，1000 帧，66.67 秒。
- ffprobe 完整解码检查：1000/1000 帧通过。
