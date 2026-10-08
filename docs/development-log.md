# Development log / 开发记录

[English overview](../README.md) · [中文首页](../README.zh-CN.md)

The project is now organized by module. Week identifiers below describe historical milestones and published artifacts, not the current code architecture. / 仓库现在按模块组织，以下周次仅保留开发历史与提交包对应关系。

| Milestone | Current module | Completed work | Historical artifacts |
| --- | --- | --- | --- |
| Week 1 | [simulation](../simulation/README.md) | Windows CARLA + WSL2 environment; traffic and manual-control validation | [Release](https://github.com/xuzihao723/xiaomi-auto-drive/releases/tag/week1-submission) |
| Week 2 | [data-pipeline](../data-pipeline/README.md) | Synchronized RGB/LiDAR; 1,000 KITTI frames; automated validation | [Release](https://github.com/xuzihao723/xiaomi-auto-drive/releases/tag/week2-submission) |
| Week 3 | [object-detection](../object-detection/README.md) | Four-class YOLOv8; grouped splits; specialized fusion; label review; backend benchmarks | [Release](https://github.com/xuzihao723/xiaomi-auto-drive/releases/tag/week3-submission) |
| Week 4 | [road-segmentation](../road-segmentation/README.md) | 1,300 recorded image/mask pairs; U-Net; night adaptation; geometry; 60-second fusion demo | [Release](https://github.com/xuzihao723/xiaomi-auto-drive/releases/tag/week4-submission) |

## Environment and data

The environment milestone validated CARLA 0.9.15 on Windows with a WSL2 Ubuntu 22.04 client. The data milestone collected frame-aligned RGB/LiDAR and actor metadata, then converted 1,000 frames into KITTI Object format with automated image, cloud, label and calibration checks.

## Detection and evaluation

The final detection deliverable uses Town05 and Town10HD_Opt with varied weather, lighting and seeds. It includes 40-epoch training records, scene/frame-block grouped splits with no recorded group or exact-image overlap, and class-specialized road-user / traffic-control model fusion.

Strict-test results: Precision 0.880, Recall 0.590, mAP50 0.704 and mAP50–95 0.403. A separate old 150-image independent test was manually reviewed: 414 existing traffic-control labels and 97 missing-label candidates were checked, with 92 deletions, 6 reclassifications and 36 additions recorded. PyTorch/ONNX/TensorRT notebook benchmarks remain separate from vehicle-level latency claims.

## Segmentation and integration — completed

The segmentation milestone is complete, replacing the old “in progress” entry. It includes 1,225 main image/mask pairs (800/250/175 train/val/test) and a separate 75-image locked-model audit. Fixed-test Road/Lane mIoU is 0.676; audit mIoU is 0.796. Night adaptation improved the recorded night scenario while slightly reducing the heavy-rain result. The final H.264 fusion demo contains 900 frames at 15 FPS and 640 × 360.

## Portfolio organization

Module directories replace the Week-based code layout. English/Chinese READMEs, a visual showcase, architecture and experiment evidence provide the public entry points. Original checkpoint filenames, release tags, ZIPs and experiment records are retained for traceability.
