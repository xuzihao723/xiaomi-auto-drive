# Road segmentation and visual integration

**English** · [简体中文](README.zh-CN.md) · [Project overview](../README.md)

A lightweight U-Net segments **Background, DrivableArea and LaneMarking** in CARLA RGB images. Geometric postprocessing extracts a drivable polygon and quadratic lane curves; the integration script renders these together with specialized YOLOv8 detections.

## Published results

| Set | Images | Road IoU | Lane IoU | Road/Lane mIoU |
| --- | ---: | ---: | ---: | ---: |
| Fixed test after night adaptation | 175 | 0.749 | 0.603 | **0.676** |
| Separate locked-model audit | 75 | 0.877 | 0.715 | **0.796** |

The main dataset uses 800/250/175 train/validation/test pairs; the audit brings the recorded total to 1,300. The audit is a separate cloudy-night scenario and does not replace the fixed test. Night test mIoU improves from 0.004 to 0.778; heavy-rain test mIoU changes from 0.659 to 0.637.

![Published fusion video preview](reports/integrated_video_contact_sheet.jpg)

## Start with the published weights

Follow the [setup and artifact placement guide](../docs/getting-started.md). Download the [segmentation package](https://github.com/xuzihao723/xiaomi-auto-drive/releases/download/week4-submission/xiaomi_week4.zip) and place `unet_week4_best.pt` in `weights/`. Restore the two detection weights under `../object-detection/weights/` for integration.

Raw segmentation image/mask pairs are not included in Git or the release; regenerate them with `segmentation/collect_carla_segmentation.py` and `configs/segmentation_scenarios.json`.

From `road-segmentation/`, after activating the perception environment:

```bash
python segmentation/inference.py \
  --weights weights/unet_week4_best.pt \
  --source /path/to/input.png --output outputs/example_overlay.png --device cpu
```

For an integrated 60-second demo, supply at least 900 RGB images:

```bash
python integration/integrated_demo.py \
  --source data/carla_segmentation/images \
  --segmentation-weights weights/unet_week4_best.pt \
  --road-user-weights ../object-detection/weights/road_user_best.pt \
  --traffic-control-weights ../object-detection/weights/traffic_control_best.pt \
  --output outputs/integrated_perception.mp4 \
  --summary outputs/integrated_video_summary.json \
  --fps 15 --seconds 60 --device cuda
```

Output FPS is the video playback rate. The script's OpenCV `mp4v` output may differ from the separately encoded published H.264/yuv420p video. Integration is a rendered perception montage, not a closed-loop driving benchmark.

## Evidence and documentation

- [Fixed-test metrics](reports/test_evaluation_night_adapt/test_metrics.json)
- [Separate audit metrics](reports/audit_evaluation/audit_metrics.json)
- [Night adaptation comparison](reports/night_adaptation_comparison.json)
- [Published video summary](reports/integrated_video_summary.json)
- [Final weight checksum](weights/README.md) / [video download and checksum](demo/README.md)
- [Detailed Chinese collection, training and recovery guide](README.zh-CN.md)

Lane fitting remains a geometric heuristic and needs temporal constraints for occlusion and extreme curvature. Semantic IDs are read from the installed CARLA API at runtime.
