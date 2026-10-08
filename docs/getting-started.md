# Getting started

[English overview](../README.md) · [中文项目首页](../README.zh-CN.md)

## Choose a starting point

- **Inspect results:** browse the README images, [experiment evidence](experiments.md) and release packages. No simulator or GPU is required.
- **Run saved-weight inference:** install the perception environment, restore the published weights and supply input RGB images.
- **Reproduce data and training:** also run CARLA 0.9.15 and regenerate the published scene configurations. Saved statistics alone are not training data.

## Directory migration

| Previous repository directory | Current directory | Historical release tag |
| --- | --- | --- |
| `week1-environment/` | `simulation/` | `week1-submission` |
| `week2-data-pipeline/` | `data-pipeline/` | `week2-submission` |
| `week3-perception/` | `object-detection/` | `week3-submission` |
| `week4-segmentation/` | `road-segmentation/` | `week4-submission` |

Release tags, ZIP names, checkpoint names and original experiment records retain their historical identifiers. Copy only the needed data/weights into the new module paths; do not overwrite the current source tree with an entire historical ZIP.

## Environment

The recorded runs used Windows 11 with CARLA Server, WSL2 Ubuntu 22.04 and Python 3.10. The detection training record specifies PyTorch 2.10.0+cu126. The module requirements provide the Python dependencies; configure a CUDA-compatible PyTorch build for your own GPU before GPU reproduction. ONNX/TensorRT are optional benchmark backends, not prerequisites for viewing results or standard PyTorch inference.

```bash
git clone https://github.com/xuzihao723/xiaomi-auto-drive.git
cd xiaomi-auto-drive/road-segmentation
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Commands below assume the current working directory is `road-segmentation/`. Use the [Chinese segmentation guide](../road-segmentation/README.zh-CN.md) for collection, training and recovery details.

## Restore published weights

Download and extract the [detection package](https://github.com/xuzihao723/xiaomi-auto-drive/releases/download/week3-submission/xiaomi_week3.zip) and [segmentation package](https://github.com/xuzihao723/xiaomi-auto-drive/releases/download/week4-submission/xiaomi_week4.zip) outside the checkout. Locate their `weights/` folders, then copy these files:

| Published file | Destination relative to repository root |
| --- | --- |
| `road_user_best.pt` | `object-detection/weights/road_user_best.pt` |
| `traffic_control_best.pt` | `object-detection/weights/traffic_control_best.pt` |
| `unet_week4_best.pt` | `road-segmentation/weights/unet_week4_best.pt` |

The [segmentation weight note](../road-segmentation/weights/README.md) includes its SHA-256; verify it after extraction. Weights remain ignored by Git.

## Run one segmentation image

Replace `/path/to/input.png` with an actual RGB image; an arbitrary image does not reproduce the published test metrics.

```bash
python segmentation/inference.py \
  --weights weights/unet_week4_best.pt \
  --source /path/to/input.png \
  --output outputs/example_overlay.png \
  --device cpu
```

Use `--device cuda` for GPU inference. The script writes an overlay and geometry JSON. It does not require a running CARLA Server for an existing image.

## Regenerate the segmentation dataset

Raw segmentation images/masks are not included in either Git or the published segmentation ZIP. Start CARLA Server 0.9.15 on Windows and use the scene configuration from this checkout:

```bash
export WIN_HOST=$(ip route | awk '/default/ {print $3; exit}')
python segmentation/collect_carla_segmentation.py \
  --host "$WIN_HOST" --port 2000 \
  --config configs/segmentation_scenarios.json \
  --output data/carla_segmentation
python tools/validate_segmentation_dataset.py \
  --data data/carla_segmentation \
  --output outputs/dataset_validation.json
```

The configuration controls maps, weather, seeds and splits. Regenerated simulations may differ from the recorded run; use the documented evaluation protocol rather than assuming identical scores.

## Render the integrated demo

The script needs at least 900 RGB images for `--fps 15 --seconds 60`. Supply the regenerated image directory or another explicitly selected input sequence. The collector's directory can contain different scenes/splits; the rendered montage is not a continuous closed-loop driving trial.

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

The script's OpenCV writer uses `mp4v`; the published H.264/yuv420p video is a separately encoded artifact. New output files may therefore have a different codec unless re-encoded. Generated outputs are deliberately separate from the archived `reports/` evidence.

## Detection training and evaluation

Run detection commands from `object-detection/`, following its [English guide](../object-detection/README.md) or [Chinese guide](../object-detection/README.zh-CN.md). Restore/regenerate the grouped dataset under `object-detection/data/yolo_grouped/` before training or evaluating its test split. Input images and weights are prerequisites; cloning the repository alone does not supply them.
