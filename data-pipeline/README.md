# Data pipeline

**English** · [简体中文](README.zh-CN.md) · [Project overview](../README.md)

Synchronize RGB images, LiDAR point clouds and CARLA actor metadata, then convert projected annotations to KITTI Object format. The recorded milestone contains 1,000 validated frames. The dataset is simulated, not real-world camera data.

## Pipeline

```text
CARLA synchronous world → frame-aligned RGB / LiDAR / actors
                      → projection and KITTI conversion
                      → counts, IDs, labels and calibration validation
```

## Setup

Start CARLA Server 0.9.15 on Windows. From the cloned repository in WSL2 Ubuntu 22.04 / Python 3.10:

```bash
cd data-pipeline
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export WIN_HOST=$(ip route | awk '/default/ {print $3; exit}')
```

## Collect and convert

```bash
python src/collect_raw.py \
  --host "$WIN_HOST" --port 2000 \
  --config configs/sensors.json --output data/raw --frames 1000
python src/convert_kitti.py \
  --input data/raw --output data/kitti --limit 1000 --debug-frames 20
python src/validate_kitti.py \
  --dataset data/kitti/training --expected-count 1000 --debug-count 20
```

The validator checks file IDs/counts, KITTI label fields, boxes, 3D dimensions, point clouds and calibration. Data is ignored by Git. The recorded sensor configuration uses 1280 × 720 RGB, a 32-channel LiDAR and a 0.05-second simulation timestep.

## Evidence and boundaries

- [Recorded data validation](docs/kitti_validation_report.json)
- [Detailed experiment guide (Chinese)](README.zh-CN.md)
- [Historical dataset release](https://github.com/xuzihao723/xiaomi-auto-drive/releases/download/week2-submission/xiaomi_week2.zip)

KITTI occlusion is recorded as unknown (`3`); the monocular setup uses the same zero-baseline projection for P0–P3. Boxes are projected from simulator ground truth. Continue with [object detection](../object-detection/README.md) for the grouped four-class dataset.
