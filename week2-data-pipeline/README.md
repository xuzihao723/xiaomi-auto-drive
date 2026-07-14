# 基于视觉的城市道路自动驾驶仿真系统

## 第二周：CARLA 仿真数据采集与 KITTI 格式转换

本项目基于 CARLA 0.9.15，在 Windows 宿主机运行 CARLA Server，在 WSL2 Ubuntu 22.04 中运行 Python Client。

第二周完成以下任务：

1. 使用 CARLA Python API 控制车辆。
2. 配置 RGB 相机和激光雷达。
3. 生成车辆和行人交通场景。
4. 同步采集图像、点云和目标标注信息。
5. 将 CARLA 原始数据转换为 KITTI Object 格式。
6. 完成1000帧仿真数据集采集和自动验收。

## 环境信息

| 项目 | 版本 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 / WSL2 |
| Python | 3.10 |
| CARLA Server | 0.9.15 Windows Package |
| CARLA Python API | 0.9.15 |
| NumPy | 1.23.5 |
| OpenCV | 4.11.0 |
| Git | 2.34+ |

## 项目结构

```text
week2/
├── configs/
│   └── sensors.json
├── src/
│   ├── 01_control_vehicle.py
│   ├── 02_sensor_test.py
│   ├── collect_raw.py
│   ├── convert_kitti.py
│   └── validate_kitti.py
├── docs/
│   ├── week2_report.md
│   └── kitti_validation_report.json
├── data/
│   ├── raw/
│   ├── kitti/
│   └── debug/
├── README.md
├── requirements.txt
└── .gitignore
```

数据集文件较大，不直接提交到普通 Git 仓库，通过单独的 ZIP 文件交付。

## CARLA Server 启动

在 Windows PowerShell 中执行：

```powershell
cd "E:\CARLA_0.9.15\WindowsNoEditor"
.\CarlaUE4.exe -quality-level=Low -windowed -ResX=1280 -ResY=720
```

## Python 环境

在 WSL2 Ubuntu 中执行：

```bash
cd ~/xiaomi_ad_project/week1
source venv/bin/activate

cd ~/xiaomi_ad_project/week2
```

安装依赖：

```bash
pip install -r requirements.txt
```

获取 Windows 宿主机地址：

```bash
export WIN_HOST=$(ip route | awk '/default/ {print $3; exit}')
echo "$WIN_HOST"
```

## 车辆控制测试

```bash
python src/01_control_vehicle.py \
  --host "$WIN_HOST" \
  --port 2000
```

## 传感器测试

```bash
python src/02_sensor_test.py \
  --host "$WIN_HOST" \
  --port 2000 \
  --config configs/sensors.json \
  --output data/sensor_test \
  --frames 20
```

## 原始数据采集

```bash
python src/collect_raw.py \
  --host "$WIN_HOST" \
  --port 2000 \
  --config configs/sensors.json \
  --output data/raw \
  --frames 1000
```

原始数据包括：

```text
data/raw/
├── image/
├── lidar/
└── meta/
```

每个样本使用六位数字编号：

```text
000000
000001
...
000999
```

## KITTI 格式转换

```bash
python src/convert_kitti.py \
  --input data/raw \
  --output data/kitti \
  --limit 1000 \
  --debug-frames 20
```

转换后的目录：

```text
data/kitti/training/
├── image_2/
├── velodyne/
├── label_2/
└── calib/
```

KITTI 标签包含15个字段：

```text
type truncated occluded alpha
bbox_left bbox_top bbox_right bbox_bottom
height width length
location_x location_y location_z
rotation_y
```

当前数据集使用的主要类别：

- `Car`
- `Pedestrian`

## 数据集验收

```bash
python src/validate_kitti.py \
  --dataset data/kitti/training \
  --expected-count 1000 \
  --debug-count 20
```

验收内容包括：

- 图像、点云、标签和标定文件数量。
- 四类文件的编号一致性。
- KITTI 标签字段数量。
- 二维包围框范围。
- 三维尺寸和相机坐标。
- 点云数据有效性。
- 标定矩阵完整性。
- 随机样本可视化。

验收报告保存在：

```text
docs/kitti_validation_report.json
```

## 数据集说明

RGB 图像配置：

- 分辨率：1280×720
- 水平视场角：90度
- 刚性安装于 Ego Vehicle

激光雷达配置：

- 32线
- 探测范围：50米
- 点频：100000点/秒
- 旋转频率：20Hz
- 上视场角：10度
- 下视场角：-30度

仿真采用同步模式和固定时间步：

```text
fixed_delta_seconds = 0.05
```

对应20 FPS。

## 已知限制

1. KITTI `occluded` 字段暂时统一设置为3，表示未知。
2. 项目使用单目相机，P0、P1、P2、P3采用相同的零基线投影矩阵。
3. 二维包围框由CARLA三维包围框投影得到。
4. 数据集用于课程学习与仿真实验，不等同于真实道路采集数据。

## 参考资料

- [CARLA 0.9.15 官方文档](https://carla.readthedocs.io/en/0.9.15/)
- [CARLA Python API](https://carla.readthedocs.io/en/0.9.15/python_api/)
- [CARLA Sensors and Data](https://carla.readthedocs.io/en/0.9.15/core_sensors/)
- [CARLA Synchrony and Time-step](https://carla.readthedocs.io/en/0.9.15/adv_synchrony_timestep/)
- [CARLA Bounding Boxes](https://carla.readthedocs.io/en/0.9.15/tuto_G_bounding_boxes/)
- [KITTI Object Detection Benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php)
