# 开发环境搭建

## 1. 搭建目标

本周完成 Ubuntu 22.04/WSL2、Python 3.10 虚拟环境和 CARLA 0.9.15 仿真环境的搭建，并验证 WSL2 中的 Python Client 能够连接和控制 Windows 宿主机运行的 CARLA Server。

## 2. 环境信息

| 项目 | 版本或配置 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS |
| 运行方式 | Windows Subsystem for Linux 2（WSL2） |
| WSL2 内核 | 6.6.87.2-microsoft-standard-WSL2 |
| Python | 3.10.12 |
| pip | 26.1.2 |
| Git | 2.34.1 |
| CARLA Server | 0.9.15 Windows Package |
| CARLA Python API | 0.9.15 |
| pygame | 2.6.1 |
| NumPy | 1.23.5 |
| Matplotlib | 3.10.9 |
| OpenCV | 4.11.0.86 |

## 3. 项目目录

项目根目录为：

```text
~/xiaomi_ad_project/week1
```

目录结构如下：

```text
week1/
├── docs/
│   ├── environment_setup.md
│   └── xiaomi_autonomous_driving_note.md
│   └── experiment_report.md
├── screenshots/
├── demo_video/
├── venv/
└── carla/
```

## 4. Ubuntu 基础环境

在 Ubuntu 22.04 终端安装基础工具：

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git wget unzip curl ffmpeg
```

版本检查命令：

```bash
lsb_release -a
python3 --version
pip3 --version
git --version
```

## 5. Python 虚拟环境

创建并激活 Python 虚拟环境：

```bash
cd ~/xiaomi_ad_project/week1
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

安装 CARLA Python API 和基础依赖：

```bash
pip install carla==0.9.15 pygame numpy==1.23.5 matplotlib opencv-python
```

验证 Python 包：

```bash
python -c "import carla, numpy, pygame, cv2, matplotlib; print('Python dependencies OK')"
```

## 6. CARLA 部署方式

CARLA 采用 Client/Server 架构。本项目最终部署方式如下：

```text
Windows 宿主机
└── CARLA 0.9.15 Server
    └── TCP 端口 2000

WSL2 Ubuntu 22.04
└── Python 3.10 虚拟环境
    └── CARLA Python Client 0.9.15
```

Windows CARLA Server 安装目录：

```text
E:\CARLA_0.9.15\WindowsNoEditor
```

在 Windows PowerShell 中启动 Server：

```powershell
cd "E:\CARLA_0.9.15\WindowsNoEditor"
.\CarlaUE4.exe -quality-level=Low -windowed -ResX=1280 -ResY=720
```

## 7. Client/Server 连接验证

在 WSL2 中获取 Windows 宿主机地址：

```bash
WIN_HOST=$(ip route | awk '/default/ {print $3; exit}')
export WIN_HOST
```

连接测试结果：

```text
Windows IP: 192.168.176.1
Connected to CARLA
Client version: 0.9.15
Server version: 0.9.15
Map: Carla/Maps/Town10HD_Opt
Actors: 23
```

该结果说明 WSL2 中的 Python Client 已成功连接 Windows CARLA Server，客户端与服务端版本一致。

## 8. 结论

Ubuntu 22.04、Python 3.10、Git、CARLA Python API 及相关依赖均已完成配置。CARLA 0.9.15 Server 可以在 Windows 宿主机正常启动，WSL2 Python Client 能够连接 Town10HD_Opt 地图并读取仿真世界信息，开发和仿真环境搭建完成。

## 9. 参考资料

- [CARLA 0.9.15 官方文档](https://carla.readthedocs.io/en/0.9.15/)
- [CARLA 0.9.15 Release](https://github.com/carla-simulator/carla/releases/tag/0.9.15)
- [CARLA Quick Start](https://carla.readthedocs.io/en/0.9.15/start_quickstart/)
