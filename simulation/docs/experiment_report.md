# 第一周实验报告

**项目名称：** 小米汽车智能驾驶项目：基于视觉的城市道路端到端自动驾驶仿真系统  
**周次：** 第 1 周  
**阶段：** 环境搭建与基础认知  
**实验主题：** Ubuntu/WSL2 + Python 3.10 环境配置、CARLA 0.9.15 启动与官方示例运行  

---

## 一、实验目的

1. 了解小米汽车智能驾驶技术路线与核心产品。
2. 完成 Ubuntu 22.04 / WSL2 与 Python 3.10 开发环境配置。
3. 完成 CARLA 0.9.15 自动驾驶仿真环境部署，并验证 CARLA Server 与 Python Client 可以正常通信。
4. 运行 CARLA 官方示例，熟悉仿真器的基本启动、交通流生成和车辆控制操作。

---

## 二、实验环境

| 项目 | 版本或配置 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS |
| 运行方式 | Windows Subsystem for Linux 2（WSL2） |
| WSL2 内核 | 6.6.87.2-microsoft-standard-WSL2 |
| Python | 3.10.12 |
| pip | 26.1.2 |
| Git | 2.34.1 |
| CARLA Server | 0.9.15 Windows Package |
| CARLA Python API | 0.9.15 |
| 主要依赖 | pygame 2.6.1、NumPy 1.23.5、Matplotlib 3.10.9、OpenCV 4.11.0.86 |

---

## 三、实验内容与过程

### 3.1 小米智能驾驶资料学习

本周首先根据项目要求阅读小米汽车官网与辅助驾驶相关页面，梳理小米辅助驾驶的技术定位、核心功能、硬件基础和安全边界。学习结果整理为《小米智能驾驶技术分析笔记》，重点包括 XLA 认知大模型、多场景辅助驾驶、多传感器融合以及辅助驾驶不能替代驾驶员操控等内容。

### 3.2 Ubuntu/WSL2 与 Python 环境配置

在 Windows 环境中使用 WSL2 安装 Ubuntu 22.04，并确认 Python 3.10、pip、Git 等开发工具可用。随后在项目目录中创建 Python 虚拟环境，并安装 CARLA Python API 0.9.15 以及 pygame、NumPy、Matplotlib、OpenCV 等基础依赖。

### 3.3 CARLA 0.9.15 部署

CARLA 采用 Client/Server 架构。本次实验中，CARLA 0.9.15 Server 在 Windows 宿主机运行，WSL2 Ubuntu 中的 Python Client 通过 TCP 端口连接 Server。该部署方式解决了 WSL2 中图形仿真运行不稳定的问题，同时保证 Python 开发环境仍在 Ubuntu 中完成。

| 组成部分 | 作用 |
|---|---|
| Windows 宿主机 | 运行 CARLA 0.9.15 Server，监听 TCP 端口 2000 |
| WSL2 Ubuntu 22.04 | 运行 Python 3.10 虚拟环境与 CARLA Python Client 0.9.15 |
| 连接方式 | 在 WSL2 中获取 Windows 宿主机 IP，并由 Python Client 连接 CARLA Server |

### 3.4 CARLA 官方示例运行

启动 CARLA Server 后，在 WSL2 中运行官方示例脚本，完成交通流生成和手动控制验证。连接验证结果显示 Client version 与 Server version 均为 0.9.15，当前地图为 Town10HD_Opt，Actors 数量为 23，说明 Python Client 已成功读取仿真世界信息。

---

## 四、实验结果

| 交付项 | 完成情况 |
|---|---|
| 小米智能驾驶技术分析笔记 | 文件为 `docs/xiaomi_autonomous_driving_note.md` |
| 开发环境搭建文档 | 文件为 `docs/environment_setup.md` |
| CARLA Server 启动 | Windows CARLA 0.9.15 Server 可正常启动 |
| Python Client 连接 | WSL2 Python Client 成功连接 Server，Client/Server 版本均为 0.9.15 |
| 官方示例运行 | 运行 `generate_traffic.py` 与 `manual_control.py` |
| 截图与演示视频 | 包含 3 张运行截图与 30 秒演示视频 `carla_demo_30s.mp4` |

---

## 五、问题与解决方法

1. **问题 1：WSL2 与 CARLA 图形仿真运行方式不完全一致。**  
   解决方法：采用 Windows 宿主机运行 CARLA Server、WSL2 运行 Python Client 的 Client/Server 分离方式，通过 TCP 端口进行连接。
3. **问题 3：WSL2 需要连接 Windows 宿主机 IP。**  
   解决方法：在 WSL2 中通过默认路由获取 Windows 宿主机地址，并将其作为 CARLA Client 的连接目标。

---

## 六、实验总结

本周实验完成了项目第一阶段要求的环境搭建与基础认知任务。通过阅读小米汽车官网资料，我对小米智能驾驶的技术定位、核心功能、多传感器融合和安全边界有了初步理解；通过搭建 Ubuntu 22.04 / WSL2、Python 3.10 和 CARLA 0.9.15 环境，我验证了仿真器 Server 与 Python Client 的通信流程，并成功运行官方示例。

本周成果为后续工作奠定了基础。第二周可以在已完成的 CARLA 环境上继续学习 Python API，编写车辆与传感器控制脚本，并进一步采集图像和标注数据，为目标检测、车道线识别和后续规划控制模块做准备。



---

## 七、参考资料

1. 小米汽车官方网站：https://www.xiaomiev.com/
2. 小米汽车官网核心技术-辅助驾驶页面：https://www.xiaomiev.com/pilot
3. CARLA 0.9.15 官方文档：https://carla.readthedocs.io/en/0.9.15/
4. CARLA 0.9.15 Release：https://github.com/carla-simulator/carla/releases/tag/0.9.15
