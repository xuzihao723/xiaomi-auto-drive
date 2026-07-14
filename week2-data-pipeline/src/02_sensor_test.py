import argparse
import json
import queue
import random
import time
from pathlib import Path

import carla
import numpy as np


def clear_sensor_queue(sensor_queue):
    """清除传感器队列中的旧数据。"""
    while True:
        try:
            sensor_queue.get_nowait()
        except queue.Empty:
            break


def get_sensor_data(
    sensor_queue,
    target_frame,
    sensor_name,
    timeout=30.0
):
    """取得与目标仿真帧对应的传感器数据。"""
    deadline = time.time() + timeout

    while True:
        remaining = deadline - time.time()

        if remaining <= 0:
            raise RuntimeError(
                f"{sensor_name} 等待超时，"
                f"目标帧为 {target_frame}"
            )

        try:
            data = sensor_queue.get(
                timeout=remaining
            )
        except queue.Empty as exc:
            raise RuntimeError(
                f"{sensor_name} 等待超时，"
                f"目标帧为 {target_frame}"
            ) from exc

        if data.frame < target_frame:
            print(
                f"{sensor_name} 丢弃旧帧："
                f"{data.frame}，"
                f"目标帧：{target_frame}"
            )
            continue

        if data.frame > target_frame:
            raise RuntimeError(
                f"{sensor_name} 出现跳帧："
                f"收到 {data.frame}，"
                f"目标 {target_frame}"
            )

        return data


def main():
    parser = argparse.ArgumentParser(
        description=(
            "CARLA 0.9.15 camera "
            "and LiDAR sensor test"
        )
    )
    parser.add_argument(
        "--host",
        required=True
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000
    )
    parser.add_argument(
        "--tm-port",
        type=int,
        default=8000
    )
    parser.add_argument(
        "--config",
        default="configs/sensors.json"
    )
    parser.add_argument(
        "--output",
        default="data/sensor_test"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=20
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output)
    image_dir = output_dir / "image"
    lidar_dir = output_dir / "lidar"

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    lidar_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件：{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    client = None
    world = None
    traffic_manager = None
    original_settings = None

    vehicle = None
    camera = None
    lidar = None

    try:
        print("正在连接 CARLA Server...")

        client = carla.Client(
            args.host,
            args.port
        )
        client.set_timeout(30.0)

        world = client.get_world()

        print(
            "Client version:",
            client.get_client_version()
        )
        print(
            "Server version:",
            client.get_server_version()
        )

        current_map = (
            world.get_map()
            .name
            .split("/")[-1]
        )
        target_map = config["map"]

        if current_map != target_map:
            print(
                f"切换地图："
                f"{current_map} -> {target_map}"
            )

            world = client.load_world(
                target_map
            )

            time.sleep(2)

        print(
            "Current map:",
            world.get_map().name
        )

        original_settings = (
            world.get_settings()
        )

        # 开启同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(
            config["fixed_delta_seconds"]
        )
        world.apply_settings(settings)

        traffic_manager = (
            client.get_trafficmanager(
                args.tm_port
            )
        )
        traffic_manager.set_synchronous_mode(
            True
        )
        traffic_manager.set_random_device_seed(
            int(config["seed"])
        )

        blueprint_library = (
            world.get_blueprint_library()
        )

        # ------------------------------
        # 创建 Ego Vehicle
        # ------------------------------
        vehicle_blueprints = list(
            blueprint_library.filter(
                "vehicle.*"
            )
        )
        spawn_points = list(
            world.get_map()
            .get_spawn_points()
        )

        random.seed(int(config["seed"]))
        random.shuffle(spawn_points)

        for spawn_point in spawn_points:
            vehicle_bp = random.choice(
                vehicle_blueprints
            )

            if vehicle_bp.has_attribute(
                "role_name"
            ):
                vehicle_bp.set_attribute(
                    "role_name",
                    "hero"
                )

            vehicle = world.try_spawn_actor(
                vehicle_bp,
                spawn_point
            )

            if vehicle is not None:
                break

        if vehicle is None:
            raise RuntimeError(
                "Ego Vehicle 生成失败"
            )

        vehicle.set_autopilot(
            True,
            args.tm_port
        )

        print(
            "Ego Vehicle:",
            vehicle.type_id
        )

        # ------------------------------
        # 创建 RGB Camera
        # ------------------------------
        camera_config = config["camera"]

        camera_bp = blueprint_library.find(
            "sensor.camera.rgb"
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(camera_config["width"])
        )
        camera_bp.set_attribute(
            "image_size_y",
            str(camera_config["height"])
        )
        camera_bp.set_attribute(
            "fov",
            str(camera_config["fov"])
        )

        # 0.0 表示每个仿真 tick 生成数据
        camera_bp.set_attribute(
            "sensor_tick",
            "0.0"
        )

        camera_transform = carla.Transform(
            carla.Location(
                x=float(camera_config["x"]),
                y=float(camera_config["y"]),
                z=float(camera_config["z"])
            ),
            carla.Rotation(
                pitch=0.0,
                yaw=0.0,
                roll=0.0
            )
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle,
            attachment_type=(
                carla.AttachmentType.Rigid
            )
        )

        print("RGB Camera 创建成功")

        # ------------------------------
        # 创建 LiDAR
        # ------------------------------
        lidar_config = config["lidar"]

        lidar_bp = blueprint_library.find(
            "sensor.lidar.ray_cast"
        )

        lidar_bp.set_attribute(
            "channels",
            str(lidar_config["channels"])
        )
        lidar_bp.set_attribute(
            "range",
            str(lidar_config["range"])
        )
        lidar_bp.set_attribute(
            "points_per_second",
            str(
                lidar_config[
                    "points_per_second"
                ]
            )
        )
        lidar_bp.set_attribute(
            "rotation_frequency",
            str(
                lidar_config[
                    "rotation_frequency"
                ]
            )
        )
        lidar_bp.set_attribute(
            "upper_fov",
            str(lidar_config["upper_fov"])
        )
        lidar_bp.set_attribute(
            "lower_fov",
            str(lidar_config["lower_fov"])
        )

        # 0.0 表示每个仿真 tick 生成数据
        lidar_bp.set_attribute(
            "sensor_tick",
            "0.0"
        )

        lidar_transform = carla.Transform(
            carla.Location(
                x=float(lidar_config["x"]),
                y=float(lidar_config["y"]),
                z=float(lidar_config["z"])
            ),
            carla.Rotation(
                pitch=0.0,
                yaw=0.0,
                roll=0.0
            )
        )

        lidar = world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=vehicle,
            attachment_type=(
                carla.AttachmentType.Rigid
            )
        )

        print("LiDAR 创建成功")

        # ------------------------------
        # 创建数据队列
        # ------------------------------
        camera_queue = queue.Queue()
        lidar_queue = queue.Queue()

        camera.listen(camera_queue.put)
        lidar.listen(lidar_queue.put)

        # ------------------------------
        # 传感器预热
        # ------------------------------
        warmup_frames = int(
            config.get(
                "warmup_frames",
                20
            )
        )

        print(
            f"正在预热传感器："
            f"{warmup_frames} 帧"
        )

        for index in range(
            warmup_frames
        ):
            frame = world.tick()

            print(
                f"预热 "
                f"{index + 1}/"
                f"{warmup_frames}，"
                f"world frame={frame}"
            )

        # 等待GPU相机完成积压处理
        time.sleep(3)

        clear_sensor_queue(
            camera_queue
        )
        clear_sensor_queue(
            lidar_queue
        )

        print(
            "预热完成，开始保存数据"
        )

        # ------------------------------
        # 正式采集
        # ------------------------------
        for sample_index in range(
            args.frames
        ):
            frame = world.tick()

            image_data = get_sensor_data(
                camera_queue,
                frame,
                "RGB Camera"
            )

            lidar_data = get_sensor_data(
                lidar_queue,
                frame,
                "LiDAR"
            )

            sample_name = (
                f"{sample_index:06d}"
            )

            image_path = (
                image_dir /
                f"{sample_name}.png"
            )
            lidar_path = (
                lidar_dir /
                f"{sample_name}.npy"
            )

            image_data.save_to_disk(
                str(image_path)
            )

            lidar_points = np.frombuffer(
                lidar_data.raw_data,
                dtype=np.float32
            ).copy()

            lidar_points = (
                lidar_points.reshape(
                    -1,
                    4
                )
            )

            np.save(
                str(lidar_path),
                lidar_points
            )

            print(
                f"[{sample_index + 1:03d}/"
                f"{args.frames:03d}] "
                f"world frame={frame}, "
                f"camera frame="
                f"{image_data.frame}, "
                f"lidar frame="
                f"{lidar_data.frame}, "
                f"lidar points="
                f"{len(lidar_points)}"
            )

        print()
        print("传感器测试完成")
        print(
            "图像目录：",
            image_dir
        )
        print(
            "点云目录：",
            lidar_dir
        )

    except Exception as exc:
        print()
        print("运行失败：", exc)

    finally:
        print()
        print(
            "正在清理传感器和车辆..."
        )

        if camera is not None:
            try:
                camera.stop()
                camera.destroy()
            except RuntimeError:
                pass

        if lidar is not None:
            try:
                lidar.stop()
                lidar.destroy()
            except RuntimeError:
                pass

        if vehicle is not None:
            try:
                vehicle.destroy()
            except RuntimeError:
                pass

        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(
                    False
                )
            except RuntimeError:
                pass

        if (
            world is not None
            and original_settings is not None
        ):
            try:
                world.apply_settings(
                    original_settings
                )
            except RuntimeError:
                pass

        print("清理完成")


if __name__ == "__main__":
    main()
