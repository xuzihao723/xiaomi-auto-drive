import argparse
import json
import math
import queue
import random
import time
from pathlib import Path

import carla
import numpy as np


def clear_queue(sensor_queue):
    if sensor_queue is None:
        return

    while True:
        try:
            sensor_queue.get_nowait()
        except queue.Empty:
            break


def get_sensor_data(sensor_queue, target_frame, sensor_name, timeout=30.0):
    deadline = time.time() + timeout

    while True:
        remaining = deadline - time.time()

        if remaining <= 0:
            raise RuntimeError(
                f"{sensor_name} 等待帧 {target_frame} 超时"
            )

        try:
            data = sensor_queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise RuntimeError(
                f"{sensor_name} 等待帧 {target_frame} 超时"
            ) from exc

        if data.frame < target_frame:
            continue

        if data.frame > target_frame:
            raise RuntimeError(
                f"{sensor_name} 跳帧："
                f"目标={target_frame}，收到={data.frame}"
            )

        return data


def location_to_dict(location):
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z)
    }


def rotation_to_dict(rotation):
    return {
        "pitch": float(rotation.pitch),
        "yaw": float(rotation.yaw),
        "roll": float(rotation.roll)
    }


def transform_to_dict(transform):
    return {
        "location": location_to_dict(transform.location),
        "rotation": rotation_to_dict(transform.rotation)
    }


def vector_to_dict(vector):
    return {
        "x": float(vector.x),
        "y": float(vector.y),
        "z": float(vector.z)
    }


def bounding_box_to_dict(bounding_box):
    return {
        "location": location_to_dict(bounding_box.location),
        "extent": vector_to_dict(bounding_box.extent),
        "rotation": rotation_to_dict(bounding_box.rotation)
    }


def build_camera_intrinsic(width, height, fov):
    focal = width / (
        2.0 * math.tan(math.radians(fov) / 2.0)
    )

    return [
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0]
    ]


def get_actor_class(type_id):
    if type_id.startswith("vehicle."):
        return "Car"

    if type_id.startswith("walker.pedestrian."):
        return "Pedestrian"

    return None


def collect_actor_metadata(world, ego_vehicle, maximum_distance=80.0):
    records = []
    ego_location = ego_vehicle.get_location()

    for actor in world.get_actors():
        if actor.id == ego_vehicle.id:
            continue

        class_name = get_actor_class(actor.type_id)

        if class_name is None:
            continue

        try:
            if not actor.is_alive:
                continue

            actor_transform = actor.get_transform()
            distance = actor_transform.location.distance(ego_location)

            if distance > maximum_distance:
                continue

            records.append({
                "id": int(actor.id),
                "type_id": actor.type_id,
                "class_name": class_name,
                "distance_to_ego": float(distance),
                "transform": transform_to_dict(actor_transform),
                "velocity": vector_to_dict(actor.get_velocity()),
                "bounding_box": bounding_box_to_dict(
                    actor.bounding_box
                )
            })

        except RuntimeError:
            continue

    return records


def choose_vehicle_blueprint(blueprint_library, rng):
    blueprints = list(
        blueprint_library.filter("vehicle.*")
    )

    if not blueprints:
        raise RuntimeError("没有找到车辆 Blueprint")

    return rng.choice(blueprints)


def spawn_ego_vehicle(world, blueprint_library, rng):
    spawn_points = list(
        world.get_map().get_spawn_points()
    )
    rng.shuffle(spawn_points)

    try:
        preferred_blueprint = blueprint_library.find(
            "vehicle.tesla.model3"
        )
    except RuntimeError:
        preferred_blueprint = None

    for spawn_point in spawn_points:
        if preferred_blueprint is not None:
            vehicle_bp = preferred_blueprint
        else:
            vehicle_bp = choose_vehicle_blueprint(
                blueprint_library,
                rng
            )

        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "hero")

        vehicle = world.try_spawn_actor(
            vehicle_bp,
            spawn_point
        )

        if vehicle is not None:
            return vehicle

    raise RuntimeError("无法生成 Ego Vehicle")


def spawn_npc_vehicles(
    world,
    blueprint_library,
    count,
    rng,
    tm_port
):
    vehicles = []
    spawn_points = list(
        world.get_map().get_spawn_points()
    )
    rng.shuffle(spawn_points)

    for spawn_point in spawn_points:
        if len(vehicles) >= count:
            break

        vehicle_bp = choose_vehicle_blueprint(
            blueprint_library,
            rng
        )

        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute(
                "role_name",
                "autopilot"
            )

        if vehicle_bp.has_attribute("color"):
            colors = vehicle_bp.get_attribute(
                "color"
            ).recommended_values

            if colors:
                vehicle_bp.set_attribute(
                    "color",
                    rng.choice(colors)
                )

        vehicle = world.try_spawn_actor(
            vehicle_bp,
            spawn_point
        )

        if vehicle is not None:
            vehicle.set_autopilot(True, tm_port)
            vehicles.append(vehicle)

    return vehicles


def spawn_walkers(
    world,
    blueprint_library,
    count,
    rng
):
    walkers = []
    controllers = []
    walker_speeds = {}
    controller_speeds = {}

    walker_blueprints = list(
        blueprint_library.filter(
            "walker.pedestrian.*"
        )
    )

    controller_bp = blueprint_library.find(
        "controller.ai.walker"
    )

    attempts = 0
    maximum_attempts = max(count * 5, 20)

    while (
        len(walkers) < count
        and attempts < maximum_attempts
    ):
        attempts += 1

        location = (
            world.get_random_location_from_navigation()
        )

        if location is None:
            continue

        walker_bp = rng.choice(walker_blueprints)

        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute(
                "is_invincible",
                "false"
            )

        speed = 1.4

        if walker_bp.has_attribute("speed"):
            speeds = walker_bp.get_attribute(
                "speed"
            ).recommended_values

            if len(speeds) > 1:
                speed = float(speeds[1])

        walker = world.try_spawn_actor(
            walker_bp,
            carla.Transform(location)
        )

        if walker is not None:
            walkers.append(walker)
            walker_speeds[walker.id] = speed

    world.tick()

    for walker in walkers:
        try:
            controller = world.spawn_actor(
                controller_bp,
                carla.Transform(),
                attach_to=walker
            )

            controllers.append(controller)
            controller_speeds[controller.id] = (
                walker_speeds.get(walker.id, 1.4)
            )

        except RuntimeError:
            continue

    world.tick()

    for controller in controllers:
        try:
            controller.start()

            destination = (
                world.get_random_location_from_navigation()
            )

            if destination is not None:
                controller.go_to_location(destination)

            controller.set_max_speed(
                controller_speeds.get(
                    controller.id,
                    1.4
                )
            )

        except RuntimeError:
            continue

    return walkers, controllers


def save_metadata(
    output_path,
    sample_index,
    world_frame,
    image_data,
    lidar_data,
    ego_vehicle,
    actors,
    config
):
    camera_config = config["camera"]

    intrinsic = build_camera_intrinsic(
        int(camera_config["width"]),
        int(camera_config["height"]),
        float(camera_config["fov"])
    )

    metadata = {
        "sample_index": int(sample_index),
        "sample_name": f"{sample_index:06d}",
        "carla_frame": int(world_frame),
        "timestamp": float(image_data.timestamp),
        "map": config["map"],
        "camera": {
            "width": int(camera_config["width"]),
            "height": int(camera_config["height"]),
            "fov": float(camera_config["fov"]),
            "intrinsic": intrinsic,
            "transform": transform_to_dict(
                image_data.transform
            )
        },
        "lidar": {
            "transform": transform_to_dict(
                lidar_data.transform
            )
        },
        "ego_vehicle": {
            "id": int(ego_vehicle.id),
            "type_id": ego_vehicle.type_id,
            "transform": transform_to_dict(
                ego_vehicle.get_transform()
            ),
            "velocity": vector_to_dict(
                ego_vehicle.get_velocity()
            )
        },
        "actors": actors
    }

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )


def get_alive_actor_ids(actor_list):
    actor_ids = []

    for actor in actor_list:
        if actor is None:
            continue

        try:
            if actor.is_alive:
                actor_ids.append(actor.id)
        except RuntimeError:
            continue

    return actor_ids


def batch_destroy(client, actors, group_name):
    actor_ids = get_alive_actor_ids(actors)

    if not actor_ids:
        return

    print(
        f"正在销毁{group_name}："
        f"{len(actor_ids)} 个"
    )

    commands = [
        carla.command.DestroyActor(actor_id)
        for actor_id in actor_ids
    ]

    responses = client.apply_batch_sync(
        commands,
        True
    )

    for response in responses:
        if response.has_error():
            print(
                "清理警告：",
                response.error
            )


def main():
    parser = argparse.ArgumentParser(
        description="CARLA 0.9.15 raw dataset collector"
    )
    parser.add_argument("--host", required=True)
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
        default="data/raw"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output)

    image_dir = output_dir / "image"
    lidar_dir = output_dir / "lidar"
    meta_dir = output_dir / "meta"

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    lidar_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    meta_dir.mkdir(
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

    if args.frames is None:
        capture_frames = int(
            config["capture_frames"]
        )
    else:
        capture_frames = args.frames

    rng = random.Random(
        int(config["seed"])
    )

    client = None
    world = None
    traffic_manager = None
    original_settings = None

    ego_vehicle = None
    npc_vehicles = []
    walkers = []
    walker_controllers = []

    camera = None
    lidar = None
    camera_queue = None
    lidar_queue = None

    try:
        print("正在连接 CARLA Server...")

        client = carla.Client(
            args.host,
            args.port
        )
        client.set_timeout(30.0)

        world = client.get_world()

        current_map = (
            world.get_map().name.split("/")[-1]
        )
        target_map = config["map"]

        if current_map != target_map:
            print(
                f"切换地图："
                f"{current_map} -> {target_map}"
            )

            world = client.load_world(target_map)
            time.sleep(2)

        print(
            "Client version:",
            client.get_client_version()
        )
        print(
            "Server version:",
            client.get_server_version()
        )
        print(
            "Current map:",
            world.get_map().name
        )

        original_settings = world.get_settings()

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
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(
            int(config["seed"])
        )

        blueprint_library = (
            world.get_blueprint_library()
        )

        print("正在生成 Ego Vehicle...")

        ego_vehicle = spawn_ego_vehicle(
            world,
            blueprint_library,
            rng
        )

        ego_vehicle.set_autopilot(
            True,
            args.tm_port
        )

        print(
            "Ego Vehicle:",
            ego_vehicle.type_id
        )

        print("正在生成 NPC 车辆...")

        npc_vehicles = spawn_npc_vehicles(
            world,
            blueprint_library,
            int(config["npc_vehicles"]),
            rng,
            args.tm_port
        )

        print(
            f"NPC 车辆数量："
            f"{len(npc_vehicles)}"
        )

        print("正在生成行人...")

        walkers, walker_controllers = (
            spawn_walkers(
                world,
                blueprint_library,
                int(config["walkers"]),
                rng
            )
        )

        print(
            f"行人数量："
            f"{len(walkers)}"
        )

        # RGB相机
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
        camera_bp.set_attribute(
            "sensor_tick",
            "0.0"
        )

        camera_transform = carla.Transform(
            carla.Location(
                x=float(camera_config["x"]),
                y=float(camera_config["y"]),
                z=float(camera_config["z"])
            )
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=ego_vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )

        # 激光雷达
        lidar_config = config["lidar"]

        lidar_bp = blueprint_library.find(
            "sensor.lidar.ray_cast"
        )

        for attribute_name in [
            "channels",
            "range",
            "points_per_second",
            "rotation_frequency",
            "upper_fov",
            "lower_fov"
        ]:
            lidar_bp.set_attribute(
                attribute_name,
                str(lidar_config[attribute_name])
            )

        lidar_bp.set_attribute(
            "sensor_tick",
            "0.0"
        )

        lidar_transform = carla.Transform(
            carla.Location(
                x=float(lidar_config["x"]),
                y=float(lidar_config["y"]),
                z=float(lidar_config["z"])
            )
        )

        lidar = world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=ego_vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )

        print(
            "RGB Camera 和 LiDAR 已创建"
        )

        camera_queue = queue.Queue()
        lidar_queue = queue.Queue()

        camera.listen(camera_queue.put)
        lidar.listen(lidar_queue.put)

        warmup_frames = int(
            config["warmup_frames"]
        )

        print(
            f"正在预热："
            f"{warmup_frames} 帧"
        )

        for index in range(warmup_frames):
            frame = world.tick()

            print(
                f"预热 {index + 1}/"
                f"{warmup_frames}，"
                f"frame={frame}"
            )

        time.sleep(3)

        clear_queue(camera_queue)
        clear_queue(lidar_queue)

        print(
            f"开始采集 "
            f"{capture_frames} 帧数据"
        )

        for sample_index in range(
            capture_frames
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
            meta_path = (
                meta_dir /
                f"{sample_name}.json"
            )

            image_data.save_to_disk(
                str(image_path)
            )

            lidar_points = np.frombuffer(
                lidar_data.raw_data,
                dtype=np.float32
            ).copy().reshape(-1, 4)

            np.save(
                str(lidar_path),
                lidar_points
            )

            actor_records = (
                collect_actor_metadata(
                    world,
                    ego_vehicle
                )
            )

            save_metadata(
                meta_path,
                sample_index,
                frame,
                image_data,
                lidar_data,
                ego_vehicle,
                actor_records,
                config
            )

            print(
                f"[{sample_index + 1:04d}/"
                f"{capture_frames:04d}] "
                f"frame={frame}, "
                f"points={len(lidar_points)}, "
                f"actors={len(actor_records)}"
            )

        print()
        print("原始数据采集完成")
        print("图像：", image_dir)
        print("点云：", lidar_dir)
        print("元数据：", meta_dir)

    except Exception as exc:
        print()
        print("采集失败：", exc)

    finally:
        print()
        print("正在安全清理仿真对象...")

        # 停止传感器回调
        for sensor in [camera, lidar]:
            if sensor is None:
                continue

            try:
                if sensor.is_alive and sensor.is_listening:
                    sensor.stop()
            except RuntimeError:
                pass

        # 停止行人控制器
        for controller in walker_controllers:
            try:
                if controller.is_alive:
                    controller.stop()
            except RuntimeError:
                pass

        # 清理传感器队列
        clear_queue(camera_queue)
        clear_queue(lidar_queue)

        # 等待回调线程完全停止
        time.sleep(1)

        if client is not None:
            # 先销毁传感器和行人控制器
            batch_destroy(
                client,
                [
                    camera,
                    lidar,
                    *walker_controllers
                ],
                "传感器和控制器"
            )

            time.sleep(0.5)

            # 再销毁行人和车辆
            batch_destroy(
                client,
                [
                    *walkers,
                    *npc_vehicles,
                    ego_vehicle
                ],
                "行人和车辆"
            )

            time.sleep(0.5)

        # 清除Python Actor引用
        camera = None
        lidar = None
        ego_vehicle = None
        camera_queue = None
        lidar_queue = None

        walker_controllers.clear()
        walkers.clear()
        npc_vehicles.clear()

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

        time.sleep(1)

        print("安全清理完成")


if __name__ == "__main__":
    main()
