import argparse
import random
import time

import carla


def main():
    parser = argparse.ArgumentParser(
        description="CARLA 0.9.15 vehicle control test"
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    args = parser.parse_args()

    vehicle = None

    try:
        print("正在连接 CARLA Server...")

        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        world = client.get_world()

        print("连接成功")
        print("Client version:", client.get_client_version())
        print("Server version:", client.get_server_version())
        print("Current map:", world.get_map().name)

        blueprint_library = world.get_blueprint_library()

        vehicle_blueprints = blueprint_library.filter("vehicle.*")
        spawn_points = world.get_map().get_spawn_points()

        if not vehicle_blueprints:
            raise RuntimeError("没有找到车辆 Blueprint")

        if not spawn_points:
            raise RuntimeError("当前地图没有车辆出生点")

        random.shuffle(spawn_points)

        for spawn_point in spawn_points:
            vehicle_bp = random.choice(vehicle_blueprints)

            if vehicle_bp.has_attribute("role_name"):
                vehicle_bp.set_attribute("role_name", "hero")

            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

            if vehicle is not None:
                break

        if vehicle is None:
            raise RuntimeError("车辆生成失败，请确认道路上没有太多车辆")

        print("车辆生成成功")
        print("Vehicle ID:", vehicle.id)
        print("Vehicle type:", vehicle.type_id)

        vehicle.set_autopilot(True, args.tm_port)

        print("自动驾驶已开启，将运行20秒")

        start_time = time.time()

        while time.time() - start_time < 20:
            world.wait_for_tick()

            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()

            speed = 3.6 * (
                velocity.x ** 2
                + velocity.y ** 2
                + velocity.z ** 2
            ) ** 0.5

            print(
                f"位置: "
                f"x={transform.location.x:.2f}, "
                f"y={transform.location.y:.2f}, "
                f"z={transform.location.z:.2f}, "
                f"速度={speed:.2f} km/h"
            )

            time.sleep(1)

    except Exception as exc:
        print("运行失败:", exc)

    finally:
        if vehicle is not None:
            print("正在销毁测试车辆...")
            vehicle.destroy()
            print("车辆已销毁")


if __name__ == "__main__":
    main()
