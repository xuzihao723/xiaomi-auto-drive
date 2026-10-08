import argparse
import json
import math
import queue
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import carla
import cv2
import numpy as np


CLASS_NAMES = {
    0: "Car",
    1: "Pedestrian",
    2: "TrafficLight",
    3: "TrafficSign",
}

# CARLA semantic tag IDs (CityScapes label mapping used by 0.9.15).
# The instance-segmentation image stores B, G, semantic-tag, A.  B/G encode
# the instance ID, so labels below are produced only from pixels that are
# actually visible to the camera; hidden map objects are not projected into
# the image.
SEMANTIC_TO_CLASS = {
    # CARLA changed these tag values in 0.9.14.  Keep car, truck and bus in
    # the project's existing KITTI-style "Car" superclass.
    14: 0,  # Car
    15: 0,  # Truck
    16: 0,  # Bus
    12: 1,  # Pedestrian
    7: 2,   # TrafficLight (light boxes only, poles are tag 6)
    8: 3,   # TrafficSign (sign faces only, poles are tag 6)
}

MIN_VISIBLE_PIXELS = {
    0: 40,
    1: 20,
    2: 10,
    3: 10,
}

MIN_BOX_SIZE = {
    0: (8, 8),
    1: (4, 8),
    2: (3, 4),
    3: (3, 3),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect grouped multi-map CARLA images with native 2D labels for "
            "vehicles, pedestrians, traffic lights, and traffic signs."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Optional scenario name; repeat to run a subset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-drive-frames",
        type=int,
        default=None,
        help="Optional smoke-test cap applied to each selected scenario.",
    )
    parser.add_argument(
        "--save-instance-masks",
        action="store_true",
        help="Save raw instance masks for label-source auditing.",
    )
    return parser.parse_args()


def build_intrinsic(width, height, fov):
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def weather_from_name(name):
    if name == "ClearNight":
        weather = carla.WeatherParameters(
            cloudiness=5.0,
            precipitation=0.0,
            sun_altitude_angle=-10.0,
            sun_azimuth_angle=15.0,
            fog_density=0.0,
            wetness=0.0,
        )
        return weather
    weather = getattr(carla.WeatherParameters, name, None)
    if weather is None:
        raise ValueError(f"Unknown CARLA weather preset: {name}")
    return weather


def clear_queue(image_queue):
    while True:
        try:
            image_queue.get_nowait()
        except queue.Empty:
            return


def image_for_frame(image_queue, target_frame, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            image = image_queue.get(timeout=max(0.1, deadline - time.time()))
        except queue.Empty as exc:
            raise RuntimeError(f"Camera timeout for frame {target_frame}") from exc
        if image.frame < target_frame:
            continue
        if image.frame > target_frame:
            raise RuntimeError(
                f"Camera skipped frame: target={target_frame}, received={image.frame}"
            )
        return image
    raise RuntimeError(f"Camera timeout for frame {target_frame}")


def project_vertices(vertices, world_to_camera, intrinsic, width, height):
    points = []
    depths = []
    for vertex in vertices:
        point_world = np.array(
            [vertex.x, vertex.y, vertex.z, 1.0], dtype=np.float64
        )
        point_ue = world_to_camera @ point_world
        point_camera = np.array(
            [point_ue[1], -point_ue[2], point_ue[0]], dtype=np.float64
        )
        if point_camera[2] <= 0.1:
            continue
        image_point = intrinsic @ point_camera
        image_point[:2] /= image_point[2]
        points.append(image_point[:2])
        depths.append(point_camera[2])

    if len(points) < 4:
        return None

    points = np.asarray(points)
    left = max(0.0, float(points[:, 0].min()))
    top = max(0.0, float(points[:, 1].min()))
    right = min(float(width - 1), float(points[:, 0].max()))
    bottom = min(float(height - 1), float(points[:, 1].max()))
    box_width = right - left
    box_height = bottom - top
    if box_width < 6.0 or box_height < 6.0:
        return None
    if right <= 0.0 or bottom <= 0.0 or left >= width or top >= height:
        return None
    return left, top, right, bottom, min(depths)


def yolo_row(class_id, box, width, height):
    left, top, right, bottom, _ = box
    return (
        class_id,
        (left + right) / 2.0 / width,
        (top + bottom) / 2.0 / height,
        (right - left) / width,
        (bottom - top) / height,
    )


def labels_from_instance_image(image):
    """Create tight YOLO boxes from visible CARLA instance-mask pixels."""
    width = int(image.width)
    height = int(image.height)
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        (height, width, 4)
    )
    semantic_tags = bgra[:, :, 2]
    instance_ids = (
        bgra[:, :, 1].astype(np.uint32) << 8
    ) | bgra[:, :, 0].astype(np.uint32)

    rows = []
    for semantic_tag, class_id in SEMANTIC_TO_CLASS.items():
        class_mask = semantic_tags == semantic_tag
        ids = np.unique(instance_ids[class_mask])
        for instance_id in ids:
            if instance_id == 0:
                continue
            visible = class_mask & (instance_ids == instance_id)
            # Some CARLA map assets reuse a 16-bit instance ID.  Splitting
            # disconnected regions prevents two distinct objects from being
            # enclosed by one oversized box.
            component_count, _, stats, _ = cv2.connectedComponentsWithStats(
                visible.astype(np.uint8), connectivity=8
            )
            for component in range(1, component_count):
                left, top, box_width, box_height, pixels = stats[component]
                if pixels < MIN_VISIBLE_PIXELS[class_id]:
                    continue
                min_width, min_height = MIN_BOX_SIZE[class_id]
                if box_width < min_width or box_height < min_height:
                    continue

                right = left + box_width - 1
                bottom = top + box_height - 1
                # A one-pixel margin avoids cutting the antialiased edge.
                left = max(0, int(left) - 1)
                top = max(0, int(top) - 1)
                right = min(width - 1, int(right) + 1)
                bottom = min(height - 1, int(bottom) + 1)
                box = (left, top, right, bottom, 0.0)
                rows.append(yolo_row(class_id, box, width, height))
    return rows


def collect_labels(
    world,
    camera_transform,
    intrinsic,
    width,
    height,
    static_boxes,
    excluded_actor_ids=None,
):
    world_to_camera = np.asarray(
        camera_transform.get_inverse_matrix(), dtype=np.float64
    )
    camera_location = camera_transform.location
    candidates = []

    excluded_actor_ids = set(excluded_actor_ids or [])
    for actor in world.get_actors():
        if actor.id in excluded_actor_ids:
            continue
        if not actor.is_alive:
            continue
        if actor.type_id.startswith("vehicle."):
            class_id = 0
        elif actor.type_id.startswith("walker.pedestrian."):
            class_id = 1
        else:
            continue
        try:
            if actor.get_location().distance(camera_location) > 80.0:
                continue
            vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
            box = project_vertices(
                vertices, world_to_camera, intrinsic, width, height
            )
            if box is not None:
                candidates.append((class_id, box))
        except RuntimeError:
            continue

    identity = carla.Transform()
    for class_id, boxes in static_boxes.items():
        for bounding_box in boxes:
            if bounding_box.location.distance(camera_location) > 80.0:
                continue
            vertices = bounding_box.get_world_vertices(identity)
            box = project_vertices(
                vertices, world_to_camera, intrinsic, width, height
            )
            if box is not None:
                candidates.append((class_id, box))

    # Keep nearer boxes first and remove almost identical boxes from the same class.
    candidates.sort(key=lambda item: item[1][4])
    rows = []
    accepted = []
    for class_id, box in candidates:
        left, top, right, bottom, _ = box
        duplicate = False
        for other_class, other in accepted:
            if class_id != other_class:
                continue
            ol, ot, oright, ob, _ = other
            inter_w = max(0.0, min(right, oright) - max(left, ol))
            inter_h = max(0.0, min(bottom, ob) - max(top, ot))
            intersection = inter_w * inter_h
            union = (right - left) * (bottom - top) + (
                (oright - ol) * (ob - ot)
            ) - intersection
            if union > 0.0 and intersection / union > 0.90:
                duplicate = True
                break
        if not duplicate:
            accepted.append((class_id, box))
            rows.append(yolo_row(class_id, box, width, height))
    return rows


def write_label(path, rows):
    path.write_text(
        "\n".join(
            f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
            for class_id, cx, cy, width, height in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def save_sample(image, rows, image_path, label_path):
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image.save_to_disk(str(image_path), carla.ColorConverter.Raw)
    write_label(label_path, rows)


def choose_vehicle_blueprints(library):
    blueprints = []
    for blueprint in library.filter("vehicle.*"):
        if blueprint.has_attribute("number_of_wheels"):
            if int(blueprint.get_attribute("number_of_wheels")) < 4:
                continue
        blueprints.append(blueprint)
    return blueprints


def spawn_traffic(
    world,
    client,
    traffic_manager,
    count,
    walker_count,
    rng,
    reference_location,
):
    library = world.get_blueprint_library()
    vehicle_blueprints = choose_vehicle_blueprints(library)
    spawn_points = list(world.get_map().get_spawn_points())
    spawn_points.sort(
        key=lambda transform: transform.location.distance(reference_location)
    )
    nearby_spawn_points = [
        transform
        for transform in spawn_points
        if transform.location.distance(reference_location) >= 8.0
    ][: max(count * 2, count)]
    rng.shuffle(nearby_spawn_points)
    actors = []

    for transform in nearby_spawn_points:
        if len(actors) >= count:
            break
        blueprint = rng.choice(vehicle_blueprints)
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", rng.choice(colors))
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            actor.set_autopilot(True, traffic_manager.get_port())
            actors.append(actor)

    walker_blueprints = list(library.filter("walker.pedestrian.*"))
    spawned_walkers = 0
    attempts = 0
    while spawned_walkers < walker_count and attempts < walker_count * 30:
        attempts += 1
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        if location.distance(reference_location) > 120.0:
            continue
        transform = carla.Transform(location)
        walker = world.try_spawn_actor(rng.choice(walker_blueprints), transform)
        if walker is not None:
            actors.append(walker)
            spawned_walkers += 1
    return actors


def spawn_camera(world, width, height, fov, transform, attach_to=None):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(fov))
    blueprint.set_attribute("sensor_tick", "0.0")
    if attach_to is None:
        return world.spawn_actor(blueprint, transform)
    return world.spawn_actor(
        blueprint,
        transform,
        attach_to=attach_to,
        attachment_type=carla.AttachmentType.Rigid,
    )


def spawn_instance_camera(world, width, height, fov, transform, attach_to=None):
    blueprint = world.get_blueprint_library().find(
        "sensor.camera.instance_segmentation"
    )
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(fov))
    blueprint.set_attribute("sensor_tick", "0.0")
    if attach_to is None:
        return world.spawn_actor(blueprint, transform)
    return world.spawn_actor(
        blueprint,
        transform,
        attach_to=attach_to,
        attachment_type=carla.AttachmentType.Rigid,
    )


def spawn_ego(world, traffic_manager, rng, spawn_index=None):
    library = world.get_blueprint_library()
    try:
        blueprint = library.find("vehicle.tesla.model3")
    except RuntimeError:
        blueprint = rng.choice(choose_vehicle_blueprints(library))
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    spawn_points = list(world.get_map().get_spawn_points())
    if spawn_index is not None:
        if spawn_index < 0 or spawn_index >= len(spawn_points):
            raise IndexError(
                f"ego_spawn_index={spawn_index} outside 0..{len(spawn_points) - 1}"
            )
        preferred = spawn_points[spawn_index]
        remaining = [
            transform
            for index, transform in enumerate(spawn_points)
            if index != spawn_index
        ]
        rng.shuffle(remaining)
        ordered_spawn_points = [preferred] + remaining
    else:
        rng.shuffle(spawn_points)
        ordered_spawn_points = spawn_points
    for transform in ordered_spawn_points:
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            actor.set_autopilot(True, traffic_manager.get_port())
            return actor
    raise RuntimeError("Could not spawn ego vehicle")


def look_at_transform(location, target):
    dx = target.x - location.x
    dy = target.y - location.y
    dz = target.z - location.z
    horizontal = max(0.001, math.hypot(dx, dy))
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = -math.degrees(math.atan2(dz, horizontal))
    return carla.Transform(location, carla.Rotation(pitch=pitch, yaw=yaw))


def camera_transform_for_sign(map_object, sign_box, distance, lateral_offset):
    waypoint = map_object.get_waypoint(
        sign_box.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    chosen = waypoint
    previous = waypoint.previous(distance)
    if previous:
        chosen = previous[0]
    base = chosen.transform
    right = base.get_right_vector()
    location = carla.Location(
        x=base.location.x + right.x * lateral_offset,
        y=base.location.y + right.y * lateral_offset,
        z=base.location.z + 2.2,
    )
    target = carla.Location(
        x=sign_box.location.x,
        y=sign_box.location.y,
        z=sign_box.location.z,
    )
    return look_at_transform(location, target)


def run_scenario(
    client,
    base_config,
    scenario,
    output_root,
    tm_port,
    max_drive_frames=None,
    save_instance_masks=False,
):
    name = scenario["name"]
    scenario_dir = output_root / "scenarios" / name
    images_dir = scenario_dir / "images"
    labels_dir = scenario_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    width = int(base_config["image_width"])
    height = int(base_config["image_height"])
    fov = float(base_config["camera_fov"])
    intrinsic = build_intrinsic(width, height, fov)
    rng = random.Random(int(scenario["seed"]))

    current_world = client.get_world()
    current_map = current_world.get_map().name.rsplit("/", 1)[-1]
    if current_map == scenario["map"]:
        world = current_world
    else:
        world = client.load_world(scenario["map"])
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(base_config["fixed_delta_seconds"])
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    world.set_weather(weather_from_name(scenario["weather"]))

    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(int(scenario["seed"]))
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    actors = []
    cameras = []
    class_counts = Counter()
    frame_counts = Counter()
    sample_records = []
    sample_index = 0

    static_boxes = {
        2: list(world.get_level_bbs(carla.CityObjectLabel.TrafficLight)),
        3: list(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)),
    }

    try:
        ego = spawn_ego(
            world,
            traffic_manager,
            rng,
            spawn_index=scenario.get("ego_spawn_index"),
        )
        actors.append(ego)
        actors.extend(
            spawn_traffic(
                world,
                client,
                traffic_manager,
                int(scenario.get("npc_vehicles", base_config["npc_vehicles"])),
                int(scenario.get("walkers", base_config["walkers"])),
                rng,
                ego.get_location(),
            )
        )
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = spawn_camera(
            world,
            width,
            height,
            fov,
            camera_transform,
            attach_to=ego,
        )
        instance_camera = spawn_instance_camera(
            world,
            width,
            height,
            fov,
            camera_transform,
            attach_to=ego,
        )
        cameras.append(camera)
        cameras.append(instance_camera)
        image_queue = queue.Queue()
        instance_queue = queue.Queue()
        camera.listen(image_queue.put)
        instance_camera.listen(instance_queue.put)

        for _ in range(20):
            world.tick()
        clear_queue(image_queue)
        clear_queue(instance_queue)

        drive_frames = int(scenario.get("drive_frames", base_config["drive_frames"]))
        if max_drive_frames is not None:
            drive_frames = min(drive_frames, max_drive_frames)
        stride = int(scenario.get("drive_stride", base_config["drive_stride"]))
        for capture_index in range(drive_frames):
            target_frame = None
            for _ in range(stride):
                target_frame = world.tick()
            image = image_for_frame(image_queue, target_frame)
            instance_image = image_for_frame(instance_queue, target_frame)
            rows = labels_from_instance_image(instance_image)
            stem = f"{name}_drive_{sample_index:05d}"
            if save_instance_masks:
                mask_path = scenario_dir / "instance_masks" / f"{stem}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                instance_image.save_to_disk(
                    str(mask_path), carla.ColorConverter.Raw
                )
            save_sample(
                image,
                rows,
                images_dir / f"{stem}.png",
                labels_dir / f"{stem}.txt",
            )
            for class_id in {row[0] for row in rows}:
                frame_counts[class_id] += 1
            for row in rows:
                class_counts[row[0]] += 1
            sample_records.append(
                {"stem": stem, "mode": "drive", "frame": target_frame}
            )
            sample_index += 1
            if (capture_index + 1) % 20 == 0:
                print(f"[{name}] drive {capture_index + 1}/{drive_frames}")

        camera.stop()
        camera.destroy()
        cameras.remove(camera)
        instance_camera.stop()
        instance_camera.destroy()
        cameras.remove(instance_camera)

        sign_boxes = static_boxes[3]
        views = int(
            scenario.get("sign_views_per_sign", base_config["sign_views_per_sign"])
        )
        if sign_boxes and views > 0:
            camera = spawn_camera(
                world,
                width,
                height,
                fov,
                carla.Transform(carla.Location(z=10.0)),
            )
            cameras.append(camera)
            image_queue = queue.Queue()
            camera.listen(image_queue.put)
            distances = [12.0, 20.0, 30.0, 40.0]
            offsets = [-1.0, 0.0, 1.0, 2.0]
            for sign_index, sign_box in enumerate(sign_boxes):
                saved_for_sign = 0
                for view_index in range(views):
                    transform = camera_transform_for_sign(
                        world.get_map(),
                        sign_box,
                        distances[view_index % len(distances)],
                        offsets[(sign_index + view_index) % len(offsets)],
                    )
                    camera.set_transform(transform)
                    clear_queue(image_queue)
                    world.tick()
                    target_frame = world.tick()
                    image = image_for_frame(image_queue, target_frame)
                    rows = collect_labels(
                        world,
                        camera.get_transform(),
                        intrinsic,
                        width,
                        height,
                        static_boxes,
                        excluded_actor_ids={ego.id},
                    )
                    if not any(row[0] == 3 for row in rows):
                        continue
                    stem = f"{name}_sign_{sample_index:05d}"
                    save_sample(
                        image,
                        rows,
                        images_dir / f"{stem}.png",
                        labels_dir / f"{stem}.txt",
                    )
                    for class_id in {row[0] for row in rows}:
                        frame_counts[class_id] += 1
                    for row in rows:
                        class_counts[row[0]] += 1
                    sample_records.append(
                        {
                            "stem": stem,
                            "mode": "sign_targeted",
                            "frame": target_frame,
                            "sign_index": sign_index,
                        }
                    )
                    sample_index += 1
                    saved_for_sign += 1
                print(
                    f"[{name}] sign {sign_index + 1}/{len(sign_boxes)} "
                    f"saved={saved_for_sign}"
                )

        summary = {
            "name": name,
            "map": scenario["map"],
            "weather": scenario["weather"],
            "seed": int(scenario["seed"]),
            "ego_spawn_index": scenario.get("ego_spawn_index"),
            "split": scenario["split"],
            "label_source": "CARLA instance segmentation visible pixels",
            "images": len(sample_records),
            "native_static_boxes": {
                "TrafficLight": len(static_boxes[2]),
                "TrafficSign": len(static_boxes[3]),
            },
            "object_counts": {
                CLASS_NAMES[class_id]: class_counts[class_id]
                for class_id in CLASS_NAMES
            },
            "frame_counts": {
                CLASS_NAMES[class_id]: frame_counts[class_id]
                for class_id in CLASS_NAMES
            },
            "samples": sample_records,
        }
        (scenario_dir / "scenario.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, indent=2))
        return summary
    finally:
        for camera in cameras:
            try:
                camera.stop()
            except RuntimeError:
                pass
        actor_ids = [actor.id for actor in cameras + actors if actor is not None]
        if actor_ids:
            client.apply_batch_sync(
                [carla.command.DestroyActor(actor_id) for actor_id in actor_ids],
                True,
            )
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    requested = set(args.scenario)
    scenarios = [
        scenario
        for scenario in config["scenarios"]
        if not requested or scenario["name"] in requested
    ]
    if requested and len(scenarios) != len(requested):
        known = {scenario["name"] for scenario in config["scenarios"]}
        raise ValueError(f"Unknown scenarios: {sorted(requested - known)}")

    if args.output.exists() and args.overwrite and not requested:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    print(f"client={client.get_client_version()} server={client.get_server_version()}")

    summaries = []
    for index, scenario in enumerate(scenarios):
        scenario_dir = args.output / "scenarios" / scenario["name"]
        if scenario_dir.exists() and args.overwrite:
            shutil.rmtree(scenario_dir)
        elif (scenario_dir / "scenario.json").exists():
            summary = json.loads(
                (scenario_dir / "scenario.json").read_text(encoding="utf-8")
            )
            summaries.append(summary)
            print(f"skip completed scenario: {scenario['name']}")
            continue
        print(f"scenario {index + 1}/{len(scenarios)}: {scenario['name']}")
        summaries.append(
            run_scenario(
                client,
                config,
                scenario,
                args.output,
                args.tm_port + index,
                max_drive_frames=args.max_drive_frames,
                save_instance_masks=args.save_instance_masks,
            )
        )

    manifest = {
        "collector": "CARLA 0.9.15 instance segmentation visible-pixel boxes",
        "config": config,
        "scenarios": [
            {key: value for key, value in summary.items() if key != "samples"}
            for summary in summaries
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"manifest={args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
