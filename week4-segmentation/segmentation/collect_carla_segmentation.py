"""Collect synchronized RGB and semantic masks from CARLA 0.9.15."""

import argparse
import json
import queue
import random
import time
from collections import Counter
from pathlib import Path

import carla
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8100)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def weather_from_name(name):
    if name in {"ClearNight", "CloudyNight"}:
        cloudy = name == "CloudyNight"
        return carla.WeatherParameters(
            cloudiness=80.0 if cloudy else 5.0,
            precipitation=0.0,
            sun_altitude_angle=-10.0,
            sun_azimuth_angle=15.0,
            fog_density=8.0 if cloudy else 0.0,
            wetness=35.0 if cloudy else 0.0,
        )
    weather = getattr(carla.WeatherParameters, name, None)
    if weather is None:
        raise ValueError(f"Unknown CARLA weather preset: {name}")
    return weather


def image_for_frame(source_queue, frame, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            image = source_queue.get(timeout=max(0.1, deadline - time.time()))
        except queue.Empty as exc:
            raise RuntimeError(f"Sensor timeout for frame {frame}") from exc
        if image.frame < frame:
            continue
        if image.frame > frame:
            raise RuntimeError(f"Sensor skipped frame {frame}; received {image.frame}")
        return image
    raise RuntimeError(f"Sensor timeout for frame {frame}")


def clear_queue(source_queue):
    while True:
        try:
            source_queue.get_nowait()
        except queue.Empty:
            return


def choose_vehicle_blueprints(library):
    result = []
    for blueprint in library.filter("vehicle.*"):
        if blueprint.has_attribute("number_of_wheels"):
            if int(blueprint.get_attribute("number_of_wheels")) < 4:
                continue
        result.append(blueprint)
    return result


def spawn_ego(world, traffic_manager, rng, spawn_index):
    library = world.get_blueprint_library()
    blueprint = library.find("vehicle.tesla.model3")
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    points = list(world.get_map().get_spawn_points())
    if not points:
        raise RuntimeError("Map has no vehicle spawn points")
    preferred = points[int(spawn_index) % len(points)]
    remaining = [point for point in points if point is not preferred]
    rng.shuffle(remaining)
    for transform in [preferred, *remaining]:
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            actor.set_autopilot(True, traffic_manager.get_port())
            return actor
    raise RuntimeError("Unable to spawn ego vehicle")


def spawn_traffic(world, traffic_manager, rng, count, reference_location):
    library = world.get_blueprint_library()
    blueprints = choose_vehicle_blueprints(library)
    points = list(world.get_map().get_spawn_points())
    points.sort(key=lambda item: item.location.distance(reference_location))
    points = [point for point in points if point.location.distance(reference_location) > 8.0]
    rng.shuffle(points)
    actors = []
    for transform in points:
        if len(actors) >= count:
            break
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", rng.choice(colors))
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            actor.set_autopilot(True, traffic_manager.get_port())
            actors.append(actor)
    return actors


def spawn_camera(world, blueprint_id, width, height, fov, transform, ego):
    blueprint = world.get_blueprint_library().find(blueprint_id)
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(fov))
    blueprint.set_attribute("sensor_tick", "0.0")
    return world.spawn_actor(
        blueprint,
        transform,
        attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid,
    )


def decode_rgb(image):
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return bgra[:, :, :3].copy()


def decode_semantic_tags(image):
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return bgra[:, :, 2].copy()


def mask_from_tags(tags, road_tag, road_line_tag):
    """Map runtime CityObjectLabel values to stable project classes."""
    mask = np.zeros(tags.shape, dtype=np.uint8)
    mask[tags == road_tag] = 1
    mask[tags == road_line_tag] = 2
    return mask


def run_scenario(client, config, scenario, output, tm_port, max_frames=None, overwrite=False):
    name = scenario["name"]
    split = scenario["split"]
    scenario_manifest = output / "scenarios" / f"{name}.json"
    if scenario_manifest.exists() and not overwrite:
        print(f"skip completed scenario: {name}")
        return json.loads(scenario_manifest.read_text(encoding="utf-8"))
    if overwrite:
        for directory_name in ("images", "masks"):
            directory = output / directory_name / split
            for stale in directory.glob(f"{name}_*.png") if directory.exists() else []:
                stale.unlink()
        if scenario_manifest.exists():
            scenario_manifest.unlink()

    width = int(config["image_width"])
    height = int(config["image_height"])
    fov = float(config["camera_fov"])
    rng = random.Random(int(scenario["seed"]))
    current_world = client.get_world()
    current_map = current_world.get_map().name.rsplit("/", 1)[-1]
    world = current_world if current_map == scenario["map"] else client.load_world(scenario["map"])
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(config["fixed_delta_seconds"])
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    world.set_weather(weather_from_name(scenario["weather"]))

    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(int(scenario["seed"]))
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    actors = []
    sensors = []
    rgb_queue = queue.Queue()
    semantic_queue = queue.Queue()
    pixel_counts = Counter()
    semantic_tag_counts = Counter()
    road_tag = int(carla.CityObjectLabel.Roads)
    road_line_tag = int(carla.CityObjectLabel.RoadLines)
    records = []
    try:
        ego = spawn_ego(world, traffic_manager, rng, scenario["ego_spawn_index"])
        actors.append(ego)
        actors.extend(
            spawn_traffic(
                world,
                traffic_manager,
                rng,
                int(scenario.get("npc_vehicles", config["npc_vehicles"])),
                ego.get_location(),
            )
        )
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        rgb = spawn_camera(world, "sensor.camera.rgb", width, height, fov, camera_transform, ego)
        semantic = spawn_camera(
            world,
            "sensor.camera.semantic_segmentation",
            width,
            height,
            fov,
            camera_transform,
            ego,
        )
        sensors.extend([rgb, semantic])
        rgb.listen(rgb_queue.put)
        semantic.listen(semantic_queue.put)
        for _ in range(int(config["warmup_frames"])):
            world.tick()
        clear_queue(rgb_queue)
        clear_queue(semantic_queue)

        frame_count = int(scenario["frames"])
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)
        stride = int(config["capture_stride"])
        image_dir = output / "images" / split
        mask_dir = output / "masks" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for index in range(frame_count):
            target_frame = None
            for _ in range(stride):
                target_frame = world.tick()
            rgb_image = image_for_frame(rgb_queue, target_frame)
            semantic_image = image_for_frame(semantic_queue, target_frame)
            image = decode_rgb(rgb_image)
            tags = decode_semantic_tags(semantic_image)
            mask = mask_from_tags(tags, road_tag, road_line_tag)
            stem = f"{name}_{index:05d}"
            image_path = image_dir / f"{stem}.png"
            mask_path = mask_dir / f"{stem}.png"
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"Failed to write {image_path}")
            if not cv2.imwrite(str(mask_path), mask):
                raise RuntimeError(f"Failed to write {mask_path}")
            values, counts = np.unique(mask, return_counts=True)
            for value, count in zip(values, counts):
                pixel_counts[int(value)] += int(count)
            tag_values, tag_counts = np.unique(tags, return_counts=True)
            for value, count in zip(tag_values, tag_counts):
                semantic_tag_counts[int(value)] += int(count)
            records.append({"stem": stem, "carla_frame": int(target_frame)})
            if (index + 1) % 25 == 0 or index + 1 == frame_count:
                print(f"[{name}] {index + 1}/{frame_count}")

        summary = {
            "name": name,
            "map": scenario["map"],
            "weather": scenario["weather"],
            "seed": int(scenario["seed"]),
            "split": split,
            "images": len(records),
            "label_source": "CARLA 0.9.15 semantic segmentation raw red-channel tags",
            "class_mapping": {
                "0": "Background",
                "1": f"DrivableArea(runtime tag={road_tag})",
                "2": f"LaneMarking(runtime tag={road_line_tag})",
            },
            "pixel_counts": {str(key): pixel_counts[key] for key in range(3)},
            "semantic_tag_counts": {
                str(key): value for key, value in sorted(semantic_tag_counts.items())
            },
            "records": records,
        }
        scenario_manifest.parent.mkdir(parents=True, exist_ok=True)
        scenario_manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
        actor_ids = [actor.id for actor in sensors + actors if actor is not None]
        if actor_ids:
            client.apply_batch_sync([carla.command.DestroyActor(actor_id) for actor_id in actor_ids], True)
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = set(args.scenario)
    scenarios = [item for item in config["scenarios"] if not selected or item["name"] in selected]
    known = {item["name"] for item in config["scenarios"]}
    if selected - known:
        raise ValueError(f"Unknown scenarios: {sorted(selected - known)}")
    args.output.mkdir(parents=True, exist_ok=True)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    print(f"client={client.get_client_version()} server={client.get_server_version()}")
    summaries = []
    for index, scenario in enumerate(scenarios):
        print(f"scenario {index + 1}/{len(scenarios)}: {scenario['name']}")
        summaries.append(
            run_scenario(
                client,
                config,
                scenario,
                args.output,
                args.tm_port + index,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
            )
        )
    completed_summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.output / "scenarios").glob("*.json"))
    ]
    manifest = {
        "collector": "CARLA 0.9.15 synchronized RGB + semantic segmentation camera",
        "images": sum(item["images"] for item in completed_summaries),
        "splits": dict(
            Counter(
                item["split"]
                for item in completed_summaries
                for _ in range(item["images"])
            )
        ),
        "scenarios": [
            {key: value for key, value in item.items() if key != "records"}
            for item in completed_summaries
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
