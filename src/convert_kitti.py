import argparse
import json
import math
import shutil
from pathlib import Path

import carla
import cv2
import numpy as np


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def dict_to_location(data):
    return carla.Location(
        x=float(data["x"]),
        y=float(data["y"]),
        z=float(data["z"])
    )


def dict_to_rotation(data):
    return carla.Rotation(
        pitch=float(data["pitch"]),
        yaw=float(data["yaw"]),
        roll=float(data["roll"])
    )


def dict_to_transform(data):
    return carla.Transform(
        dict_to_location(data["location"]),
        dict_to_rotation(data["rotation"])
    )


def dict_to_bounding_box(data):
    location = dict_to_location(
        data["location"]
    )

    extent = carla.Vector3D(
        x=float(data["extent"]["x"]),
        y=float(data["extent"]["y"]),
        z=float(data["extent"]["z"])
    )

    bounding_box = carla.BoundingBox(
        location,
        extent
    )

    bounding_box.rotation = dict_to_rotation(
        data["rotation"]
    )

    return bounding_box


def project_world_point(
    world_location,
    world_to_camera,
    intrinsic
):
    point_world = np.array([
        world_location.x,
        world_location.y,
        world_location.z,
        1.0
    ], dtype=np.float64)

    # CARLA世界坐标转CARLA相机坐标
    point_camera_ue = (
        world_to_camera @ point_world
    )

    # CARLA相机坐标：
    # x向前、y向右、z向上
    #
    # KITTI相机坐标：
    # x向右、y向下、z向前
    point_camera_kitti = np.array([
        point_camera_ue[1],
        -point_camera_ue[2],
        point_camera_ue[0]
    ], dtype=np.float64)

    depth = point_camera_kitti[2]

    if depth <= 0.0:
        return None, point_camera_kitti

    point_image = (
        intrinsic @ point_camera_kitti
    )

    point_image[0] /= point_image[2]
    point_image[1] /= point_image[2]

    return point_image[:2], point_camera_kitti


def get_box_center_world(
    actor_transform,
    bounding_box
):
    center = carla.Location(
        x=bounding_box.location.x,
        y=bounding_box.location.y,
        z=bounding_box.location.z
    )

    actor_transform.transform(center)

    return center


def get_rotation_y(
    actor_transform,
    world_to_camera
):
    forward = (
        actor_transform.get_forward_vector()
    )

    forward_world = np.array([
        forward.x,
        forward.y,
        forward.z
    ], dtype=np.float64)

    camera_rotation = (
        world_to_camera[:3, :3]
    )

    forward_camera_ue = (
        camera_rotation @ forward_world
    )

    forward_camera_kitti = np.array([
        forward_camera_ue[1],
        -forward_camera_ue[2],
        forward_camera_ue[0]
    ])

    rotation_y = math.atan2(
        -forward_camera_kitti[2],
        forward_camera_kitti[0]
    )

    return normalize_angle(rotation_y)


def convert_actor_to_kitti(
    actor_data,
    camera_transform,
    intrinsic,
    image_width,
    image_height,
    minimum_box_height
):
    actor_transform = dict_to_transform(
        actor_data["transform"]
    )

    bounding_box = dict_to_bounding_box(
        actor_data["bounding_box"]
    )

    world_to_camera = np.array(
        camera_transform.get_inverse_matrix(),
        dtype=np.float64
    )

    center_world = get_box_center_world(
        actor_transform,
        bounding_box
    )

    _, center_camera = project_world_point(
        center_world,
        world_to_camera,
        intrinsic
    )

    # 目标中心必须在相机前方
    if center_camera[2] <= 0.0:
        return None

    vertices_world = (
        bounding_box.get_world_vertices(
            actor_transform
        )
    )

    image_points = []

    for vertex in vertices_world:
        image_point, camera_point = (
            project_world_point(
                vertex,
                world_to_camera,
                intrinsic
            )
        )

        # 简化处理：包围框顶点在相机后方时跳过
        if (
            image_point is None
            or camera_point[2] <= 0.1
        ):
            continue

        image_points.append(image_point)

    # 至少需要4个有效投影点
    if len(image_points) < 4:
        return None

    image_points = np.asarray(
        image_points,
        dtype=np.float64
    )

    raw_left = float(
        np.min(image_points[:, 0])
    )
    raw_top = float(
        np.min(image_points[:, 1])
    )
    raw_right = float(
        np.max(image_points[:, 0])
    )
    raw_bottom = float(
        np.max(image_points[:, 1])
    )

    raw_width = raw_right - raw_left
    raw_height = raw_bottom - raw_top

    if raw_width <= 0.0 or raw_height <= 0.0:
        return None

    left = max(
        0.0,
        min(raw_left, image_width - 1.0)
    )
    top = max(
        0.0,
        min(raw_top, image_height - 1.0)
    )
    right = max(
        0.0,
        min(raw_right, image_width - 1.0)
    )
    bottom = max(
        0.0,
        min(raw_bottom, image_height - 1.0)
    )

    clipped_width = right - left
    clipped_height = bottom - top

    if (
        clipped_width <= 1.0
        or clipped_height < minimum_box_height
    ):
        return None

    raw_area = raw_width * raw_height
    clipped_area = (
        clipped_width * clipped_height
    )

    truncated = 1.0 - (
        clipped_area / raw_area
    )
    truncated = max(
        0.0,
        min(1.0, truncated)
    )

    extent = bounding_box.extent

    height = 2.0 * float(extent.z)
    width = 2.0 * float(extent.y)
    length = 2.0 * float(extent.x)

    location_x = float(center_camera[0])

    # KITTI的3D位置使用包围框底部中心
    location_y = float(
        center_camera[1] + height / 2.0
    )
    location_z = float(center_camera[2])

    rotation_y = get_rotation_y(
        actor_transform,
        world_to_camera
    )

    alpha = normalize_angle(
        rotation_y
        - math.atan2(
            location_x,
            location_z
        )
    )

    class_name = actor_data.get(
        "class_name",
        "Misc"
    )

    # 当前版本不精确计算遮挡，3表示unknown
    occluded = 3

    label_line = (
        f"{class_name} "
        f"{truncated:.6f} "
        f"{occluded:d} "
        f"{alpha:.6f} "
        f"{left:.6f} "
        f"{top:.6f} "
        f"{right:.6f} "
        f"{bottom:.6f} "
        f"{height:.6f} "
        f"{width:.6f} "
        f"{length:.6f} "
        f"{location_x:.6f} "
        f"{location_y:.6f} "
        f"{location_z:.6f} "
        f"{rotation_y:.6f}"
    )

    return {
        "line": label_line,
        "class_name": class_name,
        "bbox": [
            left,
            top,
            right,
            bottom
        ]
    }


def save_lidar_bin(
    source_path,
    output_path
):
    points = np.load(source_path)

    if (
        points.ndim != 2
        or points.shape[1] != 4
    ):
        raise ValueError(
            f"点云格式错误：{source_path}，"
            f"shape={points.shape}"
        )

    points = points.astype(
        np.float32,
        copy=True
    )

    # CARLA：x前、y右、z上
    # KITTI：x前、y左、z上
    points[:, 1] *= -1.0

    points.tofile(output_path)


def matrix_to_line(name, matrix):
    values = matrix.reshape(-1)

    value_text = " ".join(
        f"{float(value):.12e}"
        for value in values
    )

    return f"{name}: {value_text}"


def save_calibration(
    output_path,
    intrinsic,
    camera_transform,
    lidar_transform
):
    identity_extrinsic = np.hstack([
        np.eye(3, dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64)
    ])

    projection = (
        intrinsic @ identity_extrinsic
    )

    rectification = np.eye(
        3,
        dtype=np.float64
    )

    world_from_camera = np.array(
        camera_transform.get_matrix(),
        dtype=np.float64
    )

    world_from_lidar = np.array(
        lidar_transform.get_matrix(),
        dtype=np.float64
    )

    camera_from_world = np.linalg.inv(
        world_from_camera
    )

    # CARLA相机坐标转KITTI相机坐标
    camera_axis_conversion = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # KITTI Velodyne坐标转CARLA LiDAR坐标
    lidar_axis_conversion = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    lidar_to_camera_4x4 = (
        camera_axis_conversion
        @ camera_from_world
        @ world_from_lidar
        @ lidar_axis_conversion
    )

    lidar_to_camera = (
        lidar_to_camera_4x4[:3, :4]
    )

    lines = [
        matrix_to_line("P0", projection),
        matrix_to_line("P1", projection),
        matrix_to_line("P2", projection),
        matrix_to_line("P3", projection),
        matrix_to_line(
            "R0_rect",
            rectification
        ),
        matrix_to_line(
            "Tr_velo_to_cam",
            lidar_to_camera
        )
    ]

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )


def draw_debug_image(
    image_path,
    labels,
    output_path
):
    image = cv2.imread(str(image_path))

    if image is None:
        return

    colors = {
        "Car": (255, 0, 0),
        "Pedestrian": (0, 255, 0),
        "Misc": (0, 255, 255)
    }

    for label in labels:
        left, top, right, bottom = (
            label["bbox"]
        )

        color = colors.get(
            label["class_name"],
            (0, 0, 255)
        )

        point1 = (
            int(round(left)),
            int(round(top))
        )
        point2 = (
            int(round(right)),
            int(round(bottom))
        )

        cv2.rectangle(
            image,
            point1,
            point2,
            color,
            2
        )

        cv2.putText(
            image,
            label["class_name"],
            (
                point1[0],
                max(15, point1[1] - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        image
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert CARLA raw data "
            "to KITTI object format"
        )
    )
    parser.add_argument(
        "--input",
        default="data/raw"
    )
    parser.add_argument(
        "--output",
        default="data/kitti"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )
    parser.add_argument(
        "--minimum-box-height",
        type=float,
        default=20.0
    )
    parser.add_argument(
        "--debug-frames",
        type=int,
        default=20
    )
    parser.add_argument(
        "--debug-output",
        default="data/debug/kitti"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    debug_dir = Path(args.debug_output)

    raw_image_dir = input_dir / "image"
    raw_lidar_dir = input_dir / "lidar"
    raw_meta_dir = input_dir / "meta"

    image_output_dir = (
        output_dir / "training" / "image_2"
    )
    lidar_output_dir = (
        output_dir / "training" / "velodyne"
    )
    label_output_dir = (
        output_dir / "training" / "label_2"
    )
    calib_output_dir = (
        output_dir / "training" / "calib"
    )

    for directory in [
        image_output_dir,
        lidar_output_dir,
        label_output_dir,
        calib_output_dir,
        debug_dir
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    metadata_files = sorted(
        raw_meta_dir.glob("*.json")
    )

    if args.limit is not None:
        metadata_files = metadata_files[
            :args.limit
        ]

    if not metadata_files:
        raise RuntimeError(
            "没有找到原始元数据文件"
        )

    total_labels = 0
    total_cars = 0
    total_pedestrians = 0

    for index, metadata_path in enumerate(
        metadata_files
    ):
        sample_name = metadata_path.stem

        image_source = (
            raw_image_dir /
            f"{sample_name}.png"
        )
        lidar_source = (
            raw_lidar_dir /
            f"{sample_name}.npy"
        )

        if not image_source.exists():
            raise FileNotFoundError(
                f"缺少图像：{image_source}"
            )

        if not lidar_source.exists():
            raise FileNotFoundError(
                f"缺少点云：{lidar_source}"
            )

        with metadata_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        image_width = int(
            metadata["camera"]["width"]
        )
        image_height = int(
            metadata["camera"]["height"]
        )

        intrinsic = np.asarray(
            metadata["camera"]["intrinsic"],
            dtype=np.float64
        )

        camera_transform = dict_to_transform(
            metadata["camera"]["transform"]
        )
        lidar_transform = dict_to_transform(
            metadata["lidar"]["transform"]
        )

        labels = []

        for actor_data in metadata["actors"]:
            converted = convert_actor_to_kitti(
                actor_data,
                camera_transform,
                intrinsic,
                image_width,
                image_height,
                args.minimum_box_height
            )

            if converted is None:
                continue

            labels.append(converted)

            if converted["class_name"] == "Car":
                total_cars += 1
            elif (
                converted["class_name"]
                == "Pedestrian"
            ):
                total_pedestrians += 1

        total_labels += len(labels)

        image_output_path = (
            image_output_dir /
            f"{sample_name}.png"
        )
        lidar_output_path = (
            lidar_output_dir /
            f"{sample_name}.bin"
        )
        label_output_path = (
            label_output_dir /
            f"{sample_name}.txt"
        )
        calib_output_path = (
            calib_output_dir /
            f"{sample_name}.txt"
        )

        shutil.copy2(
            image_source,
            image_output_path
        )

        save_lidar_bin(
            lidar_source,
            lidar_output_path
        )

        label_text = "\n".join(
            label["line"]
            for label in labels
        )

        if label_text:
            label_text += "\n"

        label_output_path.write_text(
            label_text,
            encoding="utf-8"
        )

        save_calibration(
            calib_output_path,
            intrinsic,
            camera_transform,
            lidar_transform
        )

        if index < args.debug_frames:
            debug_path = (
                debug_dir /
                f"{sample_name}.png"
            )

            draw_debug_image(
                image_source,
                labels,
                debug_path
            )

        print(
            f"[{index + 1:04d}/"
            f"{len(metadata_files):04d}] "
            f"{sample_name}："
            f"{len(labels)} 个标签"
        )

    print()
    print("KITTI 转换完成")
    print("样本数量：", len(metadata_files))
    print("标签总数：", total_labels)
    print("车辆标签：", total_cars)
    print("行人标签：", total_pedestrians)
    print("输出目录：", output_dir)
    print("检查图片：", debug_dir)


if __name__ == "__main__":
    main()
