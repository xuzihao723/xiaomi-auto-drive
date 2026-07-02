import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np


REQUIRED_CALIBRATION_KEYS = {
    "P0": 12,
    "P1": 12,
    "P2": 12,
    "P3": 12,
    "R0_rect": 9,
    "Tr_velo_to_cam": 12
}


def parse_label_file(label_path):
    labels = []

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return labels

    for line_number, line in enumerate(
        text.splitlines(),
        start=1
    ):
        values = line.split()

        if len(values) != 15:
            raise ValueError(
                f"{label_path.name} 第"
                f"{line_number}行不是15列，"
                f"实际为{len(values)}列"
            )

        label = {
            "type": values[0],
            "truncated": float(values[1]),
            "occluded": int(float(values[2])),
            "alpha": float(values[3]),
            "left": float(values[4]),
            "top": float(values[5]),
            "right": float(values[6]),
            "bottom": float(values[7]),
            "height": float(values[8]),
            "width": float(values[9]),
            "length": float(values[10]),
            "location_x": float(values[11]),
            "location_y": float(values[12]),
            "location_z": float(values[13]),
            "rotation_y": float(values[14])
        }

        labels.append(label)

    return labels


def parse_calibration_file(calib_path):
    matrices = {}

    text = calib_path.read_text(
        encoding="utf-8"
    )

    for line in text.splitlines():
        if ":" not in line:
            continue

        name, value_text = line.split(
            ":",
            maxsplit=1
        )

        values = [
            float(value)
            for value in value_text.split()
        ]

        matrices[name.strip()] = values

    return matrices


def validate_label(
    label,
    image_width,
    image_height
):
    errors = []

    numeric_values = [
        label["truncated"],
        label["alpha"],
        label["left"],
        label["top"],
        label["right"],
        label["bottom"],
        label["height"],
        label["width"],
        label["length"],
        label["location_x"],
        label["location_y"],
        label["location_z"],
        label["rotation_y"]
    ]

    if not all(
        math.isfinite(value)
        for value in numeric_values
    ):
        errors.append(
            "标签中包含NaN或无穷值"
        )

    if not (
        0.0 <= label["truncated"] <= 1.0
    ):
        errors.append(
            "truncated不在[0,1]范围"
        )

    if label["occluded"] not in {
        0, 1, 2, 3
    }:
        errors.append(
            "occluded不是0、1、2或3"
        )

    if not (
        -math.pi - 0.001
        <= label["alpha"]
        <= math.pi + 0.001
    ):
        errors.append(
            "alpha不在[-pi,pi]范围"
        )

    if not (
        -math.pi - 0.001
        <= label["rotation_y"]
        <= math.pi + 0.001
    ):
        errors.append(
            "rotation_y不在[-pi,pi]范围"
        )

    if label["left"] < 0.0:
        errors.append("bbox left小于0")

    if label["top"] < 0.0:
        errors.append("bbox top小于0")

    if label["right"] > image_width:
        errors.append(
            "bbox right超出图像"
        )

    if label["bottom"] > image_height:
        errors.append(
            "bbox bottom超出图像"
        )

    if label["right"] <= label["left"]:
        errors.append(
            "bbox宽度小于或等于0"
        )

    if label["bottom"] <= label["top"]:
        errors.append(
            "bbox高度小于或等于0"
        )

    if label["height"] <= 0.0:
        errors.append(
            "3D高度小于或等于0"
        )

    if label["width"] <= 0.0:
        errors.append(
            "3D宽度小于或等于0"
        )

    if label["length"] <= 0.0:
        errors.append(
            "3D长度小于或等于0"
        )

    if label["location_z"] <= 0.0:
        errors.append(
            "目标位于相机后方"
        )

    return errors


def draw_debug_image(
    image,
    labels,
    output_path
):
    colors = {
        "Car": (255, 0, 0),
        "Pedestrian": (0, 255, 0),
        "Cyclist": (0, 255, 255),
        "Misc": (255, 0, 255)
    }

    for label in labels:
        color = colors.get(
            label["type"],
            (0, 0, 255)
        )

        left = int(
            round(label["left"])
        )
        top = int(
            round(label["top"])
        )
        right = int(
            round(label["right"])
        )
        bottom = int(
            round(label["bottom"])
        )

        cv2.rectangle(
            image,
            (left, top),
            (right, bottom),
            color,
            2
        )

        text = (
            f"{label['type']} "
            f"z={label['location_z']:.1f}m"
        )

        cv2.putText(
            image,
            text,
            (left, max(18, top - 5)),
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


def get_stems(directory, suffix):
    return {
        path.stem
        for path in directory.glob(
            f"*{suffix}"
        )
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate KITTI object dataset"
        )
    )
    parser.add_argument(
        "--dataset",
        default="data/kitti/training"
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None
    )
    parser.add_argument(
        "--debug-count",
        type=int,
        default=20
    )
    parser.add_argument(
        "--debug-output",
        default="data/debug/validation"
    )
    parser.add_argument(
        "--report",
        default=(
            "docs/kitti_validation_report.json"
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)

    image_dir = dataset_dir / "image_2"
    lidar_dir = dataset_dir / "velodyne"
    label_dir = dataset_dir / "label_2"
    calib_dir = dataset_dir / "calib"

    debug_dir = Path(args.debug_output)
    report_path = Path(args.report)

    errors = []
    warnings = []

    for directory in [
        image_dir,
        lidar_dir,
        label_dir,
        calib_dir
    ]:
        if not directory.exists():
            errors.append(
                f"目录不存在：{directory}"
            )

    if errors:
        for error in errors:
            print("错误：", error)

        sys.exit(1)

    image_stems = get_stems(
        image_dir,
        ".png"
    )
    lidar_stems = get_stems(
        lidar_dir,
        ".bin"
    )
    label_stems = get_stems(
        label_dir,
        ".txt"
    )
    calib_stems = get_stems(
        calib_dir,
        ".txt"
    )

    all_stems = (
        image_stems
        | lidar_stems
        | label_stems
        | calib_stems
    )

    common_stems = (
        image_stems
        & lidar_stems
        & label_stems
        & calib_stems
    )

    if image_stems != all_stems:
        errors.append(
            "image_2文件编号不完整"
        )

    if lidar_stems != all_stems:
        errors.append(
            "velodyne文件编号不完整"
        )

    if label_stems != all_stems:
        errors.append(
            "label_2文件编号不完整"
        )

    if calib_stems != all_stems:
        errors.append(
            "calib文件编号不完整"
        )

    sample_count = len(common_stems)

    if (
        args.expected_count is not None
        and sample_count
        != args.expected_count
    ):
        errors.append(
            f"样本数量应为"
            f"{args.expected_count}，"
            f"实际为{sample_count}"
        )

    sample_pattern = re.compile(
        r"^\d{6}$"
    )

    for stem in all_stems:
        if not sample_pattern.match(stem):
            errors.append(
                f"文件编号不是6位数字：{stem}"
            )

    sorted_stems = sorted(common_stems)

    total_labels = 0
    car_labels = 0
    pedestrian_labels = 0
    empty_label_files = 0
    total_lidar_points = 0

    valid_labels_by_stem = {}

    print(
        f"开始检查 {sample_count} 个样本"
    )

    for index, stem in enumerate(
        sorted_stems,
        start=1
    ):
        image_path = (
            image_dir / f"{stem}.png"
        )
        lidar_path = (
            lidar_dir / f"{stem}.bin"
        )
        label_path = (
            label_dir / f"{stem}.txt"
        )
        calib_path = (
            calib_dir / f"{stem}.txt"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            errors.append(
                f"{stem}：图像无法读取"
            )
            continue

        image_height, image_width = (
            image.shape[:2]
        )

        try:
            lidar_values = np.fromfile(
                lidar_path,
                dtype=np.float32
            )

            if (
                lidar_values.size % 4
                != 0
            ):
                errors.append(
                    f"{stem}：点云数据"
                    f"不能重塑为N×4"
                )
            else:
                lidar_points = (
                    lidar_values.reshape(
                        -1,
                        4
                    )
                )

                if len(lidar_points) == 0:
                    errors.append(
                        f"{stem}：点云为空"
                    )

                if not np.isfinite(
                    lidar_points
                ).all():
                    errors.append(
                        f"{stem}：点云包含"
                        f"NaN或无穷值"
                    )

                total_lidar_points += len(
                    lidar_points
                )

        except Exception as exc:
            errors.append(
                f"{stem}：点云读取失败："
                f"{exc}"
            )

        try:
            labels = parse_label_file(
                label_path
            )

            valid_labels_by_stem[
                stem
            ] = labels

            if not labels:
                empty_label_files += 1

            for label_index, label in enumerate(
                labels,
                start=1
            ):
                label_errors = validate_label(
                    label,
                    image_width,
                    image_height
                )

                for label_error in label_errors:
                    errors.append(
                        f"{stem} 标签"
                        f"{label_index}："
                        f"{label_error}"
                    )

                total_labels += 1

                if label["type"] == "Car":
                    car_labels += 1
                elif (
                    label["type"]
                    == "Pedestrian"
                ):
                    pedestrian_labels += 1

        except Exception as exc:
            errors.append(
                f"{stem}：标签读取失败："
                f"{exc}"
            )

        try:
            calibration = (
                parse_calibration_file(
                    calib_path
                )
            )

            for key, expected_values in (
                REQUIRED_CALIBRATION_KEYS.items()
            ):
                if key not in calibration:
                    errors.append(
                        f"{stem}：标定缺少"
                        f"{key}"
                    )
                    continue

                actual_values = len(
                    calibration[key]
                )

                if (
                    actual_values
                    != expected_values
                ):
                    errors.append(
                        f"{stem}：{key}应有"
                        f"{expected_values}个值，"
                        f"实际为{actual_values}"
                    )

                if not np.isfinite(
                    np.asarray(
                        calibration[key],
                        dtype=np.float64
                    )
                ).all():
                    errors.append(
                        f"{stem}：{key}包含"
                        f"NaN或无穷值"
                    )

        except Exception as exc:
            errors.append(
                f"{stem}：标定读取失败："
                f"{exc}"
            )

        if (
            index % 100 == 0
            or index == sample_count
        ):
            print(
                f"已检查 "
                f"{index}/{sample_count}"
            )

    if sample_count > 0:
        empty_ratio = (
            empty_label_files
            / sample_count
        )

        if empty_ratio > 0.5:
            warnings.append(
                f"空标签样本比例较高："
                f"{empty_ratio:.2%}"
            )

    if total_labels == 0:
        warnings.append(
            "整个数据集没有有效目标标签"
        )

    # 随机生成验收图片
    rng = random.Random(args.seed)

    debug_stems = list(sorted_stems)
    rng.shuffle(debug_stems)

    debug_stems = debug_stems[
        :min(
            args.debug_count,
            len(debug_stems)
        )
    ]

    debug_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for stem in debug_stems:
        image_path = (
            image_dir / f"{stem}.png"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        labels = valid_labels_by_stem.get(
            stem,
            []
        )

        draw_debug_image(
            image,
            labels,
            debug_dir / f"{stem}.png"
        )

    average_lidar_points = 0.0

    if sample_count > 0:
        average_lidar_points = (
            total_lidar_points
            / sample_count
        )

    report = {
        "dataset": str(dataset_dir),
        "sample_count": sample_count,
        "expected_count":
            args.expected_count,
        "total_labels": total_labels,
        "car_labels": car_labels,
        "pedestrian_labels":
            pedestrian_labels,
        "empty_label_files":
            empty_label_files,
        "average_lidar_points":
            average_lidar_points,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("========== 验收结果 ==========")
    print("样本数量：", sample_count)
    print("标签总数：", total_labels)
    print("车辆标签：", car_labels)
    print(
        "行人标签：",
        pedestrian_labels
    )
    print(
        "空标签文件：",
        empty_label_files
    )
    print(
        "平均点云数量：",
        f"{average_lidar_points:.2f}"
    )
    print("错误数量：", len(errors))
    print("警告数量：", len(warnings))
    print("验收图片：", debug_dir)
    print("验收报告：", report_path)

    for warning in warnings:
        print("警告：", warning)

    if errors:
        print()
        print("前20个错误：")

        for error in errors[:20]:
            print("-", error)

        sys.exit(1)

    print()
    print("KITTI 数据集验收通过")


if __name__ == "__main__":
    main()
