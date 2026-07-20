"""Extract drivable-area polygons and polynomial lane curves from a class mask."""

import cv2
import numpy as np


COLORS = {
    1: (0, 180, 0),
    2: (0, 220, 255),
}


def clean_binary(mask, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def largest_drivable_polygon(class_mask, min_area=500):
    binary = clean_binary(class_mask == 1, 7)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    epsilon = 0.005 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def _row_clusters(x_values, max_gap=7, max_width=70):
    """Return compact horizontal marking-cluster centers for a narrow row band."""
    values = np.unique(np.sort(x_values.astype(np.int32)))
    if len(values) == 0:
        return []
    boundaries = np.where(np.diff(values) > max_gap)[0] + 1
    groups = np.split(values, boundaries)
    return [float(np.median(group)) for group in groups if 1 <= len(group) and group[-1] - group[0] <= max_width]


def _track_nearest_lane(lane, side, row_step=6):
    """Track the ego-lane boundary from the image bottom toward the horizon."""
    height, width = lane.shape
    center = width / 2.0
    expected = None
    tracked = []
    for y_value in range(height - 1, int(height * 0.35), -row_step):
        top = max(int(height * 0.35), y_value - row_step + 1)
        _, x_values = np.where(lane[top : y_value + 1] > 0)
        if side == "left":
            x_values = x_values[x_values < center]
        else:
            x_values = x_values[x_values >= center]
        clusters = _row_clusters(x_values, max_width=max(20, int(width * 0.14)))
        if not clusters:
            continue
        if expected is None:
            anchor = width * (0.30 if side == "left" else 0.70)
            selected = min(clusters, key=lambda item: abs(item - anchor))
            if abs(selected - anchor) > width * 0.22:
                continue
        else:
            projected = 0.92 * expected + 0.08 * center
            selected = min(clusters, key=lambda item: abs(item - projected))
            if abs(selected - projected) > max(24, width * 0.13):
                continue
        tracked.append((selected, float((top + y_value) / 2.0)))
        expected = selected if expected is None else 0.65 * expected + 0.35 * selected
    return np.asarray(tracked, dtype=np.float64)


def fit_lane_curves(class_mask, min_points=40, samples=40):
    lane = clean_binary(class_mask == 2, 3)
    height, width = lane.shape
    lane[: int(height * 0.35)] = 0
    if int(lane.sum()) < min_points:
        return []
    curves = []
    for side in ("left", "right"):
        tracked = _track_nearest_lane(lane, side)
        if len(tracked) < 6:
            continue
        xs, ys = tracked[:, 0], tracked[:, 1]
        inliers = np.ones(len(tracked), dtype=bool)
        try:
            for _ in range(3):
                coefficients = np.polyfit(ys[inliers], xs[inliers], 2)
                residuals = np.abs(xs - np.polyval(coefficients, ys))
                threshold = max(8.0, float(np.median(residuals[inliers]) * 3.0))
                updated = residuals <= threshold
                if updated.sum() < 6 or np.array_equal(updated, inliers):
                    break
                inliers = updated
            coefficients = np.polyfit(ys[inliers], xs[inliers], 2)
        except (np.linalg.LinAlgError, ValueError):
            continue
        y_values = np.linspace(float(ys[inliers].min()), float(ys[inliers].max()), samples)
        x_values = np.polyval(coefficients, y_values)
        valid = (x_values >= 0) & (x_values < width)
        if side == "left":
            valid &= x_values <= width / 2.0
        else:
            valid &= x_values >= width / 2.0
        points = np.column_stack([x_values[valid], y_values[valid]]).round().astype(np.int32)
        if len(points) >= 2:
            curves.append({"side": side, "coefficients": coefficients.tolist(), "points": points})
    return curves


def overlay_segmentation(image_bgr, class_mask, alpha=0.35):
    color = np.zeros_like(image_bgr)
    for class_id, bgr in COLORS.items():
        color[class_mask == class_id] = bgr
    output = cv2.addWeighted(image_bgr, 1.0, color, alpha, 0.0)
    polygon = largest_drivable_polygon(class_mask)
    if polygon is not None:
        cv2.polylines(output, [polygon], True, (60, 255, 60), 2, cv2.LINE_AA)
    curves = fit_lane_curves(class_mask)
    for curve in curves:
        cv2.polylines(output, [curve["points"]], False, (255, 80, 255), 4, cv2.LINE_AA)
    return output, polygon, curves
