"""Streaming confusion-matrix metrics for semantic segmentation."""

import torch


class SegmentationMetrics:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, prediction, target):
        prediction = prediction.detach().to("cpu").reshape(-1)
        target = target.detach().to("cpu").reshape(-1)
        valid = (target >= 0) & (target < self.num_classes)
        indices = self.num_classes * target[valid] + prediction[valid]
        self.confusion += torch.bincount(indices, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self):
        matrix = self.confusion.double()
        true_positive = matrix.diag()
        target_count = matrix.sum(dim=1)
        predicted_count = matrix.sum(dim=0)
        union = target_count + predicted_count - true_positive
        iou = true_positive / union.clamp_min(1)
        dice = 2 * true_positive / (target_count + predicted_count).clamp_min(1)
        pixel_accuracy = true_positive.sum() / matrix.sum().clamp_min(1)
        return {
            "pixel_accuracy": float(pixel_accuracy),
            "per_class_iou": [float(value) for value in iou],
            "per_class_dice": [float(value) for value in dice],
            "mean_iou_road_lane": float(iou[1:].mean()),
            "confusion_matrix": self.confusion.tolist(),
        }
