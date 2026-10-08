"""Train U-Net for background, drivable area and lane marking classes."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CarlaSegmentationDataset
from metrics import SegmentationMetrics
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weights", nargs=3, type=float, default=[0.2, 1.0, 4.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5, help="Early-stop patience; 0 disables")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a saved checkpoint")
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Load model weights but restart optimizer/scheduler for a stable recovery",
    )
    parser.add_argument(
        "--reset-best",
        action="store_true",
        help="Select a new best checkpoint when the validation distribution changed",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits, target, num_classes=3):
    probabilities = logits.softmax(dim=1)
    one_hot = nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice[1:].mean()


def run_epoch(model, loader, optimizer, device, criterion, scaler, amp, training):
    model.train(training)
    metrics = SegmentationMetrics(3)
    total_loss = 0.0
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks) + 0.5 * dice_loss(logits, masks)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach()) * images.shape[0]
        metrics.update(logits.argmax(dim=1), masks)
    result = metrics.compute()
    result["loss"] = total_loss / len(loader.dataset)
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    train_dataset = CarlaSegmentationDataset(args.data, "train", (args.width, args.height), augment=True)
    val_dataset = CarlaSegmentationDataset(args.data, "val", (args.width, args.height), augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model(base_channels=args.base_channels).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    history_path = args.output / "results.csv"
    history = []
    if history_path.exists():
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            history = [dict(row) for row in csv.DictReader(handle)]
        for row in history:
            for key in row:
                row[key] = int(row[key]) if key == "epoch" else float(row[key])
    start_epoch = 1
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        history = [row for row in history if int(row["epoch"]) < start_epoch]

    best_miou = max((float(row["val_miou"]) for row in history), default=-1.0)
    if resume_checkpoint is not None:
        best_miou = max(
            best_miou,
            float(resume_checkpoint.get("metrics", {}).get("mean_iou_road_lane", -1.0)),
        )
    if args.reset_best:
        best_miou = -1.0

    remaining_epochs = max(1, args.epochs - start_epoch + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    if resume_checkpoint is not None:
        has_full_state = "optimizer" in resume_checkpoint and "scheduler" in resume_checkpoint
        if has_full_state and not args.reset_optimizer:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        if args.amp and "scaler" in resume_checkpoint and not args.reset_optimizer:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        print(
            f"resumed={args.resume} start_epoch={start_epoch} amp={args.amp} "
            f"optimizer_restored={has_full_state and not args.reset_optimizer}",
            flush=True,
        )

    if start_epoch > args.epochs:
        raise ValueError(f"Checkpoint epoch {start_epoch - 1} already reaches requested epochs={args.epochs}")

    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_result = run_epoch(model, train_loader, optimizer, device, criterion, scaler, args.amp, True)
        val_result = run_epoch(model, val_loader, optimizer, device, criterion, scaler, args.amp, False)
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_result["loss"],
            "train_miou": train_result["mean_iou_road_lane"],
            "val_loss": val_result["loss"],
            "val_miou": val_result["mean_iou_road_lane"],
            "val_road_iou": val_result["per_class_iou"][1],
            "val_lane_iou": val_result["per_class_iou"][2],
            "val_pixel_accuracy": val_result["pixel_accuracy"],
        }
        history.append(row)
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": val_result,
            "model_config": {"num_classes": 3, "base_channels": args.base_channels},
            "image_size": [args.width, args.height],
            "classes": ["Background", "DrivableArea", "LaneMarking"],
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
        torch.save(checkpoint, args.output / "last.pt")
        if val_result["mean_iou_road_lane"] > best_miou:
            best_miou = val_result["mean_iou_road_lane"]
            torch.save(checkpoint, args.output / "best.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        (args.output / "train_config.json").write_text(
            json.dumps(config | {"device_used": str(device)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"early_stop epoch={epoch} patience={args.patience} best_miou={best_miou:.6f}",
                flush=True,
            )
            break
    print(f"best_miou={best_miou:.6f} weights={args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
