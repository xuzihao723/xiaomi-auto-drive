"""Compare two segmentation evaluation JSON files and report metric deltas."""

import argparse
import json
from pathlib import Path


def compact(metrics):
    return {
        "pixel_accuracy": metrics["pixel_accuracy"],
        "road_iou": metrics["per_class_iou"][1],
        "lane_iou": metrics["per_class_iou"][2],
        "road_lane_miou": metrics["mean_iou_road_lane"],
    }


def delta(before, after):
    return {key: after[key] - before[key] for key in before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before_raw = json.loads(args.before.read_text(encoding="utf-8"))
    after_raw = json.loads(args.after.read_text(encoding="utf-8"))
    if before_raw["samples"] != after_raw["samples"]:
        raise ValueError("Evaluation files use different test-set sizes")
    before_overall = compact(before_raw["overall"])
    after_overall = compact(after_raw["overall"])
    scenarios = {}
    for name in sorted(set(before_raw["by_scenario"]) & set(after_raw["by_scenario"])):
        old = compact(before_raw["by_scenario"][name])
        new = compact(after_raw["by_scenario"][name])
        scenarios[name] = {"before": old, "after": new, "delta": delta(old, new)}
    result = {
        "test_samples": before_raw["samples"],
        "before_weights": before_raw["weights"],
        "after_weights": after_raw["weights"],
        "overall": {
            "before": before_overall,
            "after": after_overall,
            "delta": delta(before_overall, after_overall),
        },
        "by_scenario": scenarios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
