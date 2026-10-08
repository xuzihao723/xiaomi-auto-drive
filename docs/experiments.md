# Experiments and evidence

[Project overview](../README.md) · [Architecture](architecture.md)

Numbers below summarize existing published experiment records. Reorganizing the repository does not rerun or replace those experiments.

## Object detection

The final dataset has 1,600 images: 960 train, 340 validation and 300 test. Final splits are grouped by scene or continuous-frame block; the recorded group and exact-image-hash overlaps across splits are zero.

| Method on the same strict 300-image test | mAP50 | mAP50–95 |
| --- | ---: | ---: |
| Previous single model for four classes | 0.473 | 0.361 |
| Class-specialized model fusion | 0.704 | 0.403 |
| Absolute difference | +0.231 | +0.042 |

The fused method has overall Precision 0.880 and Recall 0.590. The test contains 3,472 targets. Traffic lights/signs remain harder than cars/pedestrians, especially at small image sizes.

Evidence: [final metrics](../object-detection/reports/evaluation_metrics_fusion_test.json), [dataset summary](../object-detection/reports/dataset_summary_grouped.json), [split validation](../object-detection/reports/grouped_validation.json), [full experiment guide](../object-detection/README.zh-CN.md).

### Separate label-review experiment

The manually reviewed 150-image **old independent test** is not the final 300-image grouped test. Corrections to 414 existing traffic-control labels and 97 missing-label candidates were recorded without overwriting the original labels. These review results establish label quality; they must not be mixed into the final test table.

Evidence: [review records](../object-detection/evaluation/traffic_review/).

### Deployment benchmark scope

The recorded benchmark uses 150 preloaded images, batch 1, image size 640 and 20 warmup iterations. Timing includes preprocessing, inference and NMS/postprocessing; disk decoding is excluded.

| Configuration | Mean latency | FPS |
| --- | ---: | ---: |
| PyTorch GPU | 5.18 ms | 192.98 |
| TensorRT GPU FP16 | 3.12 ms | 320.63 |
| Dual-model fusion GPU | 9.65 ms | 103.60 |

The TensorRT entry benchmarks a single exported model, not the complete fused pipeline. These are notebook software benchmarks, not automotive camera-to-control latency.

Evidence: [benchmark JSON records](../object-detection/reports/inference_benchmarks/), [comparison chart](../object-detection/reports/inference_benchmark_comparison.png).

## Road segmentation

The three labels are Background, DrivableArea and LaneMarking. The main 1,225 image/mask pairs use train/val/test = 800/250/175. Another 75 images form the separate locked-model audit, bringing total recorded pairs to 1,300.

| Evaluation set | Images | Pixel accuracy | Road IoU | Lane IoU | Road/Lane mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed test after night adaptation | 175 | 0.901 | 0.749 | 0.603 | 0.676 |
| Separate audit after model fixation | 75 | 0.960 | 0.877 | 0.715 | 0.796 |

Road/Lane mIoU averages the two foreground classes and excludes background. The audit consists of the `town10_cloudy_night_audit` scenario; it is not a broader replacement for the fixed multi-scenario test. Neither metric establishes real-road generalization.

| Fixed-test scenario | Before adaptation mIoU | After adaptation mIoU |
| --- | ---: | ---: |
| Town10 clear night | 0.004 | 0.778 |
| Town05 heavy rain | 0.659 | 0.637 |

Evidence: [fixed test](../road-segmentation/reports/test_evaluation_night_adapt/test_metrics.json), [separate audit](../road-segmentation/reports/audit_evaluation/audit_metrics.json), [adaptation comparison](../road-segmentation/reports/night_adaptation_comparison.json), [full guide](../road-segmentation/README.zh-CN.md).

## Visual integration

The published video contains 900 frames at 640 × 360 and 15 FPS, for 60 seconds of playback. It combines detection boxes, drivable-area visualization and fitted lane geometry. This is a rendered perception demonstration, not closed-loop autonomous driving. The playback FPS does not measure model inference throughput.

Evidence: [video summary](../road-segmentation/reports/integrated_video_summary.json), [video validation](../road-segmentation/reports/video_validation.json), [download and checksum](../road-segmentation/demo/README.md).
