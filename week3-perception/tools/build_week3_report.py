import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build the Week 3 evaluation PDF.")
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument("--evaluation-metrics", type=Path, required=True)
    parser.add_argument("--fps-summary", type=Path, required=True)
    parser.add_argument("--video-summary", type=Path, required=True)
    parser.add_argument("--train-results", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def best_training_row(path):
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: float(row[key]))
    return {
        "epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "mAP50": float(best["metrics/mAP50(B)"]),
        "mAP50_95": float(best[key]),
    }


def main():
    args = parse_args()
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font_candidates = [
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    font_path = next((path for path in font_candidates if path.exists()), None)
    if font_path is not None:
        pdfmetrics.registerFont(TTFont("Chinese", str(font_path)))
        font_name = "Chinese"
    else:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        font_name = "STSong-Light"

    dataset = load_json(args.dataset_summary)
    evaluation = load_json(args.evaluation_metrics)
    fps = load_json(args.fps_summary)
    video = load_json(args.video_summary)
    train = best_training_row(args.train_results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="第三周目标检测模型性能评估报告",
        author="Xiaomi AD Week 3",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ChineseTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=22,
        leading=32,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
    )
    h1 = ParagraphStyle(
        "ChineseH1",
        parent=styles["Heading1"],
        fontName=font_name,
        fontSize=15,
        leading=22,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=8,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        "ChineseBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=17,
        spaceAfter=6,
    )
    small = ParagraphStyle(
        "ChineseSmall", parent=body, fontSize=9, leading=14
    )

    def p(text, style=body):
        return Paragraph(text, style)

    def table(rows, widths):
        rendered = [[p(str(value), small) for value in row] for row in rows]
        item = Table(rendered, colWidths=widths, repeatRows=1)
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9AA7B2")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return item

    story = [
        Spacer(1, 32 * mm),
        p("小米汽车智能驾驶项目", title),
        Spacer(1, 8 * mm),
        p("第三周目标检测模型性能评估报告", title),
        Spacer(1, 16 * mm),
        p("基于 YOLOv8 的车辆、行人、交通灯与交通标志检测", ParagraphStyle("subtitle", parent=body, alignment=TA_CENTER, fontSize=13)),
        Spacer(1, 45 * mm),
        table(
            [
                ["项目", "内容"],
                ["开发阶段", "第三周 - 感知模块开发"],
                ["数据来源", "CARLA Town10HD_Opt 自制仿真数据集"],
                ["模型", "YOLOv8n 微调模型"],
                ["报告内容", "mAP、FPS、混淆矩阵、PR 曲线与局限性"],
            ],
            [42 * mm, 115 * mm],
        ),
        PageBreak(),
        p("一、项目要求与完成情况", h1),
        table(
            [
                ["第三周要求", "完成情况"],
                ["学习 YOLOv8 原理", "工程说明中记录网络结构、训练和评估流程"],
                ["使用自制仿真数据微调", f"使用 {dataset['source_images']} 帧 CARLA 图像完成微调"],
                ["车辆、行人、交通标志等实时检测", "实现 Car、Pedestrian、TrafficLight、TrafficSign 四类检测"],
                ["评估 mAP、FPS", "已完成独立 test 集评估与 CPU FPS benchmark"],
                ["混淆矩阵与 PR 曲线", "图表已在本报告中完整呈现"],
                ["1 分钟不同场景演示", f"{video['duration_seconds']:.2f} 秒，覆盖连续城市道路场景"],
            ],
            [63 * mm, 94 * mm],
        ),
        Spacer(1, 5 * mm),
        p("二、数据集与标注方法", h1),
        p(
            "车辆和行人框来自第二周 CARLA/KITTI 真值标签。交通灯和交通标志由 COCO 预训练 YOLOv8 在同一批自制仿真图像上生成伪标注，并通过置信度、尺寸规则和可视化抽查过滤。该处理用于补齐原始数据没有交通控制目标标签的问题。"
        ),
        table(
            [["类别", "目标数量"]]
            + [[name, value] for name, value in dataset["class_counts"].items()],
            [80 * mm, 77 * mm],
        ),
        Spacer(1, 4 * mm),
        p(
            f"数据按固定随机种子 {dataset['seed']} 划分为 train/val/test："
            f"{dataset['splits']['train']}/{dataset['splits']['val']}/{dataset['splits']['test']}。"
        ),
        p("三、训练配置与训练结果", h1),
        p("训练使用 YOLOv8n 预训练权重、768 像素输入、固定随机种子；关闭 AMP 并使用 workers=0，以避免原环境中的 CUDA 中断。"),
        table(
            [
                ["指标", "最佳训练轮结果"],
                ["Epoch", train["epoch"]],
                ["Precision", f"{train['precision']:.4f}"],
                ["Recall", f"{train['recall']:.4f}"],
                ["mAP50", f"{train['mAP50']:.4f}"],
                ["mAP50-95", f"{train['mAP50_95']:.4f}"],
            ],
            [80 * mm, 77 * mm],
        ),
        PageBreak(),
        p("四、独立测试集性能", h1),
    ]

    metric_rows = [["类别", "Precision", "Recall", "mAP50", "mAP50-95"]]
    for class_name, values in evaluation["per_class"].items():
        metric_rows.append(
            [
                class_name,
                f"{values['precision']:.3f}",
                f"{values['recall']:.3f}",
                f"{values['mAP50']:.3f}",
                f"{values['mAP50_95']:.3f}",
            ]
        )
    metric_rows.append(
        [
            "Overall",
            "-",
            "-",
            f"{evaluation['overall']['mAP50']:.3f}",
            f"{evaluation['overall']['mAP50_95']:.3f}",
        ]
    )
    story.extend(
        [
            table(metric_rows, [37 * mm, 30 * mm, 30 * mm, 30 * mm, 30 * mm]),
            Spacer(1, 6 * mm),
            p("五、PR 曲线", h1),
            Image(str(args.evaluation_dir / "BoxPR_curve.png"), width=158 * mm, height=105 * mm),
            p("PR 曲线展示各类别在不同置信度阈值下的精确率与召回率关系。", small),
            PageBreak(),
            p("六、混淆矩阵", h1),
            Image(str(args.evaluation_dir / "confusion_matrix.png"), width=158 * mm, height=125 * mm),
            p("混淆矩阵用于观察正确检测、类别混淆以及漏检情况。", small),
            Spacer(1, 5 * mm),
            p("七、推理速度", h1),
            table(
                [
                    ["指标", "结果"],
                    ["设备", fps["device"]],
                    ["测试图像", fps["images"]],
                    ["输入尺寸", fps["imgsz"]],
                    ["平均延迟", f"{fps['mean_latency_ms']:.2f} ms"],
                    ["中位延迟", f"{fps['median_latency_ms']:.2f} ms"],
                    ["FPS", f"{fps['fps']:.2f}"],
                ],
                [80 * mm, 77 * mm],
            ),
            PageBreak(),
            p("八、演示视频", h1),
            table(
                [
                    ["属性", "结果"],
                    ["文件", Path(video["output"]).name],
                    ["分辨率", f"{video['width']} x {video['height']}"],
                    ["帧数", video["frames"]],
                    ["帧率", f"{video['fps']:.2f} FPS"],
                    ["时长", f"{video['duration_seconds']:.2f} 秒"],
                ],
                [80 * mm, 77 * mm],
            ),
            Spacer(1, 5 * mm),
            p("演示视频覆盖城区直道、路口、密集车辆和交通灯/标志场景，并叠加四类目标框、类别名称与置信度。"),
            p("九、当前局限性", h1),
            p("1. 交通灯与交通标志使用伪标注，测试指标反映模型对伪标注规则的拟合程度，不能完全替代人工标注测试集。"),
            p("2. 数据来自单一 CARLA 地图与连续采集序列，天气、光照和地图多样性仍有限。"),
            p("3. TrafficSign 样本量明显少于其他类别，小目标和远距离标志仍可能漏检。"),
            p("4. CPU FPS 包含图像读取和 Python 调度开销；不同硬件上的实时性能会有差异。"),
            p("十、结论", h1),
            p("本周已完成从自制仿真数据整理、四类目标标注、YOLOv8 微调、独立测试集评估到一分钟演示视频的完整闭环，并提交训练代码、权重、混淆矩阵、PR 曲线和性能报告。"),
        ]
    )

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawCentredString(A4[0] / 2, 9 * mm, f"第 {document.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Report written: {args.output}")


if __name__ == "__main__":
    main()
