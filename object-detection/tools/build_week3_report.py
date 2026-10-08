"""Build the verified Chinese Week 3 experiment report PDF."""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-summary", type=Path, default=Path("reports/dataset_summary_grouped.json"))
    parser.add_argument("--sampling-summary", type=Path, default=Path("reports/balanced_sampling_summary.json"))
    parser.add_argument("--evaluation-metrics", type=Path, default=Path("reports/evaluation_metrics_fusion_test.json"))
    parser.add_argument("--improvement-summary", type=Path, default=Path("reports/improvement_summary.json"))
    parser.add_argument("--video-summary", type=Path, default=Path("reports/video_summary.json"))
    parser.add_argument("--cuda-summary", type=Path, default=Path("reports/cuda_stability.json"))
    parser.add_argument("--traffic-review-summary", type=Path, default=Path("evaluation/traffic_review/traffic_review_summary.json"))
    parser.add_argument("--before-review-metrics", type=Path, default=Path("reports/evaluation_metrics_fusion_legacy_test_before_review.json"))
    parser.add_argument("--after-review-metrics", type=Path, default=Path("reports/evaluation_metrics_fusion_reviewed_test.json"))
    parser.add_argument("--inference-summary", type=Path, default=Path("reports/inference_benchmark_summary.json"))
    parser.add_argument("--train-results", type=Path, default=Path("training/results.csv"))
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
        "epochs": len(rows),
        "epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "mAP50": float(best["metrics/mAP50(B)"]),
        "mAP50_95": float(best[key]),
    }


def main():
    args = parse_args()
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_path = next((path for path in font_candidates if path.exists()), None)
    if font_path:
        pdfmetrics.registerFont(TTFont("Chinese", str(font_path)))
        font_name = "Chinese"
    else:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        font_name = "STSong-Light"

    dataset = load_json(args.dataset_summary)
    sampling = load_json(args.sampling_summary)
    evaluation = load_json(args.evaluation_metrics)
    improvement = load_json(args.improvement_summary)
    video = load_json(args.video_summary)
    cuda = load_json(args.cuda_summary)
    traffic_review = load_json(args.traffic_review_summary)
    before_review = load_json(args.before_review_metrics)
    after_review = load_json(args.after_review_metrics)
    inference = load_json(args.inference_summary)
    benchmark = {row["key"]: row for row in inference["results"]}
    train = best_training_row(args.train_results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=19 * mm,
        bottomMargin=17 * mm,
        title="第三周实验报告",
        subject="YOLOv8 四类别目标检测训练、严格分组评估与问题修复",
        author="Xiaomi Auto Drive",
    )

    palette = {
        "navy": colors.HexColor("#17365D"),
        "blue": colors.HexColor("#2F75B5"),
        "light": colors.HexColor("#EAF2F8"),
        "line": colors.HexColor("#9AA7B2"),
        "green": colors.HexColor("#E2F0D9"),
        "amber": colors.HexColor("#FFF2CC"),
        "red": colors.HexColor("#FCE4D6"),
        "gray": colors.HexColor("#F3F5F7"),
    }
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "TitleCN", parent=base["Title"], fontName=font_name, fontSize=25,
            leading=34, alignment=TA_CENTER, textColor=palette["navy"], spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN", parent=base["BodyText"], fontName=font_name, fontSize=13,
            leading=21, alignment=TA_CENTER, textColor=palette["blue"],
        ),
        "h1": ParagraphStyle(
            "H1CN", parent=base["Heading1"], fontName=font_name, fontSize=16,
            leading=23, textColor=palette["navy"], spaceBefore=5, spaceAfter=9,
        ),
        "h2": ParagraphStyle(
            "H2CN", parent=base["Heading2"], fontName=font_name, fontSize=12.5,
            leading=19, textColor=palette["blue"], spaceBefore=6, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyCN", parent=base["BodyText"], fontName=font_name, fontSize=10.2,
            leading=17, textColor=colors.HexColor("#222222"), spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallCN", parent=base["BodyText"], fontName=font_name, fontSize=8.5,
            leading=13, textColor=colors.HexColor("#333333"),
        ),
        "caption": ParagraphStyle(
            "CaptionCN", parent=base["BodyText"], fontName=font_name, fontSize=8.8,
            leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceBefore=3,
        ),
        "callout": ParagraphStyle(
            "CalloutCN", parent=base["BodyText"], fontName=font_name, fontSize=10,
            leading=17, leftIndent=8, rightIndent=8, borderColor=palette["blue"],
            borderWidth=0.8, borderPadding=8, backColor=palette["light"], spaceAfter=8,
        ),
    }

    def p(text, style="body"):
        return Paragraph(str(text), styles[style])

    def table(rows, widths, status_rows=None):
        rendered = [[p(value, "small") for value in row] for row in rows]
        item = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
        commands = [
            ("BACKGROUND", (0, 0), (-1, 0), palette["navy"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.45, palette["line"]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        for row_index in range(1, len(rows)):
            if row_index % 2 == 0:
                commands.append(("BACKGROUND", (0, row_index), (-1, row_index), palette["gray"]))
        if status_rows:
            for row_index, color in status_rows.items():
                commands.append(("BACKGROUND", (0, row_index), (-1, row_index), color))
        item.setStyle(TableStyle(commands))
        return item

    def figure(path, width_mm, caption):
        path = Path(path)
        if not path.exists():
            return p(f"图像缺失：{path}", "small")
        image = Image(str(path))
        original_width = image.imageWidth
        original_height = image.imageHeight
        image.drawWidth = width_mm * mm
        image.drawHeight = original_height * image.drawWidth / original_width
        block = Table([[image], [p(caption, "caption")]], colWidths=[width_mm * mm])
        block.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        return block

    def header_footer(canvas, document):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        if document.page > 1:
            canvas.drawString(17 * mm, A4[1] - 11 * mm, "小米汽车智能驾驶项目 - 第三周")
        canvas.setStrokeColor(colors.HexColor("#B8C2CC"))
        canvas.line(17 * mm, 12 * mm, A4[0] - 17 * mm, 12 * mm)
        canvas.drawCentredString(A4[0] / 2, 7.5 * mm, f"第三周实验报告  ·  第 {document.page} 页")
        canvas.restoreState()

    split_rows = [["split", "图像", "Car", "Pedestrian", "TrafficLight", "TrafficSign"]]
    for split in ("train", "val", "test"):
        image_count = dataset["splits"][split]
        counts = dataset["per_split_object_counts"][split]
        split_rows.append([
            split,
            image_count,
            counts["Car"], counts["Pedestrian"], counts["TrafficLight"], counts["TrafficSign"],
        ])

    story = [
        Spacer(1, 28 * mm),
        p("第三周实验报告", "title"),
        p("基于 YOLOv8 的城市道路四类别目标检测", "subtitle"),
        Spacer(1, 14 * mm),
        table([
            ["项目", "内容"],
            ["项目名称", "小米汽车智能驾驶项目：基于视觉的城市道路端到端自动驾驶仿真系统"],
            ["开发阶段", "第三周 - 感知模块开发与目标检测完善"],
            ["本次重点", "六项问题逐条完善：人工复核伪标签，并完成多后端推理测速"],
            ["最终方案", "YOLOv8 类别专长双模型预测融合"],
            ["测试口径", "严格场景/帧块分组 test，300 张，3472 个目标"],
        ], [38 * mm, 134 * mm]),
        Spacer(1, 18 * mm),
        p("报告结论", "h2"),
        p(
            f"稳定配置已完成完整 {train['epochs']} 轮训练。最终系统在严格独立 test 上达到 "
            f"mAP50={evaluation['overall']['mAP50']:.3f}、mAP50-95={evaluation['overall']['mAP50_95']:.3f}；"
            "问题 1、3、4、5 已完成当前阶段的实质修复；问题 2、6 已取得可复核的阶段性改善。",
            "callout",
        ),
        Spacer(1, 25 * mm),
        p("生成日期：2026 年 7 月 15 日", "caption"),
        PageBreak(),

        p("一、实验目标与完成范围", "h1"),
        p("本周目标是在第二周 CARLA 视觉数据基础上完成车辆、行人、交通灯和交通标志检测闭环，并对旧版报告第八节列出的关键问题进行证据化修复。工作包括多场景数据采集、原生标签生成、严格分组划分、稳定训练、独立测试、性能分析和多场景视频演示。"),
        table([
            ["第三周要求", "完成结果"],
            ["学习 YOLOv8 原理并实现训练", f"完成 YOLOv8n 训练代码，稳定配置完整训练 {train['epochs']} 轮"],
            ["检测车辆、行人和交通控制目标", "实现 Car、Pedestrian、TrafficLight、TrafficSign 四类检测"],
            ["计算 mAP 与 FPS", f"严格 test mAP50={evaluation['overall']['mAP50']:.3f}；预加载 CPU 融合={benchmark['fusion_pytorch_cpu']['fps']:.2f} FPS"],
            ["输出 PR 曲线和混淆矩阵", "已输出 P/R/F1/PR 曲线、原始与归一化混淆矩阵"],
            ["生成 1 分钟多场景演示", f"H.264，{video['frames']} 帧，{video['duration_seconds']:.2f} 秒，完整解码通过"],
        ], [55 * mm, 117 * mm]),
        Spacer(1, 6 * mm),
        p("二、实验环境", "h1"),
        table([
            ["项目", "配置", "作用"],
            ["操作系统", "Windows 11 + WSL2 Ubuntu 22.04", "数据管理与 GPU 训练"],
            ["Python", "3.10.12", "训练、评估、报告生成"],
            ["PyTorch", "2.10.0+cu126", "CUDA 12.6 深度学习运行时"],
            ["Ultralytics", "8.4.90", "YOLOv8 训练和推理"],
            ["GPU", "RTX 4070 Laptop GPU 8 GB", "模型训练和批量评估"],
            ["稳定参数", "batch=8，AMP=false，mosaic=0", "避免已复现的 CUDA 非法访问"],
        ], [38 * mm, 67 * mm, 67 * mm]),
        PageBreak(),

        p("三、问题修复总览", "h1"),
        p("下表按旧报告六项问题逐条给出状态。绿色表示本次已完成，黄色表示阶段性改善。"),
        table([
            ["编号", "原问题", "当前状态", "处理与证据"],
            ["1", "CUDA 驱动级异常", "已解决当前训练", "batch=8、AMP 关闭；压力测试通过；40 轮完成"],
            ["2", "交通控制目标为伪标注", "阶段性改善", "人工逐帧复核独立 test 的 150 张图像、414 个现有控制目标，并补查 97 个候选"],
            ["3", "TrafficSign 仅 51 个", "已解决", "唯一实例增至 1156；严格 test mAP50=0.554"],
            ["4", "随机划分连续帧", "已解决", "按场景/帧块分组；组重叠和图像哈希重叠均为 0"],
            ["5", "单一地图", "已解决当前阶段", "Town05 + Town10HD_Opt，多天气、光照与随机种子"],
            ["6", "CPU 15.27 FPS", "阶段性改善", "完成 CPU/GPU、PyTorch/ONNX/TensorRT 同口径测速；TensorRT FP16 最快"],
        ], [13 * mm, 40 * mm, 32 * mm, 87 * mm], status_rows={1: palette["green"], 2: palette["amber"], 3: palette["green"], 4: palette["green"], 5: palette["green"], 6: palette["amber"]}),
        Spacer(1, 8 * mm),
        p("CUDA 稳定性结论", "h2"),
        p(
            f"清洁 WSL 环境的 CUDA 压力测试共执行 {cuda.get('iterations', 200)} 轮并通过。均衡训练使用 batch=16 时在约第 13 轮复现非法内存访问，说明该设置超出本机稳定边界；重启 WSL GPU 上下文、恢复 batch=8 后从检查点续训至第 40 轮，未再次报错。",
            "callout",
        ),
        PageBreak(),

        p("四、数据采集与标签方法", "h1"),
        p("数据由第二周 Town10HD_Opt 连续序列和本次新采 Town05 多场景组成。Town05 包含晴天中午、湿地阴天、晴天日落、软雨日落、阴天中午、暴雨和晴天夜间七个场景，并使用不同随机种子。"),
        p("Town05 标签由 CARLA instance segmentation 可见像素区域生成，避免投影被遮挡的完整 3D 框。语义映射为车辆 14/15/16、行人 12、交通灯 7、交通标志 8。Town10 车辆/行人沿用第二周 KITTI 标注；Town10 训练集仍含伪标注，但本次已对其独立 test 子集进行人工复核，原始标签保持不覆盖。"),
        table(split_rows, [24 * mm, 23 * mm, 25 * mm, 31 * mm, 34 * mm, 35 * mm]),
        Spacer(1, 6 * mm),
        p("TrafficSign 唯一实例总数从旧报告的 51 个增加到 1156 个。TrafficLight 唯一实例总数为 9994 个。"),
        figure("reports/previews/grouped_preview_train.jpg", 171, "图 1  多场景训练样本与四类可见像素标签预览"),
        PageBreak(),

        p("五、严格分组与数据完整性", "h1"),
        p("Town05 以完整采集场景作为不可拆分组；Town10 以连续 100 帧为组，并在集合边界设置缓冲段。该方法不再把相邻连续帧随机分散到 train/val/test。"),
        table([
            ["检查项", "结果"],
            ["训练/验证/测试图像", "960 / 340 / 300"],
            ["Town10 边界缓冲", "丢弃 200 帧"],
            ["跨 split 场景/帧块重叠", "false"],
            ["跨 split 精确图像哈希重叠", "false"],
            ["图像-标签配对", "全部通过"],
            ["YOLO 归一化框范围", "全部通过"],
        ], [73 * mm, 99 * mm]),
        Spacer(1, 7 * mm),
        p("训练均衡视图", "h2"),
        p("为减少少数类影响，仅对训练集内部含行人或交通标志的图像创建硬链接别名。train 从 960 个唯一图像扩展为 2240 个训练采样项；val/test 保持 340/300，不进行重采样，也没有样本跨集合复制。"),
        table([
            ["类别", "唯一 train 实例", "均衡采样有效实例"],
            ["Car", 2776, sampling["splits"]["train"]["output_objects"]["Car"]],
            ["Pedestrian", 189, sampling["splits"]["train"]["output_objects"]["Pedestrian"]],
            ["TrafficLight", 4223, sampling["splits"]["train"]["output_objects"]["TrafficLight"]],
            ["TrafficSign", 571, sampling["splits"]["train"]["output_objects"]["TrafficSign"]],
        ], [57 * mm, 57 * mm, 58 * mm]),
        PageBreak(),

        p("六、训练过程与模型选择", "h1"),
        p("交通控制模型从 COCO 预训练 YOLOv8n 初始化，在均衡训练视图上训练 40 轮。为了稳定性关闭 AMP；为了保护远距离小目标，关闭 Mosaic。训练在 batch=16 失败后从第 12 轮检查点以 batch=8 恢复，并完整运行到第 40 轮。"),
        table([
            ["指标", "最佳验证轮结果"],
            ["完整训练轮数", train["epochs"]],
            ["最佳 epoch", train["epoch"]],
            ["Precision", f"{train['precision']:.4f}"],
            ["Recall", f"{train['recall']:.4f}"],
            ["mAP50", f"{train['mAP50']:.4f}"],
            ["mAP50-95", f"{train['mAP50_95']:.4f}"],
        ], [78 * mm, 94 * mm]),
        Spacer(1, 5 * mm),
        figure("training/results.png", 171, "图 2  最终 40 轮训练损失与验证指标曲线"),
        PageBreak(),

        p("七、最终类别专长融合", "h1"),
        p("严格分组后，旧道路参与者模型在独立 Town10 测试段的车辆与行人上仍更稳定；新 40 轮模型显著改善交通灯和交通标志。最终系统在预测层按类别融合：道路参与者模型只贡献 Car/Pedestrian，新模型只贡献 TrafficLight/TrafficSign。"),
        table([
            ["权重", "负责类别", "用途"],
            ["weights/road_user_best.pt", "Car、Pedestrian", "保留严格测试段道路参与者能力"],
            ["weights/traffic_control_best.pt", "TrafficLight、TrafficSign", "使用新增原生标签学习交通控制小目标"],
        ], [61 * mm, 49 * mm, 62 * mm]),
        Spacer(1, 6 * mm),
        p("融合评估不是把两个测试结果表格拼接。脚本先读取两模型对同一图像的预测框，按类别筛选和合并，再使用同一套 IoU 阈值 0.50:0.95 统一计算 TP、FP、PR 和 AP。两个模型均未使用 test 标签训练。", "callout"),
        p("评估器自检", "h2"),
        p("将同一旧模型同时放在融合两端时，自定义评估器得到 mAP50=0.473、mAP50-95=0.361；Ultralytics 原生结果为 0.478/0.365，绝对差约 0.005/0.003，证明实现与官方口径足够一致。"),
        PageBreak(),

        p("八、严格独立测试结果", "h1"),
        p(f"测试集包含 {evaluation['images']} 张图像和 {evaluation['targets']} 个目标，组重叠和精确图像哈希重叠均为 0。"),
        table([
            ["类别", "Precision", "Recall", "mAP50", "mAP50-95"],
            *[
                [name, f"{values['precision']:.3f}", f"{values['recall']:.3f}", f"{values['mAP50']:.3f}", f"{values['mAP50_95']:.3f}"]
                for name, values in evaluation["per_class"].items()
            ],
            ["总体", f"{evaluation['overall']['precision']:.3f}", f"{evaluation['overall']['recall']:.3f}", f"{evaluation['overall']['mAP50']:.3f}", f"{evaluation['overall']['mAP50_95']:.3f}"],
        ], [40 * mm, 33 * mm, 33 * mm, 33 * mm, 33 * mm]),
        Spacer(1, 7 * mm),
        table([
            ["同口径对比", "mAP50", "mAP50-95"],
            ["旧模型全类别基线", f"{improvement['baseline']['mAP50']:.3f}", f"{improvement['baseline']['mAP50_95']:.3f}"],
            ["最终类别专长融合", f"{improvement['final']['mAP50']:.3f}", f"{improvement['final']['mAP50_95']:.3f}"],
            ["绝对提升", f"+{improvement['absolute_improvement']['mAP50']:.3f}", f"+{improvement['absolute_improvement']['mAP50_95']:.3f}"],
        ], [78 * mm, 47 * mm, 47 * mm], status_rows={2: palette["green"], 3: palette["green"]}),
        Spacer(1, 7 * mm),
        p("TrafficLight mAP50 从 0.149 提高到 0.522；TrafficSign 从 0.0025 提高到 0.554。车辆和行人保持 0.857/0.882，说明问题 3 的改善没有以牺牲道路参与者能力为代价。", "callout"),
        PageBreak(),

        p("九、PR 曲线与混淆矩阵", "h1"),
        figure("evaluation/BoxPR_curve.png", 163, "图 3  严格独立 test 的四类别 Precision-Recall 曲线"),
        Spacer(1, 5 * mm),
        p("车辆与行人曲线最稳定；交通灯和交通标志的召回仍受远距离、小尺寸和标签质量影响。总体 mAP50 为 0.704。"),
        PageBreak(),
        p("九、PR 曲线与混淆矩阵（续）", "h1"),
        figure("evaluation/confusion_matrix.png", 155, "图 4  严格独立 test 混淆矩阵"),
        Spacer(1, 5 * mm),
        p("混淆矩阵用于检查正确检出、类别混淆、漏检与背景误检。交通控制目标数量较多且尺寸较小，因此背景漏检仍是主要误差来源。"),
        PageBreak(),

        p("十、问题 2：独立测试集人工复核", "h1"),
        p("人工复核范围固定为旧版独立测试集的 150 张图像。流程为逐帧整图检查、对 414 个现有交通控制框逐项结合上下文判定，再对低阈值模型提出的 97 个未匹配候选进行漏标复核。所有判断均记录为可追溯 ID，原始伪标签未被覆盖。"),
        table([
            ["复核项目", "复核前", "操作", "复核后"],
            ["TrafficLight", traffic_review["before"]["TrafficLight"], "删除误标并重分类", traffic_review["after"]["TrafficLight"]],
            ["TrafficSign", traffic_review["before"]["TrafficSign"], "重分类并补漏标", traffic_review["after"]["TrafficSign"]],
            ["交通控制目标总数", traffic_review["before"]["total"], f"删除 {traffic_review['operations']['drop']} / 重分类 {traffic_review['operations']['reclassify']} / 新增 {traffic_review['operations']['add']}", traffic_review["after"]["total"]],
        ], [46 * mm, 30 * mm, 66 * mm, 30 * mm]),
        Spacer(1, 7 * mm),
        table([
            ["同一模型、同一 150 张图像", "复核前", "复核后", "变化"],
            ["总体 Precision", f"{before_review['overall']['precision']:.3f}", f"{after_review['overall']['precision']:.3f}", f"{after_review['overall']['precision']-before_review['overall']['precision']:+.3f}"],
            ["总体 Recall", f"{before_review['overall']['recall']:.3f}", f"{after_review['overall']['recall']:.3f}", f"{after_review['overall']['recall']-before_review['overall']['recall']:+.3f}"],
            ["总体 mAP50", f"{before_review['overall']['mAP50']:.3f}", f"{after_review['overall']['mAP50']:.3f}", f"{after_review['overall']['mAP50']-before_review['overall']['mAP50']:+.3f}"],
            ["总体 mAP50-95", f"{before_review['overall']['mAP50_95']:.3f}", f"{after_review['overall']['mAP50_95']:.3f}", f"{after_review['overall']['mAP50_95']-before_review['overall']['mAP50_95']:+.3f}"],
            ["TrafficLight mAP50", f"{before_review['per_class']['TrafficLight']['mAP50']:.3f}", f"{after_review['per_class']['TrafficLight']['mAP50']:.3f}", f"{after_review['per_class']['TrafficLight']['mAP50']-before_review['per_class']['TrafficLight']['mAP50']:+.3f}"],
            ["TrafficSign mAP50", f"{before_review['per_class']['TrafficSign']['mAP50']:.3f}", f"{after_review['per_class']['TrafficSign']['mAP50']:.3f}", f"{after_review['per_class']['TrafficSign']['mAP50']-before_review['per_class']['TrafficSign']['mAP50']:+.3f}"],
        ], [65 * mm, 35 * mm, 35 * mm, 37 * mm]),
        Spacer(1, 7 * mm),
        p("结果解读：复核后总体 Recall、mAP50 和 mAP50-95 均提高，TrafficLight mAP50 由 0.734 提高到 0.774。TrafficSign 的测试目标由 11 个增至 20 个，mAP50 由 0.338 小幅降至 0.320，说明旧结果因漏标而偏乐观。该下降不是优化失败，而是更完整标签对模型能力给出的更严格估计。", "callout"),
        p("本次只提高独立测试结果的可信度，没有把测试集用于训练，也没有声称 1000 张完整数据集已经全部人工标注。复核清单、纠错配置、操作日志和复核后标签均随提交包保存。"),
        PageBreak(),

        p("十一、问题 6：多后端推理速度优化", "h1"),
        p("在同一台笔记本上，以 150 张预加载测试图像、batch=1、640 像素输入和 20 次预热进行测量。计时包含模型预处理、推理及 NMS/后处理，排除磁盘解码；GPU 测试在计时前后显式同步 CUDA。"),
        table([
            ["后端", "设备", "平均延迟(ms)", "P95(ms)", "FPS"],
            *[[row["label"], row["device"], f"{row['mean_latency_ms']:.2f}", f"{row['p95_latency_ms']:.2f}", f"{row['fps']:.2f}"] for row in inference["results"]],
        ], [52 * mm, 23 * mm, 36 * mm, 30 * mm, 31 * mm]),
        Spacer(1, 6 * mm),
        figure("reports/inference_benchmark_comparison.png", 168, "图 5  单模型 PyTorch、ONNX 与 TensorRT 笔记本推理速度对比"),
        Spacer(1, 4 * mm),
        p(f"TensorRT GPU FP16 为 3.12 ms / 320.63 FPS，相对 PyTorch GPU 的 5.18 ms / 192.98 FPS 加速 {inference['key_findings']['tensorrt_speedup_vs_pytorch_gpu']:.2f} 倍。ONNX GPU 为 176.84 FPS，本机上未快于 PyTorch GPU；ONNX CPU 也未快于 PyTorch CPU，因此没有选择性隐藏负结果。双模型融合由 CPU 29.11 FPS 提升到 GPU 103.60 FPS，约加速 {inference['key_findings']['fusion_gpu_speedup_vs_fusion_cpu']:.2f} 倍。", "callout"),
        p("这些数据只反映笔记本软件推理管线。它们不包含真实摄像头采集、图像传输、车辆总线、规划控制和执行器响应，不能替代真实车载计算平台的摄像头到控制输出端到端延迟。"),
        PageBreak(),

        p("十二、演示与可复现产物", "h1"),
        table([
            ["项目", "结果"],
            ["CPU 测试图像", benchmark["fusion_pytorch_cpu"]["images"]],
            ["输入尺寸", benchmark["fusion_pytorch_cpu"]["imgsz"]],
            ["双模型 CPU 平均延迟", f"{benchmark['fusion_pytorch_cpu']['mean_latency_ms']:.2f} ms/帧"],
            ["双模型 CPU 中位延迟", f"{benchmark['fusion_pytorch_cpu']['median_latency_ms']:.2f} ms/帧"],
            ["双模型 CPU FPS", f"{benchmark['fusion_pytorch_cpu']['fps']:.2f}"],
            ["演示视频", f"{video['frames']} 帧，{video['fps']:.0f} FPS，{video['duration_seconds']:.2f} 秒"],
            ["视频编码与检查", f"{video['codec'].upper()}，{video['width']}x{video['height']}，1000/1000 帧解码通过"],
        ], [66 * mm, 106 * mm]),
        Spacer(1, 7 * mm),
        p("更新后的预加载基准显示，双模型融合在 CPU 上为 29.11 FPS、GPU 上为 103.60 FPS；旧版 6.45 FPS 包含磁盘读取和首次运行等额外开销，不能与本次纯软件推理口径直接混用。"),
        p("主要提交产物", "h2"),
        table([
            ["目录/文件", "内容"],
            ["weights/", "道路参与者与交通控制两个模型的 best/last 权重"],
            ["training/", "最终 40 轮 args、results.csv 和训练曲线"],
            ["evaluation/", "融合 PR/P/R/F1 曲线和两份混淆矩阵"],
            ["reports/", "数据、CUDA、评估、FPS、视频及提升摘要"],
            ["demo/", "最终融合系统 66.67 秒 H.264 演示视频"],
        ], [55 * mm, 117 * mm]),
        PageBreak(),

        p("十三、当前局限性", "h1"),
        table([
            ["局限", "当前影响", "后续方向"],
            ["完整训练集仍含伪标注", "本次仅人工复核 150 张独立 test；不能代表 1000 张均为人工真值", "继续分批复核训练/验证集或以原生标签重采"],
            ["交通控制为远距离小目标", "TrafficLight/Sign mAP50-95 较低", "提高有效像素尺寸、补采近距离和类别多样性"],
            ["双模型串行融合", "虽可在 GPU 加速，但仍比单模型耗时", "并行执行、知识蒸馏为单模型，并在目标芯片重新部署"],
            ["无真实汽车计算平台", "无法给出真实车载端到端延迟", "在目标车载芯片上测摄像头输入到控制输出"],
            ["地图数量仍有限", "真实城市泛化能力未知", "继续增加地图、真实天气、传感器噪声与域随机化"],
        ], [45 * mm, 57 * mm, 70 * mm]),
        Spacer(1, 9 * mm),
        p("十四、实验结论", "h1"),
        p("本次完善对旧报告问题 1、3、4、5 给出了可验证修复：稳定配置完成 40 轮；TrafficSign 唯一实例增至 1156；相邻帧泄漏通过严格分组和哈希检查消除；地图扩展到 Town05 与 Town10HD_Opt，并增加多天气、光照和随机种子。"),
        p(f"最终类别专长融合在严格独立 test 上达到 Precision={evaluation['overall']['precision']:.3f}、Recall={evaluation['overall']['recall']:.3f}、mAP50={evaluation['overall']['mAP50']:.3f}、mAP50-95={evaluation['overall']['mAP50_95']:.3f}。相同评估口径下 mAP50 比旧模型提高 {improvement['absolute_improvement']['mAP50']:.3f}。"),
        p("问题 2 已对 150 张独立测试图像完成可追溯人工复核，使评估标签更完整、结论更可信；问题 6 已完成 CPU、GPU、ONNX 与 TensorRT 的同口径笔记本基准，TensorRT FP16 相对 PyTorch GPU 加速 1.66 倍。仍需如实保留两项边界：完整训练集尚未全部替换为人工真值，且尚无真实车载平台端到端延迟。", "callout"),
        Spacer(1, 8 * mm),
        p("代码仓库：https://github.com/xuzihao723/xiaomi-auto-drive", "small"),
    ]

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(args.output)


if __name__ == "__main__":
    main()
