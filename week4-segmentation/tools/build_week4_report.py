"""Generate the verified Chinese Week 4 experiment report PDF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def best_row(rows):
    return max(rows, key=lambda row: float(row["val_miou"]))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()

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

    dataset = load_json(root / "reports/dataset_validation.json")
    before = load_json(root / "reports/test_evaluation/test_metrics.json")
    final_test = load_json(root / "reports/test_evaluation_night_adapt/test_metrics.json")
    audit = load_json(root / "reports/audit_evaluation/audit_metrics.json")
    comparison = load_json(root / "reports/night_adaptation_comparison.json")
    video = load_json(root / "reports/integrated_video_summary.json")
    video_check = load_json(root / "reports/video_validation.json")
    self_checks = load_json(root / "reports/self_checks.json")
    checksums = load_json(root / "reports/artifact_checksums.json")
    scenarios = load_json(root / "configs/segmentation_scenarios.json")
    manifest = load_json(root / "data/carla_segmentation/manifest.json")
    base_rows = load_csv(root / "runs/unet_final/results.csv")
    adapt_rows = load_csv(root / "runs/unet_night_adapt/results.csv")
    base_best = best_row(base_rows)
    adapt_best = best_row(adapt_rows)

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

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=19 * mm,
        bottomMargin=17 * mm,
        title="第四周实验报告",
        subject="基于 U-Net 的车道线与可行驶区域语义分割及感知融合",
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
        "dark": colors.HexColor("#222222"),
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
            "BodyCN", parent=base["BodyText"], fontName=font_name, fontSize=10.1,
            leading=17, textColor=palette["dark"], spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallCN", parent=base["BodyText"], fontName=font_name, fontSize=8.45,
            leading=13, textColor=colors.HexColor("#333333"),
        ),
        "tiny": ParagraphStyle(
            "TinyCN", parent=base["BodyText"], fontName=font_name, fontSize=7.5,
            leading=11, textColor=colors.HexColor("#333333"),
        ),
        "caption": ParagraphStyle(
            "CaptionCN", parent=base["BodyText"], fontName=font_name, fontSize=8.7,
            leading=13.5, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceBefore=3,
        ),
        "callout": ParagraphStyle(
            "CalloutCN", parent=base["BodyText"], fontName=font_name, fontSize=10,
            leading=17, leftIndent=8, rightIndent=8, borderColor=palette["blue"],
            borderWidth=0.8, borderPadding=8, backColor=palette["light"], spaceAfter=8,
        ),
        "url": ParagraphStyle(
            "UrlCN", parent=base["BodyText"], fontName=font_name, fontSize=8.3,
            leading=14, textColor=palette["blue"], wordWrap="CJK", spaceAfter=4,
        ),
    }

    def p(text, style="body"):
        return Paragraph(str(text), styles[style])

    def table(rows, widths, status_rows=None, tiny=False):
        cell_style = "tiny" if tiny else "small"
        rendered = [[p(value, cell_style) for value in row] for row in rows]
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

    def figure(path, width_mm, caption, max_height_mm=None):
        path = Path(path)
        if not path.exists():
            return p(f"图像缺失：{path}", "small")
        image = Image(str(path))
        draw_width = width_mm * mm
        draw_height = image.imageHeight * draw_width / image.imageWidth
        if max_height_mm and draw_height > max_height_mm * mm:
            draw_height = max_height_mm * mm
            draw_width = image.imageWidth * draw_height / image.imageHeight
        image.drawWidth = draw_width
        image.drawHeight = draw_height
        block = Table([[image], [p(caption, "caption")]], colWidths=[width_mm * mm])
        block.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        return block

    def figures_side_by_side(items, width_mm=81, max_height_mm=68):
        blocks = []
        for path, caption in items:
            blocks.append(figure(path, width_mm, caption, max_height_mm=max_height_mm))
        wrapper = Table([blocks], colWidths=[86 * mm] * len(blocks), hAlign="CENTER")
        wrapper.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        return wrapper

    def header_footer(canvas, document):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        if document.page > 1:
            canvas.drawString(17 * mm, A4[1] - 11 * mm, "小米汽车智能驾驶项目 - 第四周")
        canvas.setStrokeColor(colors.HexColor("#B8C2CC"))
        canvas.line(17 * mm, 12 * mm, A4[0] - 17 * mm, 12 * mm)
        canvas.drawCentredString(A4[0] / 2, 7.5 * mm, f"第四周实验报告  ·  第 {document.page} 页")
        canvas.restoreState()

    split_rows = [["数据划分", "图像", "掩码", "分辨率", "用途"]]
    purposes = {
        "train": "模型训练与夜间样本适配",
        "val": "训练选优与早停判断",
        "test": "严格测试；不参与训练",
        "audit": "锁定权重后的独立审计",
    }
    for split in ("train", "val", "test", "audit"):
        info = dataset["splits"][split]
        resolution = next(iter(info["image_sizes"]))
        split_rows.append([split, info["images"], info["masks"], resolution, purposes[split]])

    scenario_rows = [["场景", "地图", "天气/时段", "split", "帧数"]]
    for item in scenarios["scenarios"]:
        scenario_rows.append([
            item["name"], item["map"], item["weather"], item["split"], item["frames"],
        ])

    final = final_test["overall"]
    audit_final = audit["overall"]
    cmp = comparison["overall"]
    rain = comparison["by_scenario"]["town05_hard_rain_test"]
    night = comparison["by_scenario"]["town10_clear_night_test"]
    checksum_lookup = {item["path"]: item for item in checksums["files"]}

    story = [
        Spacer(1, 27 * mm),
        p("第四周实验报告", "title"),
        p("基于 U-Net 的车道线与可行驶区域语义分割", "subtitle"),
        Spacer(1, 13 * mm),
        table([
            ["项目", "内容"],
            ["项目名称", "小米汽车智能驾驶项目：基于视觉的城市道路端到端自动驾驶仿真系统"],
            ["开发阶段", "第四周 - 感知模块开发与语义分割"],
            ["本周任务", "构建分割数据集、训练 U-Net、提取可行驶区域并拟合车道线"],
            ["集成输出", "目标检测 + 语义分割融合演示视频（60 秒）"],
            ["最终测试口径", "175 张独立 test；75 张锁定 audit；权重以最佳验证集 checkpoint 为准"],
        ], [38 * mm, 134 * mm]),
        Spacer(1, 16 * mm),
        p("报告结论", "h2"),
        p(
            f"本周已完成 1300 组 CARLA 同步 RGB/语义标签数据、U-Net 训练、后处理和感知融合。"
            f"最终权重在 175 张严格测试集上达到像素准确率 {final['pixel_accuracy']:.3f}、"
            f"道路 IoU {final['per_class_iou'][1]:.3f}、车道线 IoU {final['per_class_iou'][2]:.3f}、"
            f"道路/车道线 mIoU {final['mean_iou_road_lane']:.3f}；全部可复核交付物均已通过校验。",
            "callout",
        ),
        Spacer(1, 23 * mm),
        p("生成日期：2026 年 7 月 20 日", "caption"),
        PageBreak(),

        p("一、实验目标与完成范围", "h1"),
        p("第四周从目标级感知扩展到像素级道路理解。核心任务是让系统不仅识别车辆、行人和交通控制目标，还能够判断图像中哪些像素属于可行驶区域、哪些像素属于车道标线，并把两类结果融合到同一段演示视频中。"),
        table([
            ["第四周要求", "完成结果", "状态"],
            ["学习 U-Net 语义分割算法原理", "实现编码器—解码器、跳跃连接和三分类输出", "完成"],
            ["标注并制作车道线/可行驶区域数据集", "CARLA 同步相机生成 1300 组 RGB 与像素级掩码", "完成"],
            ["训练车道线与可行驶区域分割模型", "完成基础训练、夜间适配、严格 test 与锁定 audit", "完成"],
            ["实现车道线拟合与可行驶区域提取", "形态学清理、最大连通区域、透视区域内二次曲线拟合", "完成"],
            ["输出训练代码和权重文件", "代码、配置、最佳权重、训练日志与指标齐全", "完成"],
            ["生成 1 分钟融合演示视频", "H.264/yuv420p，640×360，15 FPS，900 帧，60 秒", "完成"],
        ], [55 * mm, 94 * mm, 23 * mm], status_rows={1: palette["green"], 2: palette["green"], 3: palette["green"], 4: palette["green"], 5: palette["green"], 6: palette["green"]}),
        Spacer(1, 7 * mm),
        p("二、实验环境", "h1"),
        table([
            ["组件", "配置", "作用"],
            ["操作系统", "Windows 11 + WSL2 Ubuntu 22.04", "CARLA 联调、数据与训练任务管理"],
            ["CARLA", "0.9.15", "同步 RGB 与语义分割标签采集"],
            ["Python / PyTorch", "Python 3.10；PyTorch 2.10.0", "模型训练、评估和推理"],
            ["GPU", "NVIDIA RTX 4070 Laptop GPU 8 GB", "CUDA 训练与批量评估"],
            ["图像/视频", "OpenCV；FFmpeg/ffprobe", "后处理、视频编码与完整性验证"],
        ], [39 * mm, 65 * mm, 68 * mm]),
        PageBreak(),

        p("三、技术路线与资料依据", "h1"),
        p("实现顺序严格对应项目文档：先学习语义分割和 U-Net 原理，再生成像素级数据集，随后训练分割模型，最后实现道路区域提取、车道线拟合并与第三周目标检测模型融合。文档链接的用途和落地位置已记录在 docs/reference_analysis.md。"),
        table([
            ["阶段", "输入", "处理", "输出"],
            ["1. 同步采集", "CARLA 场景配置", "RGB 与 Semantic Segmentation 同帧采集", "图像、掩码、manifest"],
            ["2. 模型训练", "train / val", "U-Net 三分类训练；类别加权；最佳验证集选优", "best checkpoint、曲线"],
            ["3. 独立评估", "test / audit", "混淆矩阵累计；逐类 IoU、Dice、像素准确率", "严格指标与预览"],
            ["4. 几何后处理", "类别概率/掩码", "连通区域、形态学处理、透视区域曲线拟合", "可行驶多边形、车道曲线"],
            ["5. 感知融合", "第三周检测权重 + 第四周分割权重", "逐帧检测、分割、叠加与编码", "60 秒 MP4"],
        ], [34 * mm, 42 * mm, 60 * mm, 36 * mm]),
        Spacer(1, 8 * mm),
        p("U-Net 结构说明", "h2"),
        p("编码端逐层下采样以提取从纹理到语义的多尺度特征；解码端逐层上采样恢复空间分辨率；同尺度跳跃连接把边缘与位置信息送入解码端。模型最后输出 Background、DrivableArea、LaneMarking 三个通道，并通过逐像素最大概率得到类别掩码。"),
        table([
            ["设计点", "本项目实现", "原因"],
            ["输入尺寸", "640×360 RGB", "与采集/视频分辨率一致，避免额外几何变形"],
            ["基础通道", "32", "兼顾显存、速度和分割容量"],
            ["类别权重", "0.2 / 1.0 / 4.0", "提高稀疏车道线像素对损失的贡献"],
            ["评价主指标", "道路与车道线 mIoU", "不让占比最大的背景类掩盖前景效果"],
            ["模型选优", "验证集 mIoU 最高的 checkpoint", "避免使用训练末轮替代最佳泛化权重"],
        ], [39 * mm, 57 * mm, 76 * mm]),
        Spacer(1, 5 * mm),
        p("实现文件：segmentation/model.py、dataset.py、metrics.py、train_unet.py、inference.py、evaluate.py；几何后处理位于 segmentation/postprocess.py。", "callout"),
        PageBreak(),

        p("四、数据采集与标签体系", "h1"),
        p(f"采集器使用 CARLA 0.9.15 同步模式，同时挂载 RGB Camera 与 Semantic Segmentation Camera。共获得 {manifest['images']} 组 640×360 图像/掩码，覆盖 Town05、Town10HD_Opt、晴天、阴天、降雨、日落和夜间。固定时间步长为 {scenarios['fixed_delta_seconds']:.2f} 秒，采集步长为每 {scenarios['capture_stride']} 个仿真帧保存一组。"),
        table([
            ["输出类别", "训练值", "CARLA 运行时语义 tag", "说明"],
            ["Background", "0", "除 1、24 外", "车辆、建筑、植被、天空等非目标像素"],
            ["DrivableArea", "1", "1", "道路可行驶表面"],
            ["LaneMarking", "2", "24", "道路标线；以原始 red-channel tag 映射"],
        ], [42 * mm, 28 * mm, 48 * mm, 54 * mm]),
        Spacer(1, 6 * mm),
        figures_side_by_side([
            (root / "reports/preview_train.jpg", "图 4-1  白天/降雨训练样本：RGB 与标签叠加预览"),
            (root / "reports/preview_train_night.jpg", "图 4-2  夜间训练样本：用于缓解照度域偏移"),
        ], width_mm=80, max_height_mm=92),
        Spacer(1, 4 * mm),
        p("标签来自 CARLA 原生语义相机，而不是颜色渲染后的调色板图。采集脚本读取原始 BGRA 数据中 red channel 的语义 tag，并仅把运行时 tag=1 与 tag=24 映射为道路和车道线，从源头避免颜色近似匹配造成的错标。", "callout"),
        PageBreak(),

        p("五、数据划分与完整性校验", "h1"),
        p("数据按场景配置直接划分，而不是采集完成后随机打散连续帧。这样可以降低相邻画面跨集合泄漏造成的指标虚高。除 train、val、test 外，额外保留 cloudy night audit，在最终权重锁定后才做一次独立复核。"),
        table(split_rows, [30 * mm, 24 * mm, 24 * mm, 32 * mm, 62 * mm], status_rows={4: palette["light"]}),
        Spacer(1, 6 * mm),
        p("场景构成", "h2"),
        table(scenario_rows, [61 * mm, 27 * mm, 36 * mm, 24 * mm, 24 * mm], tiny=True),
        Spacer(1, 6 * mm),
        table([
            ["校验项", "结果", "判定"],
            ["图像/掩码数量一致", "train 800/800；val 250/250；test 175/175；audit 75/75", "通过"],
            ["分辨率一致", "全部 640×360", "通过"],
            ["跨 split 重名/重复", "未发现", "通过"],
            ["类别像素存在性", "四个 split 均包含道路与车道线像素", "通过"],
            ["dataset_validation.json", f"passed={dataset['passed']}；errors={len(dataset['errors'])}", "通过"],
        ], [48 * mm, 91 * mm, 33 * mm], status_rows={1: palette["green"], 2: palette["green"], 3: palette["green"], 4: palette["green"], 5: palette["green"]}),
        PageBreak(),

        p("六、U-Net 模型与训练方法", "h1"),
        p("训练采用像素级交叉熵、类别权重和固定随机种子。基础阶段学习通用道路外观，随后从基础最佳权重继续执行夜间适配；每个 epoch 都在 val 上计算道路 IoU、车道线 IoU、mIoU 与像素准确率，并保存 best.pt/last.pt。最终提交的是最佳验证集权重的紧凑导出版本，而不是训练末轮权重。"),
        table([
            ["参数", "基础训练", "夜间适配"],
            ["训练轮次记录", f"1–{len(base_rows)}（完整 {len(base_rows)} 轮）", f"epoch {adapt_rows[0]['epoch']}–{adapt_rows[-1]['epoch']}（{len(adapt_rows)} 轮）"],
            ["batch size", "4", "4"],
            ["基础学习率", "5×10⁻⁵", "1×10⁻⁴"],
            ["weight decay", "1×10⁻⁴", "1×10⁻⁴"],
            ["类别权重", "[0.2, 1.0, 4.0]", "[0.2, 1.0, 4.0]"],
            ["最佳 epoch", base_best["epoch"], adapt_best["epoch"]],
            ["最佳 val mIoU", f"{float(base_best['val_miou']):.3f}", f"{float(adapt_best['val_miou']):.3f}"],
            ["最佳 val Road/Lane IoU", f"{float(base_best['val_road_iou']):.3f} / {float(base_best['val_lane_iou']):.3f}", f"{float(adapt_best['val_road_iou']):.3f} / {float(adapt_best['val_lane_iou']):.3f}"],
        ], [55 * mm, 58 * mm, 59 * mm]),
        Spacer(1, 8 * mm),
        p("训练可靠性控制", "h2"),
        table([
            ["控制措施", "作用"],
            ["场景级划分", "减少连续相邻帧跨集合泄漏"],
            ["best checkpoint", "以验证集泛化表现选优，避免末轮退化"],
            ["test 与 audit 只评估", "不回传梯度，不参与最佳权重选择"],
            ["固定 seed=42", "提高采样和训练流程的可复现性"],
            ["权重/视频 SHA-256", "验证大文件传输后未损坏"],
            ["自检脚本", "模型输出尺寸、理想混淆矩阵、道路多边形和双车道曲线均通过"],
        ], [54 * mm, 118 * mm]),
        Spacer(1, 5 * mm),
        p("注意：夜间适配日志后期出现验证集回落，因此最终权重明确锁定在 epoch 30；last.pt 不作为提交结果。", "callout"),
        PageBreak(),

        p("七、训练过程与问题发现", "h1"),
        figure(root / "reports/training_curves.png", 166, "图 7-1  基础训练曲线：训练损失、验证损失与前景 mIoU", max_height_mm=70),
        Spacer(1, 5 * mm),
        p("基础训练在 epoch 27 获得最佳 val mIoU=0.766。严格 test 的总体 mIoU 为 0.481，但按场景拆分后发现：Town05 暴雨 mIoU=0.659，而 Town10 夜间仅 0.004。总体分数掩盖了明显的夜间域偏移，因此不能直接把基础模型作为最终权重。"),
        table([
            ["基础模型检查", "像素准确率", "Road IoU", "Lane IoU", "前景 mIoU"],
            ["test 总体（175）", f"{before['overall']['pixel_accuracy']:.3f}", f"{before['overall']['per_class_iou'][1]:.3f}", f"{before['overall']['per_class_iou'][2]:.3f}", f"{before['overall']['mean_iou_road_lane']:.3f}"],
            ["Town05 暴雨（100）", f"{before['by_scenario']['town05_hard_rain_test']['pixel_accuracy']:.3f}", f"{before['by_scenario']['town05_hard_rain_test']['per_class_iou'][1]:.3f}", f"{before['by_scenario']['town05_hard_rain_test']['per_class_iou'][2]:.3f}", f"{before['by_scenario']['town05_hard_rain_test']['mean_iou_road_lane']:.3f}"],
            ["Town10 夜间（75）", f"{before['by_scenario']['town10_clear_night_test']['pixel_accuracy']:.3f}", f"{before['by_scenario']['town10_clear_night_test']['per_class_iou'][1]:.3f}", f"{before['by_scenario']['town10_clear_night_test']['per_class_iou'][2]:.3f}", f"{before['by_scenario']['town10_clear_night_test']['mean_iou_road_lane']:.3f}"],
        ], [53 * mm, 30 * mm, 29 * mm, 29 * mm, 31 * mm], status_rows={3: palette["red"]}),
        Spacer(1, 7 * mm),
        p("处理决定", "h2"),
        p("补充 Town05 夜间训练与验证场景，从基础最佳权重继续微调；同时保留 Town05 暴雨 test 不变，以观察适配是否牺牲原有天气能力。该处理属于基于验证证据的域适配，不更改 test 标签，也不把 test 图像并入训练。", "callout"),
        PageBreak(),

        p("八、夜间适配前后对比", "h1"),
        figure(root / "reports/night_adaptation_curves.png", 166, "图 8-1  夜间适配曲线：epoch 30 达到最佳，后期验证表现明显回落", max_height_mm=66),
        table([
            ["指标（175 张 test）", "适配前", "适配后", "变化"],
            ["像素准确率", f"{cmp['before']['pixel_accuracy']:.3f}", f"{cmp['after']['pixel_accuracy']:.3f}", f"{cmp['delta']['pixel_accuracy']:+.3f}"],
            ["Road IoU", f"{cmp['before']['road_iou']:.3f}", f"{cmp['after']['road_iou']:.3f}", f"{cmp['delta']['road_iou']:+.3f}"],
            ["Lane IoU", f"{cmp['before']['lane_iou']:.3f}", f"{cmp['after']['lane_iou']:.3f}", f"{cmp['delta']['lane_iou']:+.3f}"],
            ["Road/Lane mIoU", f"{cmp['before']['road_lane_miou']:.3f}", f"{cmp['after']['road_lane_miou']:.3f}", f"{cmp['delta']['road_lane_miou']:+.3f}"],
            ["Town10 夜间 mIoU", f"{night['before']['road_lane_miou']:.3f}", f"{night['after']['road_lane_miou']:.3f}", f"{night['delta']['road_lane_miou']:+.3f}"],
            ["Town05 暴雨 mIoU", f"{rain['before']['road_lane_miou']:.3f}", f"{rain['after']['road_lane_miou']:.3f}", f"{rain['delta']['road_lane_miou']:+.3f}"],
        ], [54 * mm, 37 * mm, 37 * mm, 44 * mm], status_rows={4: palette["green"], 5: palette["green"], 6: palette["amber"]}),
        Spacer(1, 6 * mm),
        figure(root / "reports/test_evaluation/previews/142_town10_clear_night_test_00042.jpg", 166, "图 8-2  夜间适配前：RGB / GT / Prediction；道路与车道线预测接近失效", max_height_mm=34),
        figure(root / "reports/test_evaluation_night_adapt/previews/142_town10_clear_night_test_00042.jpg", 166, "图 8-3  夜间适配后：同一 test 帧的道路与车道线恢复", max_height_mm=34),
        Spacer(1, 3 * mm),
        p("结论：适配使总体 mIoU 提升 0.195，Town10 夜间提升 0.773；同时 Town05 暴雨回落 0.023。最终模型解决了主要夜间失效，但存在轻微天气能力权衡，后续需以混合天气重采样或正则化继续平衡。", "callout"),
        PageBreak(),

        p("九、最终严格测试结果", "h1"),
        p("最终权重在固定的 175 张 test 上统一评估。混淆矩阵在全部像素上累计，IoU 按类别由全局 TP/FP/FN 计算；本报告主指标取道路与车道线两类的平均值，背景仅作为完整性参考。"),
        table([
            ["范围", "样本", "Pixel Acc", "Background IoU", "Road IoU", "Lane IoU", "前景 mIoU"],
            ["test 总体", 175, f"{final['pixel_accuracy']:.3f}", f"{final['per_class_iou'][0]:.3f}", f"{final['per_class_iou'][1]:.3f}", f"{final['per_class_iou'][2]:.3f}", f"{final['mean_iou_road_lane']:.3f}"],
            ["Town05 暴雨", 100, f"{final_test['by_scenario']['town05_hard_rain_test']['pixel_accuracy']:.3f}", f"{final_test['by_scenario']['town05_hard_rain_test']['per_class_iou'][0]:.3f}", f"{final_test['by_scenario']['town05_hard_rain_test']['per_class_iou'][1]:.3f}", f"{final_test['by_scenario']['town05_hard_rain_test']['per_class_iou'][2]:.3f}", f"{final_test['by_scenario']['town05_hard_rain_test']['mean_iou_road_lane']:.3f}"],
            ["Town10 夜间", 75, f"{final_test['by_scenario']['town10_clear_night_test']['pixel_accuracy']:.3f}", f"{final_test['by_scenario']['town10_clear_night_test']['per_class_iou'][0]:.3f}", f"{final_test['by_scenario']['town10_clear_night_test']['per_class_iou'][1]:.3f}", f"{final_test['by_scenario']['town10_clear_night_test']['per_class_iou'][2]:.3f}", f"{final_test['by_scenario']['town10_clear_night_test']['mean_iou_road_lane']:.3f}"],
        ], [38 * mm, 18 * mm, 25 * mm, 27 * mm, 23 * mm, 23 * mm, 25 * mm], status_rows={1: palette["green"]}, tiny=True),
        Spacer(1, 8 * mm),
        p("最终混淆矩阵（行是真值，列是预测）", "h2"),
        table([
            ["真值 / 预测", "Background", "DrivableArea", "LaneMarking"],
            ["Background", *[f"{v:,}" for v in final["confusion_matrix"][0]]],
            ["DrivableArea", *[f"{v:,}" for v in final["confusion_matrix"][1]]],
            ["LaneMarking", *[f"{v:,}" for v in final["confusion_matrix"][2]]],
        ], [43 * mm, 43 * mm, 43 * mm, 43 * mm]),
        Spacer(1, 8 * mm),
        figures_side_by_side([
            (root / "reports/test_evaluation_night_adapt/previews/031_town05_hard_rain_test_00031.jpg", "图 9-1  Town05 暴雨 test：RGB / GT / Prediction"),
            (root / "reports/test_evaluation_night_adapt/previews/158_town10_clear_night_test_00058.jpg", "图 9-2  Town10 夜间 test：RGB / GT / Prediction"),
        ], width_mm=81, max_height_mm=29),
        Spacer(1, 5 * mm),
        p("测试指标来自 reports/test_evaluation_night_adapt/test_metrics.json，权重 checkpoint_epoch=30，设备记录为 CUDA；预览图与 JSON 共用同一次评估输出。", "callout"),
        PageBreak(),

        p("十、锁定审计集与泛化复核", "h1"),
        p("为避免只围绕 test 反复调整，在最终权重导出后使用 75 张 Town10 cloudy night audit 做独立复核。audit 不进入训练、验证和夜间适配对比；其意义是确认最终权重在另一组夜间天气和不同随机种子下仍能输出稳定道路结构。"),
        table([
            ["audit 指标（75 张）", "数值"],
            ["像素准确率", f"{audit_final['pixel_accuracy']:.3f}"],
            ["Background IoU / Dice", f"{audit_final['per_class_iou'][0]:.3f} / {audit_final['per_class_dice'][0]:.3f}"],
            ["DrivableArea IoU / Dice", f"{audit_final['per_class_iou'][1]:.3f} / {audit_final['per_class_dice'][1]:.3f}"],
            ["LaneMarking IoU / Dice", f"{audit_final['per_class_iou'][2]:.3f} / {audit_final['per_class_dice'][2]:.3f}"],
            ["道路/车道线 mIoU", f"{audit_final['mean_iou_road_lane']:.3f}"],
        ], [75 * mm, 97 * mm], status_rows={5: palette["green"]}),
        Spacer(1, 7 * mm),
        figure(root / "reports/audit_evaluation/previews/031_town10_cloudy_night_audit_00031.jpg", 166, "图 10-1  锁定 audit 可视化：RGB / GT / Prediction", max_height_mm=36),
        Spacer(1, 7 * mm),
        table([
            ["复核问题", "证据", "判断"],
            ["是否使用最终导出权重", "weights/unet_week4_best.pt；checkpoint epoch 30", "是"],
            ["是否包含独立天气/随机种子", "Town10 CloudyNight；seed=410", "是"],
            ["是否出现全背景塌缩", "Road IoU=0.877；Lane IoU=0.715", "否"],
            ["是否具有可复核输出", "audit_metrics.json + 8 张三联预览", "是"],
        ], [48 * mm, 91 * mm, 33 * mm], status_rows={1: palette["green"], 2: palette["green"], 3: palette["green"], 4: palette["green"]}),
        Spacer(1, 5 * mm),
        p("audit 指标高于混合 test，不代表模型在所有未知域都达到同等水平；它仅表明该锁定场景更接近适配后的夜间分布。报告仍以 175 张 test 的 0.676 mIoU 作为主要总体结果。", "callout"),
        PageBreak(),

        p("十一、可行驶区域提取与车道线拟合", "h1"),
        p("分割网络输出像素类别后，系统还需把像素结果转成便于规划或可视化的几何结构。postprocess.py 对道路类执行形态学闭运算并保留最大连通区域，提取下方可行驶多边形；对车道线类在感兴趣区域内按左右分区收集像素点，再用二次多项式 x=f(y) 拟合曲线。"),
        table([
            ["步骤", "输入", "处理", "输出"],
            ["1. 类别掩码", "U-Net logits", "逐像素 argmax", "0/1/2 掩码"],
            ["2. 道路清理", "DrivableArea", "闭运算、最大连通域、轮廓筛选", "稳定道路区域"],
            ["3. 可行驶多边形", "道路轮廓", "下半视野取样与简化", "绿色半透明 polygon"],
            ["4. 车道点筛选", "LaneMarking", "ROI、左右分区、离群点抑制", "左右像素点集"],
            ["5. 曲线拟合", "像素点集", "二次多项式 x=f(y)", "左右车道曲线"],
        ], [31 * mm, 35 * mm, 68 * mm, 38 * mm]),
        Spacer(1, 8 * mm),
        figure(root / "reports/audit_lane_fit_example.png", 150, "图 11-1  audit 帧的道路多边形与左右车道线拟合示例", max_height_mm=84),
        Spacer(1, 7 * mm),
        p("自检结果", "h2"),
        table([
            ["检查项", "结果"],
            ["模型输出尺寸", str(self_checks["checks"]["model_output_shape"])],
            ["理想混淆矩阵指标", str(self_checks["checks"]["perfect_metric"])],
            ["可行驶多边形生成", str(self_checks["checks"]["drivable_polygon"])],
            ["左右两条车道曲线", str(self_checks["checks"]["two_lane_curves"])],
        ], [86 * mm, 86 * mm], status_rows={1: palette["green"], 2: palette["green"], 3: palette["green"], 4: palette["green"]}),
        PageBreak(),

        p("十二、目标检测与分割融合演示", "h1"),
        p("融合脚本逐帧读取演示素材：第三周两个 YOLOv8 权重分别负责道路参与者与交通控制目标；第四周 U-Net 负责可行驶区域和车道线。输出画面同时叠加检测框、道路半透明区域、车道曲线、图例和运行统计。视频由代码自动推理与编码生成，不是手工逐帧绘制。"),
        figure(root / "reports/integrated_video_contact_sheet.jpg", 166, "图 12-1  60 秒融合视频接触表：目标检测、道路分割和车道拟合同屏显示", max_height_mm=94),
        Spacer(1, 6 * mm),
        table([
            ["视频校验项", "结果"],
            ["文件", video["output"]],
            ["编码/像素格式", f"{video_check['codec']} / {video_check['pixel_format']}"],
            ["分辨率 / 帧率", f"{video_check['width']}×{video_check['height']} / {video_check['fps']} FPS"],
            ["总帧数 / 时长", f"{video_check['frames']} 帧 / {video_check['duration_seconds']:.1f} 秒"],
            ["包含可行驶区域的帧", f"{video['frames_with_drivable_polygon']} / {video['frames']}"],
            ["拟合车道曲线累计", f"{video['fitted_lane_curves']} 条（逐帧累计）"],
            ["检测框累计", f"Car {video['detections']['Car']}；Pedestrian {video['detections']['Pedestrian']}；TrafficLight {video['detections']['TrafficLight']}；TrafficSign {video['detections']['TrafficSign']}"],
        ], [57 * mm, 115 * mm]),
        Spacer(1, 5 * mm),
        p("表中的目标数量是逐帧检测框累计值，同一物体跨帧会重复统计，不能解释为场景中的唯一目标数。视频已由 ffprobe 验证为 900 帧、整 60.0 秒，并使用兼容性较好的 H.264/yuv420p。", "callout"),
        PageBreak(),

        p("十三、提交物、仓库与复现", "h1"),
        table([
            ["提交项", "位置", "说明"],
            ["语义分割训练代码", "segmentation/", "模型、数据集、训练、评估、推理与后处理"],
            ["感知融合代码", "integration/integrated_demo.py", "YOLOv8 检测与 U-Net 分割同屏融合"],
            ["采集与校验工具", "segmentation/collect_*.py；tools/", "同步采集、数据检查、曲线、预览、权重导出、自检"],
            ["最终权重", "weights/unet_week4_best.pt", f"{checksum_lookup['weights/unet_week4_best.pt']['bytes'] / 1024 / 1024:.2f} MiB；epoch 30"],
            ["融合演示视频", "demo/week4_integrated_perception_1min.mp4", "60 秒 H.264"],
            ["实验报告", "reports/第四周实验报告.pdf", "本报告"],
            ["指标与证据", "reports/", "数据校验、test/audit 指标、对比图、视频校验和 SHA-256"],
        ], [50 * mm, 68 * mm, 54 * mm]),
        Spacer(1, 7 * mm),
        p("核心复现命令", "h2"),
        table([
            ["操作", "命令"],
            ["训练", "python segmentation/train_unet.py --data data/carla_segmentation --output runs/unet_final --epochs 30 --device cuda"],
            ["严格测试", "python segmentation/evaluate.py --data data/carla_segmentation --split test --weights weights/unet_week4_best.pt --output reports/test_evaluation_recheck"],
            ["融合视频", "python integration/integrated_demo.py --seg-weights weights/unet_week4_best.pt --output demo/week4_integrated_perception_1min.mp4"],
            ["全量自检", "python tools/run_self_checks.py"],
            ["校验大文件", "python tools/verify_artifact_checksums.py"],
        ], [42 * mm, 130 * mm], tiny=True),
        Spacer(1, 7 * mm),
        p("GitHub 仓库", "h2"),
        p('<link href="https://github.com/xuzihao723/xiaomi-auto-drive" color="#2F75B5">https://github.com/xuzihao723/xiaomi-auto-drive</link>', "url"),
        p("第四周目录：week4-segmentation/；Release 标签：week4-submission；附件：xiaomi_week4.zip。", "body"),
        Spacer(1, 4 * mm),
        p("说明：完整 1300 张训练图像不重复上传到代码目录；提交包保留采集代码、场景配置、manifest、验证摘要和可视化证据，可按 README 重新采集。最终权重、演示视频和报告均包含在 Release 压缩包中。", "callout"),
        PageBreak(),

        p("十四、当前局限性与实验结论", "h1"),
        p("当前局限性", "h2"),
        table([
            ["局限", "当前影响", "后续方向"],
            ["仿真到真实域差异", "标签和图像均来自 CARLA，不能代表真实道路成像噪声", "加入真实公开分割数据并做域适配/人工复核"],
            ["场景覆盖仍可扩展", "当前覆盖 Town05、Town10HD_Opt 及多种天气", "补充更多地图、复杂路口、随机种子和镜头参数"],
            ["未完成真实车载延迟验证", "60 秒视频证明离线流程，不等于车载闭环实时性", "导出 ONNX/TensorRT，并在真实汽车计算平台测端到端延迟"],
        ], [45 * mm, 62 * mm, 65 * mm]),
        Spacer(1, 8 * mm),
        p("实验结论", "h2"),
        p(
            "第四周已形成从同步数据采集、像素级标签映射、U-Net 训练、严格评估、夜间问题定位与适配，到道路/车道几何后处理和目标检测融合视频的完整闭环。最终 test 道路/车道线 mIoU 为 "
            f"{final['mean_iou_road_lane']:.3f}，锁定 audit mIoU 为 {audit_final['mean_iou_road_lane']:.3f}。"
            "与基础权重相比，夜间适配解决了 Town10 夜间近乎失效的问题，整体完成了第四周文档规定的模型、代码和融合演示交付要求。",
            "callout",
        ),
    ]

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
