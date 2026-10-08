# 第三周模型权重

为避免 Git 主分支重复存放二进制模型，完整权重和部署导出文件放在第三周 Release 压缩包中：

https://github.com/xuzihao723/xiaomi-auto-drive/releases/download/week3-submission/xiaomi_week3.zip

压缩包内包括：

- `road_user_best.pt`、`road_user_last.pt`
- `traffic_control_best.pt`、`traffic_control_last.pt`
- `traffic_control_best_fp32.onnx`
- `traffic_control_best_fp16.engine`

导出清单见 `weights/exported/export_summary.json`。
