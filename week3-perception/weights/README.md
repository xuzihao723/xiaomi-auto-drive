# 模型权重

模型权重属于二进制训练产物，不放入 Git 主分支。完整提交包中包含：

- `best.pt`：验证表现最佳的四类别 YOLOv8n 权重，用于评估和推理。
- `last.pt`：训练结束前最后保存的 checkpoint，用于恢复训练或对比。

下载并解压 [xiaomi_week3.zip](https://github.com/xuzihao723/xiaomi-auto-drive/releases/download/week3-submission/xiaomi_week3.zip) 后即可获得这两个文件。

