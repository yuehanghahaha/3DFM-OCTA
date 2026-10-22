# 3D 流匹配基线

独立组别：`01_fm3d`。完整共同协议见上一级 `README.md`。

表示：`voxel`；损失权重 FM=1、Z 投影=0.0、图像重建=0.0。

从实验根目录独立启动：

```bash
python run.py --arm 01_fm3d --stage one_case --resume
python run.py --arm 01_fm3d --stage few_case --resume
```

两阶段各从头初始化，结果分别保存到 `one_case/` 和 `few_case/`。每阶段含 `latest.pt`、`best_validation.pt`、`steps.jsonl`、`history.jsonl`、`final_metrics.json` 与 `predictions/`。
