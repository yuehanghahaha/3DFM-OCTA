# 3D 流匹配 + 三维 Haar 小波 + Z 轴投影约束

独立组别：`04_fm3d_wavelet_zproj`。完整共同协议见上一级 `README.md`。

表示：`haar3d`；损失权重 FM=1、Z 投影=0.1、图像重建=0.0。

从实验根目录独立启动：

```bash
python run.py --arm 04_fm3d_wavelet_zproj --stage one_case --resume
python run.py --arm 04_fm3d_wavelet_zproj --stage few_case --resume
```

两阶段各从头初始化，结果分别保存到 `one_case/` 和 `few_case/`。每阶段含 `latest.pt`、`best_validation.pt`、`steps.jsonl`、`history.jsonl`、`final_metrics.json` 与 `predictions/`。
