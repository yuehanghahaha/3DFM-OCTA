# 三项诊断：服务器存储位置

日期：2026-10-21，上海时间。三项实验、附加同主干监督对照、推理后分层评价均已完成。

服务器：`research-server.example.org`。本次计算限定物理 GPU0；原基线和投影模型在 GPU1 的训练保持运行。

总目录：`/srv/oct_research/three_diagnostics_20261021`。

| 内容 | 相对于总目录的位置 |
|---|---|
| 基线第106轮固定权重 | `snapshots/voxel.pt` |
| 小波第195轮固定权重 | `snapshots/haar.pt` |
| 实际生产代码的冻结副本 | `source/voxel/fm3d/`、`source/haar/fm3d/` |
| 8个固定诊断体块 | `inputs/` |
| 小型纯监督U-Net，2400次成功更新 | `experiment1_supervised/update_2400.pt` |
| 原主干纯监督，2400次成功更新 | `experiment1_same_backbone/update_2400.pt` |
| 三项主实验日志 | `run.log` |
| 同主干监督日志 | `same_backbone.log` |
| 推理后ILM–OPL评价 | `posthoc_layers/result.json`、`layer_view.log` |
| 冻结协议与输入哈希 | `protocol.json`、`input_audit.json` |
| 完成状态与最终复核 | `status.json`、`final_audit.json` |

两份监督模型都仅训练一个固定体块，不应当作全数据训练模型或替代正式模型。`experiment1_same_backbone` 是固定t=0的直接回归；它没有学好多步速度场，不能直接改为ODE采样。

原基线仍在续训，生产目录的 `best.pt` 之后可能正常更新；本报告及复核始终使用上述固定副本。

本地保存了所有小型指标、曲线、图像及相应脚本。权重、源代码副本和三维输入保留在服务器。主程序需要这些服务器文件，因此本地的小型结果包不是离线训练环境。

阅读入口：[诊断报告](诊断报告.md)。
