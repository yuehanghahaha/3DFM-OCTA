# Z 轴投影约束 + 96 横向块：200 轮训练

2026-10-21 已在服务器 **GPU 1** 从头启动，计划总计 **200 轮**。
状态为训练进行中，尚无最终性能结论。启动及检查快照见 `launch_status.json`。

## 实验配置

| 项目 | 本次设置 |
|---|---|
| 数据 | 原 3M 公共划分，140 训练 / 10 验证 / 50 测试 |
| 训练块 [Z,Y,X] | **256×96×96**；保持完整 256 深度 |
| 网络 | 原体素条件 3D U-Net，base16，2,097,521 参数 |
| 损失 | 原 FM 速度 MSE + **0.1 × Z 均值投影 L1** |
| 投影监督位置 | `OCT + v(OCT,0,OCT)` 的单步终点估计，投影分支只输入 OCT |
| 训练预算 | 从第 1 轮起，共 200 轮，每轮 200 批次，batch1 |
| 优化器 | AdamW，学习率 2e-4，weight decay 0，clip1，AMP |
| 验证 | 每轮全部 10 例，8 步 Euler，50% 重叠全卷滑窗 |
| 记录 | 三维 MAE/PSNR、Z 投影 MAE/PSNR/PCC，两项训练损失与总损失 |
| 保存 | 每轮 latest、验证 MAE 最优 best、每 5 轮当前及累计最佳快照 |

均值投影沿 `[B,C,Z,Y,X]` 的 `dim=2` 计算，不使用额外层标注。
投影分支的单步预测不裁剪，保持梯度；它不是对 8 步积分结果直接反传。
0.1 为初始实验权重，详见项目 `fm3d/PROJECTION.md`。

## 查看训练

服务器为 `research-server.example.org`。连接后执行：

```bash
tmux attach -t lyh_fm3d_zproj96_200
```

按 Ctrl+B、松开后按 D，可离开窗口并保留训练；SSH 断开也不影响 tmux 训练。

直接查看日志：

```bash
tail -f /srv/oct_research/fm3d_zproj_96_200/logs/train_3M_200.log
```

## 服务器位置

| 内容 | 路径 |
|---|---|
| 代码与启动脚本 | `/srv/oct_research/fm3d_zproj_96_200` |
| 权重、配置与逐轮指标 | `/srv/oct_research/fm3d_zproj_96_200/runs/fm3d_zproj_3M_96_200` |
| 固定数据清单 | `/srv/oct_research/OCTA500_common_v1/split_3M_z256.csv` |
| Python 环境 | `/srv/oct_research/conda_envs/lyh_fm3d/bin/python` |
| 启动器 | `/srv/oct_research/fm3d_zproj_96_200/run_3M_200.sh` |

这次使用独立代码和输出目录，原基线继续运行。两者共用此前使用的 GPU 1。
启动时两进程合计显存约 11.55 GiB，GPU 0 未用于此实验。

## 已完成检查

- 本地 22 项测试全部通过，包括投影轴、源输入、梯度、损失记录、配置冻结、恢复与连续训练逐参数一致性。
- 服务器 21 项通过、1 项可选 PyWavelets 对照因未安装该包跳过。
  初次部署漏带旧数据审计测试需要的辅助文件；已补入原文件，单项重跑通过。
  原失败日志和重跑日志均保留，不影响正式训练源代码。
- 服务器逐项校验了 31 个部署源文件及代码压缩包的 SHA256。
- RTX 3090 实测：峰值 allocated 8,108,194,304 bytes（7.55 GiB），
  reserved 9,258,926,080 bytes（8.62 GiB），4 次含预热更新均成功。
- 真实训练病例 10301 的 256×96×96 体块已完成 3 次更新及 8 步推理，
  无 AMP 跳过、无非有限结果。这仅验证运行，不是泛化性能评价。

## 中断后恢复

若训练确实已经停止，且运行目录已有 `latest.pt`，在上述 tmux 窗口中执行：

```bash
cd /srv/oct_research/fm3d_zproj_96_200
bash train_in_tmux.sh --resume
```

恢复目标仍是总计 200 轮。已经运行时不要重复启动。
`best.pt` 仍以三维验证 MAE 选择；投影指标和五轮快照用于进一步分析。
本次同时增加投影约束并扩大横向块，性能变化需要后续单项对照才能归因。
