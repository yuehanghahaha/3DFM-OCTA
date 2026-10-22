# 服务器上的三维小波 FM：200 轮 tmux 训练

服务器：`research-server.example.org`。

代码已经放到你的目录：

```text
/srv/oct_research/fm3d_wavelet_200
```

使用原基线的 `lyh_fm3d` 环境及同一份 3M 数据划分。启动脚本已经写好环境和数据绝对路径，无需重新激活环境或修改数据位置。

2026-10-21 已在 **GPU 1** 启动正式训练，tmux 会话为 **`lyh_fm3d_wavelet200`**。已确认训练进程存在并输出第一批损失。不要再次执行新训练启动命令。

## 查看训练

连接服务器后进入训练窗口：

```bash
tmux attach -t lyh_fm3d_wavelet200
```

退出窗口但保持训练：先按 **Ctrl+B**，松开，再按 **D**。直接断开 SSH 也不会终止 tmux 中的训练。训练窗口中按 **Ctrl+C** 会中断训练。

也可以不进入 tmux，直接查看实时日志：

```bash
tail -f /srv/oct_research/fm3d_wavelet_200/logs/train_3M_200.log
```

在单独的 `tail -f` 日志查看窗口中按 `Ctrl+C`，只会退出查看，训练继续。

## 实验参数

- 3D Haar 小波域条件 FM，从头训练。
- 总计 200 轮，每轮 200 个训练批次，batch=1。
- 原始体块 `256×64×64`，小波后每个子带 `128×32×32`。
- base_channels=16，学习率 `2e-4`，AMP 开启。
- 每 5 轮进行完整验证，Euler 8 步，滑窗重叠 50%。
- 按最低验证 MAE 保存 `best.pt`，每轮保存 `latest.pt`。
- 每 5 轮保存当前权重及截至该轮的最佳权重，包括 `best_through_0100.pt` 和 `best_through_0200.pt`。如果验证 MAE 没有刷新，多个 `best_through_*.pt` 的模型会相同。

原训练进程启动时的保存间隔是 50 轮。2026-10-21 已新增 tmux 的 `snapshots` 窗口，在不中断训练的情况下从第 185 轮起每 5 轮补存两类权重；第 180 轮已准确补存。可用 `tmux select-window -t lyh_fm3d_wavelet200:snapshots` 查看保存窗口，或查看 `logs/snapshot_watch.log`。启动代码已改成每 5 轮保存，若训练意外中断后续训，原训练程序也会直接按 5 轮保存。

之前保存过的最佳模型若恰好与历史上某个五轮节点一致，已依据验证记录准确补齐。第 5–40、60、65、70 轮当时的最佳权重已被旧规则覆盖，无法恢复；不能用后来的模型冒充。记录见 `verification/backfill_best_5.json`。

## 断点恢复

若训练意外停止，确认原训练进程已经结束并且已有 `latest.pt`，再回到 tmux 窗口执行：

```bash
cd /srv/oct_research/fm3d_wavelet_200
bash -o pipefail -c 'CUDA_VISIBLE_DEVICES=1 bash run_3M_200.sh --resume 2>&1 | tee -a logs/train_3M_200.log'
```

目标是总计 200 轮，不是额外增加 200 轮。

## 服务器路径

| 内容 | 路径 |
|---|---|
| 启动脚本 | `/srv/oct_research/fm3d_wavelet_200/run_3M_200.sh` |
| Python 环境 | `/srv/oct_research/conda_envs/lyh_fm3d/bin/python` |
| 固定数据清单 | `/srv/oct_research/OCTA500_common_v1/split_3M_z256.csv` |
| 新实验输出 | `/srv/oct_research/fm3d_wavelet_200/runs/fm3d_haar3d_3M_200` |
| 已完成的基线 | `/srv/oct_research/fm3d_ubuntu/runs/fm3d_3M_run01` |
| 服务器测试日志 | `/srv/oct_research/fm3d_wavelet_200/logs/server_unit_tests.log` |
| 服务器显存测试 | `/srv/oct_research/fm3d_wavelet_200/verification/server_3090_benchmark.json` |
| 五轮快照保存日志 | `/srv/oct_research/fm3d_wavelet_200/logs/snapshot_watch.log` |

tmux 能承受 SSH 断开；服务器重启会中断进程，届时需要从已保存的 checkpoint 恢复。
