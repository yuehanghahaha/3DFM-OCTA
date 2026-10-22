# 服务器实验已完成

主机：`research-server.example.org`；物理 GPU：`0`；显卡 RTX 3090 24 GiB。

独立目录：`/srv/oct_research/fm3d_validation_20261021`。

五组的 `one_case/` 与 `few_case/` 共 10 个运行全部完成，6,000 次成功更新、AMP 跳过 0 次。
最终审计逐一核对了 125 份三维预测，全部通过。
本轮训练进程已退出，GPU0 已释放；原基线与旧投影训练仍在 GPU1 运行。

## 保存了什么

- 各组各阶段的 `latest.pt` 与 `best_validation.pt`，共 20 份检查点，包含模型、优化器、AMP 状态和配置签名。
- 各组各阶段 `predictions/` 下共 125 份三维 NPZ 与对应 PNG。
- 初始指标、分项训练损失、评估记录、最终逐病例/逐患者指标、冻结训练顺序和时间序列。

上述大体积检查点与三维 NPZ 保留在服务器。本地同步了所有小型指标、训练日志、预测 PNG、对照报告和学习曲线。
[完整服务器文件清单](server_artifacts.json) 记录 145 份大文件的准确路径、大小和 SHA256，可按需取回。

## 原地恢复

在服务器的这个实验根目录中运行，例如：

```bash
/srv/oct_research/conda_envs/lyh_fm3d/bin/python run.py --arm 05_fm3d_wavelet_zproj_reconstruction --stage few_case --resume
```

已完成的相同预算会提示完成并跳过。更改实验预算、损失、代码或数据应创建新运行版本。
本地报告镜像没有 `.pt`，不能直接把它当作包含检查点的恢复目录。
启动时使用过的持久会话名为 `lyh_fm3d_validate5_20261021`；完成后该会话已退出，无需继续挂接。

## 新增 FlowOCT 与诊断对照

同一服务器根目录的 `06_flowoct/` 与 `diagnostics/` 已完成后续验证，共新增 5,600 次实际优化更新。
延长实验从原 400 次检查点续训，只把后续 1,200 次计入新增预算。原五组 20 份权重及训练代码哈希核对未变。

- `06_flowoct/one_case/`、`few_case/`：400/800 次更新，分别导出原生三步与统一八步结果。
- `diagnostics/time_scale_1/`：同 400 次时间序列，只把时间编码尺度由 1000 改为 1。
- `diagnostics/start_only_mse/`：纯 OCT 起点直接拟合 1600 次，作为网络记忆能力检查。
- `diagnostics/uniform_fm_1600/`、`flowoct_1600/`：各自原方法的单病例训练延长到总计 1600 次。

FlowOCT 环境为 `/srv/oct_research/conda_envs/lyh_flowoct/bin/python`。
新增 8 份检查点、95 份三维预测保存在服务器；[补充文件清单](flowoct_followup_artifacts.json) 记录全部 103 份大文件的准确路径、大小和 SHA256。
本地已同步所有新增小型指标、日志、PNG 和报告，仍不包含这些 `.pt` 与三维预测 NPZ。
[新增审计](verification/flowoct_followup_audit.json) 验证了全部预测的真值一致性、范围及误差记录。

最新入口：[排查结论](reports/FLOWOCT_DIAGNOSIS.md)、[同数据图表](reports/flowoct_comparison.html)。
