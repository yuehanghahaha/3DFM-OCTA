# 基线 3D FM：从第 100 轮严格续训到第 200 轮

用户要求每 10 轮保留该区间内验证 MAE 最低的一份权重，因此 101–200 轮**每轮都做完整 3M 验证**。每个不重叠的区间（101–110、111–120、……、191–200）形成 `best_window_起始_结束.pt` 和同名 JSON 记录。`best.pt` 始终是从第 1 轮至当前轮的全程最佳，`latest.pt` 是最后完成的轮次。模型选择指标为 10 位验证患者的完整三维体积、患者宏平均 MAE；测试集不参与选择。

原来的 101–133 轮曾按每 5 轮验证运行。由于缺少其余轮次的模型及指标，不能从中恢复严格的每 10 轮最佳；该运行已保存在服务器的 `/srv/oct_research/fm3d_ubuntu/runs/fm3d_3M_run01_superseded_val5_20261021`。正式运行目录从原始第 100 轮检查点恢复后重新训练。原第 100 轮检查点、第 90 轮最佳权重、配置、历史和训练代码均保存在 `/srv/oct_research/fm3d_ubuntu/verification/pre_resume_20261021`。

2026-10-21 按用户要求清理了上述作废分支的 5 份权重、旧启动脚本和一份重复日志，
合计 126,662,113 bytes（120.79 MiB）。旧目录只保留配置、历史与验证指标，历史记录中
引用的旧检查点文件已删除。当前基线和投影实验的权重均未删除；第 100 轮恢复点与
第 90 轮原始最佳权重也继续保留。逐文件 SHA256、删除清单及训练进程核对记录见
服务器 `verification/cleanup_superseded_20261021/cleanup.json`，本地副本为同目录 `cleanup.json`。

服务器使用 `tmux` 会话 `lyh_fm3d_baseline200`，窗口 `train`，物理 GPU 1。启动脚本为 `/srv/oct_research/fm3d_ubuntu/resume_to_200.sh`，日志为 `/srv/oct_research/fm3d_ubuntu/logs/fm3d_3M_run01_resume_101_200_val_each_epoch_20261021.log`。正式运行目录为 `/srv/oct_research/fm3d_ubuntu/runs/fm3d_3M_run01`。
